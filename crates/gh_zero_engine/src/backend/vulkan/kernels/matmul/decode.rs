/*
 * gh_zero_engine — Vulkan decode matmul dispatch
 * Selects the retained per-format FP16-output GEMV and records one projection.
 * It owns no environment settings, batched policy, pipeline, or buffer lifetime.
 */

use ash::vk;

use super::{project, trace};
use crate::backend::vulkan::buffers::{GpuBuffer, WeightFormat};
use crate::backend::vulkan::device::Device;
use crate::backend::vulkan::pipeline::{Kernel, PipelineRegistry};

pub(crate) fn matmul(
    dev: &Device,
    reg: &PipelineRegistry,
    cmd: vk::CommandBuffer,
    out: &GpuBuffer,
    a: &GpuBuffer,
    w: &GpuBuffer,
    in_dim: u32,
    out_dim: u32,
) {
    let kernel = match w.quant {
        WeightFormat::F16 => Kernel::MatmulF16,
        WeightFormat::Q4K => Kernel::MatmulQ4KTiled,
        WeightFormat::Q6K => Kernel::MatmulQ6K,
        WeightFormat::Q5K => Kernel::MatmulQ5K,
    };
    trace::log_path_once(kernel);
    project(dev, reg, cmd, kernel, out, a, w, in_dim, out_dim);
}
// The parity test needs the CPU Q4_K oracle (`compute::matmul`, cpu-gated) and the
// Vulkan device, so it only compiles when both backends are present (the hybrid
// build); a vulkan-only test build skips it.
#[cfg(all(test, any(feature = "cpu", feature = "vulcan-hybrid")))]
mod tests {
    use crate::backend::Backend;
    use crate::backend::cpu::buffer::{f16_to_f32, f32_to_f16_bytes};
    use crate::backend::cpu::compute;
    use crate::backend::cpu::{CpuBuffer, CpuFormat};
    use crate::backend::vulkan::VulkanBackend;
    use crate::backend::vulkan::buffers::{GpuBuffer, WeightFormat};

    // Parity: the tiled Q4_K kernel (routed by `matmul` for Q4_K) matches the CPU
    // Q4_K oracle (`compute::matmul`) on random super-blocks. out_dim is NOT a
    // multiple of 64 (boundary guard) and in_dim spans 2 super-blocks. Skips when no
    // Vulkan device is present (headless CI); the GPU gate validates the kernel.
    #[test]
    fn tiled_q4k_matches_cpu_oracle() {
        let backend = match VulkanBackend::bare() {
            Ok(b) => b,
            Err(_) => {
                eprintln!("tiled Q4_K parity skipped: no Vulkan device");
                return;
            }
        };
        let in_dim = 512usize; // 2 Q4_K super-blocks per row
        let out_dim = 70usize; // not a multiple of 64 → exercises the row boundary

        // Random Q4_K weight bytes: 144 bytes per super-block, 2 per row.
        let nsb = in_dim / 256;
        let mut seed = 0x9e3779b9u32;
        let mut rng = || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            seed
        };
        let mut qbytes: Vec<u8> = (0..out_dim * nsb * 144)
            .map(|_| (rng() >> 16) as u8)
            .collect();
        // Random bytes give pathological f16 super-block scales (d/dmin) that overflow
        // f16 in the result. Pin them to small realistic magnitudes so the dequantized
        // weights — and the dot — stay inside the f16 range the kernels store.
        let f16le = |v: f32| -> [u8; 2] {
            let b = f32_to_f16_bytes(&v.to_le_bytes());
            [b[0], b[1]]
        };
        for blk in (0..qbytes.len()).step_by(144) {
            qbytes[blk..blk + 2].copy_from_slice(&f16le(0.03));
            qbytes[blk + 2..blk + 4].copy_from_slice(&f16le(0.01));
        }
        let a: Vec<f32> = (0..in_dim)
            .map(|_| ((rng() >> 8) as f32 / u32::MAX as f32) - 0.5)
            .collect();

        // CPU oracle over the same bytes.
        let cpu_w = CpuBuffer::from_bytes(qbytes.clone(), CpuFormat::Q4_K);
        let want = compute::matmul(&cpu_w, &a, in_dim, out_dim);

        // GPU: upload weight + FP16 activation, run the (tiled) matmul, read back.
        let mut w = GpuBuffer::alloc(&backend.dev, qbytes.len() as u64, false).expect("w");
        w.quant = WeightFormat::Q4K;
        backend.upload_bytes(&w, &qbytes).expect("upload w");
        let a_f16 = f32_to_f16_bytes(&a.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>());
        let in_buf = backend.alloc_buffer((in_dim * 2) as u64).expect("in");
        backend.upload_bytes(&in_buf, &a_f16).expect("upload a");
        let out_buf = backend.alloc_buffer((out_dim * 2) as u64).expect("out");
        let enc = backend.begin().expect("begin");
        backend.matmul(&enc, &out_buf, &in_buf, &w, in_dim as u32, out_dim as u32);
        backend.submit(enc).expect("submit");
        let raw = backend.read_bytes(&out_buf, out_dim * 2).expect("read");
        let got: Vec<f32> = raw
            .chunks_exact(2)
            .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect();

        // FP16 activation + FP16 result rounding ⇒ compare within the I2 tolerance.
        for o in 0..out_dim {
            let (g, w) = (got[o], want[o]);
            let ok = (g - w).abs() <= 1e-2 || (g - w).abs() <= 1e-3 * w.abs().max(1.0);
            assert!(ok, "row {o}: got={g} want={w}");
        }
        w.destroy(&backend.dev);
        in_buf.destroy(&backend.dev);
        out_buf.destroy(&backend.dev);
    }

    // The mmvq Q4_K decode GEMV (quant A → int8 Q8_1, then the dp4a GEMV) produces
    // FINITE logits within tolerance of the CPU Q4_K/f32 oracle. Drives the two
    // kernels directly (no forward) so it isolates the kernel from the integration.
    // Skips when no Vulkan device is present, OR when the device exposes no dp4a
    // (the kernels are then not built and decode keeps the float GEMV).
    #[test]
    fn mmvq_q4k_matches_cpu_oracle() {
        use crate::backend::vulkan::pipeline::{Kernel, dispatch};

        let backend = match VulkanBackend::bare() {
            Ok(b) => b,
            Err(_) => {
                eprintln!("mmvq parity skipped: no Vulkan device");
                return;
            }
        };
        let dev = &backend.dev;
        if !dev.dp4a {
            eprintln!("mmvq parity skipped: device has no integer dot product (dp4a)");
            return;
        }
        let reg = &backend.reg;
        let in_dim = 512usize; // 2 Q4_K super-blocks per row, 16 Q8_1 blocks
        let out_dim = 70usize; // not a multiple of 64 → exercises the row boundary

        let nsb = in_dim / 256;
        let mut seed = 0x6d_6d_76_71u32;
        let mut rng = || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            seed
        };
        let mut qbytes: Vec<u8> = (0..out_dim * nsb * 144)
            .map(|_| (rng() >> 16) as u8)
            .collect();
        let f16le = |v: f32| -> [u8; 2] {
            let b = f32_to_f16_bytes(&v.to_le_bytes());
            [b[0], b[1]]
        };
        for blk in (0..qbytes.len()).step_by(144) {
            qbytes[blk..blk + 2].copy_from_slice(&f16le(0.03)); // d
            qbytes[blk + 2..blk + 4].copy_from_slice(&f16le(0.01)); // dmin
        }
        let a: Vec<f32> = (0..in_dim)
            .map(|_| ((rng() >> 8) as f32 / u32::MAX as f32) - 0.5)
            .collect();

        let cpu_w = CpuBuffer::from_bytes(qbytes.clone(), CpuFormat::Q4_K);
        let want = compute::matmul(&cpu_w, &a, in_dim, out_dim);

        // GPU buffers use the retained FP16 activation/output interface.
        let a_bytes = f32_to_f16_bytes(
            &a.iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<u8>>(),
        );
        let ain = backend.alloc_buffer((in_dim * 2) as u64).expect("ain");
        backend.upload_bytes(&ain, &a_bytes).expect("upload a");
        let qs = backend.alloc_buffer(in_dim as u64).expect("qs"); // in_dim/4 uints = in_dim bytes
        let nblocks = in_dim / 32;
        let ds = backend.alloc_buffer((nblocks * 2 * 4) as u64).expect("ds");
        let mut w = GpuBuffer::alloc(&backend.dev, qbytes.len() as u64, false).expect("w");
        w.quant = WeightFormat::Q4K;
        backend.upload_bytes(&w, &qbytes).expect("upload w");
        let out = backend.alloc_buffer((out_dim * 2) as u64).expect("out");

        let cmd = dev.begin_commands().expect("cmd");
        let qpush = (in_dim as u32).to_le_bytes();
        dispatch(
            dev,
            reg,
            cmd,
            Kernel::QuantAQ8F16,
            &[
                (ain.buffer, ain.offset, ain.size),
                (qs.buffer, qs.offset, qs.size),
                (ds.buffer, ds.offset, ds.size),
            ],
            &qpush,
            (nblocks as u32).div_ceil(64),
        );
        let mut mpush = Vec::with_capacity(8);
        mpush.extend_from_slice(&(in_dim as u32).to_le_bytes());
        mpush.extend_from_slice(&(out_dim as u32).to_le_bytes());
        dispatch(
            dev,
            reg,
            cmd,
            Kernel::MatmulQ4KMmvqF16Out,
            &[
                (qs.buffer, qs.offset, qs.size),
                (ds.buffer, ds.offset, ds.size),
                (w.buffer, w.offset, w.size),
                (out.buffer, out.offset, out.size),
            ],
            &mpush,
            (out_dim as u32).div_ceil(64),
        );
        dev.submit_wait(cmd).expect("submit");

        let raw = backend.read_bytes(&out, out_dim * 2).expect("read");
        let got: Vec<f32> = raw
            .chunks_exact(2)
            .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect();

        // int8 Q8_1 activations round each element to ~1/127, so a length-512 dot carries a
        // wider error than the f16 path; assert FINITE (the historical mmvq bug was inf) plus
        // a combined Q4_K+int8 tolerance (abs ≤ 1.5e-1 OR rel ≤ 3%).
        for o in 0..out_dim {
            let (g, w) = (got[o], want[o]);
            assert!(g.is_finite(), "row {o}: non-finite logit {g}");
            let ok = (g - w).abs() <= 1.5e-1 || (g - w).abs() <= 3e-2 * w.abs().max(1.0);
            assert!(ok, "row {o}: got={g} want={w}");
        }
        ain.destroy(dev);
        qs.destroy(dev);
        ds.destroy(dev);
        w.destroy(dev);
        out.destroy(dev);
    }
}
