/*
 * graph_horizon_engine — Vulkan decode matmul dispatch
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
    let (kernel, output_rows) = match w.quant {
        WeightFormat::F16 => (Kernel::MatmulF16, 64),
        WeightFormat::Q4K => (Kernel::MatmulQ4KTiled, 16),
        WeightFormat::Q6K => (Kernel::MatmulQ6K, 64),
        WeightFormat::Q5K => (Kernel::MatmulQ5K, 64),
    };
    trace::log_path_once(kernel);
    project(
        dev,
        reg,
        cmd,
        kernel,
        out,
        a,
        w,
        in_dim,
        out_dim,
        output_rows,
    );
}
// The parity test needs the CPU Q4_K oracle (`compute::matmul`, cpu-gated) and the
// Vulkan device, so it only compiles when both backends are present (the hybrid
// build); a vulkan-only test build skips it.
#[cfg(test)]
#[cfg(any(feature = "cpu", feature = "vulkan-hybrid"))]
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

    #[test]
    fn parallel_q6k_projections_match_cpu_oracle() {
        let backend = match VulkanBackend::bare() {
            Ok(backend) => backend,
            Err(_) => {
                eprintln!("parallel Q6_K parity skipped: no Vulkan device");
                return;
            }
        };
        let in_dim = 512usize;
        let out_dim = 70usize;
        let nsb = in_dim / 256;
        let mut seed = 0x71_36_6b_21u32;
        let mut rng = || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            seed
        };
        let mut qbytes: Vec<u8> = (0..out_dim * nsb * 210)
            .map(|_| (rng() >> 16) as u8)
            .collect();
        let f16le = |value: f32| {
            let bytes = f32_to_f16_bytes(&value.to_le_bytes());
            [bytes[0], bytes[1]]
        };
        for block in qbytes.chunks_exact_mut(210) {
            for (index, scale) in block[192..208].iter_mut().enumerate() {
                *scale = (index % 7 + 1) as u8;
            }
            block[208..210].copy_from_slice(&f16le(0.01));
        }
        let activation: Vec<f32> = (0..in_dim)
            .map(|_| ((rng() >> 8) as f32 / u32::MAX as f32) - 0.5)
            .collect();
        let want = compute::matmul(
            &CpuBuffer::from_bytes(qbytes.clone(), CpuFormat::Q6_K),
            &activation,
            in_dim,
            out_dim,
        );

        let activation_bytes = f32_to_f16_bytes(
            &activation
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<_>>(),
        );
        let input = backend
            .alloc_buffer(activation_bytes.len() as u64)
            .expect("input");
        backend
            .upload_bytes(&input, &activation_bytes)
            .expect("upload input");
        let mut weights =
            GpuBuffer::alloc(&backend.dev, qbytes.len() as u64, false).expect("weights");
        weights.quant = WeightFormat::Q6K;
        backend
            .upload_bytes(&weights, &qbytes)
            .expect("upload weights");
        let output = backend.alloc_buffer((out_dim * 2) as u64).expect("output");
        let encoder = backend.begin().expect("begin");
        backend.matmul(
            &encoder,
            &output,
            &input,
            &weights,
            in_dim as u32,
            out_dim as u32,
        );
        backend.submit(encoder).expect("submit");
        let raw = backend
            .read_bytes(&output, out_dim * 2)
            .expect("read output");
        for (index, bytes) in raw.chunks_exact(2).enumerate() {
            let got = f16_to_f32(u16::from_le_bytes([bytes[0], bytes[1]]));
            let reference = want[index];
            assert!(got.is_finite(), "row {index}: non-finite result");
            assert!(
                (got - reference).abs() <= 5e-2
                    || (got - reference).abs() <= 5e-3 * reference.abs().max(1.0),
                "row {index}: got={got} want={reference}"
            );
        }

        let logits = backend.alloc_buffer((out_dim * 4) as u64).expect("logits");
        let encoder = backend.begin().expect("begin logits");
        backend.logits(
            &encoder,
            &logits,
            &input,
            &weights,
            in_dim as u32,
            out_dim as u32,
        );
        backend.submit(encoder).expect("submit logits");
        let raw = backend
            .read_bytes(&logits, out_dim * 4)
            .expect("read logits");
        for (index, bytes) in raw.chunks_exact(4).enumerate() {
            let got = f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
            let reference = want[index];
            assert!(got.is_finite(), "logit {index}: non-finite result");
            assert!(
                (got - reference).abs() <= 5e-2
                    || (got - reference).abs() <= 5e-3 * reference.abs().max(1.0),
                "logit {index}: got={got} want={reference}"
            );
        }
        input.destroy(&backend.dev);
        weights.destroy(&backend.dev);
        output.destroy(&backend.dev);
        logits.destroy(&backend.dev);
    }

    // The MMVQ Q4_K decode GEMV (quant A → per-8 int8 Q8, then DP4A) produces
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
        let in_dim = 512usize; // 2 Q4_K super-blocks per row, 64 Q8 blocks
        let out_dim = 70usize; // not a multiple of 32 → exercises the row boundary

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
        let nblocks = in_dim / 8;
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
            (out_dim as u32).div_ceil(32),
        );
        dev.submit_wait(cmd).expect("submit");

        let raw = backend.read_bytes(&out, out_dim * 2).expect("read");
        let got: Vec<f32> = raw
            .chunks_exact(2)
            .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect();

        // int8 Q8 activations round each element to ~1/127, so a length-512 dot carries a
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
