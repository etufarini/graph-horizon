/*
 * gh_zero_engine — matmul kernel dispatch
 * Records the matmul (FP16 activations × weights) and the final FP32 logits
 * projection. Batch 1: y = W · a with W in ggml row-major [out, in]. The weight
 * dtype is read off the buffer's WeightFormat and selects the kernel: FP16
 * or a quantized path (Q4_K/Q6_K/Q8_0) that dequantizes the ggml blocks on the
 * fly. The call interface is identical across formats so the graph never changes.
*/

// AGENTS deroga K: kernel/dispatch matmul GPU (selezione coopmat/mmvq/per-formato della stessa operazione GEMM).

// Dispatch wrappers mirror the kernels' (buffers, dims, strides) interface, so
// wide argument lists are intrinsic and expected.
#![allow(clippy::too_many_arguments)]

use std::sync::Once;

use ash::vk;

use crate::backend::vulkan::buffers::{GpuBuffer, WeightFormat};
use crate::backend::vulkan::device::Device;
use crate::backend::vulkan::pipeline::{Kernel, PipelineRegistry, dispatch, dispatch_2d};

// Load-path diagnostics are opt-in because stderr would corrupt the interactive
// terminal. This setting lives here because matmul is its only consumer.
fn load_trace() -> bool {
    use std::sync::OnceLock;
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        matches!(
            std::env::var("GH_ZERO_LOAD_TRACE").ok().as_deref(),
            Some("1") | Some("true") | Some("yes")
        )
    })
}

// y[out_dim] = W[out_dim, in_dim] · a[in_dim], result FP16. One invocation per
// output row; the kernel is chosen by the weight format (gated at load time, so
// the match is total over the supported formats).
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
    // Runtime path selection (2.3): Q4_K — the benchmark weight format — uses the
    // tiled kernel (shared-memory activation staging, 4-lane row split); the other
    // formats keep the proven per-format GEMV as fallback until the tiled path
    // covers them. The chosen path is logged once. coopmat is not present on
    // the dev GPU (RDNA2), so the selection here is always tiled-subgroup.
    let kernel = match w.quant {
        WeightFormat::F16 => Kernel::MatmulF16,
        WeightFormat::Q4K => Kernel::MatmulQ4KTiled,
        WeightFormat::Q6K => Kernel::MatmulQ6K,
        WeightFormat::Q8 => Kernel::MatmulQ8,
        WeightFormat::Q5K => Kernel::MatmulQ5K,
    };
    log_path_once(kernel);
    project(dev, reg, cmd, kernel, out, a, w, in_dim, out_dim);
}

// Logs the matmul path once at the first dispatch (the selection is deterministic
// per build/device), so the run declares which kernel it uses without per-call noise.
pub(crate) fn log_path_once(kernel: Kernel) {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        if !load_trace() {
            return;
        }
        let path = match kernel {
            Kernel::MatmulQ4KTiled => "Q4_K tiled-subgroup",
            Kernel::MatmulQ4KMmvqF16Out => "Q4_K mmvq (int8/dp4a)",
            _ => "per-format GEMV",
        };
        eprintln!("matmul: path={path}");
    });
}

// Logs the batched (prefill) Q4_K path once: coopmat tensor-core vs the f16-out batched
// fallback. Same load-trace gating as `log_path_once`; lets a run declare whether the
// prefill engaged coopmat without per-dispatch noise.
fn log_batched_path_once(coopmat: bool) {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        if !load_trace() {
            return;
        }
        let path = if coopmat {
            "Q4_K coopmat (f16-out)"
        } else {
            "Q4_K batched f16-out"
        };
        eprintln!("matmul_batched: path={path}");
    });
}

// Whether the dense prefill may use the coopmat tensor-core GEMM. OFF by default:
// measured on a large Q4_K_M dense prompt, the coopmat prefill was slower than the
// f16-out batched GEMM on a PCIe-streaming-bound configuration. The single-subgroup
// 16x16x16 kernel also had lower occupancy than the 256-thread tiled batched kernel.
// `GH_ZERO_PREFILL_COOPMAT=1` enables it (read once) for re-measurement on a
// compute-bound config or after the kernel is tuned for occupancy. The verified-correct
// kernel stays available; the default keeps the faster path.
fn prefill_coopmat_enabled() -> bool {
    use std::sync::OnceLock;
    static FLAG: OnceLock<bool> = OnceLock::new();
    *FLAG.get_or_init(|| {
        matches!(
            std::env::var("GH_ZERO_PREFILL_COOPMAT").ok().as_deref(),
            Some("1") | Some("true") | Some("yes")
        )
    })
}

// Batched FP16-in/FP16-out Q4_K GEMM: Y[n][out_dim] = W · A[n][in_dim] in ONE
// dispatch, reading each Q4_K weight row once for the whole token batch (the
// prefill path; the per-token `matmul_batched` re-read every weight n times). The
// output layout (token-major [n][out_dim] FP16) matches the per-token path, so it
// is a drop-in. 2D grid over (out_dim/64 row-tiles, n/64 col-tiles).
pub(crate) fn matmul_batched_q4k(
    dev: &Device,
    reg: &PipelineRegistry,
    cmd: vk::CommandBuffer,
    out: &GpuBuffer,
    a: &GpuBuffer,
    w: &GpuBuffer,
    in_dim: u32,
    out_dim: u32,
    n: u32,
) {
    // Coopmat tensor-core path: the dense prefill GEMM on the tensor cores when the
    // device exposes the kernel's 16×16×16 f16→f32 shape. A is ALREADY f16 here (this path is
    // f16-in/f16-out), so the f16-out coopmat kernel is a direct drop-in — no staging; W is
    // dequantized in-shader. The fallback (MatmulQ4KBatchF16Out) is itself an f16 path and
    // accepted dense weights are O(1) within f16 range, so the synthetic-weight NaN that forced
    // no extra numeric gate is required. Edge tiles are zero-padded in the shader; falls back
    // when caps/shape/format/in_dim don't qualify.
    // Ablation (misurato): sulla Radeon 760M (RADV espone coopmat 16×16×16) questo path
    // è ~+30% sul prefill vs il fallback batched f16-out, con logit bit-identici.
    // Il default resta batched f16-out; l'opzione serve per rimisurare.
    let caps = dev.coopmat;
    if prefill_coopmat_enabled()
        && w.quant == WeightFormat::Q4K
        && caps.available
        && caps.m == 16
        && caps.n == 16
        && caps.k == 16
        && in_dim.is_multiple_of(256)
    {
        log_batched_path_once(true);
        super::coopmat::dispatch_coopmat(dev, reg, cmd, out, a, w, in_dim, out_dim, n, caps);
        return;
    }
    log_batched_path_once(false);
    let mut push = Vec::with_capacity(12);
    push.extend_from_slice(&in_dim.to_le_bytes());
    push.extend_from_slice(&out_dim.to_le_bytes());
    push.extend_from_slice(&n.to_le_bytes());
    dispatch_2d(
        dev,
        reg,
        cmd,
        Kernel::MatmulQ4KBatchF16Out,
        &[
            (a.buffer, a.offset, a.size),
            (w.buffer, w.offset, w.size),
            (out.buffer, out.offset, out.size),
        ],
        &push,
        out_dim.div_ceil(64),
        n.div_ceil(64),
    );
}

// Same as matmul but the output is FP32 (the vocab-sized logits). The lm_head
// (output.weight) may be FP16 or quantized; the format selects the kernel.
pub(crate) fn logits(
    dev: &Device,
    reg: &PipelineRegistry,
    cmd: vk::CommandBuffer,
    out: &GpuBuffer,
    x: &GpuBuffer,
    w: &GpuBuffer,
    in_dim: u32,
    out_dim: u32,
) {
    let kernel = match w.quant {
        WeightFormat::F16 => Kernel::Logits,
        WeightFormat::Q4K => Kernel::LogitsQ4K,
        WeightFormat::Q6K => Kernel::LogitsQ6K,
        // Q8_0 lm_head occurs on a Q8_0 causal model with tied embeddings, where
        // the lm_head reuses the Q8_0 token_embd (tied to token_embd).
        WeightFormat::Q8 => Kernel::LogitsQ8,
        WeightFormat::Q5K => Kernel::LogitsQ5K,
    };
    project(dev, reg, cmd, kernel, out, x, w, in_dim, out_dim);
}

// Shared dispatch for the matmul/logits kernels: same bindings (a, w, out) and
// push constants (in_dim, out_dim), one workgroup of 64 rows per group.
fn project(
    dev: &Device,
    reg: &PipelineRegistry,
    cmd: vk::CommandBuffer,
    kernel: Kernel,
    out: &GpuBuffer,
    a: &GpuBuffer,
    w: &GpuBuffer,
    in_dim: u32,
    out_dim: u32,
) {
    let mut push = Vec::with_capacity(8);
    push.extend_from_slice(&in_dim.to_le_bytes());
    push.extend_from_slice(&out_dim.to_le_bytes());
    dispatch(
        dev,
        reg,
        cmd,
        kernel,
        &[
            (a.buffer, a.offset, a.size),
            (w.buffer, w.offset, w.size),
            (out.buffer, out.offset, out.size),
        ],
        &push,
        out_dim.div_ceil(64),
    );
}

// The parity test needs the CPU Q4_K oracle (`compute::matmul`, cpu-gated) and the
// Vulkan device, so it only compiles when both backends are present (the hybrid
// build); a vulkan-only test build skips it.
#[cfg(all(test, feature = "cpu"))]
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
