/*
 * graph_horizon_engine — Vulkan backend compute dispatch
 * Free-fn bodies of the `Backend` compute methods that do more than one kernel call:
 * `embed` (per-format gather), `matmul` (mmvq decode routing + per-format GEMV
 * fallback), and `matmul_batched` (specialized Q4_K/Q6_K batches plus fallback). Records
 * kernels only, with no submit or resource ownership.
*/

use ash::vk;
use color_eyre::eyre::Result;

use crate::backend::Backend;
use crate::backend::vulkan::VulkanBackend;
use crate::backend::vulkan::buffers::{GpuBuffer, WeightFormat};
use crate::backend::vulkan::kernels;
use crate::backend::vulkan::pipeline::{Kernel, dispatch};

// Embedding lookup into the FP32 residual stream. FP16 token_embd is a widening copy;
// Q4_K/Q5_K/Q6_K are dequantized on the fly. The backend surface is broader
// than the Ministral profile gate, so Q5_K remains reachable for format coverage.
pub(in crate::backend::vulkan) fn embed(
    b: &VulkanBackend,
    enc: &vk::CommandBuffer,
    x: &GpuBuffer,
    token_embd: &GpuBuffer,
    token: u32,
    embd: u32,
) -> Result<()> {
    let kernel = match token_embd.quant {
        WeightFormat::F16 => Kernel::EmbedF16,
        WeightFormat::Q4K => Kernel::EmbedQ4K,
        WeightFormat::Q5K => Kernel::EmbedQ5K,
        WeightFormat::Q6K => Kernel::EmbedQ6K,
    };
    let mut push = Vec::with_capacity(8);
    push.extend_from_slice(&token.to_le_bytes());
    push.extend_from_slice(&embd.to_le_bytes());
    // EmbedF16 uses one invocation per element; every retained quantized format
    // uses one invocation per 256-element super-block.
    let groups = match token_embd.quant {
        WeightFormat::Q4K | WeightFormat::Q5K | WeightFormat::Q6K => (embd / 256).div_ceil(64),
        _ => embd.div_ceil(64),
    };
    dispatch(
        &b.dev,
        &b.reg,
        *enc,
        kernel,
        &[
            (token_embd.buffer, token_embd.offset, token_embd.size),
            (x.buffer, x.offset, x.size),
        ],
        &push,
        groups,
    );
    Ok(())
}

pub(in crate::backend::vulkan) fn matmul(
    b: &VulkanBackend,
    enc: &vk::CommandBuffer,
    out: &GpuBuffer,
    a: &GpuBuffer,
    w: &GpuBuffer,
    in_dim: u32,
    out_dim: u32,
) {
    // MMVQ int8/DP4A decode path for the dense Q4_K GEMV; FP16 interfaces use
    // reused scratch. Unsupported devices and shapes keep the per-format GEMV.
    if kernels::mmvq::dispatch_mmvq(
        &b.dev, &b.reg, *enc, out, a, &b.mmvq_qs, &b.mmvq_ds, w, in_dim, out_dim,
    ) {
        return;
    }
    kernels::matmul::matmul(&b.dev, &b.reg, *enc, out, a, w, in_dim, out_dim);
}

// Override the per-token default: a real batch first tries MMQ, then the Q4_K or
// Q6_K tiled path. Decode and remaining formats use the per-token loop with
// barrier elision. Output stays token-major `[n][out_dim]` FP16 on every route.
#[allow(clippy::too_many_arguments)]
pub(in crate::backend::vulkan) fn matmul_batched(
    b: &VulkanBackend,
    enc: &vk::CommandBuffer,
    out: &GpuBuffer,
    a: &GpuBuffer,
    w: &GpuBuffer,
    in_dim: u32,
    out_dim: u32,
    n: u32,
) {
    if n > 1 {
        if kernels::mmvq::dispatch_mmq_batched(
            &b.dev, &b.reg, *enc, out, a, &b.mmvq_qs, &b.mmvq_ds, w, in_dim, out_dim, n,
        ) {
            return;
        }
        // These format-specific entry points may assume their weight format;
        // all other formats retain the per-token fallback below.
        match w.quant {
            WeightFormat::Q4K => {
                kernels::matmul::matmul_batched_q4k(
                    &b.dev, &b.reg, *enc, out, a, w, in_dim, out_dim, n,
                );
                return;
            }
            WeightFormat::Q6K => {
                kernels::matmul::matmul_batched_q6k(
                    &b.dev, &b.reg, *enc, out, a, w, in_dim, out_dim, n,
                );
                return;
            }
            WeightFormat::F16 | WeightFormat::Q5K => {}
        }
    }
    let a_stride = in_dim as u64 * 2; // FP16 activation row
    let o_stride = out_dim as u64 * 2; // FP16 output row
    for i in 0..n as u64 {
        let ai = b.view(a, i * a_stride, a_stride);
        let oi = b.view(out, i * o_stride, o_stride);
        if i + 1 < n as u64 {
            b.no_barrier();
        }
        matmul(b, enc, &oi, &ai, w, in_dim, out_dim);
    }
}
