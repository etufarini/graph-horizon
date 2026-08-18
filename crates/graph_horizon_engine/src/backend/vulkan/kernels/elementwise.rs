/*
 * graph_horizon_engine — elementwise / normalization kernel dispatch
 * Records the retained RMSNorm-X, RoPE/YaRN, and residual-add pipelines.
 * Parameters come from validated model configuration; this module owns no
 * dispatch selection or buffers.
*/

// Dispatch wrappers mirror the kernels' (buffers, dims, strides) interface, so
// wide argument lists are intrinsic and expected.
#![allow(clippy::too_many_arguments)]

use ash::vk;

use crate::backend::vulkan::buffers::GpuBuffer;
use crate::backend::vulkan::device::Device;
use crate::backend::vulkan::pipeline::{Kernel, PipelineRegistry, dispatch};

// RMSNorm over the FP32 residual stream with FP16 output.
pub(crate) fn rmsnorm_x(
    dev: &Device,
    reg: &PipelineRegistry,
    cmd: vk::CommandBuffer,
    out: &GpuBuffer,
    x: &GpuBuffer,
    weight: &GpuBuffer,
    n: u32,
    eps: f32,
    rows: u32,
) {
    let mut push = Vec::with_capacity(8);
    push.extend_from_slice(&n.to_le_bytes());
    push.extend_from_slice(&eps.to_le_bytes());
    dispatch(
        dev,
        reg,
        cmd,
        Kernel::RmsNormX,
        &[
            (x.buffer, x.offset, x.size),
            (weight.buffer, weight.offset, weight.size),
            (out.buffer, out.offset, out.size),
        ],
        &push,
        rows,
    );
}

// In-place YaRN RoPE. `post_scale` is supplied by the caller: Q receives the
// approved attention-temperature factor after rotation; K passes 1.0 and therefore
// stays neutral. With `rope_dim == head_dim`, `factor == 1` and `post_scale == 1`
// the same shader reduces to ordinary Mistral NORM rotation.
pub(crate) fn rope_yarn(
    dev: &Device,
    reg: &PipelineRegistry,
    cmd: vk::CommandBuffer,
    x: &GpuBuffer,
    n_heads: u32,
    head_dim: u32,
    rope_dim: u32,
    pos: u32,
    rows: u32,
    freq_base: f32,
    factor: f32,
    beta_fast: f32,
    beta_slow: f32,
    original_context: u32,
    post_scale: f32,
) {
    let mut push = Vec::with_capacity(44);
    for v in [n_heads, head_dim, rope_dim, pos, rows] {
        push.extend_from_slice(&v.to_le_bytes());
    }
    for v in [
        freq_base,
        factor,
        beta_fast,
        beta_slow,
        original_context as f32,
        post_scale,
    ] {
        push.extend_from_slice(&v.to_le_bytes());
    }
    let pairs = rows * n_heads * (head_dim / 2);
    dispatch(
        dev,
        reg,
        cmd,
        Kernel::Rope,
        &[(x.buffer, x.offset, x.size)],
        &push,
        pairs.div_ceil(64),
    );
}

// In-place residual: x[i] += y[i].
pub(crate) fn residual_add(
    dev: &Device,
    reg: &PipelineRegistry,
    cmd: vk::CommandBuffer,
    x: &GpuBuffer,
    y: &GpuBuffer,
    n: u32,
) {
    dispatch(
        dev,
        reg,
        cmd,
        Kernel::Residual,
        &[(x.buffer, x.offset, x.size), (y.buffer, y.offset, y.size)],
        &n.to_le_bytes(),
        n.div_ceil(64),
    );
}
