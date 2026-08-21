/*
 * Vulkan INT8 decode-attention dispatch: selects exact 4:1 GQA cache reuse when
 * the split wave32 pipeline and shared scratch are available, otherwise records
 * the established portable per-query-head kernel.
 */

use ash::vk;

use super::super::GQA_DECODE_SPLITS;
use crate::backend::vulkan::buffers::GpuBuffer;
use crate::backend::vulkan::device::Device;
use crate::backend::vulkan::pipeline::{Kernel, PipelineRegistry, dispatch, dispatch_2d};

#[allow(clippy::too_many_arguments)]
pub(crate) fn run(
    dev: &Device,
    reg: &PipelineRegistry,
    cmd: vk::CommandBuffer,
    out: &GpuBuffer,
    q: &GpuBuffer,
    kc: &GpuBuffer,
    vc: &GpuBuffer,
    partial: &GpuBuffer,
    state: &GpuBuffer,
    head_dim: u32,
    kv_heads: u32,
    q_heads: u32,
    pos: u32,
    layer: u32,
    context: u32,
    meta_base: u32,
) {
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    let scratch_fits = partial.size >= u64::from(q_heads) * 128 * u64::from(GQA_DECODE_SPLITS) * 4
        && state.size >= u64::from(q_heads) * u64::from(GQA_DECODE_SPLITS) * 2 * 4;
    if head_dim == 128
        && kv_heads.checked_mul(4) == Some(q_heads)
        && scratch_fits
        && reg.contains(Kernel::AttentionDecodeGqaInt8Split)
    {
        let mut split_push = Vec::with_capacity(32);
        for value in [head_dim, kv_heads, q_heads, pos, layer, context, meta_base] {
            split_push.extend_from_slice(&value.to_le_bytes());
        }
        split_push.extend_from_slice(&scale.to_le_bytes());
        dispatch_2d(
            dev,
            reg,
            cmd,
            Kernel::AttentionDecodeGqaInt8Split,
            &[
                (q.buffer, q.offset, q.size),
                (kc.buffer, kc.offset, kc.size),
                (vc.buffer, vc.offset, vc.size),
                (partial.buffer, partial.offset, partial.size),
                (state.buffer, state.offset, state.size),
            ],
            &split_push,
            kv_heads,
            GQA_DECODE_SPLITS,
        );

        let mut reduce_push = Vec::with_capacity(28);
        for value in [head_dim, kv_heads, q_heads, pos, layer, context] {
            reduce_push.extend_from_slice(&value.to_le_bytes());
        }
        reduce_push.extend_from_slice(&scale.to_le_bytes());
        dispatch(
            dev,
            reg,
            cmd,
            Kernel::AttentionDecodeGqaReduce,
            &[
                (partial.buffer, partial.offset, partial.size),
                (state.buffer, state.offset, state.size),
                (out.buffer, out.offset, out.size),
            ],
            &reduce_push,
            q_heads,
        );
        return;
    }

    let mut push = Vec::with_capacity(32);
    for value in [head_dim, kv_heads, q_heads, pos, layer, context, meta_base] {
        push.extend_from_slice(&value.to_le_bytes());
    }
    push.extend_from_slice(&scale.to_le_bytes());
    dispatch(
        dev,
        reg,
        cmd,
        Kernel::AttentionDecodeInt8,
        &[
            (q.buffer, q.offset, q.size),
            (kc.buffer, kc.offset, kc.size),
            (vc.buffer, vc.offset, vc.size),
            (out.buffer, out.offset, out.size),
        ],
        &push,
        q_heads,
    );
}
