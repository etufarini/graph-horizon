/*
 * Vulkan decode-attention dispatch: routes exact-shape F16 GQA through the
 * split/reduce kernels and keeps the established F16/INT8 fallbacks explicit.
 */

#![allow(clippy::too_many_arguments)]

use ash::vk;

use super::{GQA_DECODE_SPLITS, GQA_DECODE_WAVE64_PARTS, GQA_DECODE_WAVE64_WORKGROUPS};
use crate::backend::vulkan::buffers::GpuBuffer;
use crate::backend::vulkan::device::Device;
use crate::backend::vulkan::pipeline::{Kernel, PipelineRegistry, dispatch, dispatch_2d};

pub(crate) fn f16(
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
) {
    let mut push = Vec::with_capacity(32);
    for value in [head_dim, kv_heads, q_heads, pos, layer, context] {
        push.extend_from_slice(&value.to_le_bytes());
    }
    push.extend_from_slice(&(1.0f32 / (head_dim as f32).sqrt()).to_le_bytes());
    let (split_kernel, reduce_kernel, workgroups_y, parts, wave64) =
        if reg.contains(Kernel::AttentionDecodeGqaWave64Split) {
            (
                Kernel::AttentionDecodeGqaWave64Split,
                Kernel::AttentionDecodeGqaWave64Reduce,
                GQA_DECODE_WAVE64_WORKGROUPS,
                GQA_DECODE_WAVE64_PARTS,
                true,
            )
        } else {
            (
                Kernel::AttentionDecodeGqaSplit,
                Kernel::AttentionDecodeGqaReduce,
                GQA_DECODE_SPLITS,
                GQA_DECODE_SPLITS,
                false,
            )
        };
    if wave64 {
        push.extend_from_slice(&parts.to_le_bytes());
    }
    let scratch_fits = partial.size >= u64::from(q_heads) * 128 * u64::from(parts) * 4
        && state.size >= u64::from(q_heads) * u64::from(parts) * 2 * 4;
    if head_dim == 128
        && kv_heads.checked_mul(4) == Some(q_heads)
        && scratch_fits
        && reg.contains(split_kernel)
    {
        // These aliases are fully consumed here before their normal projection
        // and sampling users can overwrite the same scratch allocations.
        dispatch_2d(
            dev,
            reg,
            cmd,
            split_kernel,
            &[
                (q.buffer, q.offset, q.size),
                (kc.buffer, kc.offset, kc.size),
                (vc.buffer, vc.offset, vc.size),
                (partial.buffer, partial.offset, partial.size),
                (state.buffer, state.offset, state.size),
            ],
            &push,
            kv_heads,
            workgroups_y,
        );
        dispatch(
            dev,
            reg,
            cmd,
            reduce_kernel,
            &[
                (partial.buffer, partial.offset, partial.size),
                (state.buffer, state.offset, state.size),
                (out.buffer, out.offset, out.size),
            ],
            &push,
            q_heads,
        );
        return;
    }
    let kernel = if reg.contains(Kernel::AttentionDecode1024) {
        Kernel::AttentionDecode1024
    } else if reg.contains(Kernel::AttentionDecodeWide) {
        Kernel::AttentionDecodeWide
    } else {
        Kernel::AttentionDecode
    };
    dispatch(
        dev,
        reg,
        cmd,
        kernel,
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

pub(crate) fn int8(
    dev: &Device,
    reg: &PipelineRegistry,
    cmd: vk::CommandBuffer,
    out: &GpuBuffer,
    q: &GpuBuffer,
    kc: &GpuBuffer,
    vc: &GpuBuffer,
    head_dim: u32,
    kv_heads: u32,
    q_heads: u32,
    pos: u32,
    layer: u32,
    context: u32,
    meta_base: u32,
) {
    let mut push = Vec::with_capacity(32);
    for value in [head_dim, kv_heads, q_heads, pos, layer, context, meta_base] {
        push.extend_from_slice(&value.to_le_bytes());
    }
    push.extend_from_slice(&(1.0f32 / (head_dim as f32).sqrt()).to_le_bytes());
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
