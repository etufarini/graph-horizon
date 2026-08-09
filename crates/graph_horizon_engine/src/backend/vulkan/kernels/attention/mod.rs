/*
 * Vulkan attention dispatch: selects causal GQA decode/prefill variants while the sibling module owns KV-cache mutation.
 */

#![allow(clippy::too_many_arguments)]

mod write;

pub(crate) use write::{kv_write, kv_write_int8};

use ash::vk;

use crate::backend::vulkan::buffers::GpuBuffer;
use crate::backend::vulkan::device::Device;
use crate::backend::vulkan::pipeline::{Kernel, PipelineRegistry, dispatch, dispatch_2d};

pub(crate) fn attention_decode_int8(
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

pub(crate) fn attention_prefill_int8(
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
    base: u32,
    n: u32,
    layer: u32,
    context: u32,
    meta_base: u32,
) {
    let mut push = Vec::with_capacity(36);
    for value in [
        head_dim, kv_heads, q_heads, base, n, layer, context, meta_base,
    ] {
        push.extend_from_slice(&value.to_le_bytes());
    }
    push.extend_from_slice(&(1.0f32 / (head_dim as f32).sqrt()).to_le_bytes());
    dispatch_2d(
        dev,
        reg,
        cmd,
        Kernel::AttentionPrefillInt8,
        &[
            (q.buffer, q.offset, q.size),
            (kc.buffer, kc.offset, kc.size),
            (vc.buffer, vc.offset, vc.size),
            (out.buffer, out.offset, out.size),
        ],
        &push,
        q_heads,
        n,
    );
}

pub(crate) fn attention_decode(
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
) {
    let mut push = Vec::with_capacity(32);
    for value in [head_dim, kv_heads, q_heads, pos, layer, context] {
        push.extend_from_slice(&value.to_le_bytes());
    }
    push.extend_from_slice(&(1.0f32 / (head_dim as f32).sqrt()).to_le_bytes());
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

pub(crate) fn attention_prefill(
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
    base: u32,
    n: u32,
    layer: u32,
    context: u32,
) {
    let mut push = Vec::with_capacity(32);
    for value in [head_dim, kv_heads, q_heads, base, n, layer, context] {
        push.extend_from_slice(&value.to_le_bytes());
    }
    push.extend_from_slice(&(1.0f32 / (head_dim as f32).sqrt()).to_le_bytes());
    let kernel = if reg.contains(Kernel::AttentionPrefillWide) {
        Kernel::AttentionPrefillWide
    } else {
        Kernel::AttentionPrefill
    };
    dispatch_2d(
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
        n.div_ceil(2),
    );
}
