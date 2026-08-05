/*
 * graph_horizon_engine — attention kernel dispatch
 * Records F16 and INT8-cache KV writes plus causal GQA decode/prefill, with
 * online softmax fused into attention. Scale and GQA head mapping stay
 * identical across the retained cache formats.
*/

// AGENTS deroga K: varianti coese write/decode/prefill della sola attention F16/INT8.

// Dispatch wrappers mirror the kernels' (buffers, dims, strides) interface, so
// wide argument lists are intrinsic and expected.
#![allow(clippy::too_many_arguments)]

use ash::vk;

use crate::backend::vulkan::buffers::GpuBuffer;
use crate::backend::vulkan::device::Device;
use crate::backend::vulkan::pipeline::{Kernel, PipelineRegistry, dispatch, dispatch_2d};

// Copies k,v (each `count` elements) into the caches at element `dst_offset`,
// the precomputed [layer][pos] origin from kv_cache.rs.
pub(crate) fn kv_write(
    dev: &Device,
    reg: &PipelineRegistry,
    cmd: vk::CommandBuffer,
    kc: &GpuBuffer,
    vc: &GpuBuffer,
    k: &GpuBuffer,
    v: &GpuBuffer,
    dst_offset: u32,
    count: u32,
) {
    let mut push = Vec::with_capacity(8);
    push.extend_from_slice(&dst_offset.to_le_bytes());
    push.extend_from_slice(&count.to_le_bytes());
    dispatch(
        dev,
        reg,
        cmd,
        Kernel::KvWrite,
        &[
            (k.buffer, k.offset, k.size),
            (v.buffer, v.offset, v.size),
            (kc.buffer, kc.offset, kc.size),
            (vc.buffer, vc.offset, vc.size),
        ],
        &push,
        count.div_ceil(64),
    );
}

// INT8 quantize-on-write: one workgroup per (token, kv_head) vector, mirroring
// kv_cache/int8.rs bit-exactly (I3). Byte offsets arrive precomputed from
// kv_cache::layout (D6).
pub(crate) fn kv_write_int8(
    dev: &Device,
    reg: &PipelineRegistry,
    cmd: vk::CommandBuffer,
    kc: &GpuBuffer,
    vc: &GpuBuffer,
    k: &GpuBuffer,
    v: &GpuBuffer,
    payload_offset: u32,
    meta_offset: u32,
    vectors: u32,
    head_dim: u32,
) {
    let mut push = Vec::with_capacity(16);
    for x in [payload_offset, meta_offset, vectors, head_dim] {
        push.extend_from_slice(&x.to_le_bytes());
    }
    dispatch(
        dev,
        reg,
        cmd,
        Kernel::KvWriteInt8,
        &[
            (k.buffer, k.offset, k.size),
            (v.buffer, v.offset, v.size),
            (kc.buffer, kc.offset, kc.size),
            (vc.buffer, vc.offset, vc.size),
        ],
        &push,
        vectors,
    );
}

// INT8 decode attention: same grid as attention_decode, dequantizing on read;
// `meta_base` is the byte base of the metadata region (kv.meta_base()).
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
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    let mut push = Vec::with_capacity(32);
    for x in [head_dim, kv_heads, q_heads, pos, layer, context, meta_base] {
        push.extend_from_slice(&x.to_le_bytes());
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

// INT8 N-query prefill attention: same 2D grid as attention_prefill,
// dequantizing on read; `meta_base` as in the decode variant.
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
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    let mut push = Vec::with_capacity(36);
    for x in [
        head_dim, kv_heads, q_heads, base, n, layer, context, meta_base,
    ] {
        push.extend_from_slice(&x.to_le_bytes());
    }
    push.extend_from_slice(&scale.to_le_bytes());
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

// out[q_heads*head_dim] = attention(q, cache[layer, 0..=pos]). One workgroup per
// query head; the kernel reads the caches directly (KV range 0..=pos).
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
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    let mut push = Vec::with_capacity(32);
    for v in [head_dim, kv_heads, q_heads, pos, layer, context] {
        push.extend_from_slice(&v.to_le_bytes());
    }
    push.extend_from_slice(&scale.to_le_bytes());
    dispatch(
        dev,
        reg,
        cmd,
        Kernel::AttentionDecode,
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

// out[n*q_heads*head_dim] = causal GQA attention for N query in one dispatch:
// row i (absolute position base+i) attends cache[layer, 0..=base+i]. 2D grid
// (q_heads, n), one workgroup per (head, row). Mirror of attention_decode with a
// per-row position and a per-row N-wide offset into q/out.
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
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    let mut push = Vec::with_capacity(32);
    for v in [head_dim, kv_heads, q_heads, base, n, layer, context] {
        push.extend_from_slice(&v.to_le_bytes());
    }
    push.extend_from_slice(&scale.to_le_bytes());
    dispatch_2d(
        dev,
        reg,
        cmd,
        Kernel::AttentionPrefill,
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
