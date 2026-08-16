/*
 * Vulkan prefill-attention dispatch and module boundary; decode routing and
 * KV-cache mutation are owned by dedicated sibling modules.
 */

#![allow(clippy::too_many_arguments)]

mod decode;
mod policy;
mod write;

pub(crate) const GQA_DECODE_SPLITS: u32 = 8;
pub(crate) const GQA_DECODE_PARTIAL_BYTES: u64 = 32 * GQA_DECODE_SPLITS as u64 * 128 * 4;
pub(crate) const GQA_DECODE_STATE_BYTES: u64 = 32 * GQA_DECODE_SPLITS as u64 * 2 * 4;

pub(crate) use decode::{f16 as attention_decode, int8 as attention_decode_int8};
pub(crate) use write::{kv_write, kv_write_int8};

use ash::vk;

use crate::backend::vulkan::buffers::GpuBuffer;
use crate::backend::vulkan::device::Device;
use crate::backend::vulkan::pipeline::{Kernel, PipelineRegistry, dispatch_2d};

fn matrix2_enabled() -> bool {
    use std::sync::OnceLock;
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        matches!(
            std::env::var("GRAPH_HORIZON_PREFILL_MATRIX2")
                .ok()
                .as_deref(),
            None | Some("1" | "true" | "yes")
        )
    })
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
    if head_dim == 128 && n.is_multiple_of(32) && matrix2_enabled() {
        if let Some(sparse) = policy::production()
            && base.saturating_add(n) >= sparse.context_threshold
            && sparse.layer_mask.checked_shr(layer).unwrap_or(0) & 1 != 0
            && reg.contains(Kernel::AttentionPrefillMatrix2Sparse)
        {
            for value in [
                sparse.window,
                0,
                sparse.global_stride_blocks,
                sparse.layer_mask,
            ] {
                push.extend_from_slice(&value.to_le_bytes());
            }
            dispatch_2d(
                dev,
                reg,
                cmd,
                Kernel::AttentionPrefillMatrix2Sparse,
                &[
                    (q.buffer, q.offset, q.size),
                    (kc.buffer, kc.offset, kc.size),
                    (vc.buffer, vc.offset, vc.size),
                    (out.buffer, out.offset, out.size),
                ],
                &push,
                q_heads,
                n / 32,
            );
            return;
        }
    }
    // The tiled shader specializes the approved Ministral K/V width; generic
    // mistral3 shapes keep the existing runtime-dimension fallback.
    let (kernel, rows) = if head_dim == 128
        && n.is_multiple_of(32)
        && matrix2_enabled()
        && reg.contains(Kernel::AttentionPrefillMatrix2)
    {
        (Kernel::AttentionPrefillMatrix2, n / 32)
    } else if head_dim == 128 && reg.contains(Kernel::AttentionPrefillTiledCoopQk) {
        (Kernel::AttentionPrefillTiledCoopQk, n.div_ceil(16))
    } else if head_dim == 128 && reg.contains(Kernel::AttentionPrefillTiled) {
        (Kernel::AttentionPrefillTiled, n.div_ceil(8))
    } else if reg.contains(Kernel::AttentionPrefillWide) {
        (Kernel::AttentionPrefillWide, n)
    } else {
        (Kernel::AttentionPrefill, n)
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
        rows,
    );
}
