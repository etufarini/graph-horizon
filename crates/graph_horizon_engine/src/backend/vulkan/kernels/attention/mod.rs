/*
 * Vulkan prefill-attention dispatch and module boundary; decode routing and
 * KV-cache mutation are owned by dedicated sibling modules.
 */

#![allow(clippy::too_many_arguments)]

mod decode;
mod policy;
mod write;

pub(crate) const GQA_DECODE_SPLITS: u32 = 8;
pub(crate) const GQA_DECODE_WAVE64_WORKGROUPS: u32 = 4;
pub(crate) const GQA_DECODE_WAVE64_PARTS: u32 = 16;
pub(crate) const GQA_DECODE_PARTIAL_BYTES: u64 = 32 * GQA_DECODE_WAVE64_PARTS as u64 * 128 * 4;
pub(crate) const GQA_DECODE_STATE_BYTES: u64 = 32 * GQA_DECODE_WAVE64_PARTS as u64 * 2 * 4;

pub(crate) use decode::{f16 as attention_decode, int8 as attention_decode_int8};
pub(crate) use write::{kv_write, kv_write_int8};

use ash::vk;

use crate::backend::vulkan::buffers::GpuBuffer;
use crate::backend::vulkan::device::Device;
use crate::backend::vulkan::pipeline::{Kernel, PipelineRegistry, dispatch_2d};

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
    // The F16 function itself is the datatype gate. The pure policy below owns
    // every device/shape/workload decision; successful pipeline construction is
    // the final resource-capability signal and absence always selects a fallback.
    let pipelines = policy::Pipelines {
        nvidia_q64: reg.contains(Kernel::AttentionPrefillMatrix2Q64),
        matrix2_q32: reg.contains(Kernel::AttentionPrefillMatrix2),
        coop_qk: reg.contains(Kernel::AttentionPrefillTiledCoopQk),
        tiled: reg.contains(Kernel::AttentionPrefillTiled),
        wide: reg.contains(Kernel::AttentionPrefillWide),
    };
    let shape = policy::Shape {
        head_dim,
        kv_heads,
        q_heads,
        rows: n,
    };
    let (route, rows) = policy::select(shape, pipelines);
    let kernel = match route {
        policy::Route::NvidiaQ64 => Kernel::AttentionPrefillMatrix2Q64,
        policy::Route::Matrix2Q32 => Kernel::AttentionPrefillMatrix2,
        policy::Route::CoopQk => Kernel::AttentionPrefillTiledCoopQk,
        policy::Route::Tiled => Kernel::AttentionPrefillTiled,
        policy::Route::Wide => Kernel::AttentionPrefillWide,
        policy::Route::Portable => Kernel::AttentionPrefill,
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
