/*
 * graph_horizon_engine — Vulkan KV-cache write dispatch
 * Dispatches F16 copies or INT8 quantize-on-write for precomputed cache ranges;
 * it contains no attention calculation or resource ownership.
 */

#![allow(clippy::too_many_arguments)]

use ash::vk;

use crate::backend::vulkan::buffers::GpuBuffer;
use crate::backend::vulkan::device::Device;
use crate::backend::vulkan::pipeline::{Kernel, PipelineRegistry, dispatch};

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
    for value in [payload_offset, meta_offset, vectors, head_dim] {
        push.extend_from_slice(&value.to_le_bytes());
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
