/*
 * graph_horizon_engine — Vulkan matmul namespace
 * Exports decode/prefill dispatch and records the shared FP32 logits projection.
 * Selection policy and numeric parity stay in focused child modules.
 */

#![allow(clippy::too_many_arguments)]

mod decode;
mod prefill;
mod trace;

use ash::vk;

use crate::backend::vulkan::buffers::{GpuBuffer, WeightFormat};
use crate::backend::vulkan::device::Device;
use crate::backend::vulkan::pipeline::{Kernel, PipelineRegistry, dispatch};

pub(crate) use decode::matmul;
pub(crate) use prefill::{matmul_batched_q4k, matmul_batched_q6k};
pub(crate) use trace::log_path_once;

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
        WeightFormat::Q5K => Kernel::LogitsQ5K,
    };
    project(dev, reg, cmd, kernel, out, x, w, in_dim, out_dim, 64);
}

pub(super) fn project(
    dev: &Device,
    reg: &PipelineRegistry,
    cmd: vk::CommandBuffer,
    kernel: Kernel,
    out: &GpuBuffer,
    a: &GpuBuffer,
    w: &GpuBuffer,
    in_dim: u32,
    out_dim: u32,
    output_rows: u32,
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
        out_dim.div_ceil(output_rows),
    );
}
