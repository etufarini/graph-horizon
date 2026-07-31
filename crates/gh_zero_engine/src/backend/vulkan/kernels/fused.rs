/*
 * gh_zero_engine — fused elementwise kernel dispatch
 * Dispatch wrapper for the fused `silu_mul` kernel, kept
 * out of `elementwise.rs` (already near the line budget). Same pattern as the
 * other kernel wrappers: select the pipeline, bind the buffers, set the push
 * constants and record the dispatch. No pipeline or buffer ownership lives here;
 * the fusion's bit-exactness with the standalone primitives is a property of the
 * shaders, not of this dispatch.
*/

#![allow(clippy::too_many_arguments)]

use ash::vk;

use crate::backend::vulkan::buffers::GpuBuffer;
use crate::backend::vulkan::device::Device;
use crate::backend::vulkan::pipeline::{Kernel, PipelineRegistry, dispatch};

// out[i] = f16(f16(silu(gate[i])) * up[i]). One invocation per element.
pub(crate) fn silu_mul(
    dev: &Device,
    reg: &PipelineRegistry,
    cmd: vk::CommandBuffer,
    out: &GpuBuffer,
    gate: &GpuBuffer,
    up: &GpuBuffer,
    n: u32,
) {
    dispatch(
        dev,
        reg,
        cmd,
        Kernel::SiluMul,
        &[
            (gate.buffer, gate.offset, gate.size),
            (up.buffer, up.offset, up.size),
            (out.buffer, out.offset, out.size),
        ],
        &n.to_le_bytes(),
        n.div_ceil(64),
    );
}
