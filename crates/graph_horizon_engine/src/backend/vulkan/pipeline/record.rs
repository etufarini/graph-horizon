/*
 * graph_horizon_engine — pipeline dispatch recording
 * Binds a registered compute pipeline, pushes its buffers and constants, launches
 * a 1D/2D grid, and records the trailing compute barrier.
 */

use ash::vk;

use super::{Kernel, PipelineRegistry};
use crate::backend::vulkan::device::Device;

pub(crate) fn dispatch(
    dev: &Device,
    reg: &PipelineRegistry,
    cmd: vk::CommandBuffer,
    k: Kernel,
    buffers: &[(vk::Buffer, u64, u64)],
    push: &[u8],
    groups: u32,
) {
    record(dev, reg, cmd, k, buffers, push, groups, 1);
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn dispatch_2d(
    dev: &Device,
    reg: &PipelineRegistry,
    cmd: vk::CommandBuffer,
    k: Kernel,
    buffers: &[(vk::Buffer, u64, u64)],
    push: &[u8],
    groups_x: u32,
    groups_y: u32,
) {
    record(dev, reg, cmd, k, buffers, push, groups_x, groups_y);
}

#[allow(clippy::too_many_arguments)]
fn record(
    dev: &Device,
    reg: &PipelineRegistry,
    cmd: vk::CommandBuffer,
    k: Kernel,
    buffers: &[(vk::Buffer, u64, u64)],
    push: &[u8],
    groups_x: u32,
    groups_y: u32,
) {
    let p = reg.get(k);
    let infos: Vec<vk::DescriptorBufferInfo> = buffers
        .iter()
        .map(|&(b, offset, range)| {
            vk::DescriptorBufferInfo::default()
                .buffer(b)
                .offset(offset)
                .range(range)
        })
        .collect();
    let writes: Vec<vk::WriteDescriptorSet> = infos
        .iter()
        .enumerate()
        .map(|(i, info)| {
            vk::WriteDescriptorSet::default()
                .dst_binding(i as u32)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(info))
        })
        .collect();
    // SAFETY: `cmd` is recording; `p`'s pipeline/layout are live and built for this device;
    // `writes` (and `push`) outlive the calls and bind exactly the kernel's declared
    // descriptors/push range before the dispatch.
    unsafe {
        dev.device
            .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, p.pipeline);
    }
    // SAFETY: the command and pipeline layout are live; `writes` outlives the call.
    unsafe {
        dev.push_desc.cmd_push_descriptor_set(
            cmd,
            vk::PipelineBindPoint::COMPUTE,
            p.layout,
            0,
            &writes,
        );
    }
    // SAFETY: `cmd` is recording and the pushed bytes match this pipeline ABI.
    unsafe {
        if !push.is_empty() {
            dev.device
                .cmd_push_constants(cmd, p.layout, vk::ShaderStageFlags::COMPUTE, 0, push);
        }
        dev.device.cmd_dispatch(cmd, groups_x, groups_y, 1);
    }
    if !dev
        .skip_next_barrier
        .swap(false, std::sync::atomic::Ordering::Relaxed)
    {
        dev.compute_barrier(cmd);
    }
}
