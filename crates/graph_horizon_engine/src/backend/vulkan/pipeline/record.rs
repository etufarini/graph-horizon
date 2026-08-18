/*
 * graph_horizon_engine — pipeline dispatch recording & single-pipeline build
 * The command-recording side of the registry: `dispatch`/`dispatch_2d` bind a
 * kernel's pipeline, push its buffers + constants and launch a 1D/2D grid, with the
 * trailing compute→compute barrier (honouring the one-shot barrier-elision flag);
 * `build_one` compiles one SPIR-V module into a cached compute pipeline with its
 * descriptor-set layout and push-constant range.
*/

use std::io::Cursor;
#[cfg(feature = "vulkan-profile")]
use std::time::Instant;

use ash::vk;
use color_eyre::eyre::{Result, eyre};

use super::kernel::{Kernel, spec};
use super::{Pipeline, PipelineRegistry};
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
    #[cfg(feature = "vulkan-profile")]
    let stamp = dev
        .profile
        .begin_kernel(&dev.device, cmd, k, groups_x, groups_y);
    // SAFETY: `cmd` is recording; `p`'s pipeline/layout are live and built for this device;
    // `writes` (and `push`) outlive the calls and bind exactly the kernel's declared
    // descriptors/push range before the dispatch.
    unsafe {
        dev.device
            .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, p.pipeline);
    }
    #[cfg(feature = "vulkan-profile")]
    let descriptor_started = Instant::now();
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
    #[cfg(feature = "vulkan-profile")]
    let descriptor_ms = descriptor_started.elapsed().as_secs_f64() * 1000.0;
    // SAFETY: `cmd` is recording and the pushed bytes match this pipeline ABI.
    unsafe {
        if !push.is_empty() {
            dev.device
                .cmd_push_constants(cmd, p.layout, vk::ShaderStageFlags::COMPUTE, 0, push);
        }
        dev.device.cmd_dispatch(cmd, groups_x, groups_y, 1);
    }
    #[cfg(feature = "vulkan-profile")]
    dev.profile
        .end_kernel(&dev.device, cmd, stamp, descriptor_ms);
    if !dev
        .skip_next_barrier
        .swap(false, std::sync::atomic::Ordering::Relaxed)
    {
        dev.compute_barrier(cmd);
    }
}

pub(super) fn build_one(dev: &Device, cache: vk::PipelineCache, k: Kernel) -> Result<Pipeline> {
    build(dev, cache, k, None)
}

pub(super) fn build_one_wave32(
    dev: &Device,
    cache: vk::PipelineCache,
    k: Kernel,
) -> Result<Pipeline> {
    build(dev, cache, k, Some(32))
}

fn build(
    dev: &Device,
    cache: vk::PipelineCache,
    k: Kernel,
    required_subgroup_size: Option<u32>,
) -> Result<Pipeline> {
    let (bytes, bindings, push_size) = spec(k);
    let code = ash::util::read_spv(&mut Cursor::new(bytes))
        .map_err(|_| eyre!("vulkan: malformed SPIR-V module"))?;
    // SAFETY: `dev.device` is alive; `code` was validated by `read_spv`.
    let module = unsafe {
        dev.device
            .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&code), None)
    }
    .map_err(|_| eyre!("vulkan: cannot create shader module"))?;

    let layout_bindings: Vec<vk::DescriptorSetLayoutBinding> = (0..bindings)
        .map(|i| {
            vk::DescriptorSetLayoutBinding::default()
                .binding(i)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
        })
        .collect();
    // SAFETY: `dev.device` is alive; `layout_bindings` outlives the call.
    let set_layout = match unsafe {
        dev.device.create_descriptor_set_layout(
            &vk::DescriptorSetLayoutCreateInfo::default()
                .flags(vk::DescriptorSetLayoutCreateFlags::PUSH_DESCRIPTOR_KHR)
                .bindings(&layout_bindings),
            None,
        )
    } {
        Ok(layout) => layout,
        Err(_) => {
            // SAFETY: `module` was created above and has no pipeline owner yet.
            unsafe { dev.device.destroy_shader_module(module, None) };
            return Err(eyre!("vulkan: cannot create descriptor set layout"));
        }
    };

    let set_layouts = [set_layout];
    let push_ranges = [vk::PushConstantRange::default()
        .stage_flags(vk::ShaderStageFlags::COMPUTE)
        .offset(0)
        .size(push_size)];
    // SAFETY: `dev.device`, `set_layouts` and `push_ranges` outlive the call.
    let layout = match unsafe {
        dev.device.create_pipeline_layout(
            &vk::PipelineLayoutCreateInfo::default()
                .set_layouts(&set_layouts)
                .push_constant_ranges(&push_ranges),
            None,
        )
    } {
        Ok(layout) => layout,
        Err(_) => {
            // SAFETY: both handles were created above and are not owned elsewhere.
            unsafe {
                dev.device.destroy_descriptor_set_layout(set_layout, None);
                dev.device.destroy_shader_module(module, None);
            }
            return Err(eyre!("vulkan: cannot create pipeline layout"));
        }
    };

    let mut stage = vk::PipelineShaderStageCreateInfo::default()
        .stage(vk::ShaderStageFlags::COMPUTE)
        .module(module)
        .name(c"main");
    let mut subgroup = vk::PipelineShaderStageRequiredSubgroupSizeCreateInfo::default();
    if let Some(size) = required_subgroup_size {
        subgroup = subgroup.required_subgroup_size(size);
        stage = stage.push_next(&mut subgroup);
    }
    let info = vk::ComputePipelineCreateInfo::default()
        .stage(stage)
        .layout(layout);
    // SAFETY: `dev.device`/`cache` are alive; `info` owns live module/layout refs.
    let pipeline = match unsafe { dev.device.create_compute_pipelines(cache, &[info], None) } {
        Ok(pipelines) => pipelines[0],
        Err(_) => {
            // SAFETY: all three handles are local to this failed build step.
            unsafe {
                dev.device.destroy_pipeline_layout(layout, None);
                dev.device.destroy_descriptor_set_layout(set_layout, None);
                dev.device.destroy_shader_module(module, None);
            }
            return Err(eyre!("vulkan: compute pipeline creation failed"));
        }
    };
    // SAFETY: `module` was created from this device and is no longer referenced once the
    // pipeline above is built, so destroying it now races nothing.
    unsafe { dev.device.destroy_shader_module(module, None) };

    Ok(Pipeline {
        pipeline,
        layout,
        set_layout,
    })
}
