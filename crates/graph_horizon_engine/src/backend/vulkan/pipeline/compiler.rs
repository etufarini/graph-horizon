/*
 * graph_horizon_engine — single compute-pipeline compilation
 * Creates one shader module, descriptor layout, pipeline layout, and cached
 * compute pipeline, optionally requiring a fixed subgroup size. Partial failures
 * release every resource created within this operation.
 */

use std::io::Cursor;

use ash::vk;
use color_eyre::eyre::{Result, eyre};

use super::Pipeline;
use super::kernel::{Kernel, spec};
use crate::backend::vulkan::device::Device;

pub(super) fn build(dev: &Device, cache: vk::PipelineCache, k: Kernel) -> Result<Pipeline> {
    build_with_subgroup(dev, cache, k, None)
}

pub(super) fn build_wave32(dev: &Device, cache: vk::PipelineCache, k: Kernel) -> Result<Pipeline> {
    build_with_subgroup(dev, cache, k, Some(32))
}

fn build_with_subgroup(
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
    // SAFETY: `module` is no longer referenced after successful pipeline creation.
    unsafe { dev.device.destroy_shader_module(module, None) };

    Ok(Pipeline {
        pipeline,
        layout,
        set_layout,
    })
}
