/*
 * graph_horizon_engine — Vulkan logical-device creation
 * The logical-device bring-up `Device::init` delegates to, after `caps` has selected
 * and validated the physical device: detect optional capabilities (memory-budget,
 * coopmat, dp4a), build the compute queue, enable FP16 features, and create the
 * command pool. Every optional feature is enabled only after its capability is
 * established. `create_device` returns a `DeviceBoot` the
 * caller spreads into `Device`.
*/

use ash::vk;
use color_eyre::eyre::{Result, eyre};

use crate::backend::vulkan::coopmat::{self, CoopmatCaps};
use crate::backend::vulkan::coopmat2::{self, Coopmat2Caps};

const PUSH_DESCRIPTOR: &std::ffi::CStr = ash::khr::push_descriptor::NAME;
const MEMORY_BUDGET: &std::ffi::CStr = ash::ext::memory_budget::NAME;

// Transient carrier of the created resources; `init` moves each field into `Device`.
pub(super) struct DeviceBoot {
    pub device: ash::Device,
    pub queue: vk::Queue,
    pub memory_budget_enabled: bool,
    pub coopmat: CoopmatCaps,
    pub coopmat2: Coopmat2Caps,
    pub dp4a: bool,
    pub wave32_control: bool,
    pub push_desc: ash::khr::push_descriptor::Device,
    pub cmd_pool: vk::CommandPool,
}

// Create the logical device: detect optional caps (memory-budget, coopmat, dp4a),
// enable the FP16 features, and create the command pool. Optional numeric
// capabilities are enabled only after their device support was observed.
pub(super) fn create_device(
    entry: &ash::Entry,
    instance: &ash::Instance,
    physical: vk::PhysicalDevice,
    queue_family: u32,
) -> Result<DeviceBoot> {
    // VK_EXT_memory_budget only when advertised; absence → free_vram returns None.
    // SAFETY: `instance`/`physical` are live handles from a successful init; the call
    // only reads the driver's extension list.
    let memory_budget_enabled = unsafe { instance.enumerate_device_extension_properties(physical) }
        .map(|exts| {
            exts.iter().any(|e| {
                // SAFETY: extension_name is a NUL-terminated C string from the driver.
                let name = unsafe { std::ffi::CStr::from_ptr(e.extension_name.as_ptr()) };
                name == MEMORY_BUDGET
            })
        })
        .unwrap_or(false);

    // Subgroup cooperative matrices and workgroup Matrix2 are independent
    // optional contracts. Either one may require the shared KHR extension.
    let coopmat = coopmat::detect(entry, instance, physical);
    let coopmat2 = coopmat2::detect(entry, instance, physical);
    let matrix_extension = coopmat.available || coopmat2.enabled();

    // Integer dot products and required subgroup sizes are optional Vulkan 1.3
    // contracts. Wave32 is retained only when 32 is in range for compute stages.
    let mut supported13 = vk::PhysicalDeviceVulkan13Features::default();
    let mut features = vk::PhysicalDeviceFeatures2::default().push_next(&mut supported13);
    let mut subgroup_control = vk::PhysicalDeviceSubgroupSizeControlProperties::default();
    let mut properties = vk::PhysicalDeviceProperties2::default().push_next(&mut subgroup_control);
    // SAFETY: every queried struct is stack-owned and outlives its driver call.
    unsafe {
        instance.get_physical_device_features2(physical, &mut features);
        instance.get_physical_device_properties2(physical, &mut properties);
    }
    let dp4a = supported13.shader_integer_dot_product != 0;
    let wave32_control = supported13.subgroup_size_control != 0
        && subgroup_control.min_subgroup_size <= 32
        && subgroup_control.max_subgroup_size >= 32
        && subgroup_control
            .required_subgroup_size_stages
            .contains(vk::ShaderStageFlags::COMPUTE);

    let priorities = [1.0f32];
    let qci = [vk::DeviceQueueCreateInfo::default()
        .queue_family_index(queue_family)
        .queue_priorities(&priorities)];
    let mut ext = vec![PUSH_DESCRIPTOR.as_ptr()];
    if memory_budget_enabled {
        ext.push(MEMORY_BUDGET.as_ptr());
    }
    // Enable coopmat only when a usable shape was found; the
    // feature structs must outlive `dci`.
    if matrix_extension {
        ext.push(coopmat::COOPERATIVE_MATRIX.as_ptr());
    }
    if coopmat2.enabled() {
        ext.push(coopmat2::COOPERATIVE_MATRIX2.as_ptr());
    }
    let mut f11 = vk::PhysicalDeviceVulkan11Features::default().storage_buffer16_bit_access(true);
    let mut f12 = vk::PhysicalDeviceVulkan12Features::default()
        .shader_float16(true)
        .vulkan_memory_model(coopmat2.enabled());
    let mut coop_feat =
        vk::PhysicalDeviceCooperativeMatrixFeaturesKHR::default().cooperative_matrix(true);
    let mut coop2_feat = coopmat2::enabled_features(coopmat2);
    let mut f13 = vk::PhysicalDeviceVulkan13Features::default()
        .shader_integer_dot_product(dp4a)
        .subgroup_size_control(wave32_control);
    let mut dci = vk::DeviceCreateInfo::default()
        .queue_create_infos(&qci)
        .enabled_extension_names(&ext)
        .push_next(&mut f11)
        .push_next(&mut f12);
    if matrix_extension {
        dci = dci.push_next(&mut coop_feat);
    }
    if coopmat2.enabled() {
        dci = dci.push_next(&mut coop2_feat);
    }
    if dp4a || wave32_control {
        dci = dci.push_next(&mut f13);
    }
    // SAFETY: `instance`/`physical` are live; `dci` and every feature/queue struct it
    // chains (`f13`, queue infos) are stack values that outlive this call.
    let device = unsafe { instance.create_device(physical, &dci, None) }
        .map_err(|_| eyre!("vulkan: failed to create logical device"))?;

    // SAFETY: `device` was just created and `queue_family`/index 0 were requested in `dci`.
    let queue = unsafe { device.get_device_queue(queue_family, 0) };
    let push_desc = ash::khr::push_descriptor::Device::new(instance, &device);

    let pool_info = vk::CommandPoolCreateInfo::default()
        .queue_family_index(queue_family)
        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
    // SAFETY: `device` is alive; `pool_info` names `queue_family` (valid on this device).
    let cmd_pool = match unsafe { device.create_command_pool(&pool_info, None) } {
        Ok(pool) => pool,
        Err(_) => {
            // SAFETY: command-pool creation was the first owned child attempted;
            // no queue work exists and this local logical device has not escaped.
            unsafe { device.destroy_device(None) };
            return Err(eyre!("vulkan: failed to create command pool"));
        }
    };

    Ok(DeviceBoot {
        device,
        queue,
        memory_budget_enabled,
        coopmat,
        coopmat2,
        dp4a,
        wave32_control,
        push_desc,
        cmd_pool,
    })
}
