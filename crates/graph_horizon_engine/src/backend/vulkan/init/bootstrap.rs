/*
 * graph_horizon_engine — Vulkan logical-device creation
 * The logical-device bring-up `Device::init` delegates to, after `caps` has selected
 * and validated the physical device: detect optional capabilities (memory-budget,
 * coopmat, dp4a), build the compute queue, enable FP16 features, and create the
 * command pool. Every
 * opt-in is conditional (INV-6 / P1). `create_device` returns a `DeviceBoot` the
 * caller spreads into `Device`. Bodies moved 1:1 from the former monolithic `device.rs`.
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

    // Coopmat shape (f16→f32) if exposed; absence means fallback.
    let coopmat = coopmat::detect(entry, instance, physical);
    let coopmat2 = if coopmat.available {
        coopmat2::detect(entry, instance, physical)
    } else {
        Coopmat2Caps::default()
    };

    // dp4a for the mmvq Q4_K decode GEMV; absence means float GEMV.
    let dp4a = {
        let mut f13 = vk::PhysicalDeviceVulkan13Features::default();
        let mut feats = vk::PhysicalDeviceFeatures2::default().push_next(&mut f13);
        // SAFETY: `instance`/`physical` are live; `feats` (with `f13` chained via
        // push_next) is a stack struct that outlives the call the driver writes into.
        unsafe { instance.get_physical_device_features2(physical, &mut feats) };
        f13.shader_integer_dot_product != 0
    };

    let priorities = [1.0f32];
    let qci = [vk::DeviceQueueCreateInfo::default()
        .queue_family_index(queue_family)
        .queue_priorities(&priorities)];
    let mut ext = vec![PUSH_DESCRIPTOR.as_ptr()];
    if memory_budget_enabled {
        ext.push(MEMORY_BUDGET.as_ptr());
    }
    // Coopmat extension + feature only when a usable shape was found (INV-6); the
    // feature structs must outlive `dci`.
    if coopmat.available {
        ext.push(coopmat::COOPERATIVE_MATRIX.as_ptr());
    }
    if coopmat.available && coopmat2.available {
        ext.push(coopmat2::COOPERATIVE_MATRIX2.as_ptr());
    }
    let mut f11 = vk::PhysicalDeviceVulkan11Features::default().storage_buffer16_bit_access(true);
    let mut f12 = vk::PhysicalDeviceVulkan12Features::default()
        .shader_float16(true)
        .vulkan_memory_model(coopmat2.available);
    let mut coop_feat =
        vk::PhysicalDeviceCooperativeMatrixFeaturesKHR::default().cooperative_matrix(true);
    let mut coop2_feat = coopmat2::enabled_features(coopmat2);
    let mut f13 = vk::PhysicalDeviceVulkan13Features::default().shader_integer_dot_product(true);
    let mut dci = vk::DeviceCreateInfo::default()
        .queue_create_infos(&qci)
        .enabled_extension_names(&ext)
        .push_next(&mut f11)
        .push_next(&mut f12);
    if coopmat.available {
        dci = dci.push_next(&mut coop_feat);
    }
    if coopmat.available && coopmat2.available {
        dci = dci.push_next(&mut coop2_feat);
    }
    if dp4a {
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
        push_desc,
        cmd_pool,
    })
}
