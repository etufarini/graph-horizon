/*
 * graph_horizon_engine — Vulkan device selection & capability validation
 * Pre-device queries on the instance/physical device that `Device::init` runs before
 * creating the logical device: deterministic physical-device choice, the host-visible
 * and FP16 requirement checks. No device is created here — only queried and validated.
*/

use ash::vk;
use color_eyre::eyre::{Result, eyre};

// Deterministic: among devices with a compute queue, prefer discrete > integrated >
// other; ties broken by lowest index.
pub(super) fn pick_device(instance: &ash::Instance) -> Result<(vk::PhysicalDevice, u32)> {
    // SAFETY: `instance` is a live handle; the call only reads the driver's device list.
    let devices = unsafe { instance.enumerate_physical_devices() }
        .map_err(|_| eyre!("vulkan: cannot enumerate devices"))?;
    let mut best: Option<(i32, vk::PhysicalDevice, u32)> = None;
    for pd in devices {
        // SAFETY: `instance` is live and `pd` comes from its own enumeration above.
        let qfam = unsafe { instance.get_physical_device_queue_family_properties(pd) };
        let compute = qfam
            .iter()
            .position(|q| q.queue_flags.contains(vk::QueueFlags::COMPUTE) && q.queue_count > 0);
        let Some(qi) = compute else { continue };
        // SAFETY: `instance` is live and `pd` is one of its enumerated devices.
        let props = unsafe { instance.get_physical_device_properties(pd) };
        let score = match props.device_type {
            vk::PhysicalDeviceType::DISCRETE_GPU => 3,
            vk::PhysicalDeviceType::INTEGRATED_GPU => 2,
            _ => 1,
        };
        if best.as_ref().map(|b| score > b.0).unwrap_or(true) {
            best = Some((score, pd, qi as u32));
        }
    }
    best.map(|(_, pd, qi)| (pd, qi))
        .ok_or_else(|| eyre!("vulkan: no device with a compute queue"))
}

pub(super) fn require_host_visible(mem: &vk::PhysicalDeviceMemoryProperties) -> Result<()> {
    let flags = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
    let ok = mem.memory_types[..mem.memory_type_count as usize]
        .iter()
        .any(|t| t.property_flags.contains(flags));
    ok.then_some(())
        .ok_or_else(|| eyre!("vulkan: no host-visible memory type (required for staging)"))
}

pub(super) fn require_fp16(instance: &ash::Instance, pd: vk::PhysicalDevice) -> Result<()> {
    let mut f11 = vk::PhysicalDeviceVulkan11Features::default();
    let mut f12 = vk::PhysicalDeviceVulkan12Features::default();
    let mut features = vk::PhysicalDeviceFeatures2::default()
        .push_next(&mut f11)
        .push_next(&mut f12);
    // SAFETY: `instance`/`pd` are live; `features` (with `f11`/`f12` chained) is a stack
    // struct that outlives the call the driver writes into.
    unsafe { instance.get_physical_device_features2(pd, &mut features) };
    if f11.storage_buffer16_bit_access == 0 || f12.shader_float16 == 0 {
        return Err(eyre!(
            "vulkan: device lacks FP16 support (16-bit storage / shaderFloat16)"
        ));
    }
    Ok(())
}
