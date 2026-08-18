/*
 * graph_horizon_engine — Vulkan device
 * Owns the brought-up Vulkan device: the logical `ash::Device`, the compute queue,
 * the memory properties for device-local VRAM and host-visible coherent RAM, the
 * optional cooperative-matrix / dp4a / memory-budget capabilities,
 * the dedicated streaming transfer queues, a command pool and the barrier-elision
 * flag. `init()` here is the orchestrator: it creates the instance, then delegates
 * physical-device selection, requirement checks and logical-device creation to
 * `bootstrap`, and assembles the `Device`. Runtime helpers (submit, barrier, budget,
 * transfer-queue lookup) live in `commands`. Errors are explicit (no Vulkan, no
 * compute queue, no suitable device, missing host-visible memory) and carry no paths.
 *
 * Beyond the compute queue the `Device` can own up to two DEDICATED transfer queues
 * for the streamed-prefill copies (H2D weights, D2H KV): families exposing `TRANSFER`
 * WITHOUT `COMPUTE`/`GRAPHICS`, chosen deterministically in family-index order. Their
 * absence is not an error: uploads use the shared compute queue
 * queue, then the compute queue, so correctness never depends on the queue choice
 * (P10) — the fences/semaphores in `streaming.rs` provide ordering.
*/

use ash::vk;
use color_eyre::eyre::{Result, eyre};

use super::bootstrap;
use crate::backend::vulkan::coopmat::CoopmatCaps;
use crate::backend::vulkan::coopmat2::Coopmat2Caps;
#[cfg(feature = "vulkan-profile")]
use crate::backend::vulkan::exec::profile::Profile;

pub(crate) const AMD_VENDOR_ID: u32 = 0x1002;

pub(crate) struct Device {
    pub device: ash::Device,
    pub physical: vk::PhysicalDevice,
    // PCI vendor identity is retained only for architecture-family capability
    // routing; product names and model IDs never enter kernel selection.
    pub vendor_id: u32,
    pub queue: vk::Queue,
    pub mem_props: vk::PhysicalDeviceMemoryProperties,
    // True when VK_EXT_memory_budget was advertised and enabled; gates `free_vram`
    // (read in `exec::commands`). Widened from pub(super) so that cross-folder module can read it.
    pub(in crate::backend::vulkan) memory_budget_enabled: bool,
    // The cooperative-matrix shape selected at bootstrap. `available = false` when
    // the device exposes no usable f16→f32 shape — the extension/feature are then NOT
    // enabled and the prefill routing keeps the f16×f16 tiled fallback.
    pub coopmat: CoopmatCaps,
    // NVIDIA workgroup cooperative-matrix2 support; never exposed outside Vulkan.
    pub coopmat2: Coopmat2Caps,
    // True when the device exposes integer dot product (dp4a, core 1.3): gates the mmvq
    // Q4_K decode GEMV. False means decode keeps the float GEMV.
    pub dp4a: bool,
    // Required start alignment (bytes) for a storage-buffer binding offset. A
    // sub-view's byte offset must be a multiple of this or the binding is
    // undefined; the prefill path (m3) validates row offsets against it before
    // constructing any view. Exposed here; not enforced at the buffer layer.
    pub min_storage_buffer_offset_alignment: u64,
    pub push_desc: ash::khr::push_descriptor::Device,
    pub cmd_pool: vk::CommandPool,
    #[cfg(feature = "vulkan-profile")]
    pub(crate) profile: Profile,
    // When set, the next recorded dispatch omits its trailing compute→compute
    // barrier (see `pipeline::record`). A run of mutually-independent dispatches
    // (e.g. the Q/K/V projections, which all read `normed` and write disjoint
    // buffers) needs only ONE global memory barrier after the last of the run, not
    // one after each: the barrier's global SHADER_WRITE→SHADER_READ scope makes all
    // prior writes visible at once. The flag is opt-in (default keeps the barrier),
    // so forgetting to set it costs performance, never correctness. Single-threaded
    // recording; the atomic only keeps `Device` `Sync` without a per-dispatch lock.
    pub skip_next_barrier: std::sync::atomic::AtomicBool,
    // Dropped last; the logical device above borrows nothing from these but the
    // instance/entry must outlive it.
    pub instance: ash::Instance,
    _entry: ash::Entry,
}

impl Device {
    // Orchestrates bring-up: create the instance, then delegate physical-device
    // selection, requirement checks and logical-device creation to `bootstrap`, and
    // assemble the `Device`. The instance creation stays here (it owns `entry`).
    pub(crate) fn init() -> Result<Device> {
        // SAFETY: loads the system Vulkan loader; failure is reported, not panic.
        let entry = unsafe { ash::Entry::load() }
            .map_err(|_| eyre!("vulkan: loader not available on this system"))?;

        // Vulkan 1.3: the shaders are SPIR-V 1.6 (build.rs) for the optional
        // coopmat/dp4a paths, consumed only at API 1.3. Pre-existing kernels are
        // unchanged; only the module version header rises.
        let app = vk::ApplicationInfo::default().api_version(vk::API_VERSION_1_3);

        let create = vk::InstanceCreateInfo::default().application_info(&app);
        // SAFETY: `entry` is the live loader; `create` and `app` are stack
        // values that outlive the call.
        let instance = unsafe { entry.create_instance(&create, None) }
            .map_err(|_| eyre!("vulkan: failed to create instance"))?;

        let cleanup_instance = |err| {
            // SAFETY: initialization has not returned a Device, so this local
            // instance has no surviving children after bootstrap cleans its device.
            unsafe { instance.destroy_instance(None) };
            err
        };
        let (physical, queue_family) =
            super::caps::pick_device(&instance).map_err(&cleanup_instance)?;
        // SAFETY: `instance` is live and `physical` is one of its enumerated devices.
        let mem_props = unsafe { instance.get_physical_device_memory_properties(physical) };
        // SAFETY: `instance` is live and `physical` is one of its enumerated devices.
        let dev_props = unsafe { instance.get_physical_device_properties(physical) };
        let min_storage_buffer_offset_alignment =
            dev_props.limits.min_storage_buffer_offset_alignment;
        // Upload staging and readback need a host-visible+coherent memory type.
        super::caps::require_host_visible(&mem_props).map_err(&cleanup_instance)?;
        super::caps::require_fp16(&instance, physical).map_err(&cleanup_instance)?;

        let boot = bootstrap::create_device(&entry, &instance, physical, queue_family)
            .map_err(&cleanup_instance)?;

        Ok(Device {
            device: boot.device,
            physical,
            vendor_id: dev_props.vendor_id,
            queue: boot.queue,
            mem_props,
            min_storage_buffer_offset_alignment,
            memory_budget_enabled: boot.memory_budget_enabled,
            coopmat: boot.coopmat,
            coopmat2: boot.coopmat2,
            dp4a: boot.dp4a,
            push_desc: boot.push_desc,
            cmd_pool: boot.cmd_pool,
            #[cfg(feature = "vulkan-profile")]
            profile: Profile::new(dev_props.limits.timestamp_period),
            skip_next_barrier: std::sync::atomic::AtomicBool::new(false),
            instance,
            _entry: entry,
        })
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        // SAFETY: drop runs once with exclusive ownership; `device_wait_idle` drains all
        // in-flight work so no GPU command still references these handles, then each is
        // destroyed in dependency order (pool → device → instance), child before parent.
        unsafe {
            let _ = self.device.device_wait_idle();
            #[cfg(feature = "vulkan-profile")]
            self.profile.finish(&self.device);
            self.device.destroy_command_pool(self.cmd_pool, None);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}
