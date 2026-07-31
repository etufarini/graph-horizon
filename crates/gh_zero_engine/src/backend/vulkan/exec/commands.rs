/*
 * gh_zero_engine — Vulkan device runtime helpers
 * Operations on an already-created `Device`: live-VRAM query (VK_EXT_memory_budget),
 * memory-type selection, one-shot command-buffer begin/submit-wait, and the
 * compute→compute barrier. Bodies moved 1:1 from the former monolithic file.
*/

use ash::vk;
use color_eyre::eyre::{Result, eyre};

use crate::backend::vulkan::device::Device;

impl Device {
    // Live free VRAM (bytes) of the largest device-local heap, or `None` when
    // VK_EXT_memory_budget is absent. Reads `heapBudget − heapUsage` for that heap;
    // a driver reporting usage > budget yields 0 (saturating). Hybrid auto split only.
    pub(crate) fn free_vram(&self) -> Option<u64> {
        if !self.memory_budget_enabled {
            return None;
        }
        let mut budget = vk::PhysicalDeviceMemoryBudgetPropertiesEXT::default();
        {
            let mut props2 = vk::PhysicalDeviceMemoryProperties2::default().push_next(&mut budget);
            // SAFETY: physical device is valid; props2 carries the budget extension
            // struct the driver fills in. Scoped so the &mut budget borrow ends here.
            unsafe {
                self.instance
                    .get_physical_device_memory_properties2(self.physical, &mut props2)
            };
        }
        let heaps = &self.mem_props.memory_heaps[..self.mem_props.memory_heap_count as usize];
        let mut best: Option<(u64, u64)> = None; // (heapBudget, free) of the chosen heap
        for (i, h) in heaps.iter().enumerate() {
            if h.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL) {
                let total = budget.heap_budget[i];
                let free = total.saturating_sub(budget.heap_usage[i]);
                if best.map(|(b, _)| total > b).unwrap_or(true) {
                    best = Some((total, free));
                }
            }
        }
        best.map(|(_, free)| free)
    }

    // Picks a memory type index supporting `flags` among `type_bits`.
    pub(crate) fn find_memory_type(
        &self,
        type_bits: u32,
        flags: vk::MemoryPropertyFlags,
    ) -> Option<u32> {
        (0..self.mem_props.memory_type_count).find(|&i| {
            type_bits & (1 << i) != 0
                && self.mem_props.memory_types[i as usize]
                    .property_flags
                    .contains(flags)
        })
    }

    // Allocates and begins a primary command buffer for a one-shot submission.
    pub(crate) fn begin_commands(&self) -> Result<vk::CommandBuffer> {
        let info = vk::CommandBufferAllocateInfo::default()
            .command_pool(self.cmd_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        // SAFETY: `device` is alive (owned by self) and `cmd_pool` is this backend's
        // own pool, not used concurrently (synchronous, batch-1 design); `info`
        // requests exactly 1 buffer, so indexing `[0]` of the returned slice is in range.
        let cmd = unsafe { self.device.allocate_command_buffers(&info) }
            .map_err(|_| eyre!("vulkan: cannot allocate command buffer"))?[0];
        let begin = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        // SAFETY: `cmd` was just allocated from `cmd_pool` and is in the initial
        // (non-recording) state, so opening a recording session on it is valid.
        if unsafe { self.device.begin_command_buffer(cmd, &begin) }.is_err() {
            // SAFETY: allocation succeeded but recording did not begin; no queue
            // references this command buffer.
            unsafe { self.device.free_command_buffers(self.cmd_pool, &[cmd]) };
            return Err(eyre!("vulkan: cannot begin command buffer"));
        }
        #[cfg(test)]
        {
            let _tracked =
                crate::backend::vulkan::fault::track(crate::backend::vulkan::fault::Point::Record);
            if let Err(error) =
                crate::backend::vulkan::fault::hit(crate::backend::vulkan::fault::Point::Record)
            {
                // SAFETY: this command buffer is local, recording, and was never submitted.
                unsafe { self.device.free_command_buffers(self.cmd_pool, &[cmd]) };
                return Err(error);
            }
        }
        Ok(cmd)
    }

    // Ends, submits and waits on `cmd`, then frees it. Synchronous by design
    // (batch 1, correctness first): one submit per forward step.
    pub(crate) fn submit_wait(&self, cmd: vk::CommandBuffer) -> Result<()> {
        // SAFETY: `cmd` is in the recording state opened by `begin_commands`; ending
        // it is the required transition before submission.
        if unsafe { self.device.end_command_buffer(cmd) }.is_err() {
            // SAFETY: the failed-to-end buffer was never submitted.
            unsafe { self.device.free_command_buffers(self.cmd_pool, &[cmd]) };
            return Err(eyre!("vulkan: cannot end command buffer"));
        }
        let cmds = [cmd];
        let submit = vk::SubmitInfo::default().command_buffers(&cmds);
        // SAFETY: `device` is alive; the default (unsignaled) fence info is valid.
        let fence = match unsafe {
            self.device
                .create_fence(&vk::FenceCreateInfo::default(), None)
        } {
            Ok(fence) => fence,
            Err(_) => {
                // SAFETY: the ended command buffer was never submitted.
                unsafe { self.device.free_command_buffers(self.cmd_pool, &cmds) };
                return Err(eyre!("vulkan: cannot create fence"));
            }
        };
        #[cfg(test)]
        {
            let _tracked =
                crate::backend::vulkan::fault::track(crate::backend::vulkan::fault::Point::Submit);
            if let Err(error) =
                crate::backend::vulkan::fault::hit(crate::backend::vulkan::fault::Point::Submit)
            {
                // SAFETY: the fence and ended command are local and no queue saw them.
                unsafe {
                    self.device.destroy_fence(fence, None);
                    self.device.free_command_buffers(self.cmd_pool, &cmds);
                }
                return Err(error);
            }
        }
        // A GPU watchdog (TDR) reset surfaces as VK_ERROR_DEVICE_LOST on the submit or the
        // fence wait. Tag those with the stable `E_DEVICE_LOST` marker so the prefill
        // driver can recognise them and retry at a smaller chunk (m1-T2) without naming
        // Vulkan in the backend-agnostic skeleton. Other failures keep their own message.
        // SAFETY: `queue` and `device` are alive; `cmd` is recorded and ended; `fence`
        // was just created unsignaled. The fence wait blocks until the GPU is idle, so
        // destroying the fence and freeing `cmd` afterwards races nothing in flight.
        unsafe {
            let submit_result = self
                .device
                .queue_submit(self.queue, &[submit], fence)
                .map_err(|e| {
                    if e == vk::Result::ERROR_DEVICE_LOST {
                        eyre!("vulkan: queue submit failed (E_DEVICE_LOST)")
                    } else {
                        eyre!("vulkan: queue submit failed")
                    }
                })
                .and_then(|_| {
                    self.device
                        .wait_for_fences(&[fence], true, u64::MAX)
                        .map_err(|e| {
                            if e == vk::Result::ERROR_DEVICE_LOST {
                                eyre!("vulkan: fence wait failed (E_DEVICE_LOST)")
                            } else {
                                eyre!("vulkan: fence wait failed")
                            }
                        })
                });
            // The wait completed or the device was lost; either way these local
            // synchronization/command objects must not escape this boundary.
            self.device.destroy_fence(fence, None);
            self.device.free_command_buffers(self.cmd_pool, &cmds);
            submit_result
        }
    }

    // Global shader-write → shader-read barrier between dependent dispatches.
    pub(crate) fn compute_barrier(&self, cmd: vk::CommandBuffer) {
        let barrier = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE);
        // SAFETY: `cmd` is in the recording state; `barrier` outlives the call (it is a
        // stack value passed by slice) and describes a self-contained memory dependency.
        unsafe {
            self.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[barrier],
                &[],
                &[],
            );
        }
    }
}
