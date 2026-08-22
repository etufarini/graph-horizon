/*
 * graph_horizon_engine — Vulkan buffer primitive
 * Defines the GpuBuffer (a VkBuffer + its memory, carrying its WeightFormat),
 * allocation and upload (device-local via a host-visible staging buffer + copy;
 * host-visible via direct mapping), views, and destruction. Persistent backend
 * assembly lives in `vulkan::loader`; generic shape holders live in
 * `backend::buffers`; weight upload lives in `vulkan::mem::weights`.
*/

use ash::vk;
use color_eyre::eyre::{Result, eyre};

use crate::backend::vulkan::device::Device;

// Weight layout for matmul/dequant dispatch. F16 covers genuine F16 weights and F32
// norms converted on upload; Q4K/Q5K/Q6K keep ggml block bytes.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum WeightFormat {
    F16,
    Q4K,
    Q5K,
    Q6K,
}

// One GPU buffer plus the memory backing it. `host_visible` records whether it
// was placed in host RAM (weight spill) so uploads pick mapping vs staging. `quant`
// records the on-GPU weight layout so the matmul dispatch picks the right kernel;
// it defaults to F16 and is set by `vulkan::mem::weights` for quantized weights.
pub(crate) struct GpuBuffer {
    pub buffer: vk::Buffer,
    pub memory: vk::DeviceMemory,
    pub size: u64,
    pub host_visible: bool,
    pub quant: WeightFormat,
    // Origin in bytes inside `buffer`/`memory`. Always 0 for buffers returned by
    // `alloc`; non-zero only for sub-views produced by `view`. A view shares the
    // parent's `buffer`/`memory` handles and MUST NOT be destroyed: `destroy`
    // would free the parent's allocation (double-free). There is deliberately no
    // `Drop` impl, so a view going out of scope is a no-op.
    pub offset: u64,
}

impl GpuBuffer {
    pub(crate) fn alloc(dev: &Device, size: u64, host: bool) -> Result<GpuBuffer> {
        let usage = vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::TRANSFER_SRC
            | vk::BufferUsageFlags::TRANSFER_DST;
        let info = vk::BufferCreateInfo::default()
            .size(size.max(1))
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        // SAFETY: `dev.device` is alive; `info` is a stack struct valid for the call.
        let buffer = unsafe { dev.device.create_buffer(&info, None) }
            .map_err(|_| eyre!("vulkan: buffer creation failed"))?;
        // SAFETY: `buffer` was just created from `dev.device` and not yet destroyed.
        let req = unsafe { dev.device.get_buffer_memory_requirements(buffer) };
        let flags = if host {
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT
        } else {
            vk::MemoryPropertyFlags::DEVICE_LOCAL
        };
        let mt = dev
            .find_memory_type(req.memory_type_bits, flags)
            .ok_or_else(|| {
                // SAFETY: `buffer` has not been bound or shared yet.
                unsafe { dev.device.destroy_buffer(buffer, None) };
                eyre!("vulkan: no compatible memory type for buffer")
            })?;
        let alloc = vk::MemoryAllocateInfo::default()
            .allocation_size(req.size)
            .memory_type_index(mt);
        // SAFETY: `dev.device` is alive; `alloc` sizes the allocation to `req.size` with a
        // memory type (`mt`) the requirements declared compatible.
        let memory = match unsafe { dev.device.allocate_memory(&alloc, None) } {
            Ok(memory) => memory,
            Err(_) => {
                // SAFETY: `buffer` has not been bound or shared yet.
                unsafe { dev.device.destroy_buffer(buffer, None) };
                return Err(eyre!("vulkan: memory allocation failed"));
            }
        };
        // SAFETY: `buffer` and `memory` were just created from this device; `memory` is at
        // least `req.size` and unbound, so binding it at offset 0 is valid.
        if unsafe { dev.device.bind_buffer_memory(buffer, memory, 0) }.is_err() {
            // SAFETY: both handles are local to this failed allocation.
            unsafe {
                dev.device.destroy_buffer(buffer, None);
                dev.device.free_memory(memory, None);
            }
            return Err(eyre!("vulkan: bind buffer memory failed"));
        }
        Ok(GpuBuffer {
            buffer,
            memory,
            size: req.size,
            host_visible: host,
            quant: WeightFormat::F16,
            offset: 0,
        })
    }

    // A sub-view aliasing this buffer's storage over `[offset, offset + len)`.
    // Copies the (shared) `buffer`/`memory`/`quant` handles, sets `size = len`
    // and composes the byte origin (`self.offset + offset`), so a view of a view
    // accumulates the offsets. No allocation, no copy. The result MUST NOT be
    // destroyed (see the `offset` field comment).
    pub(crate) fn view(&self, offset: u64, len: u64) -> GpuBuffer {
        debug_assert!(
            offset + len <= self.size,
            "GpuBuffer::view out of bounds: offset + len > size"
        );
        GpuBuffer {
            buffer: self.buffer,
            memory: self.memory,
            size: len,
            host_visible: self.host_visible,
            quant: self.quant,
            offset: self.offset + offset,
        }
    }

    // Uploads `data` (≤ size). Host-visible: map and copy. Device-local: stage
    // through a temporary host buffer and a GPU copy.
    pub(crate) fn upload(&self, dev: &Device, data: &[u8]) -> Result<()> {
        if data.len() as u64 > self.size {
            return Err(eyre!("vulkan: upload larger than buffer"));
        }
        if self.host_visible {
            return self.write_mapped(dev, data);
        }
        let staging = GpuBuffer::alloc(dev, data.len() as u64, true)?;
        let result = (|| {
            #[cfg(test)]
            {
                let _tracked = crate::backend::vulkan::fault::track(
                    crate::backend::vulkan::fault::Point::Transfer,
                );
                crate::backend::vulkan::fault::hit(crate::backend::vulkan::fault::Point::Transfer)?;
            }
            staging.write_mapped(dev, data)?;
            let cmd = dev.begin_commands()?;
            // Honour this buffer's byte origin so a sub-view uploads into its window.
            let region = vk::BufferCopy::default()
                .dst_offset(self.offset)
                .size(data.len() as u64);
            // SAFETY: both buffers are live and the checked copy fits this window.
            unsafe {
                dev.device
                    .cmd_copy_buffer(cmd, staging.buffer, self.buffer, &[region])
            };
            dev.submit_wait(cmd)
        })();
        // The staging allocation is temporary even when recording/submission fails.
        staging.destroy(dev);
        result
    }

    fn write_mapped(&self, dev: &Device, data: &[u8]) -> Result<()> {
        // SAFETY: host-coherent mapping of exactly this allocation; we copy at
        // most `size` bytes and unmap before returning.
        unsafe {
            let ptr = dev
                .device
                .map_memory(self.memory, 0, self.size, vk::MemoryMapFlags::empty())
                .map_err(|_| eyre!("vulkan: memory map failed"))?;
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr as *mut u8, data.len());
            dev.device.unmap_memory(self.memory);
        }
        Ok(())
    }

    pub(crate) fn destroy(&self, dev: &Device) {
        // SAFETY: caller guarantees this is a real allocation (never a view — see the
        // `offset` field contract) and no in-flight GPU work still references it; the
        // buffer is destroyed before its backing memory is freed.
        unsafe {
            dev.device.destroy_buffer(self.buffer, None);
            dev.device.free_memory(self.memory, None);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // GpuBuffer::view is pure handle arithmetic (no device needed): it shares the
    // VkBuffer/memory, sets size = len and composes the byte offset. A real
    // write-through-then-read test needs a GPU, so it is out of scope for the unit
    // tests; the CPU backend covers the aliasing semantics end to end.
    #[test]
    fn view_composes_offset_and_shares_handles() {
        let base = GpuBuffer {
            buffer: vk::Buffer::null(),
            memory: vk::DeviceMemory::null(),
            size: 100,
            host_visible: false,
            quant: WeightFormat::F16,
            offset: 0,
        };
        let v = base.view(16, 32);
        assert_eq!(v.offset, 16);
        assert_eq!(v.size, 32);
        assert_eq!(v.buffer, base.buffer);
        assert_eq!(v.memory, base.memory);
        // A view of a view accumulates the offsets.
        let v2 = v.view(8, 8);
        assert_eq!(v2.offset, 24);
        assert_eq!(v2.size, 8);
    }
}
