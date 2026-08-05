/*
 * graph_horizon_engine — Vulkan backend host read-back
 * Copies FP32 logits through the persistent host-visible mirror. Tests also use
 * the bounded raw-byte reader for exact buffer parity checks; production decode
 * never allocates a read-back staging buffer.
*/

use ash::vk;
#[cfg(test)]
use color_eyre::eyre::bail;
use color_eyre::eyre::{Result, eyre};

use crate::backend::vulkan::VulkanBackend;
use crate::backend::vulkan::buffers::GpuBuffer;

// Copies the FP32 logits buffer to the persistent host-visible mirror and
// reads it — no per-token buffer allocation.
pub(in crate::backend::vulkan) fn read_logits(
    b: &VulkanBackend,
    logits: &GpuBuffer,
    vocab: usize,
) -> Result<Vec<f32>> {
    let bytes = vocab as u64 * 4;
    let host = &b.logits_host;
    let cmd = b.dev.begin_commands()?;
    let region = vk::BufferCopy::default().size(bytes);
    // SAFETY: `cmd` is recording (from begin_commands); `logits` and `host` are live
    // buffers and `bytes = vocab*4` is within both (`host` is the FP32 vocab mirror).
    unsafe {
        b.dev
            .device
            .cmd_copy_buffer(cmd, logits.buffer, host.buffer, &[region])
    };
    b.dev.submit_wait(cmd)?;

    let mut out = vec![0f32; vocab];
    // SAFETY: host-coherent mapping of `bytes`; we copy exactly `bytes` out.
    unsafe {
        let ptr = b
            .dev
            .device
            .map_memory(host.memory, 0, host.size, vk::MemoryMapFlags::empty())
            .map_err(|_| eyre!("vulkan: logits map failed"))?;
        std::ptr::copy_nonoverlapping(
            ptr as *const u8,
            out.as_mut_ptr() as *mut u8,
            bytes as usize,
        );
        b.dev.device.unmap_memory(host.memory);
    }
    Ok(out)
}

// Raw byte read-back from `buf`'s window start through a transient host-visible
// staging buffer (D2H). Dev-tool primitive (KV payload dumps): a per-call
// staging allocation is fine off the hot path. Honours the source view's byte
// origin so a `view` window reads only itself.
#[cfg(test)]
pub(in crate::backend::vulkan) fn read_bytes(
    b: &VulkanBackend,
    buf: &GpuBuffer,
    bytes: usize,
) -> Result<Vec<u8>> {
    if bytes as u64 > buf.size {
        bail!("vulkan: read_bytes past buffer window");
    }
    let staging = GpuBuffer::alloc(&b.dev, bytes as u64, true)?;
    let result = (|| {
        #[cfg(test)]
        {
            let _tracked = crate::backend::vulkan::fault::track(
                crate::backend::vulkan::fault::Point::Readback,
            );
            crate::backend::vulkan::fault::hit(crate::backend::vulkan::fault::Point::Readback)?;
        }
        let cmd = b.dev.begin_commands()?;
        let region = vk::BufferCopy::default()
            .src_offset(buf.offset)
            .size(bytes as u64);
        // SAFETY: the checked source window and staging buffer both hold `bytes`.
        unsafe {
            b.dev
                .device
                .cmd_copy_buffer(cmd, buf.buffer, staging.buffer, &[region])
        };
        b.dev.submit_wait(cmd)?;
        let mut out = vec![0u8; bytes];
        // SAFETY: host-coherent mapping of staging.size; copy exactly `bytes`.
        unsafe {
            let ptr = b
                .dev
                .device
                .map_memory(staging.memory, 0, staging.size, vk::MemoryMapFlags::empty())
                .map_err(|_| eyre!("vulkan: read_bytes map failed"))?;
            std::ptr::copy_nonoverlapping(ptr as *const u8, out.as_mut_ptr(), bytes);
            b.dev.device.unmap_memory(staging.memory);
        }
        Ok(out)
    })();
    staging.destroy(&b.dev);
    result
}
