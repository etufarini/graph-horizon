/*
 * gh_zero_engine — Vulkan reduction readback
 * Records the compute-to-transfer dependency and a bounded device-to-host copy,
 * then extracts bytes from the coherent host mirror after submission completes.
 * Callers own command submission and result decoding; this file owns no buffers.
 */

use ash::vk;
use color_eyre::eyre::{Result, bail, eyre};

use crate::backend::vulkan::buffers::GpuBuffer;
use crate::backend::vulkan::device::Device;

pub(super) fn record(
    dev: &Device,
    cmd: vk::CommandBuffer,
    source: &GpuBuffer,
    host: &GpuBuffer,
    bytes: u64,
) -> Result<()> {
    validate(source, host, bytes)?;

    let barrier = vk::MemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::SHADER_WRITE)
        .dst_access_mask(vk::AccessFlags::TRANSFER_READ);
    let region = vk::BufferCopy::default()
        .src_offset(source.offset)
        .dst_offset(0)
        .size(bytes);
    // SAFETY: `cmd` is recording; source and host are live for the command's
    // lifetime; the checks above prove the copy range fits both buffer windows.
    unsafe {
        dev.device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[barrier],
            &[],
            &[],
        );
        dev.device
            .cmd_copy_buffer(cmd, source.buffer, host.buffer, &[region]);
    }
    Ok(())
}

pub(super) fn validate(source: &GpuBuffer, host: &GpuBuffer, bytes: u64) -> Result<()> {
    require_window(source, bytes)?;
    require_host_window(host, bytes)
}

pub(super) fn completed(dev: &Device, host: &GpuBuffer, bytes: usize) -> Result<Vec<u8>> {
    let bytes_u64 = u64::try_from(bytes).map_err(|_| eyre!("vulkan: reduce readback too large"))?;
    require_host_window(host, bytes_u64)?;
    let mut out = vec![0u8; bytes];
    // SAFETY: the host allocation is coherent and no command remains in flight;
    // the range check proves `bytes` fits the full offset-zero mapped mirror.
    unsafe {
        let ptr = dev
            .device
            .map_memory(host.memory, 0, host.size, vk::MemoryMapFlags::empty())
            .map_err(|_| eyre!("vulkan: reduce map failed"))?;
        std::ptr::copy_nonoverlapping(ptr.cast::<u8>(), out.as_mut_ptr(), bytes);
        dev.device.unmap_memory(host.memory);
    }
    Ok(out)
}

fn require_window(buffer: &GpuBuffer, bytes: u64) -> Result<()> {
    if bytes > buffer.size || buffer.offset.checked_add(bytes).is_none() {
        bail!("vulkan: reduce readback past buffer window");
    }
    Ok(())
}

fn require_host_window(host: &GpuBuffer, bytes: u64) -> Result<()> {
    if !host.host_visible || host.offset != 0 {
        bail!("vulkan: invalid reduce host mirror");
    }
    require_window(host, bytes)
}
