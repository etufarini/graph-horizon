/*
 * gh_zero_engine — Vulkan deterministic argmax
 * Records the existing argmax kernel and four-byte transfer on a caller-owned
 * command buffer, decodes a completed host mirror, and composes both steps for
 * standalone reads. The backend retains command and persistent-buffer ownership.
 */

use ash::vk;
use color_eyre::eyre::{Result, bail, eyre};

use super::readback;
use crate::backend::vulkan::buffers::GpuBuffer;
use crate::backend::vulkan::device::Device;
use crate::backend::vulkan::pipeline::{Kernel, PipelineRegistry, dispatch};

pub(crate) fn record(
    dev: &Device,
    reg: &PipelineRegistry,
    cmd: vk::CommandBuffer,
    logits: &GpuBuffer,
    reduce: &GpuBuffer,
    host: &GpuBuffer,
    vocab: usize,
) -> Result<()> {
    let vocab = validate(dev, reduce, host, vocab)?;
    dispatch(
        dev,
        reg,
        cmd,
        Kernel::Argmax,
        &[
            (logits.buffer, logits.offset, logits.size),
            (reduce.buffer, reduce.offset, reduce.size),
        ],
        &vocab.to_le_bytes(),
        1,
    );
    readback::record(dev, cmd, reduce, host, 4)
}

pub(crate) fn completed(dev: &Device, host: &GpuBuffer, vocab: usize) -> Result<u32> {
    let vocab = checked_vocab(vocab)?;
    let bytes = readback::completed(dev, host, 4)?;
    let token = u32::from_le_bytes(
        bytes
            .try_into()
            .map_err(|_| eyre!("vulkan: incomplete argmax readback"))?,
    );
    if token >= vocab {
        bail!("vulkan: invalid argmax result");
    }
    Ok(token)
}

pub(crate) fn read(
    dev: &Device,
    reg: &PipelineRegistry,
    logits: &GpuBuffer,
    reduce: &GpuBuffer,
    host: &GpuBuffer,
    vocab: usize,
) -> Result<u32> {
    validate(dev, reduce, host, vocab)?;
    let cmd = dev.begin_commands()?;
    record(dev, reg, cmd, logits, reduce, host, vocab)?;
    dev.submit_wait(cmd)?;
    completed(dev, host, vocab)
}

fn validate(dev: &Device, reduce: &GpuBuffer, host: &GpuBuffer, vocab: usize) -> Result<u32> {
    let vocab = checked_vocab(vocab)?;
    require_aligned(dev, reduce)?;
    readback::validate(reduce, host, 4)?;
    Ok(vocab)
}

fn checked_vocab(vocab: usize) -> Result<u32> {
    if vocab == 0 {
        bail!("vulkan: read_argmax on empty logits");
    }
    u32::try_from(vocab).map_err(|_| eyre!("vulkan: argmax vocabulary too large"))
}

fn require_aligned(dev: &Device, reduce: &GpuBuffer) -> Result<()> {
    let align = dev.min_storage_buffer_offset_alignment;
    if !reduce.offset.is_multiple_of(align) {
        bail!("vulkan: reduce buffer offset not aligned to {align} bytes");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::Backend;
    use crate::backend::vulkan::{VulkanBackend, fault};

    #[test]
    fn fused_argmax_uses_one_submit_and_does_not_retry_failure() {
        let mut backend = match VulkanBackend::bare() {
            Ok(backend) => backend,
            Err(_) => {
                eprintln!("external verification: no Vulkan device");
                return;
            }
        };
        let host = GpuBuffer::alloc(&backend.dev, 16, true).expect("host mirror");
        backend.logits_host.destroy(&backend.dev);
        backend.logits_host = host;
        let logits = backend.alloc_buffer(16).expect("logits");
        let values = [1.0_f32, 3.0, 3.0, 2.0];
        let bytes: Vec<u8> = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        backend
            .upload_bytes(&logits, &bytes)
            .expect("upload logits");

        crate::backend::vulkan::reset_submit_count();
        let enc = backend.begin().expect("begin fused argmax");
        assert_eq!(backend.submit_argmax(enc, &logits, 4).unwrap(), 1);
        assert_eq!(crate::backend::vulkan::submit_count(), 1);

        crate::backend::vulkan::reset_submit_count();
        let enc = backend.begin().expect("begin failing fused argmax");
        fault::arm(fault::Point::Submit);
        let error = backend
            .submit_argmax(enc, &logits, 4)
            .expect_err("submit failure is terminal");
        assert_eq!(error.to_string(), "vulkan: injected test failure");
        assert_eq!(crate::backend::vulkan::submit_count(), 1);
        assert_eq!(fault::live(), [0; 7]);

        backend.free_buffer(logits);
    }
}
