/*
 * gh_zero_engine — synchronous Metal command lifecycle
 * Owns one command buffer and compute encoder, ends exactly once, commits,
 * waits, and accepts only completed status. It owns no dispatch policy.
 */

use color_eyre::eyre::{Result, eyre};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder, MTLCommandQueue,
    MTLComputeCommandEncoder,
};

use super::super::Device;

pub(crate) struct MetalEncoder {
    command: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    compute: Option<Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>>,
}

impl MetalEncoder {
    pub(crate) fn begin(device: &Device) -> Result<Self> {
        let command = device.queue.commandBuffer().ok_or_else(failed)?;
        let compute = command.computeCommandEncoder().ok_or_else(failed)?;
        Ok(Self {
            command,
            compute: Some(compute),
        })
    }

    pub(crate) fn compute(&self) -> Result<&ProtocolObject<dyn MTLComputeCommandEncoder>> {
        self.compute.as_deref().ok_or_else(failed)
    }

    pub(crate) fn submit(mut self) -> Result<()> {
        self.end()?;
        self.command.commit();
        self.command.waitUntilCompleted();
        completed(self.command.status())
    }

    fn end(&mut self) -> Result<()> {
        let compute = self.compute.take().ok_or_else(failed)?;
        compute.endEncoding();
        Ok(())
    }
}

impl Drop for MetalEncoder {
    fn drop(&mut self) {
        if let Some(compute) = self.compute.take() {
            compute.endEncoding();
        }
    }
}

fn completed(status: MTLCommandBufferStatus) -> Result<()> {
    if status == MTLCommandBufferStatus::Completed {
        Ok(())
    } else {
        Err(failed())
    }
}

fn failed() -> color_eyre::Report {
    eyre!("metal: command buffer failed")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_completed_status_is_accepted() {
        assert!(completed(MTLCommandBufferStatus::Completed).is_ok());
        for status in [
            MTLCommandBufferStatus::NotEnqueued,
            MTLCommandBufferStatus::Enqueued,
            MTLCommandBufferStatus::Committed,
            MTLCommandBufferStatus::Scheduled,
            MTLCommandBufferStatus::Error,
        ] {
            assert_eq!(
                completed(status).unwrap_err().to_string(),
                "metal: command buffer failed"
            );
        }
    }

    #[test]
    fn empty_command_buffer_completes_on_the_qualified_device() -> Result<()> {
        MetalEncoder::begin(&Device::acquire()?)?.submit()
    }
}
