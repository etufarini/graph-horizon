/*
 * graph_horizon_engine — synchronous Metal command lifecycle
 * Owns one command buffer and compute encoder, ends exactly once, commits,
 * waits, and accepts only completed status. It owns no dispatch policy.
 */

use color_eyre::eyre::{Result, eyre};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandQueue, MTLComputeCommandEncoder,
};

#[cfg(not(feature = "metal-profile"))]
use objc2_metal::MTLCommandEncoder;

#[cfg(feature = "metal-profile")]
use std::{cell::RefCell, time::Instant};

use super::super::Device;
use super::super::pipeline::Kernel;
#[cfg(feature = "metal-profile")]
use super::profile::ProfileCommand;

pub(crate) struct MetalEncoder {
    command: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    #[cfg(not(feature = "metal-profile"))]
    compute: Option<Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>>,
    #[cfg(feature = "metal-profile")]
    profile: RefCell<ProfileCommand>,
    #[cfg(feature = "metal-profile")]
    started: Instant,
}

impl MetalEncoder {
    pub(crate) fn begin(device: &Device) -> Result<Self> {
        let command = device.queue.commandBuffer().ok_or_else(failed)?;
        #[cfg(not(feature = "metal-profile"))]
        let compute = command.computeCommandEncoder().ok_or_else(failed)?;
        Ok(Self {
            command,
            #[cfg(not(feature = "metal-profile"))]
            compute: Some(compute),
            #[cfg(feature = "metal-profile")]
            profile: RefCell::new(device.profile.command()),
            #[cfg(feature = "metal-profile")]
            started: Instant::now(),
        })
    }

    #[cfg(not(feature = "metal-profile"))]
    pub(crate) fn compute(&self) -> Result<&ProtocolObject<dyn MTLComputeCommandEncoder>> {
        self.compute.as_deref().ok_or_else(failed)
    }

    pub(crate) fn encode(
        &self,
        _kernel: Kernel,
        _constants: &[u8],
        body: impl FnOnce(&ProtocolObject<dyn MTLComputeCommandEncoder>),
    ) -> Result<()> {
        #[cfg(feature = "metal-profile")]
        {
            let compute = self
                .profile
                .borrow_mut()
                .encoder(&self.command, _kernel, _constants)?;
            body(&compute);
        }
        #[cfg(not(feature = "metal-profile"))]
        body(self.compute()?);
        Ok(())
    }

    pub(crate) fn submit(mut self) -> Result<()> {
        self.end()?;
        #[cfg(feature = "metal-profile")]
        let record_ms = self.started.elapsed().as_secs_f64() * 1000.0;
        #[cfg(feature = "metal-profile")]
        let submit_start = Instant::now();
        self.command.commit();
        #[cfg(feature = "metal-profile")]
        let submit_ms = submit_start.elapsed().as_secs_f64() * 1000.0;
        #[cfg(feature = "metal-profile")]
        let wait_start = Instant::now();
        self.command.waitUntilCompleted();
        #[cfg(feature = "metal-profile")]
        let wait_ms = wait_start.elapsed().as_secs_f64() * 1000.0;
        completed(self.command.status())?;
        #[cfg(feature = "metal-profile")]
        self.profile.get_mut().resolve(
            (self.command.GPUEndTime() - self.command.GPUStartTime()) * 1000.0,
            [record_ms, submit_ms, wait_ms],
        )?;
        Ok(())
    }

    #[cfg(not(feature = "metal-profile"))]
    fn end(&mut self) -> Result<()> {
        let compute = self.compute.take().ok_or_else(failed)?;
        compute.endEncoding();
        Ok(())
    }

    #[cfg(feature = "metal-profile")]
    fn end(&mut self) -> Result<()> {
        self.profile.get_mut().finish_encoder();
        Ok(())
    }
}

impl Drop for MetalEncoder {
    fn drop(&mut self) {
        #[cfg(feature = "metal-profile")]
        self.profile.get_mut().finish_encoder();
        #[cfg(not(feature = "metal-profile"))]
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
