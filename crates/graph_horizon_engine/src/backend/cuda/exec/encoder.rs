/*
 * graph_horizon_engine — one-stream CUDA encoder and first-error latch.
 * Void trait operations latch a sanitized launch failure; submission always
 * synchronizes the stream exactly once before reporting success or failure.
 */

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use color_eyre::eyre::{Result, eyre};
use cudarc::driver::CudaStream;

use super::super::Device;

pub(crate) struct CudaEncoder {
    pub(crate) stream: Arc<CudaStream>,
    failed: AtomicBool,
    #[cfg(test)]
    launches: std::sync::atomic::AtomicUsize,
}

impl CudaEncoder {
    pub(crate) fn begin(device: &Device) -> Self {
        Self {
            stream: device.stream.clone(),
            failed: AtomicBool::new(false),
            #[cfg(test)]
            launches: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    pub(crate) fn ready(&self) -> bool {
        !self.failed.load(Ordering::Acquire)
    }

    pub(crate) fn latch(&self, result: Result<()>) {
        if result.is_err() {
            self.failed.store(true, Ordering::Release);
        }
    }

    pub(crate) fn submitted_launch(&self) {
        #[cfg(test)]
        self.launches.fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn submit(self) -> Result<()> {
        let synchronized = self.stream.synchronize().is_ok();
        if synchronized && self.ready() {
            Ok(())
        } else {
            Err(failed())
        }
    }
}

fn failed() -> color_eyre::Report {
    eyre!("cuda: command submission failed")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::cuda::kernels;
    use crate::backend::cuda::mem::buffer::{CudaBuffer, CudaFormat};
    use crate::backend::cuda::module::Module;

    #[test]
    fn first_failure_suppresses_later_dispatches() -> Result<()> {
        let device = Device::acquire()?;
        let module = Module::load(&device.context)?;
        let x = CudaBuffer::allocate(&device, 4, CudaFormat::F32)?;
        let y = CudaBuffer::allocate(&device, 2, CudaFormat::F16)?;
        let encoder = CudaEncoder::begin(&device);
        assert!(encoder.ready());
        encoder.latch(Err(failed()));
        encoder.latch(Ok(()));
        assert!(!encoder.ready());
        kernels::residual_add::encode(&encoder, &module, &x, &y, 1)?;
        assert_eq!(encoder.launches.load(Ordering::Relaxed), 0);
        assert_eq!(
            encoder.submit().unwrap_err().to_string(),
            "cuda: command submission failed"
        );
        Ok(())
    }

    #[test]
    fn empty_submission_synchronizes_successfully() -> Result<()> {
        let device = Device::acquire()?;
        CudaEncoder::begin(&device).submit()
    }
}
