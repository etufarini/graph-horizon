/*
 * CUDA buffer transfers. Every CPU/device copy binds the owning context and
 * completes its required synchronization before host or device storage releases.
 */

use color_eyre::eyre::{Result, eyre};
use cudarc::driver::{DevicePtr, result};

use super::{CudaBuffer, CudaFormat, arithmetic};
use crate::backend::cuda::Device;

impl CudaBuffer {
    pub(crate) fn upload(device: &Device, bytes: &[u8], format: CudaFormat) -> Result<Self> {
        let allocation = device
            .stream
            .clone_htod(bytes)
            .map_err(|_| eyre!("cuda: weight upload failed"))?;
        Ok(Self::from_allocation(allocation, bytes.len(), format))
    }

    pub(crate) fn read(&self, device: &Device, bytes: usize) -> Result<Vec<u8>> {
        if bytes > self.len {
            return Err(eyre!("cuda: readback failed"));
        }
        device
            .context
            .bind_to_thread()
            .map_err(|_| eyre!("cuda: readback failed"))?;
        let (base, guard) = self.allocation.device_ptr(&device.stream);
        let source = base
            .checked_add(u64::try_from(self.offset).map_err(|_| arithmetic())?)
            .ok_or_else(arithmetic)?;
        let mut output = vec![0; bytes];
        // SAFETY: Arc keeps the u8 allocation live, the checked view bounds the
        // source range, the current context owns the sole stream, and output
        // remains fixed until the immediate synchronization completes.
        unsafe { result::memcpy_dtoh_async(&mut output, source, device.stream.cu_stream()) }
            .map_err(|_| eyre!("cuda: readback failed"))?;
        device
            .stream
            .synchronize()
            .map_err(|_| eyre!("cuda: readback failed"))?;
        drop(guard);
        Ok(output)
    }

    #[cfg(any(feature = "cuda-hybrid", test))]
    pub(crate) fn write(&self, device: &Device, bytes: &[u8]) -> Result<()> {
        if bytes.len() > self.len {
            return Err(eyre!("cuda: residual upload failed"));
        }
        device
            .context
            .bind_to_thread()
            .map_err(|_| eyre!("cuda: residual upload failed"))?;
        // The synchronous Driver copy is not enqueued on the backend stream;
        // complete prior work so a queued kernel or initialization cannot
        // overwrite the residual after this method returns.
        device
            .stream
            .synchronize()
            .map_err(|_| eyre!("cuda: residual upload failed"))?;
        let (base, guard) = self.allocation.device_ptr(&device.stream);
        let target = base
            .checked_add(u64::try_from(self.offset).map_err(|_| arithmetic())?)
            .ok_or_else(arithmetic)?;
        // SAFETY: the bound context owns the live allocation, the composed
        // view bounds the complete destination span, `bytes` remains fixed,
        // and the synchronous Driver call completes before either can drop.
        unsafe { result::memcpy_htod_sync(target, bytes) }
            .map_err(|_| eyre!("cuda: residual upload failed"))?;
        drop(guard);
        Ok(())
    }
}
