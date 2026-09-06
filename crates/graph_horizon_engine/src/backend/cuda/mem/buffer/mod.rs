/*
 * graph_horizon_engine — owning CUDA byte allocations and checked alias views.
 * Every handle keeps the sole allocation alive through Arc; logical windows
 * preserve format and never expose CUDA addresses outside the backend tree.
 */

mod transfer;

use std::sync::Arc;

use color_eyre::eyre::{Result, eyre};
use cudarc::driver::{CudaSlice, CudaStream, DevicePtr, SyncOnDrop, sys};

use super::super::Device;

#[cfg(test)]
thread_local! {
    static TEST_COUNTS: std::cell::Cell<(usize, usize)> = const { std::cell::Cell::new((0, 0)) };
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CudaFormat {
    F16,
    F32,
    Q4K,
    Q5K,
    Q6K,
    Q6KCached,
    Raw,
}

#[derive(Debug)]
pub(crate) struct CudaBuffer {
    allocation: Arc<CudaSlice<u8>>,
    offset: usize,
    len: usize,
    format: CudaFormat,
    prefill_cache: Option<Arc<CudaBuffer>>,
}

impl CudaBuffer {
    pub(crate) fn allocate(device: &Device, bytes: u64, format: CudaFormat) -> Result<Self> {
        let len = usize::try_from(bytes).map_err(|_| arithmetic())?;
        let allocation = device
            .stream
            .alloc_zeros::<u8>(len)
            .map_err(|_| eyre!("cuda: model allocation failed"))?;
        Ok(Self::from_allocation(allocation, len, format))
    }

    fn from_allocation(allocation: CudaSlice<u8>, len: usize, format: CudaFormat) -> Self {
        #[cfg(test)]
        TEST_COUNTS.with(|counts| {
            let (allocations, releases) = counts.get();
            counts.set((allocations + 1, releases));
        });
        Self {
            allocation: Arc::new(allocation),
            offset: 0,
            len,
            format,
            prefill_cache: None,
        }
    }

    pub(crate) fn view(&self, offset: u64, len: u64) -> Result<Self> {
        let offset = usize::try_from(offset).map_err(|_| arithmetic())?;
        let len = usize::try_from(len).map_err(|_| arithmetic())?;
        let end = offset.checked_add(len).ok_or_else(arithmetic)?;
        let alignment = if matches!(self.format, CudaFormat::Q6KCached) {
            4
        } else {
            2
        };
        // Cached f32 coefficients require stronger alignment than raw half metadata.
        if offset % alignment != 0 || end > self.len {
            return Err(arithmetic());
        }
        Ok(Self {
            allocation: self.allocation.clone(),
            offset: self.offset.checked_add(offset).ok_or_else(arithmetic)?,
            len,
            format: self.format,
            prefill_cache: if offset == 0 && len == self.len {
                self.prefill_cache.clone()
            } else {
                None
            },
        })
    }

    pub(crate) fn device_ptr<'a>(
        &'a self,
        stream: &'a CudaStream,
        bytes: usize,
    ) -> Result<(sys::CUdeviceptr, SyncOnDrop<'a>)> {
        if bytes > self.len {
            return Err(arithmetic());
        }
        let (base, guard) = self.allocation.device_ptr(stream);
        let offset = u64::try_from(self.offset).map_err(|_| arithmetic())?;
        let pointer = base.checked_add(offset).ok_or_else(arithmetic)?;
        Ok((pointer, guard))
    }

    pub(super) fn with_prefill(mut self, cache: CudaBuffer) -> Result<Self> {
        // The loader creates both on one Device. Reject nesting and nonmatching representations.
        let expected = (self.len / 210).checked_mul(320);
        if self.format != CudaFormat::Q6K
            || cache.format != CudaFormat::Q6KCached
            || self.len == 0
            || self.len % 210 != 0
            || expected != Some(cache.len)
            || self.offset != 0
            || cache.offset != 0
            || self.prefill_cache.is_some()
            || cache.prefill_cache.is_some()
        {
            return Err(arithmetic());
        }
        self.prefill_cache = Some(Arc::new(cache));
        Ok(self)
    }

    pub(crate) fn prefill(&self) -> &Self {
        self.prefill_cache.as_deref().unwrap_or(self)
    }

    #[cfg(any(feature = "cuda", test))]
    pub(crate) fn retained_bytes(&self) -> Option<u64> {
        // The public plan counts aligned immutable representations, not live allocator pages.
        let raw = u64::try_from(self.len).ok()?.checked_add(31)? / 32 * 32;
        let cached = match &self.prefill_cache {
            Some(cache) => u64::try_from(cache.len).ok()?.checked_add(31)? / 32 * 32,
            None => 0,
        };
        raw.checked_add(cached)
    }

    pub(crate) fn len(&self) -> usize {
        self.len
    }

    pub(crate) fn format(&self) -> CudaFormat {
        self.format
    }
}

#[cfg(test)]
impl Drop for CudaBuffer {
    fn drop(&mut self) {
        if Arc::strong_count(&self.allocation) == 1 {
            TEST_COUNTS.with(|counts| {
                let (allocations, releases) = counts.get();
                counts.set((allocations, releases + 1));
            });
        }
    }
}

#[cfg(test)]
pub(crate) fn reset_test_counts() {
    TEST_COUNTS.with(|counts| counts.set((0, 0)));
}

#[cfg(test)]
pub(crate) fn test_counts() -> (usize, usize) {
    TEST_COUNTS.with(std::cell::Cell::get)
}

fn arithmetic() -> color_eyre::Report {
    eyre!("cuda: buffer arithmetic overflow")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::cuda::exec::encoder::CudaEncoder;
    use crate::backend::cuda::{kernels, module::Module};
    use crate::backend::f16::f32_to_f16;

    #[test]
    fn views_compose_and_keep_one_allocation_alive() -> Result<()> {
        let device = Device::acquire()?;
        let owner = CudaBuffer::upload(&device, &[0, 1, 2, 3, 4, 5, 6, 7], CudaFormat::Raw)?;
        let allocation = Arc::downgrade(&owner.allocation);
        let view = owner.view(2, 6)?.view(2, 2)?;
        assert_eq!(view.format(), CudaFormat::Raw);
        drop(owner);
        assert!(allocation.upgrade().is_some());
        assert_eq!(view.read(&device, 2)?, [4, 5]);
        drop(view);
        assert!(allocation.upgrade().is_none());
        Ok(())
    }

    #[test]
    fn windows_and_transfers_are_bounded() -> Result<()> {
        let device = Device::acquire()?;
        let buffer = CudaBuffer::allocate(&device, 16, CudaFormat::Raw)?;
        for result in [
            buffer.view(1, 4),
            buffer.view(14, 4),
            buffer.view(u64::MAX, 2),
        ] {
            assert_eq!(
                result.unwrap_err().to_string(),
                "cuda: buffer arithmetic overflow"
            );
        }
        assert_eq!(
            buffer.read(&device, 17).unwrap_err().to_string(),
            "cuda: readback failed"
        );
        Ok(())
    }

    #[test]
    fn kernel_writes_through_a_view_are_visible_to_the_owner() -> Result<()> {
        let device = Device::acquire()?;
        let module = Module::load(&device.context)?;
        let values = [1.0f32, 2.0, 3.0, 4.0];
        let owner = CudaBuffer::upload(
            &device,
            &values
                .into_iter()
                .flat_map(f32::to_le_bytes)
                .collect::<Vec<_>>(),
            CudaFormat::F32,
        )?;
        let view = owner.view(4, 8)?;
        let add = CudaBuffer::upload(
            &device,
            &[f32_to_f16(10.0), f32_to_f16(20.0)]
                .into_iter()
                .flat_map(u16::to_le_bytes)
                .collect::<Vec<_>>(),
            CudaFormat::F16,
        )?;
        let encoder = CudaEncoder::begin(&device);
        kernels::residual_add::encode(&encoder, &module, &view, &add, 2)?;
        encoder.submit()?;
        let actual = owner
            .read(&device, 16)?
            .chunks_exact(4)
            .map(|bytes| f32::from_le_bytes(bytes.try_into().expect("four-byte float")))
            .collect::<Vec<_>>();
        assert_eq!(actual, [1.0, 12.0, 23.0, 4.0]);
        Ok(())
    }

    #[test]
    fn residual_write_targets_nested_view_and_rejects_overflow() -> Result<()> {
        let device = Device::acquire()?;
        let owner = CudaBuffer::allocate(&device, 16, CudaFormat::F32)?;
        let view = owner.view(4, 12)?.view(4, 4)?;
        view.write(&device, &7.0f32.to_le_bytes())?;
        let before_overflow = owner.read(&device, 16)?;
        assert_eq!(
            f32::from_le_bytes(before_overflow[8..12].try_into().unwrap()),
            7.0
        );
        assert_eq!(
            view.write(&device, &[1, 2, 3, 4, 5])
                .unwrap_err()
                .to_string(),
            "cuda: residual upload failed"
        );
        assert_eq!(owner.read(&device, 16)?, before_overflow);
        Ok(())
    }
}
