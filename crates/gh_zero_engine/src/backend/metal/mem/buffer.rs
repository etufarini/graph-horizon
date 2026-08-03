/*
 * gh_zero_engine — shared Metal buffer ownership
 * Distinguishes one logical owning allocation from retained non-owning views,
 * validates byte windows, and performs bounded shared-memory copies. No dispatch.
 */

use color_eyre::eyre::{Result, eyre};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLDevice, MTLResourceOptions};

#[cfg(test)]
use std::sync::Arc;
#[cfg(test)]
use std::sync::atomic::{AtomicUsize, Ordering};

#[cfg(test)]
thread_local! {
    static TEST_COUNTS: std::cell::Cell<(usize, usize)> = const { std::cell::Cell::new((0, 0)) };
}

use super::super::Device;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum MetalFormat {
    F16,
    F32,
    Q4K,
    Q5K,
    Q6K,
    Raw,
}

pub(crate) struct MetalBuffer {
    raw: Retained<ProtocolObject<dyn MTLBuffer>>,
    offset: usize,
    len: usize,
    format: MetalFormat,
    owner: bool,
    #[cfg(test)]
    lifecycle: Arc<TestLifecycle>,
}

// SAFETY: MTLBuffer resources may cross threads; this wrapper mutates only the
// Metal allocation contents, whose engine-level request lifecycle is serialized.
unsafe impl Send for MetalBuffer {}
// SAFETY: immutable handles and checked shared-storage windows are thread-safe;
// graph execution owns the ordering of writes and completed readback.
unsafe impl Sync for MetalBuffer {}

#[cfg(test)]
struct TestLifecycle {
    owner_drops: AtomicUsize,
}

impl MetalBuffer {
    pub(crate) fn allocate(device: &Device, bytes: u64, format: MetalFormat) -> Result<Self> {
        let len = usize::try_from(bytes).map_err(|_| arithmetic())?;
        let raw = device
            .raw
            .newBufferWithLength_options(len.max(1), MTLResourceOptions::StorageModeShared)
            .ok_or_else(|| eyre!("metal: model allocation failed"))?;
        #[cfg(test)]
        TEST_COUNTS.with(|counts| {
            let (allocations, drops) = counts.get();
            counts.set((allocations + 1, drops));
        });
        Ok(Self {
            raw,
            offset: 0,
            len,
            format,
            owner: true,
            #[cfg(test)]
            lifecycle: Arc::new(TestLifecycle {
                owner_drops: AtomicUsize::new(0),
            }),
        })
    }

    pub(crate) fn view(&self, offset: u64, len: u64) -> Result<Self> {
        let offset = usize::try_from(offset).map_err(|_| arithmetic())?;
        let len = usize::try_from(len).map_err(|_| arithmetic())?;
        let end = offset.checked_add(len).ok_or_else(arithmetic)?;
        if offset % 4 != 0 || end > self.len {
            return Err(arithmetic());
        }
        Ok(Self {
            raw: self.raw.clone(),
            offset: self.offset.checked_add(offset).ok_or_else(arithmetic)?,
            len,
            format: self.format,
            owner: false,
            #[cfg(test)]
            lifecycle: self.lifecycle.clone(),
        })
    }

    pub(crate) fn alias(&self) -> Self {
        Self {
            raw: self.raw.clone(),
            offset: self.offset,
            len: self.len,
            format: self.format,
            owner: false,
            #[cfg(test)]
            lifecycle: self.lifecycle.clone(),
        }
    }

    pub(crate) fn write(&self, bytes: &[u8]) -> Result<()> {
        if bytes.len() > self.len {
            return Err(eyre!("metal: weight upload failed"));
        }
        // SAFETY: shared MTLBuffer contents is valid for its allocation; the
        // checked logical window and source length bound this non-overlapping copy.
        unsafe {
            std::ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                self.raw.contents().as_ptr().cast::<u8>().add(self.offset),
                bytes.len(),
            );
        }
        Ok(())
    }

    pub(crate) fn read(&self, bytes: usize) -> Result<Vec<u8>> {
        if bytes > self.len {
            return Err(eyre!("metal: readback past buffer window"));
        }
        let mut output = vec![0; bytes];
        // SAFETY: shared storage remains CPU-visible after command completion;
        // the checked window bounds both the source and destination slices.
        unsafe {
            std::ptr::copy_nonoverlapping(
                self.raw.contents().as_ptr().cast::<u8>().add(self.offset),
                output.as_mut_ptr(),
                bytes,
            );
        }
        Ok(output)
    }

    pub(crate) fn raw(&self) -> &ProtocolObject<dyn MTLBuffer> {
        &self.raw
    }

    pub(crate) fn offset(&self) -> usize {
        self.offset
    }

    pub(crate) fn len(&self) -> usize {
        self.len
    }

    pub(crate) fn format(&self) -> MetalFormat {
        self.format
    }

    pub(crate) fn is_owner(&self) -> bool {
        self.owner
    }
}

#[cfg(test)]
impl Drop for MetalBuffer {
    fn drop(&mut self) {
        if self.owner {
            self.lifecycle.owner_drops.fetch_add(1, Ordering::Relaxed);
            TEST_COUNTS.with(|counts| {
                let (allocations, drops) = counts.get();
                counts.set((allocations, drops + 1));
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
    eyre!("metal: buffer arithmetic overflow")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn views_keep_storage_alive_and_never_drop_as_owner() -> Result<()> {
        let device = Device::acquire()?;
        let owner = MetalBuffer::allocate(&device, 16, MetalFormat::Raw)?;
        let lifecycle = owner.lifecycle.clone();
        owner.write(&[0, 1, 2, 3, 4, 5, 6, 7])?;
        let view = owner.view(4, 4)?;
        let alias = view.alias();

        assert!(owner.is_owner());
        assert!(!view.is_owner());
        drop(owner);
        assert_eq!(lifecycle.owner_drops.load(Ordering::Relaxed), 1);
        assert_eq!(view.read(4)?, vec![4, 5, 6, 7]);
        drop(view);
        drop(alias);
        assert_eq!(lifecycle.owner_drops.load(Ordering::Relaxed), 1);
        Ok(())
    }

    #[test]
    fn views_reject_misaligned_and_out_of_range_windows() -> Result<()> {
        let device = Device::acquire()?;
        let buffer = MetalBuffer::allocate(&device, 16, MetalFormat::Raw)?;
        for result in [
            buffer.view(1, 4),
            buffer.view(12, 8),
            buffer.view(u64::MAX, 4),
        ] {
            assert_eq!(
                result.err().expect("invalid view").to_string(),
                "metal: buffer arithmetic overflow"
            );
        }
        assert_eq!(
            buffer.read(17).unwrap_err().to_string(),
            "metal: readback past buffer window"
        );
        Ok(())
    }
}
