/*
 * graph_horizon_engine — CUDA context, stream, capability, and memory snapshot.
 * Device acquisition never reads the device name and exposes only sanitized
 * failures; ordinal zero is interpreted by CUDA after visibility filtering.
 */

use std::sync::Arc;

use color_eyre::eyre::{Result, bail, eyre};
use cudarc::driver::{CudaContext, CudaStream, sys};

const REQUIRED_MAJOR: i32 = 7;
const REQUIRED_MINOR: i32 = 5;
const REQUIRED_WARP: i32 = 32;
const REQUIRED_THREADS: i32 = 256;
const REQUIRED_SHARED_MEMORY: i32 = 48 * 1024;

#[cfg(all(test, feature = "cuda-hybrid"))]
static PROBE_COUNT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

pub(crate) struct Device {
    pub(crate) context: Arc<CudaContext>,
    pub(crate) stream: Arc<CudaStream>,
    pub(crate) free_bytes: u64,
    #[cfg(any(feature = "cuda", test))]
    pub(crate) total_bytes: u64,
}

impl Device {
    #[cfg(any(feature = "cuda", test))]
    pub(crate) fn acquire() -> Result<Self> {
        Self::acquire_optional()?.ok_or_else(unavailable)
    }

    pub(crate) fn acquire_optional() -> Result<Option<Self>> {
        #[cfg(all(test, feature = "cuda-hybrid"))]
        PROBE_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let Ok(context) = CudaContext::new(0) else {
            return Ok(None);
        };
        if context.bind_to_thread().is_err() {
            return Ok(None);
        }
        let Ok((major, minor)) = context.compute_capability() else {
            return Ok(None);
        };
        let Ok(warp) = context.attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_WARP_SIZE)
        else {
            return Ok(None);
        };
        let Ok(threads) =
            context.attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)
        else {
            return Ok(None);
        };
        let Ok(shared_memory) = context
            .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK)
        else {
            return Ok(None);
        };
        if Self::validate_capabilities(major, minor, warp, threads, shared_memory).is_err() {
            return Ok(None);
        }
        let (free, total) = context.mem_get_info().map_err(|_| unavailable())?;
        #[cfg(not(any(feature = "cuda", test)))]
        let _ = total;

        // SAFETY: the backend creates exactly one stream and synchronizes it
        // before every readback or request completion. This call precedes all
        // allocations, so no slice can depend on cudarc event tracking.
        unsafe { context.disable_event_tracking() };
        let stream = context.new_stream().map_err(|_| unavailable())?;
        Ok(Some(Self {
            context,
            stream,
            free_bytes: u64::try_from(free).map_err(|_| arithmetic())?,
            #[cfg(any(feature = "cuda", test))]
            total_bytes: u64::try_from(total).map_err(|_| arithmetic())?,
        }))
    }

    fn validate_capabilities(
        major: i32,
        minor: i32,
        warp: i32,
        threads: i32,
        shared_memory: i32,
    ) -> Result<()> {
        let compute =
            major > REQUIRED_MAJOR || (major == REQUIRED_MAJOR && minor >= REQUIRED_MINOR);
        if compute
            && warp == REQUIRED_WARP
            && threads >= REQUIRED_THREADS
            && shared_memory >= REQUIRED_SHARED_MEMORY
        {
            Ok(())
        } else {
            bail!("CUDA device does not satisfy the required capabilities")
        }
    }
}

#[cfg(all(test, feature = "cuda-hybrid"))]
pub(crate) fn reset_probe_count() {
    PROBE_COUNT.store(0, std::sync::atomic::Ordering::Relaxed);
}

#[cfg(all(test, feature = "cuda-hybrid"))]
pub(crate) fn probe_count() -> usize {
    PROBE_COUNT.load(std::sync::atomic::Ordering::Relaxed)
}

fn unavailable() -> color_eyre::Report {
    eyre!("CUDA backend is unavailable")
}

fn arithmetic() -> color_eyre::Report {
    eyre!("cuda: buffer arithmetic overflow")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capability_boundary_and_each_predicate_are_checked() {
        assert!(
            Device::validate_capabilities(
                REQUIRED_MAJOR,
                REQUIRED_MINOR,
                REQUIRED_WARP,
                REQUIRED_THREADS,
                REQUIRED_SHARED_MEMORY,
            )
            .is_ok()
        );
        for row in [
            (7, 4, 32, 256, 49_152),
            (7, 5, 31, 256, 49_152),
            (7, 5, 32, 255, 49_152),
            (7, 5, 32, 256, 49_151),
        ] {
            assert_eq!(
                Device::validate_capabilities(row.0, row.1, row.2, row.3, row.4)
                    .unwrap_err()
                    .to_string(),
                "CUDA device does not satisfy the required capabilities"
            );
        }
        assert!(Device::validate_capabilities(8, 0, 32, 256, 49_152).is_ok());
    }

    #[test]
    fn acquisition_uses_visible_ordinal_zero_and_one_stream() -> Result<()> {
        let device = Device::acquire()?;
        assert_eq!(device.context.ordinal(), 0);
        assert!(!device.context.is_event_tracking());
        assert!(device.free_bytes > 0);
        assert!(device.total_bytes >= device.free_bytes);
        Ok(())
    }
}
