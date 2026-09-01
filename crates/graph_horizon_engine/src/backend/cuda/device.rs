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

pub(crate) struct Device {
    pub(crate) context: Arc<CudaContext>,
    pub(crate) stream: Arc<CudaStream>,
    pub(crate) free_bytes: u64,
    pub(crate) total_bytes: u64,
}

impl Device {
    pub(crate) fn acquire() -> Result<Self> {
        let context = CudaContext::new(0).map_err(|_| unavailable())?;
        context.bind_to_thread().map_err(|_| unavailable())?;
        let (major, minor) = context.compute_capability().map_err(|_| unavailable())?;
        let warp = context
            .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_WARP_SIZE)
            .map_err(|_| unavailable())?;
        let threads = context
            .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)
            .map_err(|_| unavailable())?;
        let shared_memory = context
            .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK)
            .map_err(|_| unavailable())?;
        Self::validate_capabilities(major, minor, warp, threads, shared_memory)?;
        let (free, total) = context.mem_get_info().map_err(|_| unavailable())?;

        // SAFETY: the backend creates exactly one stream and synchronizes it
        // before every readback or request completion. This call precedes all
        // allocations, so no slice can depend on cudarc event tracking.
        unsafe { context.disable_event_tracking() };
        let stream = context.new_stream().map_err(|_| unavailable())?;
        Ok(Self {
            context,
            stream,
            free_bytes: u64::try_from(free).map_err(|_| arithmetic())?,
            total_bytes: u64::try_from(total).map_err(|_| arithmetic())?,
        })
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
