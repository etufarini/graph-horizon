/*
 * graph_horizon_engine — qualified Metal device ownership
 * Acquires the system default device, validates required Metal capabilities,
 * snapshots budget inputs, and owns one command queue. It allocates no model data.
 */

#[cfg(any(test, feature = "metal"))]
use color_eyre::eyre::bail;
use color_eyre::eyre::{Result, eyre};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLCommandQueue, MTLCreateSystemDefaultDevice, MTLDevice, MTLGPUFamily};

const REQUIRED_THREADS: usize = 128;
const REQUIRED_THREADGROUP_MEMORY: usize = 16 * 1024;

pub(crate) struct Device {
    pub(crate) raw: Retained<ProtocolObject<dyn MTLDevice>>,
    pub(crate) queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    pub(crate) recommended_max: u64,
    pub(crate) current_allocated: u64,
}

// SAFETY: Metal devices and command queues are thread-safe resource factories;
// mutable command encoding is confined to a fresh per-call MetalEncoder.
unsafe impl Send for Device {}
// SAFETY: shared access exposes only thread-safe device/queue creation methods.
unsafe impl Sync for Device {}

impl Device {
    #[cfg(any(test, feature = "metal"))]
    pub(crate) fn acquire() -> Result<Self> {
        let raw =
            MTLCreateSystemDefaultDevice().ok_or_else(|| eyre!("Metal backend is unavailable"))?;
        if !Self::qualified(&raw) {
            bail!("Metal device does not satisfy the required capabilities");
        }
        let recommended_max = raw.recommendedMaxWorkingSetSize();
        let current_allocated = u64::try_from(raw.currentAllocatedSize())
            .map_err(|_| eyre!("metal: buffer arithmetic overflow"))?;
        let queue = raw
            .newCommandQueue()
            .ok_or_else(|| eyre!("Metal backend is unavailable"))?;
        Ok(Self {
            raw,
            queue,
            recommended_max,
            current_allocated,
        })
    }

    #[cfg(feature = "metal-hybrid")]
    pub(crate) fn acquire_optional() -> Result<Option<Self>> {
        #[cfg(all(test, feature = "metal-hybrid"))]
        super::record_probe();
        let Some(raw) = MTLCreateSystemDefaultDevice() else {
            return Ok(None);
        };
        if !Self::qualified(&raw) {
            return Ok(None);
        }
        let recommended_max = raw.recommendedMaxWorkingSetSize();
        let current_allocated = u64::try_from(raw.currentAllocatedSize())
            .map_err(|_| eyre!("metal: buffer arithmetic overflow"))?;
        let queue = raw
            .newCommandQueue()
            .ok_or_else(|| eyre!("metal: command queue creation failed"))?;
        Ok(Some(Self {
            raw,
            queue,
            recommended_max,
            current_allocated,
        }))
    }

    fn qualified(raw: &ProtocolObject<dyn MTLDevice>) -> bool {
        Self::validate_requirements(
            raw.hasUnifiedMemory(),
            raw.supportsFamily(MTLGPUFamily::Apple9),
            raw.maxThreadsPerThreadgroup().width,
            raw.maxThreadgroupMemoryLength(),
        )
        .is_ok()
    }

    fn validate_requirements(
        unified: bool,
        family: bool,
        threads: usize,
        threadgroup_memory: usize,
    ) -> Result<()> {
        if unified
            && family
            && threads >= REQUIRED_THREADS
            && threadgroup_memory >= REQUIRED_THREADGROUP_MEMORY
        {
            Ok(())
        } else {
            Err(eyre!(
                "Metal device does not satisfy the required capabilities"
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn device_contract_rejects_every_missing_capability() {
        assert!(
            Device::validate_requirements(
                true,
                true,
                REQUIRED_THREADS,
                REQUIRED_THREADGROUP_MEMORY
            )
            .is_ok()
        );
        for row in [
            (false, true, REQUIRED_THREADS, REQUIRED_THREADGROUP_MEMORY),
            (true, false, REQUIRED_THREADS, REQUIRED_THREADGROUP_MEMORY),
            (
                true,
                true,
                REQUIRED_THREADS - 1,
                REQUIRED_THREADGROUP_MEMORY,
            ),
            (
                true,
                true,
                REQUIRED_THREADS,
                REQUIRED_THREADGROUP_MEMORY - 1,
            ),
        ] {
            assert_eq!(
                Device::validate_requirements(row.0, row.1, row.2, row.3)
                    .unwrap_err()
                    .to_string(),
                "Metal device does not satisfy the required capabilities"
            );
        }
    }
}
