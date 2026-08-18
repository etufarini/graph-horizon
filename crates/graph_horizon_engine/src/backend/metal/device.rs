/*
 * graph_horizon_engine — qualified Metal device ownership
 * Acquires the system default Apple M4, validates unified Metal capability,
 * snapshots budget inputs, and owns one command queue. It allocates no model data.
 */

#[cfg(any(test, feature = "metal"))]
use color_eyre::eyre::bail;
use color_eyre::eyre::{Result, eyre};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLCommandQueue, MTLCreateSystemDefaultDevice, MTLDevice, MTLGPUFamily};

#[cfg(feature = "metal-profile")]
use std::sync::Arc;

#[cfg(feature = "metal-profile")]
use super::exec::profile::Profile;

pub(crate) struct Device {
    pub(crate) raw: Retained<ProtocolObject<dyn MTLDevice>>,
    pub(crate) queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    pub(crate) recommended_max: u64,
    pub(crate) current_allocated: u64,
    #[cfg(feature = "metal-profile")]
    pub(crate) profile: Arc<Profile>,
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
        let name = raw.name().to_string();
        if name != "Apple M4"
            || !raw.hasUnifiedMemory()
            || !raw.supportsFamily(MTLGPUFamily::Apple9)
        {
            bail!("Metal device is unsupported: Apple M4 with unified memory is required");
        }
        let recommended_max = raw.recommendedMaxWorkingSetSize();
        let current_allocated = u64::try_from(raw.currentAllocatedSize())
            .map_err(|_| eyre!("metal: buffer arithmetic overflow"))?;
        let queue = raw
            .newCommandQueue()
            .ok_or_else(|| eyre!("Metal backend is unavailable"))?;
        #[cfg(feature = "metal-profile")]
        let profile = Profile::new(&raw)?;
        Ok(Self {
            raw,
            queue,
            recommended_max,
            current_allocated,
            #[cfg(feature = "metal-profile")]
            profile,
        })
    }

    #[cfg(feature = "metal-hybrid")]
    pub(crate) fn acquire_optional() -> Result<Option<Self>> {
        #[cfg(all(test, feature = "metal-hybrid"))]
        super::record_probe();
        let Some(raw) = MTLCreateSystemDefaultDevice() else {
            return Ok(None);
        };
        let name = raw.name().to_string();
        if name != "Apple M4"
            || !raw.hasUnifiedMemory()
            || !raw.supportsFamily(MTLGPUFamily::Apple9)
        {
            return Ok(None);
        }
        let recommended_max = raw.recommendedMaxWorkingSetSize();
        let current_allocated = u64::try_from(raw.currentAllocatedSize())
            .map_err(|_| eyre!("metal: buffer arithmetic overflow"))?;
        let queue = raw
            .newCommandQueue()
            .ok_or_else(|| eyre!("metal: command queue creation failed"))?;
        #[cfg(feature = "metal-profile")]
        let profile = Profile::new(&raw)?;
        Ok(Some(Self {
            raw,
            queue,
            recommended_max,
            current_allocated,
            #[cfg(feature = "metal-profile")]
            profile,
        }))
    }

    #[cfg(test)]
    pub(crate) fn validate(name: &str, unified: bool, family: bool) -> Result<()> {
        if name == "Apple M4" && unified && family {
            Ok(())
        } else {
            bail!("Metal device is unsupported: Apple M4 with unified memory is required")
        }
    }
}

#[cfg(feature = "metal-profile")]
impl Drop for Device {
    fn drop(&mut self) {
        self.profile.finish();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn device_contract_rejects_every_missing_capability() {
        assert!(Device::validate("Apple M4", true, true).is_ok());
        for row in [
            ("Apple M3", true, true),
            ("Apple M4", false, true),
            ("Apple M4", true, false),
        ] {
            assert_eq!(
                Device::validate(row.0, row.1, row.2)
                    .unwrap_err()
                    .to_string(),
                "Metal device is unsupported: Apple M4 with unified memory is required"
            );
        }
    }
}
