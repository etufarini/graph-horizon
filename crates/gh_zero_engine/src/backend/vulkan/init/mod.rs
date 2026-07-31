/*
 * gh_zero_engine — Vulkan initialization boundary
 * Exports device bring-up and the pure-loader availability mapping used before
 * persistent model resources exist. It does not select CPU fallback policy; pure
 * Vulkan maps unavailable initialization to E14, while hybrid can inspect the same
 * initialization result for its later placement policy.
 */

use color_eyre::eyre::{Report, eyre};

mod bootstrap;
mod caps;
pub(crate) mod device;
mod setup;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::backend::vulkan) enum Unavailable {
    Loader,
    Device,
    Capability,
}

pub(in crate::backend::vulkan) fn classify_unavailable(err: &Report) -> Option<Unavailable> {
    let msg = err.to_string();
    if msg.contains("loader not available") {
        Some(Unavailable::Loader)
    } else if msg.contains("no device") || msg.contains("cannot enumerate devices") {
        Some(Unavailable::Device)
    } else if msg.contains("lacks FP16")
        || msg.contains("workgroup")
        || msg.contains("push-constant")
    {
        Some(Unavailable::Capability)
    } else {
        None
    }
}

#[cfg(any(test, not(feature = "hybrid")))]
pub(in crate::backend::vulkan) fn pure_loader_unavailable(err: Report) -> Report {
    let _unavailable = classify_unavailable(&err);
    eyre!("Vulkan backend is unavailable")
}

#[cfg(feature = "hybrid")]
pub(crate) fn hybrid_device() -> color_eyre::eyre::Result<Option<device::Device>> {
    #[cfg(test)]
    PROBE_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    match device::Device::init() {
        Ok(device) => Ok(Some(device)),
        Err(error) if classify_unavailable(&error).is_some() => Ok(None),
        Err(_) => Err(eyre!("Vulkan initialization failed")),
    }
}

#[cfg(all(test, feature = "hybrid"))]
static PROBE_COUNT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

#[cfg(all(test, feature = "hybrid"))]
pub(crate) fn reset_probe_count() {
    PROBE_COUNT.store(0, std::sync::atomic::Ordering::Relaxed);
}

#[cfg(all(test, feature = "hybrid"))]
pub(crate) fn probe_count() -> usize {
    PROBE_COUNT.load(std::sync::atomic::Ordering::Relaxed)
}

#[cfg(test)]
mod tests {
    use color_eyre::eyre::eyre;

    use super::Unavailable;

    #[test]
    fn initialization_classifies_unavailable_loader_device_and_capability() {
        let loader = eyre!("vulkan: loader not available on this system");
        assert_eq!(
            super::classify_unavailable(&loader),
            Some(Unavailable::Loader)
        );

        let device = eyre!("vulkan: no device with a compute queue");
        assert_eq!(
            super::classify_unavailable(&device),
            Some(Unavailable::Device)
        );

        let capability =
            eyre!("vulkan: device lacks FP16 support (16-bit storage / shaderFloat16)");
        assert_eq!(
            super::classify_unavailable(&capability),
            Some(Unavailable::Capability)
        );
        assert_eq!(
            super::classify_unavailable(&eyre!("vulkan: injected operational failure")),
            None
        );
    }

    #[test]
    fn error_matrix_e14_maps_unavailable_initialization() {
        let err = super::pure_loader_unavailable(eyre!("vulkan: no device with a compute queue"));
        assert_eq!(err.to_string(), "Vulkan backend is unavailable");
    }
}
