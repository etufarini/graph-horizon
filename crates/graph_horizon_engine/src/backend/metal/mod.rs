/*
 * graph_horizon_engine — native Metal backend namespace
 * Wires the statically selected Metal implementation on qualified Apple hosts.
 * Resource ownership, execution, memory, and kernels live in focused children.
 */

#![deny(clippy::undocumented_unsafe_blocks)]

mod backend;
mod device;
mod exec;
mod kernels;
mod loader;
mod mem;
mod pipeline;

pub(crate) use device::Device;
pub(crate) use exec::encoder::MetalEncoder;
#[cfg(feature = "metal")]
pub(crate) use loader::load;
pub(crate) use mem::buffer::{MetalBuffer, MetalFormat};

use crate::backend::buffers::Buffers;

pub(crate) struct MetalBackend {
    pub(crate) device: Device,
    pub(crate) pipelines: pipeline::PipelineRegistry,
    pub(crate) buffers: Buffers<MetalBuffer>,
    pub(crate) reduce: MetalBuffer,
    // Kept alive for the backend lifetime so future transfers cannot outlive
    // their shared staging allocation; no dispatch reads the handle directly.
    #[allow(dead_code)]
    pub(crate) staging: MetalBuffer,
}

#[cfg(test)]
static PROBE_COUNT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

#[cfg(test)]
pub(crate) fn reset_probe_count() {
    PROBE_COUNT.store(0, std::sync::atomic::Ordering::Relaxed);
}

#[cfg(test)]
pub(crate) fn probe_count() -> usize {
    PROBE_COUNT.load(std::sync::atomic::Ordering::Relaxed)
}

#[cfg(test)]
fn record_probe() {
    PROBE_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

#[cfg(feature = "metal-hybrid")]
impl crate::backend::hybrid::contract::HybridDevice for MetalBackend {
    type Device = Device;

    fn host_available() -> color_eyre::eyre::Result<u64> {
        objc2_foundation::NSProcessInfo::processInfo()
            .physicalMemory()
            .checked_mul(9)
            .and_then(|value| value.checked_div(10))
            .ok_or_else(|| color_eyre::eyre::eyre!("hybrid placement arithmetic overflow"))
    }

    fn acquire() -> color_eyre::eyre::Result<Option<Self::Device>> {
        Device::acquire_optional()
    }

    fn budget(
        device: &Self::Device,
    ) -> color_eyre::eyre::Result<crate::backend::hybrid::placement::BudgetInput> {
        Ok(crate::backend::hybrid::placement::BudgetInput::Unified {
            physical_memory: objc2_foundation::NSProcessInfo::processInfo().physicalMemory(),
            recommended_working_set: device.recommended_max,
            current_allocated: device.current_allocated,
        })
    }

    fn topology() -> crate::backend::hybrid::placement::MemoryTopology {
        crate::backend::hybrid::placement::MemoryTopology::Unified
    }

    fn all_mode_name() -> &'static str {
        "all-metal"
    }

    fn invalid_percentage_error() -> &'static str {
        "invalid Metal weight percentage"
    }

    fn fixed_bytes(
        shape: &crate::backend::hybrid::weights::runtime::RuntimeShape,
    ) -> color_eyre::eyre::Result<crate::backend::hybrid::weights::runtime::DeviceFixedBytes> {
        let device = (shape.vocab as u64)
            .checked_mul(4)
            .and_then(|bytes| bytes.checked_add(16 * 1024))
            .ok_or_else(|| color_eyre::eyre::eyre!("hybrid placement arithmetic overflow"))?;
        Ok(crate::backend::hybrid::weights::runtime::DeviceFixedBytes {
            host: 0,
            device,
            staging: 16 * 1024,
        })
    }

    fn load_selected(
        device: Self::Device,
        meta: &crate::gguf::metadata::ModelMetadata,
        source: &dyn crate::backend::source::WeightSource,
        gguf: &crate::gguf::loader::GgufFile,
        selection: &crate::backend::source::WeightSelection,
    ) -> color_eyre::eyre::Result<Self> {
        loader::load_selected(device, gguf, source, meta, selection)
    }

    fn buffer_bytes(buffer: &Self::Buffer) -> u64 {
        buffer.len() as u64
    }

    fn upload_residual(&self, target: &Self::Buffer, bytes: &[u8]) -> color_eyre::eyre::Result<()> {
        target.write(bytes)
    }
}
