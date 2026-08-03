/*
 * gh_zero_engine — native Metal backend namespace
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
    pub(crate) staging: MetalBuffer,
}

#[cfg(feature = "metal-hybrid")]
impl crate::backend::hybrid::contract::HybridDevice for MetalBackend {
    type Device = Device;

    fn acquire() -> color_eyre::eyre::Result<Option<Self::Device>> {
        Device::acquire().map(Some)
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
            .and_then(|bytes| bytes.checked_add(32 * 1024))
            .ok_or_else(|| color_eyre::eyre::eyre!("hybrid placement arithmetic overflow"))?;
        Ok(crate::backend::hybrid::weights::runtime::DeviceFixedBytes { host: 0, device })
    }

    fn load_selected(
        _device: Self::Device,
        _meta: &crate::gguf::metadata::ModelMetadata,
        _source: &dyn crate::backend::source::WeightSource,
        _gguf: &crate::gguf::loader::GgufFile,
        _selection: &crate::backend::source::WeightSelection,
    ) -> color_eyre::eyre::Result<Self> {
        color_eyre::eyre::bail!("metal hybrid selected loading is unavailable")
    }

    fn buffer_bytes(buffer: &Self::Buffer) -> u64 {
        buffer.len() as u64
    }

    fn upload_residual(&self, target: &Self::Buffer, bytes: &[u8]) -> color_eyre::eyre::Result<()> {
        target.write(bytes)
    }
}
