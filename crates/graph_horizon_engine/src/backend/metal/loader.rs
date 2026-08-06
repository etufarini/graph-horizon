/*
 * graph_horizon_engine — standalone Metal load transaction
 * Validates configuration and memory before persistent allocation, then builds
 * device, immutable placement state, pipelines, weights, and runtime buffers.
 */

use color_eyre::eyre::Result;
#[cfg(feature = "metal")]
use objc2_foundation::NSProcessInfo;

#[cfg(feature = "metal")]
use super::mem::budget;
use super::mem::{buffers, weights};
use super::{Device, MetalBackend};
#[cfg(feature = "metal")]
use crate::backend::hybrid::weights::runtime::RuntimeShape;
use crate::backend::source::{WeightSelection, WeightSource};
use crate::gguf::loader::GgufFile;
use crate::gguf::metadata::ModelMetadata;
#[cfg(feature = "metal")]
use crate::kv_cache::scheme::KvQuant;

#[cfg(feature = "metal")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn load(
    file: &GgufFile,
    source: &dyn WeightSource,
    metadata: &ModelMetadata,
    shape: RuntimeShape,
    context: usize,
    scheme: KvQuant,
    weights_percent: Option<u8>,
    reserve_mib: Option<u64>,
) -> Result<MetalBackend> {
    // The standalone zero/invalid dial must fail before even probing Metal.
    let percent = standalone_percentage(weights_percent)?;
    let plan = budget::MemoryPlan::new(source, shape, context, scheme)?;
    let device = Device::acquire()?;
    let physical_memory = NSProcessInfo::processInfo().physicalMemory();
    let gross = physical_memory
        .checked_mul(9)
        .and_then(|value| value.checked_div(10))
        .map(|value| value.min(device.recommended_max))
        .ok_or_else(|| color_eyre::eyre::eyre!("metal: buffer arithmetic overflow"))?;
    let reserve = budget::reserve_bytes(gross, reserve_mib)?;
    budget::preflight(
        physical_memory,
        device.recommended_max,
        device.current_allocated,
        reserve,
        percent,
        &plan,
        context,
    )?;
    load_selected(
        device,
        file,
        source,
        metadata,
        &WeightSelection::full(metadata.block_count),
        false,
    )
}

pub(crate) fn load_selected(
    device: Device,
    file: &GgufFile,
    source: &dyn WeightSource,
    metadata: &ModelMetadata,
    selection: &WeightSelection,
    mixed_placement: bool,
) -> Result<MetalBackend> {
    let pipelines = super::pipeline::PipelineRegistry::load(&device)?;
    let weights = weights::load_selected(&device, file, source, selection)?;
    let (buffers, reduce, staging) = buffers::allocate(&device, metadata, weights)?;
    Ok(MetalBackend {
        mixed_placement,
        device,
        pipelines,
        buffers,
        reduce,
        staging,
    })
}

#[cfg(feature = "metal")]
fn standalone_percentage(percent: Option<u8>) -> Result<u8> {
    budget::validate_percentage(percent)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_send_sync<T: Send + Sync>() {}

    #[test]
    #[cfg(feature = "metal")]
    fn zero_percentage_stops_at_the_first_configuration_gate() {
        assert_eq!(
            standalone_percentage(Some(0)).unwrap_err().to_string(),
            "invalid Metal weight percentage"
        );
    }

    #[test]
    fn published_backend_crosses_the_engine_worker_boundary() {
        assert_send_sync::<MetalBackend>();
    }
}
