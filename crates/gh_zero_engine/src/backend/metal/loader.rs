/*
 * gh_zero_engine — standalone Metal load transaction
 * Validates configuration and memory before persistent allocation, then builds
 * device, immutable pipelines, weights, and runtime buffers before publication.
 */

use color_eyre::eyre::Result;
use objc2_foundation::NSProcessInfo;

use super::mem::{budget, buffers, weights};
use super::{Device, MetalBackend};
use crate::backend::hybrid::weights::runtime::RuntimeShape;
use crate::backend::source::WeightSource;
use crate::gguf::loader::GgufFile;
use crate::gguf::metadata::ModelMetadata;
use crate::kv_cache::scheme::KvQuant;

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
    let percent = budget::validate_percentage(weights_percent)?;
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
    let pipelines = super::pipeline::PipelineRegistry::load(&device)?;
    let weights = weights::load(&device, file, source)?;
    let (buffers, reduce, staging) = buffers::allocate(&device, metadata, weights)?;
    Ok(MetalBackend {
        device,
        pipelines,
        buffers,
        reduce,
        staging,
    })
}
