/*
 * graph_horizon_engine — standalone CUDA load transaction.
 * Configuration and full-context capacity are fixed before module or model
 * allocation; only a fully constructed backend crosses this boundary.
 */

use color_eyre::eyre::{Result, bail};

use super::mem::{budget, buffers, weights};
use super::{CudaBackend, Device};
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
) -> Result<CudaBackend> {
    if source.groups().layers.len() != metadata.block_count {
        bail!("cuda: model allocation failed");
    }
    let percent = budget::validate_percentage(weights_percent)?;
    let plan = budget::MemoryPlan::new(source, shape, context, scheme)?;
    let device = Device::acquire()?;
    let reserve = budget::reserve_bytes(device.total_bytes, reserve_mib)?;
    budget::preflight(device.free_bytes, reserve, percent, &plan)?;
    let module = super::module::Module::load(&device.context)?;
    let weights = weights::load(&device, file, source)?;
    let (buffers, reduce) = buffers::allocate(&device, metadata, weights)?;
    Ok(CudaBackend {
        device,
        module,
        buffers,
        reduce,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_send_sync<T: Send + Sync>() {}

    #[test]
    fn published_backend_crosses_the_engine_worker_boundary() {
        assert_send_sync::<CudaBackend>();
    }

    #[test]
    fn invalid_percentage_precedes_device_acquisition() {
        assert_eq!(
            budget::validate_percentage(Some(0))
                .unwrap_err()
                .to_string(),
            "invalid CUDA weight percentage"
        );
    }
}
