/*
 * graph_horizon_engine — standalone CUDA backend namespace and owned state.
 * The backend owns one context and stream; child modules own loading, memory,
 * execution, and operation dispatch without exposing CUDA outside this tree.
 */

#![deny(clippy::undocumented_unsafe_blocks)]

mod backend;
mod device;
mod exec;
mod kernels;
mod loader;
mod mem;
mod module;

pub(crate) use device::Device;
pub(crate) use exec::encoder::CudaEncoder;
#[cfg(feature = "cuda")]
pub(crate) use loader::load;
pub(crate) use mem::buffer::{CudaBuffer, CudaFormat};

// Standalone graph batches and their memory preflight must use the same capacity.
pub(crate) const PREFILL_ROWS: usize = 64;

pub(crate) struct CudaBackend {
    pub(crate) device: Device,
    pub(crate) module: module::Module,
    pub(crate) buffers: crate::backend::buffers::Buffers<CudaBuffer>,
    pub(crate) reduce: CudaBuffer,
}

#[cfg(feature = "cuda")]
impl CudaBackend {
    pub(crate) fn weight_bytes(&self) -> color_eyre::eyre::Result<u64> {
        let weights = &self.buffers.weights;
        // Immutable representation spans, aligned like the public plan; tied output is absent.
        [
            weights.token_embd.as_ref(),
            weights.output_norm.as_ref(),
            weights.output.as_ref(),
        ]
        .into_iter()
        .flatten()
        .chain(weights.layers.iter().flat_map(|layer| {
            [
                &layer.attn_norm,
                &layer.attn_q,
                &layer.attn_k,
                &layer.attn_v,
                &layer.attn_output,
                &layer.ffn_norm,
                &layer.ffn_gate,
                &layer.ffn_up,
                &layer.ffn_down,
            ]
        }))
        .try_fold(0u64, |sum, buffer| {
            buffer
                .retained_bytes()
                .and_then(|bytes| sum.checked_add(bytes))
                .ok_or_else(|| color_eyre::eyre::eyre!("model memory accounting overflow"))
        })
    }
}

#[cfg(all(test, feature = "cuda-hybrid"))]
impl CudaBackend {
    pub(crate) fn bare() -> color_eyre::eyre::Result<Self> {
        let device = Device::acquire()?;
        let module = module::Module::load(&device.context)?;
        let weights = crate::backend::buffers::WeightSet {
            token_embd: None,
            output_norm: None,
            output: None,
            layers: Vec::new(),
        };
        let metadata = crate::gguf::metadata::ModelMetadata {
            block_count: 0,
            embedding_length: 4,
            head_count: 1,
            head_count_kv: 1,
            head_dim: 4,
            feed_forward_length: 8,
            vocab_size: 8,
        };
        let (buffers, reduce) = mem::buffers::allocate(&device, &metadata, weights)?;
        Ok(Self {
            device,
            module,
            buffers,
            reduce,
        })
    }
}

#[cfg(all(test, feature = "cuda-hybrid"))]
pub(crate) use device::{probe_count, reset_probe_count};

#[cfg(feature = "cuda-hybrid")]
impl crate::backend::hybrid::contract::HybridDevice for CudaBackend {
    type Device = Device;

    fn host_available() -> color_eyre::eyre::Result<u64> {
        Ok(mem::budget::host_available())
    }

    fn acquire() -> color_eyre::eyre::Result<Option<Self::Device>> {
        Device::acquire_optional()
    }

    fn budget(
        device: &Self::Device,
    ) -> color_eyre::eyre::Result<crate::backend::hybrid::placement::BudgetInput> {
        Ok(crate::backend::hybrid::placement::BudgetInput::Separate {
            gpu_available: device.free_bytes,
        })
    }

    fn topology() -> crate::backend::hybrid::placement::MemoryTopology {
        crate::backend::hybrid::placement::MemoryTopology::Separate
    }

    fn all_mode_name() -> &'static str {
        "all-gpu"
    }

    fn invalid_percentage_error() -> &'static str {
        "invalid CUDA weight percentage"
    }

    fn prefill_rows(_: &Self::Device, _: usize, _: usize, _: usize) -> usize {
        PREFILL_ROWS
    }

    fn active_prefill_rows(&self, _: usize, _: usize, _: usize) -> usize {
        PREFILL_ROWS
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
            staging: 0,
        })
    }

    fn load_selected(
        device: Self::Device,
        meta: &crate::gguf::metadata::ModelMetadata,
        source: &dyn crate::backend::source::WeightSource,
        gguf: &crate::gguf::loader::GgufFile,
        selection: &crate::backend::source::WeightSelection,
    ) -> color_eyre::eyre::Result<Self> {
        loader::load_selected(device, meta, source, gguf, selection)
    }

    fn buffer_bytes(buffer: &Self::Buffer) -> u64 {
        buffer.len() as u64
    }

    fn upload_residual(&self, target: &Self::Buffer, bytes: &[u8]) -> color_eyre::eyre::Result<()> {
        target.write(&self.device, bytes)
    }
}

#[cfg(all(test, feature = "cuda-hybrid"))]
mod tests {
    use super::*;
    use crate::backend::hybrid::contract::HybridDevice;
    use crate::backend::hybrid::placement::BudgetInput;
    use crate::backend::hybrid::weights::runtime::RuntimeShape;

    #[test]
    fn hybrid_budget_and_fixed_bytes_match_cuda_ownership() -> color_eyre::eyre::Result<()> {
        let device = Device::acquire()?;
        assert!(matches!(
            CudaBackend::budget(&device)?,
            BudgetInput::Separate { gpu_available } if gpu_available == device.free_bytes
        ));
        let shape = RuntimeShape {
            block_count: 2,
            embedding: 8,
            q: 8,
            k: 4,
            v: 4,
            attention: 8,
            feed_forward: 16,
            vocab: 32,
            kv_heads: 1,
            key_length: 4,
            value_length: 4,
            cpu_prefill_rows: 4,
            gpu_prefill_rows: 32,
            mixed_prefill_rows: 4,
        };
        let fixed = CudaBackend::fixed_bytes(&shape)?;
        assert_eq!(fixed.host, 0);
        assert_eq!(fixed.device, 32 * 4 + 16 * 1024);
        assert_eq!(fixed.staging, 0);
        assert_eq!(CudaBackend::prefill_rows(&device, 2, 16, 4), 32);
        Ok(())
    }
}
