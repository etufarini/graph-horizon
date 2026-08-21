/*
 * graph_horizon_engine — immutable Metal pipeline registry
 * Loads the embedded offline metallib and transactionally resolves one explicit
 * pipeline per operation. It owns no command buffers, dispatch, or model memory.
 */

use color_eyre::eyre::{Result, eyre};
use dispatch2::DispatchData;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{MTLComputePipelineState, MTLDevice, MTLLibrary};

#[cfg(test)]
use std::sync::Arc;
#[cfg(test)]
use std::sync::atomic::{AtomicUsize, Ordering};

use super::Device;

#[cfg(feature = "metal")]
const GQA_DECODE_THREADGROUP_MEMORY: usize = 20_544;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(usize)]
pub(crate) enum Kernel {
    Embedding,
    Matmul,
    MatmulBatched,
    MatmulBatchedWide,
    Rmsnorm,
    Rope,
    SiluMul,
    ResidualAdd,
    KvWrite,
    Attention,
    Argmax,
    Topk,
    #[cfg(feature = "metal")]
    AttentionGqaDecode,
    #[cfg(feature = "metal")]
    AttentionGqaSplit,
    #[cfg(feature = "metal")]
    AttentionGqaReduce,
    #[cfg(feature = "metal")]
    AttentionPrefillMatrix,
    MatmulBatchedTensor,
}

const KERNELS: &[Kernel] = &[
    Kernel::Embedding,
    Kernel::Matmul,
    Kernel::MatmulBatched,
    Kernel::MatmulBatchedWide,
    Kernel::Rmsnorm,
    Kernel::Rope,
    Kernel::SiluMul,
    Kernel::ResidualAdd,
    Kernel::KvWrite,
    Kernel::Attention,
    Kernel::Argmax,
    Kernel::Topk,
    #[cfg(feature = "metal")]
    Kernel::AttentionGqaDecode,
    #[cfg(feature = "metal")]
    Kernel::AttentionGqaSplit,
    #[cfg(feature = "metal")]
    Kernel::AttentionGqaReduce,
    #[cfg(feature = "metal")]
    Kernel::AttentionPrefillMatrix,
    Kernel::MatmulBatchedTensor,
];

impl Kernel {
    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::Embedding => "metal_embedding",
            Self::Matmul => "metal_matmul",
            Self::MatmulBatched => "metal_matmul_batched",
            Self::MatmulBatchedWide => "metal_matmul_batched_wide",
            Self::Rmsnorm => "metal_rmsnorm",
            Self::Rope => "metal_rope",
            Self::SiluMul => "metal_silu_mul",
            Self::ResidualAdd => "metal_residual_add",
            Self::KvWrite => "metal_kv_write",
            Self::Attention => "metal_attention",
            Self::Argmax => "metal_argmax",
            Self::Topk => "metal_topk",
            #[cfg(feature = "metal")]
            Self::AttentionGqaDecode => "metal_attention_gqa_decode",
            #[cfg(feature = "metal")]
            Self::AttentionGqaSplit => "metal_attention_gqa_split",
            #[cfg(feature = "metal")]
            Self::AttentionGqaReduce => "metal_attention_gqa_reduce",
            #[cfg(feature = "metal")]
            Self::AttentionPrefillMatrix => "metal_attention_prefill_matrix",
            Self::MatmulBatchedTensor => "metal_matmul_batched_tensor",
        }
    }
}

pub(crate) struct Pipeline {
    pub(crate) raw: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pub(crate) width: usize,
    pub(crate) max_threads: usize,
    #[cfg(feature = "metal")]
    pub(crate) threadgroup_memory: usize,
    #[cfg(test)]
    drops: Arc<AtomicUsize>,
}

#[cfg(test)]
impl Drop for Pipeline {
    fn drop(&mut self) {
        self.drops.fetch_add(1, Ordering::Relaxed);
    }
}

pub(crate) struct PipelineRegistry {
    values: Vec<Pipeline>,
}

// SAFETY: compiled pipeline states are immutable Metal resources and the
// registry never mutates after transactional construction.
unsafe impl Send for PipelineRegistry {}
// SAFETY: all shared access is indexed immutable pipeline lookup.
unsafe impl Sync for PipelineRegistry {}

impl PipelineRegistry {
    pub(crate) fn load(device: &Device) -> Result<Self> {
        Self::load_inner(
            device,
            #[cfg(test)]
            None,
            #[cfg(test)]
            Arc::new(AtomicUsize::new(0)),
        )
    }

    fn load_inner(
        device: &Device,
        #[cfg(test)] fail_at: Option<usize>,
        #[cfg(test)] drops: Arc<AtomicUsize>,
    ) -> Result<Self> {
        let bytes = include_bytes!(concat!(env!("OUT_DIR"), "/graph_horizon_engine.metallib"));
        let data = DispatchData::from_static_bytes(bytes);
        let library = device
            .raw
            .newLibraryWithData_error(&data)
            .map_err(|_| eyre!("metal: pipeline library load failed"))?;
        let mut values = Vec::with_capacity(KERNELS.len());
        for &kernel in KERNELS {
            if matches!(kernel, Kernel::MatmulBatchedTensor) && !device.supports_metal4() {
                continue;
            }
            #[cfg(test)]
            if fail_at == Some(values.len()) {
                return Err(eyre!("metal: pipeline creation failed"));
            }
            let name = kernel.name();
            let function = library
                .newFunctionWithName(&NSString::from_str(name))
                .ok_or_else(|| eyre!("metal: missing shader function '{name}'"))?;
            let raw = device
                .raw
                .newComputePipelineStateWithFunction_error(&function)
                .map_err(|_| eyre!("metal: pipeline creation failed"))?;
            #[cfg(feature = "metal")]
            if matches!(kernel, Kernel::MatmulBatchedWide)
                && (raw.threadExecutionWidth() != 32 || raw.maxTotalThreadsPerThreadgroup() < 256)
            {
                return Err(eyre!("metal: wide prefill pipeline is unavailable"));
            }
            if matches!(kernel, Kernel::MatmulBatchedTensor)
                && (raw.threadExecutionWidth() != 32 || raw.maxTotalThreadsPerThreadgroup() < 128)
            {
                return Err(eyre!("metal: tensor prefill pipeline is unavailable"));
            }
            #[cfg(feature = "metal")]
            if matches!(kernel, Kernel::AttentionGqaDecode)
                && (raw.threadExecutionWidth() != 32
                    || raw.maxTotalThreadsPerThreadgroup() < 256
                    || raw.staticThreadgroupMemoryLength() != GQA_DECODE_THREADGROUP_MEMORY
                    || GQA_DECODE_THREADGROUP_MEMORY > device.raw.maxThreadgroupMemoryLength())
            {
                return Err(eyre!("metal: grouped decode pipeline is unavailable"));
            }
            values.push(Pipeline {
                width: raw.threadExecutionWidth(),
                max_threads: raw.maxTotalThreadsPerThreadgroup(),
                #[cfg(feature = "metal")]
                threadgroup_memory: raw.staticThreadgroupMemoryLength(),
                raw,
                #[cfg(test)]
                drops: drops.clone(),
            });
        }
        #[cfg(test)]
        if fail_at == Some(KERNELS.len()) {
            return Err(eyre!("metal: pipeline creation failed"));
        }
        Ok(Self { values })
    }

    pub(crate) fn get(&self, kernel: Kernel) -> &Pipeline {
        &self.values[kernel as usize]
    }

    pub(crate) fn supports_tensor_matmul(&self) -> bool {
        self.values.len() == KERNELS.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loads_every_embedded_pipeline_on_the_qualified_device() -> Result<()> {
        let device = Device::acquire()?;
        let registry = PipelineRegistry::load(&device)?;
        for &kernel in KERNELS {
            let pipeline = registry.get(kernel);
            assert!(pipeline.width > 0);
            assert!(pipeline.max_threads >= pipeline.width);
        }
        Ok(())
    }

    #[test]
    fn pipeline_failures_release_every_prior_pipeline() -> Result<()> {
        let device = Device::acquire()?;
        for fail_at in 0..=KERNELS.len() {
            let drops = Arc::new(AtomicUsize::new(0));
            let error = PipelineRegistry::load_inner(&device, Some(fail_at), drops.clone())
                .err()
                .expect("injected pipeline failure");
            assert_eq!(error.to_string(), "metal: pipeline creation failed");
            assert_eq!(drops.load(Ordering::Relaxed), fail_at.min(KERNELS.len()));
        }
        Ok(())
    }
}
