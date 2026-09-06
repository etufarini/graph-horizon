/*
 * graph_horizon_engine — embedded PTX module and fixed CUDA function table.
 * This is the sole module-load boundary; model data and runtime strings can
 * never become code or symbol names.
 */

use std::ffi::CString;
use std::sync::Arc;

use color_eyre::eyre::{Result, eyre};
use cudarc::driver::{CudaContext, result, sys};

const FUNCTION_NAMES: [&str; 23] = [
    "cuda_weight_cache",
    "cuda_embedding",
    "cuda_matmul",
    "cuda_matmul_batched",
    "cuda_matmul_tensor",
    "cuda_matmul_tensor_wide",
    "cuda_matmul_tensor_paired",
    "cuda_logits",
    "cuda_rmsnorm",
    "cuda_rope",
    "cuda_silu_mul",
    "cuda_residual_add",
    "cuda_kv_write_f16",
    "cuda_kv_write_int8",
    "cuda_attention_decode_f16",
    "cuda_attention_decode_int8",
    "cuda_attention_decode_split_f16",
    "cuda_attention_decode_split_int8",
    "cuda_attention_prefill_f16",
    "cuda_attention_prefill_tensor_f16",
    "cuda_attention_prefill_int8",
    "cuda_argmax",
    "cuda_topk",
];

#[derive(Clone, Copy)]
pub(crate) enum Kernel {
    WeightCache,
    Embedding,
    Matmul,
    MatmulBatched,
    MatmulTensor,
    MatmulTensorWide,
    MatmulTensorPaired,
    Logits,
    RmsNorm,
    Rope,
    SiluMul,
    ResidualAdd,
    KvWriteF16,
    KvWriteInt8,
    AttentionDecodeF16,
    AttentionDecodeInt8,
    AttentionDecodeSplitF16,
    AttentionDecodeSplitInt8,
    AttentionPrefillF16,
    AttentionPrefillTensorF16,
    AttentionPrefillInt8,
    Argmax,
    TopK,
}

pub(crate) struct Functions {
    entries: [sys::CUfunction; FUNCTION_NAMES.len()],
}

pub(crate) struct Module {
    raw: sys::CUmodule,
    context: Arc<CudaContext>,
    pub(crate) functions: Functions,
    pub(crate) split_attention: bool,
    #[cfg(test)]
    pub(crate) tensor_mma16: bool,
}

// SAFETY: CUDA module/function handles may be used across threads only after
// binding their owning context; dispatch.rs performs that binding per launch.
unsafe impl Send for Module {}
// SAFETY: the module is immutable after construction and unload occurs only
// after all owned function handles become unreachable with the backend.
unsafe impl Sync for Module {}

impl Module {
    pub(crate) fn load(context: &Arc<CudaContext>) -> Result<Self> {
        Self::load_inner(context, None, true)
    }

    pub(super) fn load_inner(
        context: &Arc<CudaContext>,
        fail_at: Option<usize>,
        allow_mma16: bool,
    ) -> Result<Self> {
        context.bind_to_thread().map_err(|_| load_error())?;
        // Optional optimization capability: query failure preserves the portable path.
        let split_attention = context
            .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)
            .unwrap_or(0)
            >= 512;
        // Select one trusted image. A failed optional query preserves the75-compatible path.
        let tensor_mma16 = allow_mma16
            && context
                .compute_capability()
                .is_ok_and(|(major, _)| major >= 8);
        let ptx: &[u8] = if tensor_mma16 {
            include_bytes!(concat!(env!("OUT_DIR"), "/cuda_kernels_80.ptx"))
        } else {
            include_bytes!(concat!(env!("OUT_DIR"), "/cuda_kernels.ptx"))
        };
        let mut image = Vec::with_capacity(ptx.len().saturating_add(1));
        image.extend_from_slice(ptx);
        image.push(0);
        // SAFETY: the embedded trusted PTX is NUL-terminated in `image`, the
        // owning context is current, and CUDA consumes the image during this
        // call; no pointer derived from the Vec is retained by Rust.
        let raw = unsafe { result::module::load_data(image.as_ptr().cast()) }
            .map_err(|_| load_error())?;
        let entries = match load_functions(raw, context, fail_at) {
            Ok(entries) => entries,
            Err(error) => {
                // SAFETY: `raw` was loaded successfully in the current context,
                // no function escaped, and unloading closes partial ownership.
                let _ = unsafe { result::module::unload(raw) };
                return Err(error);
            }
        };
        Ok(Self {
            raw,
            context: context.clone(),
            functions: Functions { entries },
            split_attention,
            #[cfg(test)]
            tensor_mma16,
        })
    }
}

impl Functions {
    pub(crate) fn get(&self, kernel: Kernel) -> sys::CUfunction {
        self.entries[kernel as usize]
    }
}

fn load_functions(
    module: sys::CUmodule,
    context: &CudaContext,
    fail_at: Option<usize>,
) -> Result<[sys::CUfunction; FUNCTION_NAMES.len()]> {
    context.bind_to_thread().map_err(|_| load_error())?;
    let mut functions = Vec::with_capacity(FUNCTION_NAMES.len());
    for (index, name) in FUNCTION_NAMES.into_iter().enumerate() {
        if fail_at == Some(index) {
            return Err(load_error());
        }
        let name = CString::new(name).map_err(|_| load_error())?;
        // SAFETY: `module` is live in the bound context, every name is a fixed
        // NUL-free ABI symbol, and the returned handle remains subordinate to
        // Module ownership for its complete lifetime.
        let function =
            unsafe { result::module::get_function(module, name) }.map_err(|_| load_error())?;
        functions.push(function);
    }
    functions.try_into().map_err(|_| load_error())
}

impl Drop for Module {
    fn drop(&mut self) {
        if self.context.bind_to_thread().is_ok() {
            // SAFETY: this is the unique owned module handle; all function
            // fields are inert handles and no launch can outlive the backend.
            let _ = unsafe { result::module::unload(self.raw) };
        }
    }
}

fn load_error() -> color_eyre::Report {
    eyre!("cuda: kernel module load failed")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::cuda::Device;

    #[test]
    fn every_function_resolution_failpoint_is_bounded_and_recoverable() -> Result<()> {
        let device = Device::acquire()?;
        for allow_mma16 in [false, true] {
            for fail_at in 0..FUNCTION_NAMES.len() {
                assert_eq!(
                    Module::load_inner(&device.context, Some(fail_at), allow_mma16)
                        .err()
                        .expect("injected CUDA function failure")
                        .to_string(),
                    "cuda: kernel module load failed"
                );
            }
            let module = Module::load_inner(&device.context, None, allow_mma16)?;
            assert_eq!(
                module.tensor_mma16,
                allow_mma16 && device.context.compute_capability()?.0 >= 8
            );
        }
        Module::load(&device.context)?;
        Ok(())
    }
}
