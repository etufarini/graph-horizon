/*
 * gh_zero_engine — CPU backend module wiring
 * This file owns CPU module wiring, concrete backend state, and construction
 * that is not part of the `Backend` trait. Hybrid-only selected loading and
 * test fixtures are gated at their actual boundaries; kernel implementations
 * and the single trait delegator stay in focused sibling modules.
*/

use crate::backend::buffers::Buffers;

pub(crate) mod buffer;
#[cfg(test)]
pub(crate) mod compute;
pub(crate) mod dequant;
mod dispatch;
mod f16;
mod kernels;
pub(crate) mod parallel;
mod pool;
mod readback;
mod weights;

#[cfg(not(any(feature = "hybrid", test)))]
use buffer::CpuBuffer;
#[cfg(any(feature = "hybrid", test))]
pub(crate) use buffer::CpuBuffer;
#[cfg(test)]
pub(crate) use buffer::CpuFormat;

pub(crate) use kernels::attention::set_no_simd;
#[cfg(all(test, feature = "vulkan"))]
pub(crate) use kernels::matmul::q4k::row_dot_q4k;
mod backend;

// CPU execution is eager, so the encoder deliberately carries no state.
pub(crate) struct CpuEncoder;

// Persistent CPU weights and scratch buffers; algorithms live in sibling modules.
pub(crate) struct CpuBackend {
    buffers: Buffers<CpuBuffer>,
}

#[cfg(feature = "hybrid")]
impl CpuBackend {
    pub(crate) fn load_selected(
        meta: &crate::gguf::metadata::ModelMetadata,
        ws: &dyn crate::backend::source::WeightSource,
        gguf: &crate::gguf::loader::GgufFile,
        selection: &crate::backend::source::WeightSelection,
    ) -> color_eyre::eyre::Result<Self> {
        Ok(Self {
            buffers: weights::load_selected(meta, ws, gguf, selection)?,
        })
    }
}

#[cfg(test)]
impl CpuBackend {
    pub(crate) fn from_buffers(buffers: Buffers<CpuBuffer>) -> Self {
        Self { buffers }
    }
}
