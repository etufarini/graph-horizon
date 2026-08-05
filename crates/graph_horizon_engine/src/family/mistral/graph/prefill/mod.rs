/*
 * graph_horizon_engine — Ministral prefill coordinator
 * This module owns the fixed CPU, homogeneous-GPU, and mixed row policy and
 * exposes bounded prompt batching. Buffer ownership, coordination, and layer
 * recording stay in their respective submodules.
 */

mod buffers;
mod encode;
mod layer;

#[cfg(any(feature = "vulkan-hybrid", feature = "metal-hybrid"))]
pub(crate) use buffers::{BatchBuffers, X};
#[cfg(any(feature = "vulkan-hybrid", feature = "metal-hybrid"))]
pub(crate) use encode::record_batch;

pub(crate) const CPU_ROWS: usize = 4;
pub(crate) const HOMOGENEOUS_GPU_ROWS: usize = 32;
pub(crate) const MIXED_ROWS: usize = 4;

pub(crate) fn prefill_with<B, F>(
    backend: &B,
    cfg: &crate::family::mistral::MistralConfig,
    kv: &crate::kv_cache::Kv<B::Buffer>,
    prompt: &[u32],
    row_capacity: usize,
    before_batch: F,
) -> color_eyre::eyre::Result<()>
where
    B: crate::backend::Backend,
    F: FnMut() -> color_eyre::eyre::Result<()>,
{
    encode::prefill(backend, cfg, kv, prompt, row_capacity, before_batch)
}
