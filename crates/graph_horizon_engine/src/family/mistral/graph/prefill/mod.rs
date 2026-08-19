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
pub(crate) use encode::prefill;
#[cfg(any(feature = "vulkan-hybrid", feature = "metal-hybrid"))]
pub(crate) use encode::record_batch;

// Thirty-two rows keep each worker's largest output tile near 192 KiB on the
// qualified 3B shape while amortizing quantized weight decode across the batch.
pub(crate) const CPU_ROWS: usize = 32;
pub(crate) const HOMOGENEOUS_GPU_ROWS: usize = 32;
pub(crate) const MIXED_ROWS: usize = 4;
