/*
 * graph_horizon_engine — neutral runtime namespace
 * Exports graph/session contracts and homogeneous or partitioned request owners.
 * It owns traversal and KV lifecycle, not model-family parsing or device setup.
 */

pub(crate) mod contract;
#[cfg(any(
    feature = "cpu",
    feature = "vulkan",
    feature = "metal",
    feature = "cuda"
))]
pub(crate) mod homogeneous;
#[cfg(any(
    feature = "vulkan-hybrid",
    feature = "metal-hybrid",
    feature = "cuda-hybrid"
))]
pub(crate) mod partitioned;

pub(crate) use contract::RuntimeSession;
