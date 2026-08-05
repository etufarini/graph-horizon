/*
 * graph_orizon_engine — neutral runtime namespace
 * Exports graph/session contracts and homogeneous or partitioned request owners.
 * It owns traversal and KV lifecycle, not model-family parsing or device setup.
 */

pub(crate) mod contract;
#[cfg(any(feature = "cpu", feature = "vulkan", feature = "metal"))]
pub(crate) mod homogeneous;
#[cfg(any(feature = "vulkan-hybrid", feature = "metal-hybrid"))]
pub(crate) mod partitioned;

pub(crate) use contract::RuntimeSession;
