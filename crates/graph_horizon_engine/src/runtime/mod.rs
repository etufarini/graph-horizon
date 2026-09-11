/*
 * graph_horizon_engine — neutral runtime namespace
 * Exports graph/session contracts and homogeneous or partitioned request owners.
 * It owns traversal and KV lifecycle, not model-family parsing or device setup.
 */

pub(crate) mod contract;
#[cfg(any(feature = "cpu", feature = "vulkan"))]
pub(crate) mod homogeneous;
#[cfg(feature = "vulkan-hybrid")]
pub(crate) mod partitioned;

pub(crate) use contract::RuntimeSession;
