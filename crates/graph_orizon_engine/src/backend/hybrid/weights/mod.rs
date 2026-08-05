/*
 * graph_orizon_engine — hybrid byte accounting namespace
 * Exports model representation and runtime-shape accounting only; placement,
 * allocation, device probes, and family semantics stay outside this module.
 */

#[cfg(any(feature = "vulkan-hybrid", feature = "metal-hybrid"))]
pub(crate) mod model;
pub(crate) mod runtime;
