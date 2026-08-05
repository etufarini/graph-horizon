/*
 * graph_orizon_engine — Vulkan compute kernels
 * Dispatch wrappers for the reachable matmul, attention, reduction,
 * normalization, and fused elementwise pipelines. No pipeline or buffer
 * ownership lives here.
*/

pub(crate) mod attention;
pub(crate) mod coopmat;
pub(crate) mod elementwise;
pub(crate) mod fused;
pub(crate) mod matmul;
pub(crate) mod mmvq;
pub(crate) mod reduce;
