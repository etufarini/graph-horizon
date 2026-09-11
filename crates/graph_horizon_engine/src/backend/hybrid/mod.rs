/*
 * graph_horizon_engine — hybrid backend ownership
 * Defines immutable placement reports and generic CPU/GPU resource variants.
 * Placement arithmetic, loading, graph traversal, and family semantics remain
 * in their owning sibling domains.
 */

#[cfg(feature = "vulkan-hybrid")]
pub(crate) mod contract;
#[cfg(feature = "vulkan-hybrid")]
pub(crate) mod crossing;
#[cfg(feature = "vulkan-hybrid")]
pub(crate) mod loader;
pub(crate) mod placement;
#[cfg(feature = "vulkan-hybrid")]
mod plan;
pub(crate) mod weights;

#[cfg(feature = "vulkan-hybrid")]
pub(crate) use plan::{BackendBytes, HybridMode, HybridPlan};

#[cfg(feature = "vulkan-hybrid")]
pub(crate) struct HybridRuntime<G> {
    pub(crate) plan: HybridPlan,
    pub(crate) backends: HybridBackends<G>,
    pub(crate) gpu_prefill_rows: usize,
}

#[cfg(feature = "vulkan-hybrid")]
#[allow(clippy::large_enum_variant)]
pub(crate) enum HybridBackends<G> {
    AllGpu(G),
    Mixed {
        cpu: crate::backend::cpu::CpuBackend,
        gpu: G,
    },
    CpuOnly(crate::backend::cpu::CpuBackend),
}
