/*
 * gh_zero_engine — hybrid backend ownership
 * Defines immutable placement reports and generic CPU/GPU resource variants.
 * Placement arithmetic, loading, graph traversal, and family semantics remain
 * in their owning sibling domains.
 */

pub(crate) mod contract;
pub(crate) mod crossing;
pub(crate) mod loader;
pub(crate) mod placement;
mod plan;
pub(crate) mod weights;

pub(crate) use plan::{BackendBytes, HybridMode, HybridPlan};

pub(crate) struct HybridRuntime<G> {
    pub(crate) plan: HybridPlan,
    pub(crate) backends: HybridBackends<G>,
}

#[allow(clippy::large_enum_variant)]
pub(crate) enum HybridBackends<G> {
    AllGpu(G),
    Mixed {
        cpu: crate::backend::cpu::CpuBackend,
        gpu: G,
    },
    CpuOnly(crate::backend::cpu::CpuBackend),
}
