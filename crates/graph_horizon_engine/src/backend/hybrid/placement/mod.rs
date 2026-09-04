/*
 * graph_horizon_engine — hybrid placement boundary
 * Defines device topology/budget inputs and dispatches pure split arithmetic.
 * It performs no probing, I/O, allocation, graph work, or family lookup.
 */

#[cfg(any(
    feature = "vulkan-hybrid",
    feature = "metal-hybrid",
    feature = "cuda-hybrid"
))]
mod input;
#[cfg(any(
    feature = "vulkan-hybrid",
    feature = "metal-hybrid",
    feature = "cuda-hybrid"
))]
mod separate;
#[cfg(any(
    feature = "vulkan-hybrid",
    feature = "metal-hybrid",
    feature = "cuda-hybrid"
))]
mod unified;

#[cfg(any(
    feature = "vulkan-hybrid",
    feature = "metal-hybrid",
    feature = "cuda-hybrid"
))]
use color_eyre::eyre::Result;

#[cfg(any(
    feature = "vulkan-hybrid",
    feature = "metal-hybrid",
    feature = "cuda-hybrid"
))]
use super::HybridPlan;
#[cfg(any(
    feature = "vulkan-hybrid",
    feature = "metal-hybrid",
    feature = "cuda-hybrid"
))]
use super::weights::model::WeightBytes;

#[cfg(any(
    feature = "vulkan-hybrid",
    feature = "metal-hybrid",
    feature = "cuda-hybrid"
))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[allow(dead_code)]
pub(crate) enum MemoryTopology {
    Separate,
    Unified,
}

#[cfg(any(
    feature = "vulkan-hybrid",
    feature = "metal-hybrid",
    feature = "cuda-hybrid"
))]
#[derive(Clone, Copy, Debug)]
#[allow(dead_code)]
pub(crate) enum BudgetInput {
    Separate {
        gpu_available: u64,
    },
    Unified {
        physical_memory: u64,
        recommended_working_set: u64,
        current_allocated: u64,
    },
}

#[cfg(any(
    feature = "vulkan-hybrid",
    feature = "metal-hybrid",
    feature = "cuda-hybrid"
))]
pub(crate) use input::PlacementInput;
#[cfg(any(
    feature = "vulkan-hybrid",
    feature = "metal-hybrid",
    feature = "cuda-hybrid"
))]
pub(crate) use input::build;

#[cfg(any(
    feature = "vulkan-hybrid",
    feature = "metal-hybrid",
    feature = "cuda-hybrid"
))]
pub(crate) fn select(
    topology: MemoryTopology,
    weights: &WeightBytes,
    input: PlacementInput,
) -> Result<HybridPlan> {
    match topology {
        MemoryTopology::Separate => separate::select(weights, input),
        MemoryTopology::Unified => unified::select(weights, input),
    }
}

#[cfg(any(
    feature = "metal",
    feature = "vulkan-hybrid",
    feature = "metal-hybrid",
    feature = "cuda-hybrid"
))]
pub(crate) fn unified_gross(physical_memory: u64, recommended_working_set: u64) -> Option<u64> {
    physical_memory
        .checked_mul(9)?
        .checked_div(10)
        .map(|value| value.min(recommended_working_set))
}

#[cfg(any(
    feature = "metal",
    feature = "vulkan-hybrid",
    feature = "metal-hybrid",
    feature = "cuda-hybrid"
))]
pub(crate) fn unified_capacity(gross: u64, current_allocated: u64) -> u64 {
    gross.saturating_sub(current_allocated)
}
