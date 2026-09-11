/*
 * graph_horizon_engine — hybrid placement boundary
 * Exposes checked host/device budgets and separate-memory split arithmetic.
 * It performs no probing, I/O, allocation, graph work, or family lookup.
 */

#[cfg(feature = "vulkan-hybrid")]
mod input;
#[cfg(feature = "vulkan-hybrid")]
mod separate;

#[cfg(feature = "vulkan-hybrid")]
#[derive(Clone, Copy, Debug)]
pub(crate) struct BudgetInput {
    pub(crate) gpu_available: u64,
}

#[cfg(feature = "vulkan-hybrid")]
pub(crate) use input::{PlacementInput, build};
#[cfg(feature = "vulkan-hybrid")]
pub(crate) use separate::select;
