/*
 * gh_zero_engine — hybrid placement boundary
 * Defines device topology/budget inputs and dispatches pure split arithmetic.
 * It performs no probing, I/O, allocation, graph work, or family lookup.
 */

mod separate;
mod unified;

use color_eyre::eyre::Result;

use super::HybridPlan;
use super::weights::model::WeightBytes;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[allow(dead_code)]
pub(crate) enum MemoryTopology {
    Separate,
    Unified,
}

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

pub(crate) use separate::PlacementInput;

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
