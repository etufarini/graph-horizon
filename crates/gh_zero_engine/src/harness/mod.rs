/*
 * gh_zero_engine — family-neutral measurement facade
 * Exposes only end-to-end chat throughput and its report/configuration types.
 * Statistical aggregation remains private implementation detail; model lookup,
 * dense profiling, graph tracing, and backend-specific concepts stay outside
 * this narrow public measurement boundary.
*/

pub mod phases;
mod stats;
pub mod throughput;

pub use throughput::{BenchConfig, Stat, ThroughputReport};

use crate::Engine;
pub use crate::family::mistral::parity::ParityReport;

/// Runs the fixed, family-neutral oracle protocol against the selected runtime.
pub fn validate_parity(
    engine: &Engine,
    prompt_ids: &str,
    completion_ids: &str,
) -> color_eyre::Result<ParityReport> {
    engine.validate_parity(prompt_ids, completion_ids)
}
