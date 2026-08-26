/*
 * graph_horizon_engine — family-neutral measurement facade
 * Exposes only end-to-end chat throughput and its report/configuration types.
 * Statistical aggregation remains private implementation detail; model lookup,
 * dense profiling, graph tracing, and backend-specific concepts stay outside
 * this narrow public measurement boundary.
*/

mod stats;
pub mod throughput;

pub use throughput::{BenchConfig, Stat, ThroughputReport};

use crate::Engine;

pub struct ParityReport {
    pub prompt_ids: Vec<u32>,
    pub local_ids: Vec<u32>,
    pub top_two: Vec<[u32; 2]>,
    pub crossings: usize,
}

/// Runs the fixed, family-neutral oracle protocol against the selected runtime.
pub fn validate_parity(
    engine: &Engine,
    prompt_ids: &str,
    completion_ids: &str,
) -> color_eyre::Result<ParityReport> {
    engine.validate_parity(prompt_ids, completion_ids)
}
