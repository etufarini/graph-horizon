/*
 * gh_zero_engine — family-neutral measurement facade
 * Exposes only end-to-end chat throughput and its report/configuration types.
 * Statistical aggregation remains private implementation detail; model lookup,
 * dense profiling, graph tracing, and backend-specific concepts stay outside
 * this narrow public measurement boundary.
*/

mod stats;
pub mod throughput;

pub use throughput::{BenchConfig, Stat, ThroughputReport};
