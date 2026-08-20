/*
 * graph_horizon_engine — backend-neutral byte accounting namespace
 * Exports retained model representation and runtime-shape accounting only;
 * placement, allocation, device probes, and family semantics stay outside.
 */

pub(crate) mod model;
pub(crate) mod runtime;
