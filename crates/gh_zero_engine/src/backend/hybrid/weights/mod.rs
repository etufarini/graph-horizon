/*
 * gh_zero_engine — hybrid byte accounting namespace
 * Exports model representation and runtime-shape accounting only; placement,
 * allocation, device probes, and family semantics stay outside this module.
 */

pub(crate) mod model;
pub(crate) mod runtime;
