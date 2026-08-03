/*
 * gh_zero_engine — neutral runtime namespace
 * Exports graph/session contracts and homogeneous or partitioned request owners.
 * It owns traversal and KV lifecycle, not model-family parsing or device setup.
 */

pub(crate) mod contract;
pub(crate) mod homogeneous;
#[cfg(feature = "vulcan-hybrid")]
pub(crate) mod partitioned;

pub(crate) use contract::RuntimeSession;
