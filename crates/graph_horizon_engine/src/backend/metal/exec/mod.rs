/*
 * graph_horizon_engine — Metal execution namespace
 * Exports command ownership, bounded dispatch, and completed-buffer readback.
 */

pub(crate) mod dispatch;
pub(crate) mod encoder;
#[cfg(feature = "metal-profile")]
pub(crate) mod profile;
pub(crate) mod readback;
