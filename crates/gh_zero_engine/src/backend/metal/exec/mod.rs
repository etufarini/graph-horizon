/*
 * gh_zero_engine — Metal execution namespace
 * Exports command ownership, bounded dispatch, and completed-buffer readback.
 */

pub(crate) mod dispatch;
pub(crate) mod encoder;
pub(crate) mod readback;
