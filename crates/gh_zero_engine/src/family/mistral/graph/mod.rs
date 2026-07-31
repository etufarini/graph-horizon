/*
 * gh_zero_engine — backend-generic Ministral graph boundary
 * Single responsibility: expose the shared dense block, range, prefill and tail
 * recorders over `Backend` buffers. It owns no concrete backend, weight format,
 * request lifecycle, sampling policy or model-size dispatch.
 */

pub(crate) mod block;
pub(crate) mod forward;
pub(crate) mod mlp;
pub(crate) mod prefill;
pub(crate) mod tail;

#[cfg(test)]
pub(crate) mod shape;
