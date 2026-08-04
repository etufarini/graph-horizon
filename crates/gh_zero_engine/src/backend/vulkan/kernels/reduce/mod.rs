/*
 * gh_zero_engine — Vulkan logit reduction domain
 * Exposes deterministic argmax, exact top-k, and the scratch-capacity constants
 * used by backend setup. Dispatch, transfer, mapping, and host merge logic remain
 * in their operation-specific siblings; this module owns no Vulkan resources.
 */

mod argmax;
mod readback;
mod topk;

pub(crate) use argmax::{completed as completed_argmax, read as argmax, record as record_argmax};
pub(crate) use topk::{MAX_K, TOPK_GROUPS, read as topk};
