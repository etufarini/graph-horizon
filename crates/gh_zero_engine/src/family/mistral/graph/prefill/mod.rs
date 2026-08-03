/*
 * gh_zero_engine — Ministral prefill coordinator
 * This module exposes bounded causal prompt prefill. Buffer ownership and
 * checked row views stay in `buffers`; graph recording stays in `encode`.
 */

mod buffers;
mod encode;

pub(crate) use buffers::{BATCH_ROWS, BatchBuffers, X};
pub(crate) use encode::record_batch;

pub(crate) fn prefill_with<B, F>(
    backend: &B,
    cfg: &crate::family::mistral::MistralConfig,
    kv: &crate::kv_cache::Kv<B::Buffer>,
    prompt: &[u32],
    before_batch: F,
) -> color_eyre::eyre::Result<()>
where
    B: crate::backend::Backend,
    F: FnMut() -> color_eyre::eyre::Result<()>,
{
    encode::prefill(backend, cfg, kv, prompt, before_batch)
}
