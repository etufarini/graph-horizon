/*
 * gh_zero_engine — Ministral prefill coordinator
 * This module exposes bounded causal prompt prefill. Buffer ownership and
 * checked row views stay in `buffers`; chunk and layer recording stay split in
 * `batch` and `layer`. Backend selection, KV ownership, and sampling are excluded.
 */

mod batch;
mod buffers;
mod layer;

pub(crate) const MAX_PREFILL_ROWS: usize = 32;

#[cfg(any(feature = "vulkan-hybrid", feature = "metal-hybrid"))]
pub(crate) use batch::record_batch;
#[cfg(any(feature = "vulkan-hybrid", feature = "metal-hybrid"))]
pub(crate) use buffers::{BatchBuffers, X};

pub(crate) fn effective_capacity(prompt_tokens: usize) -> color_eyre::eyre::Result<usize> {
    if prompt_tokens == 0 {
        color_eyre::eyre::bail!("mistral graph: empty prompt");
    }
    Ok(prompt_tokens.min(MAX_PREFILL_ROWS))
}

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
    batch::prefill(backend, cfg, kv, prompt, before_batch)
}
