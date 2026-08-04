/*
 * gh_zero_engine — Ministral prefill batch coordinator
 * Owns prompt validation, request-local batch buffers, consecutive chunks,
 * embedding/tail recording, cancellation boundaries, and one submit per chunk.
 * Layer math, KV ownership, backend selection, retry, and sampling stay elsewhere.
 */

use color_eyre::eyre::{Result, bail, eyre};

use super::buffers::*;
use super::{effective_capacity, layer};
use crate::backend::Backend;
use crate::family::mistral::MistralConfig;
use crate::family::mistral::graph::tail;
use crate::kv_cache::Kv;

pub(crate) fn prefill<B: Backend, F: FnMut() -> Result<()>>(
    backend: &B,
    cfg: &MistralConfig,
    kv: &Kv<B::Buffer>,
    prompt: &[u32],
    mut before_batch: F,
) -> Result<()> {
    let capacity = effective_capacity(prompt.len())?;
    let buffers = BatchBuffers::new(backend, cfg, capacity)?;
    for (batch_index, tokens) in prompt.chunks(capacity).enumerate() {
        before_batch()?;
        let base = batch_index
            .checked_mul(capacity)
            .ok_or_else(|| eyre!("mistral prefill: batch offset overflow"))?;
        record_batch(backend, cfg, kv, &buffers, tokens, base, true, true)?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn record_batch<B: Backend>(
    backend: &B,
    cfg: &MistralConfig,
    kv: &Kv<B::Buffer>,
    batch: &BatchBuffers<'_, B>,
    tokens: &[u32],
    base: usize,
    with_embedding: bool,
    with_tail: bool,
) -> Result<()> {
    if tokens.is_empty() {
        bail!("mistral graph: empty prompt batch");
    }
    let rows = tokens.len();
    let encoder = backend.begin()?;
    if with_embedding {
        for (row, &token) in tokens.iter().enumerate() {
            if token as usize >= cfg.vocab_size {
                bail!("mistral graph: token beyond vocabulary");
            }
            backend.embed(
                &encoder,
                &batch.row(X, row, cfg.embedding_length, 4)?,
                backend
                    .buffers()
                    .weights
                    .token_embd
                    .as_ref()
                    .expect("embedding range owns token_embd"),
                token,
                cfg.embedding_length as u32,
            )?;
        }
    }
    for (layer_index, weights) in backend.buffers().weights.layers.iter().enumerate() {
        layer::record(
            backend,
            &encoder,
            cfg,
            kv,
            batch,
            weights,
            layer_index,
            base,
            rows,
        )?;
    }
    if with_tail {
        tail::record_buffers(
            backend,
            &encoder,
            cfg,
            &batch.row(X, rows - 1, cfg.embedding_length, 4)?,
            &batch.row(NORMED, rows - 1, cfg.embedding_length, 2)?,
        );
    }
    backend.submit(encoder)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::family::mistral::generation::tests::numeric;
    use crate::family::mistral::graph::shape::{ShapeBackend, config};
    use crate::kv_cache;
    use crate::kv_cache::scheme::KvQuant;

    #[test]
    fn dynamic_chunks_submit_once_for_sixteen_twice_for_thirty_three_and_sixty_four_times_for_long()
    {
        for (rows, expected) in [(16, 1), (33, 2), (2048, 64)] {
            let cfg = config(1, 32, 64);
            let backend = ShapeBackend::new(&cfg, false);
            let kv = kv_cache::alloc_shape(
                &backend,
                cfg.block_count,
                cfg.context_length,
                cfg.kv_head_count,
                cfg.key_length,
                cfg.value_length,
                KvQuant::F16,
            )
            .unwrap();
            prefill(&backend, &cfg, &kv, &vec![3; rows], || Ok(())).unwrap();
            assert_eq!(
                backend.trace().iter().filter(|op| *op == "submit").count(),
                expected
            );
            kv_cache::free(&backend, kv);
            assert_eq!(backend.live_allocations(), 0);
        }
    }

    #[test]
    fn batched_final_logits_match_sequential_for_both_kv_schemes() {
        numeric::batched_final_logits_match_sequential_for_both_kv_schemes();
    }

    #[test]
    fn quantized_profiles_drive_decode_and_short_prefill() {
        numeric::quantized_profiles_drive_decode_and_short_prefill();
    }
}
