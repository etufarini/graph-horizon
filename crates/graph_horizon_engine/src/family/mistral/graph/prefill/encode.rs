/*
 * graph_horizon_engine — bounded causal Ministral prefill encoder
 * Records packed multi-token embedding, dense blocks, causal attention and
 * final-row logits through `Backend`; checked views and KV layout remain elsewhere.
 */

use color_eyre::eyre::{Result, bail};

use super::buffers::{BatchBuffers, NORMED, X};
use super::layer;
use crate::backend::Backend;
use crate::family::mistral::MistralConfig;
use crate::family::mistral::graph::tail;
use crate::kv_cache::Kv;

pub(crate) fn prefill<B: Backend, F: FnMut() -> Result<()>>(
    backend: &B,
    cfg: &MistralConfig,
    kv: &Kv<B::Buffer>,
    prompt: &[u32],
    base: usize,
    row_capacity: usize,
    mut before_batch: F,
) -> Result<()> {
    if prompt.is_empty() {
        bail!("mistral graph: empty prompt");
    }
    let buffers = BatchBuffers::new(backend, cfg, row_capacity)?;
    for (batch_index, tokens) in prompt.chunks(row_capacity).enumerate() {
        before_batch()?;
        let offset = batch_index
            .checked_mul(row_capacity)
            .ok_or_else(|| color_eyre::eyre::eyre!("mistral prefill: buffer size overflow"))?;
        let position = base
            .checked_add(offset)
            .ok_or_else(|| color_eyre::eyre::eyre!("mistral prefill: buffer size overflow"))?;
        record_batch(backend, cfg, kv, &buffers, tokens, position, true, true)?;
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
    let rows = tokens.len();
    batch.validate_rows(rows)?;
    base.checked_add(rows)
        .ok_or_else(|| color_eyre::eyre::eyre!("mistral prefill: buffer size overflow"))?;
    let enc = backend.begin()?;
    if with_embedding {
        for (row, &token) in tokens.iter().enumerate() {
            if token as usize >= cfg.vocab_size {
                bail!("mistral graph: token beyond vocabulary");
            }
            backend.embed(
                &enc,
                &batch.row(X, row, cfg.embedding_length, 4),
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
    for (layer_index, layer) in backend.buffers().weights.layers.iter().enumerate() {
        layer::record(
            backend,
            &enc,
            cfg,
            kv,
            batch,
            layer,
            layer_index,
            base,
            rows,
        )?;
    }
    if with_tail {
        tail::record_buffers(
            backend,
            &enc,
            cfg,
            &batch.row(X, rows - 1, cfg.embedding_length, 4),
            &batch.row(NORMED, rows - 1, cfg.embedding_length, 2),
        );
    }
    backend.submit(enc)
}

#[cfg(test)]
#[cfg(any(feature = "cpu", feature = "vulkan-hybrid", feature = "metal-hybrid"))]
mod tests {
    use std::cell::Cell;

    use crate::family::mistral::graph::shape::{ShapeBackend, config};
    use crate::kv_cache;
    use crate::kv_cache::scheme::KvQuant;

    use super::{BatchBuffers, prefill, record_batch};

    #[test]
    fn explicit_batch_capacity_bounds_allocation_and_chunks() {
        let cfg = config(1, 8, 8);
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
        let calls = Cell::new(0);
        prefill(&backend, &cfg, &kv, &[1, 1, 1], 0, 2, || {
            calls.set(calls.get() + 1);
            Ok(())
        })
        .unwrap();
        assert_eq!(calls.get(), 2);
        assert_eq!(
            BatchBuffers::new(&backend, &cfg, 0)
                .err()
                .unwrap()
                .to_string(),
            "mistral prefill: zero batch capacity"
        );
        let batch = BatchBuffers::new(&backend, &cfg, 2).unwrap();
        assert_eq!(
            record_batch(&backend, &cfg, &kv, &batch, &[1, 1, 1], 0, true, true)
                .unwrap_err()
                .to_string(),
            "mistral prefill: batch exceeds capacity"
        );
        drop(batch);
        kv_cache::free(&backend, kv);
        assert_eq!(backend.live_allocations(), 0);
    }

    #[test]
    fn incremental_prefill_applies_base_to_every_batch() {
        let mut cfg = config(1, 8, 8);
        cfg.context_length = 4;
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

        // Base 2 fits the first two rows; the second batch starts at position 4.
        assert!(prefill(&backend, &cfg, &kv, &[1, 1, 1], 2, 2, || Ok(())).is_err());
        kv_cache::free(&backend, kv);
        assert_eq!(backend.live_allocations(), 0);
    }

    #[test]
    fn batched_final_logits_match_sequential_for_both_kv_schemes() {
        crate::family::mistral::generation::tests::numeric::
            batched_final_logits_match_sequential_for_both_kv_schemes();
    }

    #[test]
    fn quantized_profiles_drive_decode_and_short_prefill() {
        crate::family::mistral::generation::tests::numeric::quantized_profiles_drive_decode_and_short_prefill();
    }
}
