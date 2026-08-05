/*
 * gh_zero_engine — bounded causal Ministral prefill encoder
 * Records packed multi-token embedding, dense blocks, causal attention and
 * final-row logits through `Backend`; checked views and KV layout remain elsewhere.
 */

use color_eyre::eyre::{Result, bail};

use super::buffers::*;
use crate::backend::Backend;
use crate::backend::buffers::LayerWeights;
use crate::backend::rope::RopeRole;
use crate::family::mistral::MistralConfig;
use crate::family::mistral::graph::{block, mlp, tail};
use crate::kv_cache::{self, Kv};

pub(crate) fn prefill<B: Backend, F: FnMut() -> Result<()>>(
    backend: &B,
    cfg: &MistralConfig,
    kv: &Kv<B::Buffer>,
    prompt: &[u32],
    mut before_batch: F,
) -> Result<()> {
    if prompt.is_empty() {
        bail!("mistral graph: empty prompt");
    }
    let buffers = BatchBuffers::new(backend, cfg)?;
    for (batch_index, tokens) in prompt.chunks(BATCH_ROWS).enumerate() {
        before_batch()?;
        let base = batch_index * BATCH_ROWS;
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
    let rows = tokens.len();
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
        record_layer(
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

#[allow(clippy::too_many_arguments)]
fn record_layer<B: Backend>(
    backend: &B,
    enc: &B::Encoder,
    cfg: &MistralConfig,
    kv: &Kv<B::Buffer>,
    batch: &BatchBuffers<'_, B>,
    layer: &LayerWeights<B::Buffer>,
    layer_index: usize,
    base: usize,
    rows: usize,
) -> Result<()> {
    let hidden = cfg.embedding_length as u32;
    let row_count = rows as u32;
    let scratch = batch.scratch(cfg, rows);
    backend.rmsnorm_x(
        enc,
        &scratch.normed,
        &scratch.x,
        &layer.attn_norm,
        hidden,
        cfg.rms_epsilon,
        row_count,
    );
    backend.no_barrier();
    backend.matmul_batched(
        enc,
        &scratch.q,
        &scratch.normed,
        &layer.attn_q,
        hidden,
        cfg.q_width as u32,
        row_count,
    );
    backend.no_barrier();
    backend.matmul_batched(
        enc,
        &scratch.k,
        &scratch.normed,
        &layer.attn_k,
        hidden,
        cfg.k_width as u32,
        row_count,
    );
    backend.matmul_batched(
        enc,
        &scratch.v,
        &scratch.normed,
        &layer.attn_v,
        hidden,
        cfg.v_width as u32,
        row_count,
    );

    let yarn = block::yarn(cfg);
    for row in 0..rows {
        backend.no_barrier();
        backend.rope_yarn(
            enc,
            &batch.row(Q, row, cfg.q_width, 2),
            cfg.head_count as u32,
            cfg.key_length as u32,
            (base + row) as u32,
            &yarn,
            RopeRole::Query,
        )?;
        backend.rope_yarn(
            enc,
            &batch.row(K, row, cfg.k_width, 2),
            cfg.kv_head_count as u32,
            cfg.key_length as u32,
            (base + row) as u32,
            &yarn,
            RopeRole::Key,
        )?;
    }
    kv_cache::append_batch(
        backend,
        enc,
        kv,
        layer_index,
        base,
        &scratch.k,
        &scratch.v,
        rows,
    )?;
    backend.attention_prefill(
        enc,
        &scratch.attn,
        &scratch.q,
        kv,
        cfg.head_count as u32,
        base as u32,
        row_count,
        layer_index as u32,
    );
    backend.matmul_batched(
        enc,
        &scratch.proj,
        &scratch.attn,
        &layer.attn_output,
        cfg.attention_width as u32,
        hidden,
        row_count,
    );
    backend.residual_add(enc, &scratch.x, &scratch.proj, hidden * row_count);
    mlp::record_rows(backend, enc, cfg, layer, &scratch, row_count);
    Ok(())
}

#[cfg(test)]
#[cfg(any(feature = "cpu", feature = "vulkan-hybrid", feature = "metal-hybrid"))]
mod tests {
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
