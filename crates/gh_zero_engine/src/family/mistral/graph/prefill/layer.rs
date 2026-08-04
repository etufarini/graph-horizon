/*
 * gh_zero_engine — one batched Ministral prefill layer
 * Records RMSNorm, Q/K/V, RoPE, KV append, causal attention, projection,
 * residual, and MLP for the actual rows in one layer. It owns no buffers,
 * prompt chunk loop, submission boundary, backend selection, retry, or sampling.
 */

use color_eyre::eyre::Result;

use super::buffers::*;
use crate::backend::Backend;
use crate::backend::buffers::LayerWeights;
use crate::backend::rope::RopeRole;
use crate::family::mistral::MistralConfig;
use crate::family::mistral::graph::{block, mlp};
use crate::kv_cache::{self, Kv};

#[allow(clippy::too_many_arguments)]
pub(super) fn record<B: Backend>(
    backend: &B,
    encoder: &B::Encoder,
    cfg: &MistralConfig,
    kv: &Kv<B::Buffer>,
    batch: &BatchBuffers<'_, B>,
    weights: &LayerWeights<B::Buffer>,
    layer_index: usize,
    base: usize,
    rows: usize,
) -> Result<()> {
    let hidden = cfg.embedding_length as u32;
    let row_count = rows as u32;
    let scratch = batch.scratch(cfg, rows)?;
    backend.rmsnorm_x(
        encoder,
        &scratch.normed,
        &scratch.x,
        &weights.attn_norm,
        hidden,
        cfg.rms_epsilon,
        row_count,
    );
    backend.no_barrier();
    backend.matmul_batched(
        encoder,
        &scratch.q,
        &scratch.normed,
        &weights.attn_q,
        hidden,
        cfg.q_width as u32,
        row_count,
    );
    backend.no_barrier();
    backend.matmul_batched(
        encoder,
        &scratch.k,
        &scratch.normed,
        &weights.attn_k,
        hidden,
        cfg.k_width as u32,
        row_count,
    );
    backend.matmul_batched(
        encoder,
        &scratch.v,
        &scratch.normed,
        &weights.attn_v,
        hidden,
        cfg.v_width as u32,
        row_count,
    );

    let yarn = block::yarn(cfg);
    for row in 0..rows {
        backend.no_barrier();
        backend.rope_yarn(
            encoder,
            &batch.row(Q, row, cfg.q_width, 2)?,
            cfg.head_count as u32,
            cfg.key_length as u32,
            (base + row) as u32,
            &yarn,
            RopeRole::Query,
        )?;
        backend.rope_yarn(
            encoder,
            &batch.row(K, row, cfg.k_width, 2)?,
            cfg.kv_head_count as u32,
            cfg.key_length as u32,
            (base + row) as u32,
            &yarn,
            RopeRole::Key,
        )?;
    }
    kv_cache::append_batch(
        backend,
        encoder,
        kv,
        layer_index,
        base,
        &scratch.k,
        &scratch.v,
        rows,
    )?;
    backend.attention_prefill(
        encoder,
        &scratch.attn,
        &scratch.q,
        kv,
        cfg.head_count as u32,
        base as u32,
        row_count,
        layer_index as u32,
    );
    backend.matmul_batched(
        encoder,
        &scratch.proj,
        &scratch.attn,
        &weights.attn_output,
        cfg.attention_width as u32,
        hidden,
        row_count,
    );
    backend.residual_add(encoder, &scratch.x, &scratch.proj, hidden * row_count);
    mlp::record_rows(backend, encoder, cfg, weights, &scratch, row_count);
    Ok(())
}
