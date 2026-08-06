/*
 * graph_horizon_engine — one Ministral prefill layer recorder
 * Records an already validated batch through one transformer layer in graph
 * order. Batch coordination, allocation, submission, and backend crossing stay
 * outside this module.
 */

use color_eyre::eyre::{Result, eyre};

use super::buffers::{BatchBuffers, K, Q};
use crate::backend::Backend;
use crate::backend::buffers::LayerWeights;
use crate::backend::rope::RopeRole;
use crate::family::mistral::MistralConfig;
use crate::family::mistral::graph::{block, mlp};
use crate::kv_cache::{self, Kv};

#[allow(clippy::too_many_arguments)]
pub(super) fn record<B: Backend>(
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
        let position = base
            .checked_add(row)
            .and_then(|position| u32::try_from(position).ok())
            .ok_or_else(|| eyre!("mistral prefill: buffer size overflow"))?;
        backend.no_barrier();
        backend.rope_yarn(
            enc,
            &batch.row(Q, row, cfg.q_width, 2),
            cfg.head_count as u32,
            cfg.key_length as u32,
            position,
            &yarn,
            RopeRole::Query,
        )?;
        backend.rope_yarn(
            enc,
            &batch.row(K, row, cfg.k_width, 2),
            cfg.kv_head_count as u32,
            cfg.key_length as u32,
            position,
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
    mlp::record(backend, enc, cfg, layer, &scratch, row_count);
    Ok(())
}
