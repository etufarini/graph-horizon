/*
 * gh_zero_engine — backend-generic Ministral dense block
 * Single responsibility: record one dense attention/MLP block in the pinned
 * operation order. It depends only on `Backend`, validated config, backend-owned
 * buffers and request KV; it owns no resources, I/O or backend selection.
 */

use color_eyre::eyre::Result;

use super::mlp;
use crate::backend::Backend;
use crate::backend::buffers::LayerWeights;
use crate::backend::rope::{RopeRole, Yarn};
use crate::family::mistral::MistralConfig;
use crate::kv_cache::{self, Kv};

pub(crate) fn record<B: Backend>(
    backend: &B,
    enc: &B::Encoder,
    cfg: &MistralConfig,
    layer: &LayerWeights<B::Buffer>,
    kv: &Kv<B::Buffer>,
    layer_index: usize,
    pos: usize,
) -> Result<()> {
    let scratch = &backend.buffers().scratch;
    let hidden = cfg.embedding_length as u32;
    backend.rmsnorm_x(
        enc,
        &scratch.normed,
        &scratch.x,
        &layer.attn_norm,
        hidden,
        cfg.rms_epsilon,
        1,
    );
    backend.no_barrier();
    backend.matmul(
        enc,
        &scratch.q,
        &scratch.normed,
        &layer.attn_q,
        hidden,
        cfg.q_width as u32,
    );
    backend.no_barrier();
    backend.matmul(
        enc,
        &scratch.k,
        &scratch.normed,
        &layer.attn_k,
        hidden,
        cfg.k_width as u32,
    );
    backend.matmul(
        enc,
        &scratch.v,
        &scratch.normed,
        &layer.attn_v,
        hidden,
        cfg.v_width as u32,
    );

    let yarn = yarn(cfg);
    backend.no_barrier();
    backend.rope_yarn(
        enc,
        &scratch.q,
        cfg.head_count as u32,
        cfg.key_length as u32,
        pos as u32,
        &yarn,
        RopeRole::Query,
    )?;
    backend.rope_yarn(
        enc,
        &scratch.k,
        cfg.kv_head_count as u32,
        cfg.key_length as u32,
        pos as u32,
        &yarn,
        RopeRole::Key,
    )?;
    // Invariant: append precedes attention for this layer and position, so the
    // causal kernel observes the current K/V exactly once.
    kv_cache::append(backend, enc, kv, layer_index, pos, &scratch.k, &scratch.v)?;
    backend.attention_decode(
        enc,
        &scratch.attn,
        &scratch.q,
        kv,
        cfg.head_count as u32,
        pos as u32,
        layer_index as u32,
    );
    backend.matmul(
        enc,
        &scratch.proj,
        &scratch.attn,
        &layer.attn_output,
        cfg.attention_width as u32,
        hidden,
    );
    backend.residual_add(enc, &scratch.x, &scratch.proj, hidden);
    mlp::record(backend, enc, cfg, layer);
    Ok(())
}

pub(crate) fn yarn(cfg: &MistralConfig) -> Yarn {
    Yarn {
        rope_dim: cfg.rope_dimension,
        original_context: cfg.yarn_original_context,
        freq_base: cfg.rope_freq_base,
        factor: cfg.yarn_factor,
        beta_fast: cfg.yarn_beta_fast,
        beta_slow: cfg.yarn_beta_slow,
        #[cfg(any(
            feature = "cpu",
            feature = "vulkan-hybrid",
            feature = "metal-hybrid",
            test
        ))]
        log_multiplier: cfg.yarn_log_multiplier,
        q_temperature_scale: cfg.attention_temperature_scale,
    }
}
