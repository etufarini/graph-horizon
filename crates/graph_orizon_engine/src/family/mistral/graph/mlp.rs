/*
 * graph_orizon_engine — backend-generic Ministral MLP
 * Single responsibility: record FFN norm, gate/up projections, SwiGLU, down
 * projection and residual addition for one dense block. It owns no attention,
 * cache, resource lifecycle or backend-specific numeric implementation.
 */

use crate::backend::Backend;
use crate::backend::buffers::{LayerWeights, Scratch};
use crate::family::mistral::MistralConfig;

pub(crate) fn record<B: Backend>(
    backend: &B,
    enc: &B::Encoder,
    cfg: &MistralConfig,
    layer: &LayerWeights<B::Buffer>,
) {
    record_rows(backend, enc, cfg, layer, &backend.buffers().scratch, 1);
}

pub(crate) fn record_rows<B: Backend>(
    backend: &B,
    enc: &B::Encoder,
    cfg: &MistralConfig,
    layer: &LayerWeights<B::Buffer>,
    scratch: &Scratch<B::Buffer>,
    rows: u32,
) {
    let hidden = cfg.embedding_length as u32;
    let ffn = cfg.feed_forward_length as u32;
    backend.rmsnorm_x(
        enc,
        &scratch.normed,
        &scratch.x,
        &layer.ffn_norm,
        hidden,
        cfg.rms_epsilon,
        rows,
    );
    backend.no_barrier();
    backend.matmul_batched(
        enc,
        &scratch.gate,
        &scratch.normed,
        &layer.ffn_gate,
        hidden,
        ffn,
        rows,
    );
    backend.matmul_batched(
        enc,
        &scratch.up,
        &scratch.normed,
        &layer.ffn_up,
        hidden,
        ffn,
        rows,
    );
    backend.silu_mul(enc, &scratch.act, &scratch.gate, &scratch.up, ffn * rows);
    backend.matmul_batched(
        enc,
        &scratch.ffn_out,
        &scratch.act,
        &layer.ffn_down,
        ffn,
        hidden,
        rows,
    );
    backend.residual_add(enc, &scratch.x, &scratch.ffn_out, hidden * rows);
}
