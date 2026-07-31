/*
 * gh_zero_engine — backend-generic Ministral graph tail
 * Single responsibility: record final RMSNorm and logits using the dedicated
 * output allocation or the tied embedding allocation. It owns no sampling,
 * readback, request state, submission or backend-specific implementation.
 */

use crate::backend::Backend;
use crate::family::mistral::MistralConfig;

pub(crate) fn record<B: Backend>(backend: &B, enc: &B::Encoder, cfg: &MistralConfig) {
    let buffers = backend.buffers();
    record_buffers(
        backend,
        enc,
        cfg,
        &buffers.scratch.x,
        &buffers.scratch.normed,
    );
}

pub(crate) fn record_buffers<B: Backend>(
    backend: &B,
    enc: &B::Encoder,
    cfg: &MistralConfig,
    x: &B::Buffer,
    normed: &B::Buffer,
) {
    let buffers = backend.buffers();
    let hidden = cfg.embedding_length as u32;
    backend.rmsnorm_x(
        enc,
        normed,
        x,
        buffers
            .weights
            .output_norm
            .as_ref()
            .expect("tail range owns output_norm"),
        hidden,
        cfg.rms_epsilon,
        1,
    );
    // Tied output aliases the one backend-owned embedding allocation.
    let output = buffers
        .weights
        .output
        .as_ref()
        .or(buffers.weights.token_embd.as_ref())
        .expect("tail range owns output or tied embedding");
    backend.logits(
        enc,
        &buffers.logits,
        normed,
        output,
        hidden,
        cfg.vocab_size as u32,
    );
}
