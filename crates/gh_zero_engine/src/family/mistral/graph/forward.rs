/*
 * gh_zero_engine — backend-generic Ministral forward range
 * Single responsibility: record embedding and a validated contiguous layer
 * range through the shared block recorder and owns homogeneous token completion.
 * Plain and greedy paths share graph recording; it exposes the range seam needed
 * by hybrid placement but owns no crossing, sampling policy, or resources.
 */

use std::ops::Range;

use color_eyre::eyre::{Result, bail};

use super::block;
use super::tail;
use crate::backend::Backend;
use crate::family::mistral::MistralConfig;
use crate::kv_cache::Kv;

pub(crate) fn embedding<B: Backend>(
    backend: &B,
    enc: &B::Encoder,
    cfg: &MistralConfig,
    token: u32,
) -> Result<()> {
    if token as usize >= cfg.vocab_size {
        bail!("mistral graph: token beyond vocabulary");
    }
    let buffers = backend.buffers();
    backend.embed(
        enc,
        &buffers.scratch.x,
        buffers
            .weights
            .token_embd
            .as_ref()
            .expect("embedding range owns token_embd"),
        token,
        cfg.embedding_length as u32,
    )
}

pub(crate) fn range<B: Backend>(
    backend: &B,
    enc: &B::Encoder,
    cfg: &MistralConfig,
    kv: &Kv<B::Buffer>,
    layers: Range<usize>,
    pos: usize,
) -> Result<()> {
    let weights = &backend.buffers().weights.layers;
    if layers.start > layers.end || layers.end > weights.len() || kv.block_count != weights.len() {
        bail!("mistral graph: invalid layer range");
    }
    for layer_index in layers {
        block::record(
            backend,
            enc,
            cfg,
            &weights[layer_index],
            kv,
            layer_index,
            pos,
        )?;
    }
    Ok(())
}

pub(crate) fn token<B: Backend>(
    backend: &B,
    cfg: &MistralConfig,
    kv: &Kv<B::Buffer>,
    token: u32,
    pos: usize,
) -> Result<()> {
    let enc = record(backend, cfg, kv, token, pos)?;
    backend.submit(enc)
}

pub(crate) fn token_argmax<B: Backend>(
    backend: &B,
    cfg: &MistralConfig,
    kv: &Kv<B::Buffer>,
    token: u32,
    pos: usize,
    vocab: usize,
) -> Result<u32> {
    let enc = record(backend, cfg, kv, token, pos)?;
    backend.submit_argmax(enc, &backend.buffers().logits, vocab)
}

fn record<B: Backend>(
    backend: &B,
    cfg: &MistralConfig,
    kv: &Kv<B::Buffer>,
    token: u32,
    pos: usize,
) -> Result<B::Encoder> {
    if pos >= cfg.context_length || pos >= kv.context {
        bail!("mistral graph: position beyond context");
    }
    if token as usize >= cfg.vocab_size {
        bail!("mistral graph: token beyond vocabulary");
    }
    let enc = backend.begin()?;
    if let Err(error) = embedding(backend, &enc, cfg, token)
        .and_then(|_| range(backend, &enc, cfg, kv, 0..cfg.block_count, pos))
    {
        // The generic backend contract has no cancel seam. Completing a partial
        // encoder releases device command resources before this terminal error.
        let _ = backend.submit(enc);
        return Err(error);
    }
    tail::record(backend, &enc, cfg);
    Ok(enc)
}
