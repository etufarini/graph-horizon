/*
 * gh_zero_engine — persistent Metal runtime buffers
 * Transactionally allocates scratch, logits, reduction, and staging around an
 * already loaded weight set. It owns no GGUF parsing, budget, or dispatch.
 */

use color_eyre::eyre::{Result, eyre};

use super::buffer::{MetalBuffer, MetalFormat};
use crate::backend::buffers::{Buffers, Scratch, WeightSet};
use crate::backend::metal::Device;
use crate::gguf::metadata::ModelMetadata;

pub(crate) fn allocate(
    device: &Device,
    meta: &ModelMetadata,
    weights: WeightSet<MetalBuffer>,
) -> Result<(Buffers<MetalBuffer>, MetalBuffer, MetalBuffer)> {
    let sizes = scratch_sizes(meta)?;
    let mut values = Vec::with_capacity(sizes.len());
    for (bytes, format) in sizes {
        values.push(MetalBuffer::allocate(device, bytes, format)?);
    }
    let mut values = values.into_iter();
    let scratch = Scratch {
        x: values.next().unwrap(),
        normed: values.next().unwrap(),
        q: values.next().unwrap(),
        k: values.next().unwrap(),
        v: values.next().unwrap(),
        attn: values.next().unwrap(),
        proj: values.next().unwrap(),
        gate: values.next().unwrap(),
        up: values.next().unwrap(),
        act: values.next().unwrap(),
        ffn_out: values.next().unwrap(),
    };
    let logits = MetalBuffer::allocate(device, bytes(meta.vocab_size, 4)?, MetalFormat::F32)?;
    let reduce = MetalBuffer::allocate(device, 16 * 1024, MetalFormat::Raw)?;
    let staging = MetalBuffer::allocate(device, 16 * 1024, MetalFormat::Raw)?;
    Ok((
        Buffers {
            weights,
            scratch,
            logits,
        },
        reduce,
        staging,
    ))
}

fn scratch_sizes(meta: &ModelMetadata) -> Result<[(u64, MetalFormat); 11]> {
    let f16 = |items| bytes(items, 2).map(|size| (size, MetalFormat::F16));
    Ok([
        (bytes(meta.embedding_length, 4)?, MetalFormat::F32),
        f16(meta.embedding_length)?,
        f16(meta
            .head_count
            .checked_mul(meta.head_dim)
            .ok_or_else(arithmetic)?)?,
        f16(meta
            .head_count_kv
            .checked_mul(meta.head_dim)
            .ok_or_else(arithmetic)?)?,
        f16(meta
            .head_count_kv
            .checked_mul(meta.head_dim)
            .ok_or_else(arithmetic)?)?,
        f16(meta
            .head_count
            .checked_mul(meta.head_dim)
            .ok_or_else(arithmetic)?)?,
        f16(meta.embedding_length)?,
        f16(meta.feed_forward_length)?,
        f16(meta.feed_forward_length)?,
        f16(meta.feed_forward_length)?,
        f16(meta.embedding_length)?,
    ])
}

fn bytes(items: usize, width: usize) -> Result<u64> {
    items
        .checked_mul(width)
        .and_then(|value| u64::try_from(value).ok())
        .ok_or_else(arithmetic)
}

fn arithmetic() -> color_eyre::Report {
    eyre!("metal: buffer arithmetic overflow")
}
