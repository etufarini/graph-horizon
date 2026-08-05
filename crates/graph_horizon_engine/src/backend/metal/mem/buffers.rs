/*
 * graph_horizon_engine — persistent Metal runtime buffers
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
    allocate_inner(device, meta, weights, None)
}

fn allocate_inner(
    device: &Device,
    meta: &ModelMetadata,
    weights: WeightSet<MetalBuffer>,
    fail_at: Option<usize>,
) -> Result<(Buffers<MetalBuffer>, MetalBuffer, MetalBuffer)> {
    let sizes = scratch_sizes(meta)?;
    let mut values = Vec::with_capacity(sizes.len());
    let mut index = 0;
    for (bytes, format) in sizes {
        values.push(allocate_step(device, bytes, format, &mut index, fail_at)?);
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
    let logits = allocate_step(
        device,
        bytes(meta.vocab_size, 4)?,
        MetalFormat::F32,
        &mut index,
        fail_at,
    )?;
    let reduce = allocate_step(device, 16 * 1024, MetalFormat::Raw, &mut index, fail_at)?;
    let staging = allocate_step(device, 16 * 1024, MetalFormat::Raw, &mut index, fail_at)?;
    if fail_at == Some(index) {
        return Err(eyre!("metal: model allocation failed"));
    }
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

fn allocate_step(
    device: &Device,
    bytes: u64,
    format: MetalFormat,
    index: &mut usize,
    fail_at: Option<usize>,
) -> Result<MetalBuffer> {
    if fail_at == Some(*index) {
        return Err(eyre!("metal: model allocation failed"));
    }
    *index += 1;
    MetalBuffer::allocate(device, bytes, format)
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::metal::mem::buffer::{reset_test_counts, test_counts};

    fn empty_weights() -> WeightSet<MetalBuffer> {
        WeightSet {
            token_embd: None,
            output_norm: None,
            output: None,
            layers: Vec::new(),
        }
    }

    #[test]
    fn every_runtime_allocation_failure_releases_prior_buffers() -> Result<()> {
        let device = Device::acquire()?;
        let meta = ModelMetadata {
            block_count: 0,
            embedding_length: 8,
            head_count: 2,
            head_count_kv: 1,
            head_dim: 4,
            feed_forward_length: 16,
            vocab_size: 32,
        };
        for fail_at in 0..=14 {
            reset_test_counts();
            let error = allocate_inner(&device, &meta, empty_weights(), Some(fail_at))
                .err()
                .expect("injected buffer allocation failure");
            assert_eq!(error.to_string(), "metal: model allocation failed");
            let (allocations, drops) = test_counts();
            assert_eq!(allocations, drops, "leak at failpoint {fail_at}");
        }
        Ok(())
    }
}
