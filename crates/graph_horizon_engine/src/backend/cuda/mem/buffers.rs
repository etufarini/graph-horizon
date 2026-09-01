/*
 * graph_horizon_engine — persistent CUDA scratch, logits, and reduction storage.
 * Allocation is transactional and publishes no partial runtime buffer set.
 */

use color_eyre::eyre::{Result, eyre};

use super::buffer::{CudaBuffer, CudaFormat};
use crate::backend::buffers::{Buffers, Scratch, WeightSet};
use crate::gguf::metadata::ModelMetadata;

use super::super::Device;

pub(crate) fn allocate(
    device: &Device,
    meta: &ModelMetadata,
    weights: WeightSet<CudaBuffer>,
) -> Result<(Buffers<CudaBuffer>, CudaBuffer)> {
    allocate_inner(device, meta, weights, None)
}

fn allocate_inner(
    device: &Device,
    meta: &ModelMetadata,
    weights: WeightSet<CudaBuffer>,
    fail_at: Option<usize>,
) -> Result<(Buffers<CudaBuffer>, CudaBuffer)> {
    let sizes = scratch_sizes(meta)?;
    let mut values = Vec::with_capacity(sizes.len());
    let mut index = 0;
    for (bytes, format) in sizes {
        values.push(allocate_step(device, bytes, format, &mut index, fail_at)?);
    }
    let mut values = values.into_iter();
    let scratch = Scratch {
        x: values.next().expect("fixed CUDA scratch inventory"),
        normed: values.next().expect("fixed CUDA scratch inventory"),
        q: values.next().expect("fixed CUDA scratch inventory"),
        k: values.next().expect("fixed CUDA scratch inventory"),
        v: values.next().expect("fixed CUDA scratch inventory"),
        attn: values.next().expect("fixed CUDA scratch inventory"),
        proj: values.next().expect("fixed CUDA scratch inventory"),
        gate: values.next().expect("fixed CUDA scratch inventory"),
        up: values.next().expect("fixed CUDA scratch inventory"),
        act: values.next().expect("fixed CUDA scratch inventory"),
        ffn_out: values.next().expect("fixed CUDA scratch inventory"),
    };
    let logits = allocate_step(
        device,
        bytes(meta.vocab_size, 4)?,
        CudaFormat::F32,
        &mut index,
        fail_at,
    )?;
    let reduce = allocate_step(device, 16 * 1024, CudaFormat::Raw, &mut index, fail_at)?;
    if fail_at == Some(index) {
        return Err(eyre!("cuda: model allocation failed"));
    }
    Ok((
        Buffers {
            weights,
            scratch,
            logits,
        },
        reduce,
    ))
}

fn allocate_step(
    device: &Device,
    bytes: u64,
    format: CudaFormat,
    index: &mut usize,
    fail_at: Option<usize>,
) -> Result<CudaBuffer> {
    if fail_at == Some(*index) {
        return Err(eyre!("cuda: model allocation failed"));
    }
    *index += 1;
    CudaBuffer::allocate(device, bytes, format)
}

fn scratch_sizes(meta: &ModelMetadata) -> Result<[(u64, CudaFormat); 11]> {
    let f16 = |items| bytes(items, 2).map(|size| (size, CudaFormat::F16));
    Ok([
        (bytes(meta.embedding_length, 4)?, CudaFormat::F32),
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
    eyre!("cuda: buffer arithmetic overflow")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::cuda::mem::buffer::{reset_test_counts, test_counts};

    fn empty_weights() -> WeightSet<CudaBuffer> {
        WeightSet {
            token_embd: None,
            output_norm: None,
            output: None,
            layers: Vec::new(),
        }
    }

    #[test]
    fn every_runtime_failpoint_releases_completed_allocations() -> Result<()> {
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
        for fail_at in 0..=13 {
            reset_test_counts();
            let error = allocate_inner(&device, &meta, empty_weights(), Some(fail_at))
                .err()
                .expect("injected CUDA allocation failure");
            assert_eq!(error.to_string(), "cuda: model allocation failed");
            assert_eq!(test_counts().0, test_counts().1, "failpoint {fail_at}");
        }
        Ok(())
    }
}
