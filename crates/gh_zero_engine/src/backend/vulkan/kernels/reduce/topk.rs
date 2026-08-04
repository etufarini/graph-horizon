/*
 * gh_zero_engine — Vulkan exact top-k reduction
 * Dispatches the unchanged partial top-k kernel, reads its bounded pair fixture,
 * and merges candidates with sampling's exact `(logit desc, index asc)` order.
 * It owns no command queue, pipeline, scratch buffer, or sampling policy.
 */

use color_eyre::eyre::{Result, bail, eyre};

use super::readback;
use crate::backend::vulkan::buffers::GpuBuffer;
use crate::backend::vulkan::device::Device;
use crate::backend::vulkan::pipeline::{Kernel, PipelineRegistry, dispatch};

pub(crate) const TOPK_GROUPS: u32 = 64;
pub(crate) const MAX_K: usize = 128;
const PAIR_BYTES: usize = 8;

pub(crate) fn read(
    dev: &Device,
    reg: &PipelineRegistry,
    logits: &GpuBuffer,
    reduce: &GpuBuffer,
    host: &GpuBuffer,
    vocab: usize,
    k: usize,
) -> Result<Vec<(u32, f32)>> {
    if vocab == 0 {
        bail!("vulkan: read_topk on empty logits");
    }
    require_aligned(dev, reduce)?;

    let shader_k = k.min(MAX_K).min(vocab).max(1);
    let vocab_u32 =
        u32::try_from(vocab).map_err(|_| eyre!("vulkan: top-k vocabulary too large"))?;
    let mut push = Vec::with_capacity(8);
    push.extend_from_slice(&vocab_u32.to_le_bytes());
    push.extend_from_slice(&(shader_k as u32).to_le_bytes());

    let pairs = TOPK_GROUPS as usize * shader_k;
    let bytes = pairs
        .checked_mul(PAIR_BYTES)
        .ok_or_else(|| eyre!("vulkan: top-k readback too large"))?;
    readback::validate(reduce, host, bytes as u64)?;
    let cmd = dev.begin_commands()?;
    dispatch(
        dev,
        reg,
        cmd,
        Kernel::TopkPartial,
        &[
            (logits.buffer, logits.offset, logits.size),
            (reduce.buffer, reduce.offset, reduce.size),
        ],
        &push,
        TOPK_GROUPS,
    );
    readback::record(dev, cmd, reduce, host, bytes as u64)?;
    dev.submit_wait(cmd)?;
    merge(&readback::completed(dev, host, bytes)?, vocab, k)
}

fn merge(raw: &[u8], vocab: usize, k: usize) -> Result<Vec<(u32, f32)>> {
    if !raw.len().is_multiple_of(PAIR_BYTES) {
        bail!("vulkan: malformed top-k readback");
    }
    let mut candidates: Vec<(u32, f32)> = raw
        .chunks_exact(PAIR_BYTES)
        .map(|pair| {
            let index = u32::from_le_bytes(pair[..4].try_into().expect("four index bytes"));
            let value = f32::from_le_bytes(pair[4..].try_into().expect("four value bytes"));
            (index, value)
        })
        .filter(|&(index, _)| (index as usize) < vocab)
        .collect();
    let order = |a: &(u32, f32), b: &(u32, f32)| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0));
    let keep = k.min(vocab);
    if keep == 0 {
        return Ok(Vec::new());
    }
    if keep < candidates.len() {
        candidates.select_nth_unstable_by(keep - 1, order);
        candidates.truncate(keep);
    }
    candidates.sort_unstable_by(order);
    Ok(candidates)
}

fn require_aligned(dev: &Device, reduce: &GpuBuffer) -> Result<()> {
    let align = dev.min_storage_buffer_offset_alignment;
    if !reduce.offset.is_multiple_of(align) {
        bail!("vulkan: reduce buffer offset not aligned to {align} bytes");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::merge;

    #[test]
    fn byte_fixture_preserves_exact_order_and_sentinels() {
        let mut raw = Vec::new();
        for (index, value) in [(3, 2.0_f32), (1, 3.0), (9, 8.0), (2, 3.0), (0, 1.0)] {
            raw.extend_from_slice(&u32::to_le_bytes(index));
            raw.extend_from_slice(&f32::to_le_bytes(value));
        }
        assert_eq!(
            merge(&raw, 4, 3).unwrap(),
            vec![(1, 3.0), (2, 3.0), (3, 2.0)]
        );
    }

    #[test]
    fn zero_k_returns_no_candidates() {
        assert!(merge(&[0; 8], 1, 0).unwrap().is_empty());
    }
}
