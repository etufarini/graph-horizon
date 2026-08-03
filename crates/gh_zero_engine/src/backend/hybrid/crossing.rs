/*
 * gh_zero_engine — single hybrid residual crossing
 * Copies one checked FP32 CPU window to a generic GPU buffer and counts only
 * successful test crossings. It owns no graph traversal, KV, weights, or retry.
 */

use color_eyre::eyre::{Result, eyre};

use super::contract::HybridDevice;
use crate::backend::cpu::CpuBuffer;

pub(crate) fn copy<G: HybridDevice>(
    source: &CpuBuffer,
    gpu: &G,
    target: &G::Buffer,
    elements: usize,
) -> Result<()> {
    let bytes = elements
        .checked_mul(4)
        .ok_or_else(|| eyre!("hybrid residual crossing overflow"))?;
    if source.byte_len() < bytes {
        return Err(eyre!("hybrid residual crossing source is too small"));
    }
    if G::buffer_bytes(target) < bytes as u64 {
        return Err(eyre!("hybrid residual crossing destination is too small"));
    }
    let values = source.read_f32();
    let mut payload = Vec::with_capacity(bytes);
    for value in values.into_iter().take(elements) {
        payload.extend_from_slice(&value.to_le_bytes());
    }
    gpu.upload_residual(target, &payload)?;
    #[cfg(test)]
    CROSSINGS.with(|count| count.set(count.get() + 1));
    Ok(())
}

#[cfg(test)]
thread_local! {
    static CROSSINGS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
pub(crate) fn reset_count() {
    CROSSINGS.with(|count| count.set(0));
}

#[cfg(test)]
pub(crate) fn count() -> usize {
    CROSSINGS.with(std::cell::Cell::get)
}
