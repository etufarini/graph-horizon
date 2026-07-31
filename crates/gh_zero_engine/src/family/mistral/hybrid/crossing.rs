/*
 * gh_zero_engine — single hybrid residual crossing
 * Copies one bounded FP32 residual row or matrix from CPU storage to Vulkan
 * storage after the CPU submission completes. It owns no logits, KV, weights,
 * graph dispatch or fallback decision.
 */

use color_eyre::eyre::{Result, eyre};

use crate::backend::cpu::CpuBuffer;
use crate::backend::vulkan::{VulkanBackend, buffers::GpuBuffer};

pub(crate) fn copy(
    source: &CpuBuffer,
    gpu: &VulkanBackend,
    target: &GpuBuffer,
    elements: usize,
) -> Result<()> {
    let mut values = source.read_f32();
    if values.len() < elements {
        return Err(eyre!("hybrid residual crossing source is too small"));
    }
    values.truncate(elements);
    let capacity = elements
        .checked_mul(4)
        .ok_or_else(|| eyre!("hybrid residual crossing overflow"))?;
    let mut bytes = Vec::with_capacity(capacity);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    // This upload is the unique residual boundary. Logits readback and KV stay
    // outside this function, so a successful call increments exactly once.
    gpu.upload_bytes(target, &bytes)?;
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
