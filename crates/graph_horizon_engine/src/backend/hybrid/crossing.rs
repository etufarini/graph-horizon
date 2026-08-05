/*
 * graph_horizon_engine — single hybrid residual crossing
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
    #[cfg(any(test, feature = "vulkan-hybrid", feature = "metal-hybrid"))]
    CROSSINGS.with(|count| count.set(count.get() + 1));
    Ok(())
}

#[cfg(any(test, feature = "vulkan-hybrid", feature = "metal-hybrid"))]
thread_local! {
    static CROSSINGS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[cfg(any(test, feature = "vulkan-hybrid", feature = "metal-hybrid"))]
pub(crate) fn reset_count() {
    CROSSINGS.with(|count| count.set(0));
}

#[cfg(any(test, feature = "vulkan-hybrid", feature = "metal-hybrid"))]
pub(crate) fn count() -> usize {
    CROSSINGS.with(std::cell::Cell::get)
}

#[cfg(all(test, feature = "vulkan-hybrid"))]
mod tests {
    use super::*;
    use crate::backend::Backend;
    use crate::backend::cpu::{CpuBuffer, CpuFormat};
    use crate::backend::vulkan::VulkanBackend;

    #[test]
    fn runtime_has_one_crossing_site_per_mixed_path() {
        let decode = include_str!("../../runtime/partitioned/session.rs");
        let prefill = include_str!("../../runtime/partitioned/prefill.rs");
        assert_eq!(decode.matches("crossing::copy(").count(), 1);
        assert_eq!(prefill.matches("crossing::copy(").count(), 1);
    }

    #[test]
    fn successful_copy_counts_once_and_rejects_short_sources() {
        reset_count();
        let gpu = match VulkanBackend::bare() {
            Ok(backend) => backend,
            Err(_) => {
                eprintln!("external verification: no Vulkan device");
                return;
            }
        };
        let source = CpuBuffer::zeroed(16, CpuFormat::F32);
        source.write_f32(&[1.0, -2.0, 3.5, 4.0]);
        let target = gpu.alloc_buffer(16).expect("crossing target");
        copy(&source, &gpu, &target, 4).expect("mixed crossing");
        assert_eq!(count(), 1);
        let bytes = gpu.read_bytes(&target, 16).expect("crossed residual");
        let values = bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
            .collect::<Vec<_>>();
        assert_eq!(values, [1.0, -2.0, 3.5, 4.0]);
        gpu.free_buffer(target);

        let short = CpuBuffer::zeroed(4, CpuFormat::F32);
        assert!(copy(&short, &gpu, &gpu.buffers().scratch.x, 2).is_err());
        assert_eq!(count(), 1);
    }
}
