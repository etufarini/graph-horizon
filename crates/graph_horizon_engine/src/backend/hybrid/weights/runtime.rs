/*
 * graph_horizon_engine — hybrid runtime byte accounting
 * Computes checked scratch, logits, KV, fixed, staging, and crossing categories
 * from neutral dimensions. It performs no family lookup, probing, or allocation.
 */

#[cfg(any(
    feature = "metal",
    feature = "vulkan-hybrid",
    feature = "metal-hybrid",
    feature = "cuda"
))]
use color_eyre::eyre::{Result, eyre};

#[cfg(any(
    feature = "metal",
    feature = "vulkan-hybrid",
    feature = "metal-hybrid",
    feature = "cuda"
))]
use crate::kv_cache::layout;
#[cfg(any(
    feature = "metal",
    feature = "vulkan-hybrid",
    feature = "metal-hybrid",
    feature = "cuda"
))]
use crate::kv_cache::scheme::{KvQuant, KvRole};

#[derive(Clone, Copy)]
#[allow(dead_code)] // Each public profile consumes only its applicable 4/32/4 row facts.
pub(crate) struct RuntimeShape {
    pub(crate) block_count: usize,
    #[cfg(any(
        feature = "metal",
        feature = "vulkan-hybrid",
        feature = "metal-hybrid",
        feature = "cuda"
    ))]
    pub(crate) embedding: usize,
    #[cfg(any(
        feature = "metal",
        feature = "vulkan-hybrid",
        feature = "metal-hybrid",
        feature = "cuda"
    ))]
    pub(crate) q: usize,
    #[cfg(any(
        feature = "metal",
        feature = "vulkan-hybrid",
        feature = "metal-hybrid",
        feature = "cuda"
    ))]
    pub(crate) k: usize,
    #[cfg(any(
        feature = "metal",
        feature = "vulkan-hybrid",
        feature = "metal-hybrid",
        feature = "cuda"
    ))]
    pub(crate) v: usize,
    #[cfg(any(
        feature = "metal",
        feature = "vulkan-hybrid",
        feature = "metal-hybrid",
        feature = "cuda"
    ))]
    pub(crate) attention: usize,
    #[cfg(any(
        feature = "metal",
        feature = "vulkan-hybrid",
        feature = "metal-hybrid",
        feature = "cuda"
    ))]
    pub(crate) feed_forward: usize,
    #[cfg(any(
        feature = "metal",
        feature = "vulkan-hybrid",
        feature = "metal-hybrid",
        feature = "cuda"
    ))]
    pub(crate) vocab: usize,
    pub(crate) kv_heads: usize,
    pub(crate) key_length: usize,
    pub(crate) value_length: usize,
    pub(crate) cpu_prefill_rows: usize,
    pub(crate) gpu_prefill_rows: usize,
    pub(crate) mixed_prefill_rows: usize,
}

#[cfg(any(feature = "vulkan-hybrid", feature = "metal-hybrid"))]
#[derive(Clone, Copy)]
pub(crate) struct DeviceFixedBytes {
    pub(crate) host: u64,
    pub(crate) device: u64,
    pub(crate) staging: u64,
}

#[cfg(any(
    feature = "metal",
    feature = "vulkan-hybrid",
    feature = "metal-hybrid",
    feature = "cuda"
))]
pub(crate) struct RuntimeBytes {
    pub(crate) scratch: u64,
    pub(crate) logits: u64,
    pub(crate) kv_per_layer: u64,
    #[cfg(any(feature = "vulkan-hybrid", feature = "metal-hybrid"))]
    pub(crate) crossing: u64,
}

#[cfg(any(
    feature = "metal",
    feature = "vulkan-hybrid",
    feature = "metal-hybrid",
    feature = "cuda"
))]
impl RuntimeBytes {
    pub(crate) fn new(
        shape: RuntimeShape,
        context: usize,
        scheme: KvQuant,
        prefill_rows: usize,
    ) -> Result<Self> {
        if prefill_rows == 0 {
            return Err(overflow());
        }
        let row = sum([
            bytes(shape.embedding, 4)?,
            bytes(shape.embedding, 2)?,
            bytes(shape.q, 2)?,
            bytes(shape.k, 2)?,
            bytes(shape.v, 2)?,
            bytes(shape.attention, 2)?,
            bytes(shape.embedding, 2)?,
            bytes(shape.feed_forward, 2)?,
            bytes(shape.feed_forward, 2)?,
            bytes(shape.feed_forward, 2)?,
            bytes(shape.embedding, 2)?,
        ])?;
        let key = layout::buffer_bytes(
            scheme,
            KvRole::Key,
            1,
            context,
            shape.kv_heads,
            shape.key_length,
        );
        let value = layout::buffer_bytes(
            scheme,
            KvRole::Value,
            1,
            context,
            shape.kv_heads,
            shape.value_length,
        );
        Ok(Self {
            scratch: row
                .checked_mul(1 + prefill_rows as u64)
                .ok_or_else(overflow)?,
            logits: bytes(shape.vocab, 4)?,
            kv_per_layer: key.checked_add(value).ok_or_else(overflow)?,
            #[cfg(any(feature = "vulkan-hybrid", feature = "metal-hybrid"))]
            crossing: bytes(shape.embedding, 4)?
                .checked_mul(prefill_rows as u64)
                .ok_or_else(overflow)?,
        })
    }
}

#[cfg(any(
    feature = "metal",
    feature = "vulkan-hybrid",
    feature = "metal-hybrid",
    feature = "cuda"
))]
fn sum<const N: usize>(values: [u64; N]) -> Result<u64> {
    values.into_iter().try_fold(0u64, |sum, value| {
        sum.checked_add(value).ok_or_else(overflow)
    })
}

#[cfg(any(
    feature = "metal",
    feature = "vulkan-hybrid",
    feature = "metal-hybrid",
    feature = "cuda"
))]
fn bytes(items: usize, item_bytes: usize) -> Result<u64> {
    items
        .checked_mul(item_bytes)
        .and_then(|total| u64::try_from(total).ok())
        .ok_or_else(overflow)
}

#[cfg(any(
    feature = "metal",
    feature = "vulkan-hybrid",
    feature = "metal-hybrid",
    feature = "cuda"
))]
fn overflow() -> color_eyre::Report {
    eyre!("hybrid placement arithmetic overflow")
}

#[cfg(all(test, any(feature = "vulkan-hybrid", feature = "metal-hybrid")))]
mod tests {
    use super::*;

    fn shape() -> RuntimeShape {
        RuntimeShape {
            block_count: 2,
            embedding: 8,
            q: 8,
            k: 4,
            v: 4,
            attention: 8,
            feed_forward: 16,
            vocab: 32,
            kv_heads: 1,
            key_length: 4,
            value_length: 4,
            cpu_prefill_rows: 4,
            gpu_prefill_rows: 32,
            mixed_prefill_rows: 4,
        }
    }

    #[test]
    fn runtime_bytes_scale_with_explicit_prefill_rows() {
        let four = RuntimeBytes::new(shape(), 16, KvQuant::F16, 4).unwrap();
        let thirty_two = RuntimeBytes::new(shape(), 16, KvQuant::F16, 32).unwrap();
        assert_eq!(four.scratch / 5, thirty_two.scratch / 33);
        assert_eq!(thirty_two.crossing, four.crossing * 8);
        assert!(RuntimeBytes::new(shape(), 16, KvQuant::F16, 0).is_err());
    }
}
