/*
 * gh_zero_engine — hybrid runtime byte accounting
 * Computes checked scratch, logits, KV, fixed, staging, and crossing categories
 * from neutral dimensions. It performs no family lookup, probing, or allocation.
 */

use color_eyre::eyre::{Result, eyre};

use crate::kv_cache::layout;
use crate::kv_cache::scheme::{KvQuant, KvRole};

#[derive(Clone, Copy)]
pub(crate) struct RuntimeShape {
    pub(crate) block_count: usize,
    pub(crate) embedding: usize,
    pub(crate) q: usize,
    pub(crate) k: usize,
    pub(crate) v: usize,
    pub(crate) attention: usize,
    pub(crate) feed_forward: usize,
    pub(crate) vocab: usize,
    pub(crate) kv_heads: usize,
    pub(crate) key_length: usize,
    pub(crate) value_length: usize,
    pub(crate) prefill_rows: usize,
}

#[derive(Clone, Copy)]
pub(crate) struct DeviceFixedBytes {
    pub(crate) host: u64,
    pub(crate) device: u64,
    pub(crate) staging: u64,
}

pub(crate) struct RuntimeBytes {
    pub(crate) scratch: u64,
    pub(crate) logits: u64,
    pub(crate) kv_per_layer: u64,
    pub(crate) crossing: u64,
}

impl RuntimeBytes {
    pub(crate) fn new(shape: RuntimeShape, context: usize, scheme: KvQuant) -> Result<Self> {
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
                .checked_mul(1 + shape.prefill_rows as u64)
                .ok_or_else(overflow)?,
            logits: bytes(shape.vocab, 4)?,
            kv_per_layer: key.checked_add(value).ok_or_else(overflow)?,
            crossing: bytes(shape.embedding, 4)?
                .checked_mul(shape.prefill_rows as u64)
                .ok_or_else(overflow)?,
        })
    }
}

fn sum<const N: usize>(values: [u64; N]) -> Result<u64> {
    values.into_iter().try_fold(0u64, |sum, value| {
        sum.checked_add(value).ok_or_else(overflow)
    })
}

fn bytes(items: usize, item_bytes: usize) -> Result<u64> {
    items
        .checked_mul(item_bytes)
        .and_then(|total| u64::try_from(total).ok())
        .ok_or_else(overflow)
}

fn overflow() -> color_eyre::Report {
    eyre!("hybrid placement arithmetic overflow")
}
