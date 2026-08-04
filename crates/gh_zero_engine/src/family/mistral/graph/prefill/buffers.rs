/*
 * gh_zero_engine — bounded Ministral prefill buffers
 * Owns one request-local set of packed batch activations and their checked row
 * views. Allocation is transactional, every buffer is freed on drop, and all
 * byte arithmetic is validated before a backend view can be created.
 */

use color_eyre::eyre::{Result, bail, eyre};

use crate::backend::Backend;
use crate::backend::buffers::Scratch;
use crate::family::mistral::MistralConfig;
use crate::family::mistral::graph::prefill::MAX_PREFILL_ROWS;

pub(crate) const X: usize = 0;
pub(crate) const NORMED: usize = 1;
pub(crate) const Q: usize = 2;
pub(crate) const K: usize = 3;
pub(crate) const V: usize = 4;
pub(crate) const ATTN: usize = 5;
pub(crate) const PROJ: usize = 6;
pub(crate) const GATE: usize = 7;
pub(crate) const UP: usize = 8;
pub(crate) const ACT: usize = 9;
pub(crate) const FFN_OUT: usize = 10;

pub(crate) struct BatchBuffers<'a, B: Backend> {
    backend: &'a B,
    items: Vec<B::Buffer>,
    capacity: usize,
}

impl<'a, B: Backend> BatchBuffers<'a, B> {
    pub(crate) fn new(backend: &'a B, cfg: &MistralConfig, capacity: usize) -> Result<Self> {
        if !(1..=MAX_PREFILL_ROWS).contains(&capacity) {
            bail!("mistral prefill: batch capacity is unsupported");
        }
        let widths = [
            (cfg.embedding_length, 4usize),
            (cfg.embedding_length, 2),
            (cfg.q_width, 2),
            (cfg.k_width, 2),
            (cfg.v_width, 2),
            (cfg.attention_width, 2),
            (cfg.embedding_length, 2),
            (cfg.feed_forward_length, 2),
            (cfg.feed_forward_length, 2),
            (cfg.feed_forward_length, 2),
            (cfg.embedding_length, 2),
        ];
        let align = backend.min_buffer_offset_alignment();
        // Validate every layout before acquiring the first owner: arithmetic or
        // alignment failures must not leave even a transient partial allocation.
        let totals = widths
            .into_iter()
            .map(|(width, element)| {
                let stride = bytes(width, element)?;
                if align == 0 || stride % align != 0 {
                    bail!("mistral prefill: row alignment is unsupported");
                }
                stride
                    .checked_mul(capacity as u64)
                    .ok_or_else(|| eyre!("mistral prefill: buffer size overflow"))
            })
            .collect::<Result<Vec<_>>>()?;
        let mut items = Vec::with_capacity(totals.len());
        for total in totals {
            match backend.alloc_buffer(total) {
                Ok(buffer) => items.push(buffer),
                Err(error) => {
                    for buffer in items {
                        backend.free_buffer(buffer);
                    }
                    return Err(error);
                }
            }
        }
        Ok(Self {
            backend,
            items,
            capacity,
        })
    }

    #[cfg(any(feature = "vulkan-hybrid", feature = "metal-hybrid"))]
    pub(crate) fn all(&self, index: usize) -> &B::Buffer {
        &self.items[index]
    }

    fn rows(&self, index: usize, rows: usize, width: usize, element: usize) -> Result<B::Buffer> {
        if rows == 0 || rows > self.capacity {
            bail!("mistral prefill: row view exceeds batch capacity");
        }
        let len = bytes(width, element)?
            .checked_mul(rows as u64)
            .ok_or_else(|| eyre!("mistral prefill: buffer size overflow"))?;
        Ok(self.backend.view(&self.items[index], 0, len))
    }

    pub(crate) fn row(
        &self,
        index: usize,
        row: usize,
        width: usize,
        element: usize,
    ) -> Result<B::Buffer> {
        if row >= self.capacity {
            bail!("mistral prefill: row view exceeds batch capacity");
        }
        let stride = bytes(width, element)?;
        let offset = stride
            .checked_mul(row as u64)
            .ok_or_else(|| eyre!("mistral prefill: buffer offset overflow"))?;
        Ok(self.backend.view(&self.items[index], offset, stride))
    }

    pub(crate) fn scratch(&self, cfg: &MistralConfig, rows: usize) -> Result<Scratch<B::Buffer>> {
        Ok(Scratch {
            x: self.rows(X, rows, cfg.embedding_length, 4)?,
            normed: self.rows(NORMED, rows, cfg.embedding_length, 2)?,
            q: self.rows(Q, rows, cfg.q_width, 2)?,
            k: self.rows(K, rows, cfg.k_width, 2)?,
            v: self.rows(V, rows, cfg.v_width, 2)?,
            attn: self.rows(ATTN, rows, cfg.attention_width, 2)?,
            proj: self.rows(PROJ, rows, cfg.embedding_length, 2)?,
            gate: self.rows(GATE, rows, cfg.feed_forward_length, 2)?,
            up: self.rows(UP, rows, cfg.feed_forward_length, 2)?,
            act: self.rows(ACT, rows, cfg.feed_forward_length, 2)?,
            ffn_out: self.rows(FFN_OUT, rows, cfg.embedding_length, 2)?,
        })
    }
}

impl<B: Backend> Drop for BatchBuffers<'_, B> {
    fn drop(&mut self) {
        for buffer in self.items.drain(..) {
            self.backend.free_buffer(buffer);
        }
    }
}

fn bytes(width: usize, element: usize) -> Result<u64> {
    width
        .checked_mul(element)
        .and_then(|n| u64::try_from(n).ok())
        .ok_or_else(|| eyre!("mistral prefill: buffer size overflow"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::family::mistral::graph::shape::{ShapeBackend, config};

    #[test]
    fn accepts_only_checked_capacities_and_sizes_allocations() {
        let cfg = config(1, 32, 64);
        for capacity in [1, 16, 31, 32] {
            let backend = ShapeBackend::new(&cfg, false);
            let buffers = BatchBuffers::new(&backend, &cfg, capacity).unwrap();
            assert_eq!(backend.allocation_bytes()[0], 32 * 4 * capacity as u64);
            drop(buffers);
            assert_eq!(backend.live_allocations(), 0);
        }
        for capacity in [0, 33] {
            let backend = ShapeBackend::new(&cfg, false);
            assert!(BatchBuffers::new(&backend, &cfg, capacity).is_err());
            assert_eq!(backend.live_allocations(), 0);
        }
    }

    #[test]
    fn rejects_overflow_misalignment_and_partial_allocation() {
        let mut cfg = config(1, 32, 64);
        let backend = ShapeBackend::new(&cfg, false);
        cfg.q_width = usize::MAX;
        assert!(BatchBuffers::new(&backend, &cfg, 32).is_err());
        assert_eq!(backend.live_allocations(), 0);
        assert!(backend.allocation_bytes().is_empty());

        let cfg = config(1, 32, 64);
        let backend = ShapeBackend::new(&cfg, false);
        backend.set_alignment(3);
        assert!(BatchBuffers::new(&backend, &cfg, 32).is_err());
        assert_eq!(backend.live_allocations(), 0);

        let backend = ShapeBackend::new(&cfg, false);
        backend.fail_allocation(3);
        assert!(BatchBuffers::new(&backend, &cfg, 32).is_err());
        assert_eq!(backend.live_allocations(), 0);
    }

    #[test]
    fn bounds_actual_views_to_the_validated_capacity() {
        let cfg = config(1, 32, 64);
        let backend = ShapeBackend::new(&cfg, false);
        let buffers = BatchBuffers::new(&backend, &cfg, 32).unwrap();
        assert!(buffers.row(X, 31, cfg.embedding_length, 4).is_ok());
        assert!(buffers.row(X, 32, cfg.embedding_length, 4).is_err());
        assert!(buffers.scratch(&cfg, 0).is_err());
        assert!(buffers.scratch(&cfg, 33).is_err());
    }
}
