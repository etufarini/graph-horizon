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

// Standalone GPU profiles keep enough rows to amortize their batched projection;
// hybrid profiles retain four rows because their split graph crosses backends.
#[cfg(any(feature = "metal", feature = "vulkan"))]
pub(crate) const BATCH_ROWS: usize = 32;
#[cfg(not(any(feature = "metal", feature = "vulkan")))]
pub(crate) const BATCH_ROWS: usize = 4;

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
}

impl<'a, B: Backend> BatchBuffers<'a, B> {
    pub(crate) fn new(backend: &'a B, cfg: &MistralConfig) -> Result<Self> {
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
        let mut items = Vec::with_capacity(widths.len());
        for (width, element) in widths {
            let stride = bytes(width, element)?;
            if stride % align != 0 {
                for buffer in items {
                    backend.free_buffer(buffer);
                }
                bail!("mistral prefill: row alignment is unsupported");
            }
            let total = stride
                .checked_mul(BATCH_ROWS as u64)
                .ok_or_else(|| eyre!("mistral prefill: buffer size overflow"))?;
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
        Ok(Self { backend, items })
    }

    #[cfg(any(feature = "vulkan-hybrid", feature = "metal-hybrid"))]
    pub(crate) fn all(&self, index: usize) -> &B::Buffer {
        &self.items[index]
    }

    fn rows(&self, index: usize, rows: usize, width: usize, element: usize) -> B::Buffer {
        let len = bytes(width, element).expect("validated prefill row") * rows as u64;
        self.backend.view(&self.items[index], 0, len)
    }

    pub(crate) fn row(&self, index: usize, row: usize, width: usize, element: usize) -> B::Buffer {
        let stride = bytes(width, element).expect("validated prefill row");
        self.backend
            .view(&self.items[index], row as u64 * stride, stride)
    }

    pub(crate) fn scratch(&self, cfg: &MistralConfig, rows: usize) -> Scratch<B::Buffer> {
        Scratch {
            x: self.rows(X, rows, cfg.embedding_length, 4),
            normed: self.rows(NORMED, rows, cfg.embedding_length, 2),
            q: self.rows(Q, rows, cfg.q_width, 2),
            k: self.rows(K, rows, cfg.k_width, 2),
            v: self.rows(V, rows, cfg.v_width, 2),
            attn: self.rows(ATTN, rows, cfg.attention_width, 2),
            proj: self.rows(PROJ, rows, cfg.embedding_length, 2),
            gate: self.rows(GATE, rows, cfg.feed_forward_length, 2),
            up: self.rows(UP, rows, cfg.feed_forward_length, 2),
            act: self.rows(ACT, rows, cfg.feed_forward_length, 2),
            ffn_out: self.rows(FFN_OUT, rows, cfg.embedding_length, 2),
        }
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
    use super::BATCH_ROWS;

    #[test]
    fn batch_capacity_matches_the_selected_profile() {
        assert_eq!(
            BATCH_ROWS,
            if cfg!(feature = "metal") {
                32
            } else if cfg!(feature = "vulkan") {
                32
            } else {
                4
            }
        );
    }
}
