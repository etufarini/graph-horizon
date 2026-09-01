/*
 * graph_horizon_engine — backend-generic Ministral graph boundary
 * Single responsibility: expose the shared dense block, range, prefill and tail
 * recorders over `Backend` buffers. It owns no concrete backend, weight format,
 * request lifecycle, sampling policy or model-size dispatch.
 */

pub(crate) mod block;
pub(crate) mod forward;
pub(crate) mod mlp;
pub(crate) mod prefill;
pub(crate) mod tail;

use color_eyre::eyre::Result;

use crate::backend::Backend;
use crate::backend::hybrid::weights::runtime::RuntimeShape;
use crate::kv_cache::Kv;
use crate::runtime::contract::LayeredGraph;

pub(crate) struct MistralGraph;

impl LayeredGraph for MistralGraph {
    type Config = super::MistralConfig;
    #[cfg(any(feature = "vulkan-hybrid", feature = "metal-hybrid"))]
    type Batch<'a, B: Backend>
        = prefill::BatchBuffers<'a, B>
    where
        B: 'a;

    fn shape(config: &Self::Config) -> RuntimeShape {
        RuntimeShape {
            block_count: config.block_count,
            #[cfg(any(
                feature = "metal",
                feature = "vulkan-hybrid",
                feature = "metal-hybrid",
                feature = "cuda"
            ))]
            embedding: config.embedding_length,
            #[cfg(any(
                feature = "metal",
                feature = "vulkan-hybrid",
                feature = "metal-hybrid",
                feature = "cuda"
            ))]
            q: config.q_width,
            #[cfg(any(
                feature = "metal",
                feature = "vulkan-hybrid",
                feature = "metal-hybrid",
                feature = "cuda"
            ))]
            k: config.k_width,
            #[cfg(any(
                feature = "metal",
                feature = "vulkan-hybrid",
                feature = "metal-hybrid",
                feature = "cuda"
            ))]
            v: config.v_width,
            #[cfg(any(
                feature = "metal",
                feature = "vulkan-hybrid",
                feature = "metal-hybrid",
                feature = "cuda"
            ))]
            attention: config.attention_width,
            #[cfg(any(
                feature = "metal",
                feature = "vulkan-hybrid",
                feature = "metal-hybrid",
                feature = "cuda"
            ))]
            feed_forward: config.feed_forward_length,
            #[cfg(any(
                feature = "metal",
                feature = "vulkan-hybrid",
                feature = "metal-hybrid",
                feature = "cuda"
            ))]
            vocab: config.vocab_size,
            kv_heads: config.kv_head_count,
            key_length: config.key_length,
            value_length: config.value_length,
            cpu_prefill_rows: prefill::CPU_ROWS,
            gpu_prefill_rows: prefill::HOMOGENEOUS_GPU_ROWS,
            mixed_prefill_rows: prefill::MIXED_ROWS,
        }
    }

    fn token<B: Backend>(
        backend: &B,
        config: &Self::Config,
        kv: &Kv<B::Buffer>,
        token: u32,
        position: usize,
    ) -> Result<()> {
        forward::token(backend, config, kv, token, position)
    }

    #[cfg(any(feature = "vulkan-hybrid", feature = "metal-hybrid"))]
    fn embedding<B: Backend>(
        backend: &B,
        encoder: &B::Encoder,
        config: &Self::Config,
        token: u32,
    ) -> Result<()> {
        forward::embedding(backend, encoder, config, token)
    }

    #[cfg(any(feature = "vulkan-hybrid", feature = "metal-hybrid"))]
    fn range<B: Backend>(
        backend: &B,
        encoder: &B::Encoder,
        config: &Self::Config,
        kv: &Kv<B::Buffer>,
        layers: std::ops::Range<usize>,
        position: usize,
    ) -> Result<()> {
        forward::range(backend, encoder, config, kv, layers, position)
    }

    #[cfg(any(feature = "vulkan-hybrid", feature = "metal-hybrid"))]
    fn tail<B: Backend>(backend: &B, encoder: &B::Encoder, config: &Self::Config) {
        tail::record(backend, encoder, config);
    }

    fn prefill<B: Backend>(
        backend: &B,
        config: &Self::Config,
        kv: &Kv<B::Buffer>,
        prompt: &[u32],
        base: usize,
        row_capacity: usize,
        before_batch: &mut dyn FnMut() -> Result<()>,
    ) -> Result<()> {
        prefill::prefill(
            backend,
            config,
            kv,
            prompt,
            base,
            row_capacity,
            before_batch,
        )
    }

    #[cfg(any(feature = "vulkan-hybrid", feature = "metal-hybrid"))]
    fn batch<'a, B: Backend>(
        backend: &'a B,
        config: &Self::Config,
        row_capacity: usize,
    ) -> Result<Self::Batch<'a, B>> {
        prefill::BatchBuffers::new(backend, config, row_capacity)
    }

    #[cfg(any(feature = "vulkan-hybrid", feature = "metal-hybrid"))]
    fn batch_residual<'a, B: Backend>(batch: &'a Self::Batch<'_, B>) -> &'a B::Buffer {
        batch.all(prefill::X)
    }

    #[cfg(any(feature = "vulkan-hybrid", feature = "metal-hybrid"))]
    fn record_batch<B: Backend>(
        backend: &B,
        config: &Self::Config,
        kv: &Kv<B::Buffer>,
        batch: &Self::Batch<'_, B>,
        tokens: &[u32],
        base: usize,
        embedding: bool,
        tail: bool,
    ) -> Result<()> {
        prefill::record_batch(backend, config, kv, batch, tokens, base, embedding, tail)
    }
}

#[cfg(test)]
pub(crate) mod shape;
