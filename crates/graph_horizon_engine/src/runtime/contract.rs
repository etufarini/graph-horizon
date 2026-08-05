/*
 * graph_horizon_engine — layered graph and request-session contracts
 * Defines static generic seams between a family graph, runtime traversal, and
 * model-agnostic backends. It owns no concrete backend, family data, or resources.
 */

#[cfg(any(feature = "vulkan-hybrid", feature = "metal-hybrid"))]
use std::ops::Range;

use color_eyre::eyre::Result;

use crate::backend::Backend;
use crate::backend::hybrid::weights::runtime::RuntimeShape;
use crate::kv_cache::Kv;

pub(crate) trait LayeredGraph: Sized {
    type Config;
    #[cfg(any(feature = "vulkan-hybrid", feature = "metal-hybrid"))]
    type Batch<'a, B: Backend>
    where
        B: 'a;

    #[cfg(any(feature = "vulkan-hybrid", feature = "metal-hybrid"))]
    const BATCH_ROWS: usize;

    fn shape(config: &Self::Config) -> RuntimeShape;

    fn token<B: Backend>(
        backend: &B,
        config: &Self::Config,
        kv: &Kv<B::Buffer>,
        token: u32,
        position: usize,
    ) -> Result<()>;
    #[cfg(any(feature = "vulkan-hybrid", feature = "metal-hybrid"))]
    fn embedding<B: Backend>(
        backend: &B,
        encoder: &B::Encoder,
        config: &Self::Config,
        token: u32,
    ) -> Result<()>;
    #[cfg(any(feature = "vulkan-hybrid", feature = "metal-hybrid"))]
    fn range<B: Backend>(
        backend: &B,
        encoder: &B::Encoder,
        config: &Self::Config,
        kv: &Kv<B::Buffer>,
        layers: Range<usize>,
        position: usize,
    ) -> Result<()>;
    #[cfg(any(feature = "vulkan-hybrid", feature = "metal-hybrid"))]
    fn tail<B: Backend>(backend: &B, encoder: &B::Encoder, config: &Self::Config);
    fn prefill<B: Backend>(
        backend: &B,
        config: &Self::Config,
        kv: &Kv<B::Buffer>,
        prompt: &[u32],
        before_batch: &mut dyn FnMut() -> Result<()>,
    ) -> Result<()>;
    #[cfg(any(feature = "vulkan-hybrid", feature = "metal-hybrid"))]
    fn batch<'a, B: Backend>(backend: &'a B, config: &Self::Config) -> Result<Self::Batch<'a, B>>;
    #[cfg(any(feature = "vulkan-hybrid", feature = "metal-hybrid"))]
    fn batch_residual<'a, B: Backend>(batch: &'a Self::Batch<'_, B>) -> &'a B::Buffer;
    #[cfg(any(feature = "vulkan-hybrid", feature = "metal-hybrid"))]
    #[allow(clippy::too_many_arguments)]
    fn record_batch<B: Backend>(
        backend: &B,
        config: &Self::Config,
        kv: &Kv<B::Buffer>,
        batch: &Self::Batch<'_, B>,
        tokens: &[u32],
        base: usize,
        embedding: bool,
        tail: bool,
    ) -> Result<()>;
}

pub(crate) trait RuntimeSession {
    type Graph: LayeredGraph;

    fn prefill(&self, prompt: &[u32], before: &mut dyn FnMut() -> Result<()>) -> Result<()>;
    fn token(&self, token: u32, position: usize) -> Result<()>;
    fn logits(&self, vocab: usize) -> Result<Vec<f32>>;
    fn argmax(&self, vocab: usize) -> Result<u32>;
    fn topk(&self, vocab: usize, k: usize) -> Result<Vec<(u32, f32)>>;
}
