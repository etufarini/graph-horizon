/*
 * gh_zero_engine — layered graph and request-session contracts
 * Defines static generic seams between a family graph, runtime traversal, and
 * model-agnostic backends. It owns no concrete backend, family data, or resources.
 */

use std::ops::Range;

use color_eyre::eyre::Result;

use crate::backend::Backend;
use crate::backend::hybrid::weights::runtime::RuntimeShape;
use crate::kv_cache::Kv;

pub(crate) trait LayeredGraph: Sized {
    type Config;
    type Batch<'a, B: Backend>
    where
        B: 'a;

    const BATCH_ROWS: usize;

    fn shape(config: &Self::Config) -> RuntimeShape;

    fn token<B: Backend>(
        backend: &B,
        config: &Self::Config,
        kv: &Kv<B::Buffer>,
        token: u32,
        position: usize,
    ) -> Result<()>;
    fn embedding<B: Backend>(
        backend: &B,
        encoder: &B::Encoder,
        config: &Self::Config,
        token: u32,
    ) -> Result<()>;
    fn range<B: Backend>(
        backend: &B,
        encoder: &B::Encoder,
        config: &Self::Config,
        kv: &Kv<B::Buffer>,
        layers: Range<usize>,
        position: usize,
    ) -> Result<()>;
    fn tail<B: Backend>(backend: &B, encoder: &B::Encoder, config: &Self::Config);
    fn prefill<B: Backend>(
        backend: &B,
        config: &Self::Config,
        kv: &Kv<B::Buffer>,
        prompt: &[u32],
        before_batch: &mut dyn FnMut() -> Result<()>,
    ) -> Result<()>;
    fn batch<'a, B: Backend>(backend: &'a B, config: &Self::Config) -> Result<Self::Batch<'a, B>>;
    fn batch_residual<'a, B: Backend>(batch: &'a Self::Batch<'_, B>) -> &'a B::Buffer;
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
