/*
 * graph_horizon_engine — homogeneous request session
 * Owns one backend's request KV, delegates full graph traversal and readback,
 * and frees state unless transferred to a homogeneous cache. It owns no family
 * parsing, loading, or placement.
 */

use std::marker::PhantomData;

use color_eyre::eyre::Result;

use super::contract::{LayeredGraph, RuntimeSession};
use crate::backend::Backend;
use crate::backend::hybrid::weights::runtime::RuntimeShape;
use crate::kv_cache::scheme::KvQuant;
use crate::kv_cache::{self, Kv};

pub(crate) struct HomogeneousSession<'a, B: Backend, G: LayeredGraph> {
    backend: &'a B,
    config: &'a G::Config,
    kv: Option<Kv<B::Buffer>>,
    row_capacity: usize,
    graph: PhantomData<G>,
}

impl<'a, B: Backend, G: LayeredGraph> HomogeneousSession<'a, B, G> {
    pub(crate) fn new(
        backend: &'a B,
        config: &'a G::Config,
        shape: RuntimeShape,
        row_capacity: usize,
        context: usize,
        scheme: KvQuant,
    ) -> Result<Self> {
        Self::with_state(backend, config, shape, row_capacity, context, scheme, None)
    }

    pub(crate) fn with_state(
        backend: &'a B,
        config: &'a G::Config,
        shape: RuntimeShape,
        row_capacity: usize,
        context: usize,
        scheme: KvQuant,
        state: Option<Kv<B::Buffer>>,
    ) -> Result<Self> {
        let kv = match state {
            Some(kv) => kv,
            None => kv_cache::alloc_shape(
                backend,
                shape.block_count,
                context,
                shape.kv_heads,
                shape.key_length,
                shape.value_length,
                scheme,
            )?,
        };
        Ok(Self {
            backend,
            config,
            kv: Some(kv),
            row_capacity,
            graph: PhantomData,
        })
    }

    fn kv(&self) -> &Kv<B::Buffer> {
        self.kv.as_ref().expect("request KV exists until drop")
    }

    #[cfg(any(feature = "vulkan", feature = "metal"))]
    pub(crate) fn into_state(mut self) -> Kv<B::Buffer> {
        self.kv.take().expect("cached session returns its KV")
    }
}

impl<B: Backend, G: LayeredGraph> RuntimeSession for HomogeneousSession<'_, B, G> {
    type Graph = G;

    fn prefill(
        &self,
        prompt: &[u32],
        base: usize,
        before: &mut dyn FnMut() -> Result<()>,
    ) -> Result<()> {
        G::prefill(
            self.backend,
            self.config,
            self.kv(),
            prompt,
            base,
            self.row_capacity,
            before,
        )
    }

    fn token(&self, token: u32, position: usize) -> Result<()> {
        G::token(self.backend, self.config, self.kv(), token, position)
    }

    fn logits(&self, vocab: usize) -> Result<Vec<f32>> {
        self.backend
            .read_logits(&self.backend.buffers().logits, vocab)
    }

    fn argmax(&self, vocab: usize) -> Result<u32> {
        self.backend
            .read_argmax(&self.backend.buffers().logits, vocab)
    }

    fn topk(&self, vocab: usize, k: usize) -> Result<Vec<(u32, f32)>> {
        self.backend
            .read_topk(&self.backend.buffers().logits, vocab, k)
    }
}

impl<B: Backend, G: LayeredGraph> Drop for HomogeneousSession<'_, B, G> {
    fn drop(&mut self) {
        if let Some(kv) = self.kv.take() {
            kv_cache::free(self.backend, kv);
        }
    }
}
