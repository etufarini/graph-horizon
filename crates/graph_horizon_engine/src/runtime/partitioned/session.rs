/*
 * graph_horizon_engine — partitioned decode session
 * Owns per-side request KV, executes an immutable CPU prefix/GPU suffix with one
 * checked crossing, routes tail/readback, and frees partial or complete state.
 */

use color_eyre::eyre::{Result, bail};

use super::{KvState, PartitionedSession, prefill};
use crate::backend::Backend;
use crate::backend::cpu::{CpuBackend, CpuBuffer};
use crate::backend::hybrid::contract::HybridDevice;
use crate::backend::hybrid::weights::runtime::RuntimeShape;
use crate::backend::hybrid::{HybridBackends, HybridRuntime};
use crate::kv_cache::scheme::KvQuant;
use crate::kv_cache::{self, Kv};
use crate::runtime::contract::{LayeredGraph, RuntimeSession};

impl<'a, D: HybridDevice, G: LayeredGraph> PartitionedSession<'a, D, G> {
    pub(crate) fn new(
        runtime: &'a HybridRuntime<D>,
        config: &'a G::Config,
        shape: RuntimeShape,
        context: usize,
        scheme: KvQuant,
    ) -> Result<Self> {
        let state = match &runtime.backends {
            HybridBackends::AllGpu(gpu) => KvState::AllGpu(alloc(
                gpu,
                runtime.plan.block_count,
                shape,
                context,
                scheme,
            )?),
            HybridBackends::CpuOnly(cpu) => KvState::CpuOnly(alloc(
                cpu,
                runtime.plan.block_count,
                shape,
                context,
                scheme,
            )?),
            HybridBackends::Mixed { cpu, gpu } => {
                let cpu_kv = alloc(cpu, runtime.plan.cpu_layers, shape, context, scheme)?;
                let gpu_kv = match alloc(gpu, runtime.plan.gpu_layers, shape, context, scheme) {
                    Ok(kv) => kv,
                    Err(error) => {
                        kv_cache::free(cpu, cpu_kv);
                        return Err(error);
                    }
                };
                KvState::Mixed {
                    cpu: cpu_kv,
                    gpu: gpu_kv,
                }
            }
        };
        Ok(Self {
            backends: &runtime.backends,
            config,
            state: Some(state),
            shape,
            graph: std::marker::PhantomData,
        })
    }

    fn mixed_token(
        cpu: &CpuBackend,
        gpu: &D,
        config: &G::Config,
        cpu_kv: &Kv<CpuBuffer>,
        gpu_kv: &Kv<D::Buffer>,
        shape: RuntimeShape,
        token: u32,
        position: usize,
    ) -> Result<()> {
        if position >= cpu_kv.context || position >= gpu_kv.context {
            bail!("runtime: position beyond context");
        }
        let cpu_encoder = cpu.begin()?;
        G::embedding(cpu, &cpu_encoder, config, token)?;
        G::range(
            cpu,
            &cpu_encoder,
            config,
            cpu_kv,
            0..cpu_kv.block_count,
            position,
        )?;
        cpu.submit(cpu_encoder)?;
        crate::backend::hybrid::crossing::copy(
            &cpu.buffers().scratch.x,
            gpu,
            &gpu.buffers().scratch.x,
            shape.embedding,
        )?;
        let gpu_encoder = gpu.begin()?;
        G::range(
            gpu,
            &gpu_encoder,
            config,
            gpu_kv,
            0..gpu_kv.block_count,
            position,
        )?;
        G::tail(gpu, &gpu_encoder, config);
        gpu.submit(gpu_encoder)
    }
}

impl<D: HybridDevice, G: LayeredGraph> RuntimeSession for PartitionedSession<'_, D, G> {
    type Graph = G;

    fn prefill(&self, prompt: &[u32], before: &mut dyn FnMut() -> Result<()>) -> Result<()> {
        prefill::run::<D, G>(self, prompt, before)
    }

    fn token(&self, token: u32, position: usize) -> Result<()> {
        match (self.backends, self.state.as_ref()) {
            (HybridBackends::AllGpu(backend), Some(KvState::AllGpu(kv))) => {
                G::token(backend, self.config, kv, token, position)
            }
            (HybridBackends::CpuOnly(backend), Some(KvState::CpuOnly(kv))) => {
                G::token(backend, self.config, kv, token, position)
            }
            (
                HybridBackends::Mixed { cpu, gpu },
                Some(KvState::Mixed {
                    cpu: cpu_kv,
                    gpu: gpu_kv,
                }),
            ) => Self::mixed_token(
                cpu,
                gpu,
                self.config,
                cpu_kv,
                gpu_kv,
                self.shape,
                token,
                position,
            ),
            _ => unreachable!("partition KV matches immutable owners"),
        }
    }

    fn logits(&self, vocab: usize) -> Result<Vec<f32>> {
        match self.backends {
            HybridBackends::AllGpu(backend) => {
                backend.read_logits(&backend.buffers().logits, vocab)
            }
            HybridBackends::Mixed { gpu, .. } => gpu.read_logits(&gpu.buffers().logits, vocab),
            HybridBackends::CpuOnly(backend) => {
                backend.read_logits(&backend.buffers().logits, vocab)
            }
        }
    }

    fn argmax(&self, vocab: usize) -> Result<u32> {
        match self.backends {
            HybridBackends::AllGpu(backend) => {
                backend.read_argmax(&backend.buffers().logits, vocab)
            }
            HybridBackends::Mixed { gpu, .. } => gpu.read_argmax(&gpu.buffers().logits, vocab),
            HybridBackends::CpuOnly(backend) => {
                backend.read_argmax(&backend.buffers().logits, vocab)
            }
        }
    }

    fn topk(&self, vocab: usize, k: usize) -> Result<Vec<(u32, f32)>> {
        match self.backends {
            HybridBackends::AllGpu(backend) => {
                backend.read_topk(&backend.buffers().logits, vocab, k)
            }
            HybridBackends::Mixed { gpu, .. } => gpu.read_topk(&gpu.buffers().logits, vocab, k),
            HybridBackends::CpuOnly(backend) => {
                backend.read_topk(&backend.buffers().logits, vocab, k)
            }
        }
    }
}

fn alloc<B: Backend>(
    backend: &B,
    layers: usize,
    shape: RuntimeShape,
    context: usize,
    scheme: KvQuant,
) -> Result<Kv<B::Buffer>> {
    kv_cache::alloc_shape(
        backend,
        layers,
        context,
        shape.kv_heads,
        shape.key_length,
        shape.value_length,
        scheme,
    )
}
