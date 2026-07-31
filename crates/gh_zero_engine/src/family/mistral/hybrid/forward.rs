/*
 * gh_zero_engine — hybrid request state and decode execution
 * Allocates partitioned request KV and decodes on immutable owners; only mixed
 * execution crosses one residual, and post-plan failures never retry.
 */

use color_eyre::eyre::{Result, bail};

use super::{HybridBackends, LoadedHybrid, crossing};
use crate::backend::Backend;
use crate::backend::cpu::{CpuBackend, CpuBuffer};
use crate::backend::vulkan::{VulkanBackend, buffers::GpuBuffer};
use crate::family::mistral::MistralConfig;
use crate::family::mistral::graph::{forward as graph, tail};
use crate::kv_cache::scheme::KvQuant;
use crate::kv_cache::{self, Kv};

pub(super) enum KvState {
    AllGpu(Kv<GpuBuffer>),
    Mixed {
        cpu: Kv<CpuBuffer>,
        gpu: Kv<GpuBuffer>,
    },
    CpuOnly(Kv<CpuBuffer>),
}

pub(crate) struct RequestKv<'a> {
    pub(super) backends: &'a HybridBackends,
    pub(super) state: Option<KvState>,
}

impl<'a> RequestKv<'a> {
    pub(crate) fn new(
        runtime: &'a LoadedHybrid,
        cfg: &MistralConfig,
        context: usize,
        scheme: KvQuant,
    ) -> Result<Self> {
        let state = match &runtime.backends {
            HybridBackends::AllGpu(gpu) => {
                KvState::AllGpu(alloc(gpu, runtime.plan.block_count, cfg, context, scheme)?)
            }
            HybridBackends::CpuOnly(cpu) => {
                KvState::CpuOnly(alloc(cpu, runtime.plan.block_count, cfg, context, scheme)?)
            }
            HybridBackends::Mixed { cpu, gpu } => {
                let cpu_kv = alloc(cpu, runtime.plan.cpu_layers, cfg, context, scheme)?;
                let gpu_kv = match alloc(gpu, runtime.plan.gpu_layers, cfg, context, scheme) {
                    Ok(kv) => kv,
                    Err(error) => {
                        kv_cache::free(cpu, cpu_kv);
                        return Err(error);
                    }
                };
                // Each cache contains only local layers; neither backend can
                // address or allocate the other side of the split.
                KvState::Mixed {
                    cpu: cpu_kv,
                    gpu: gpu_kv,
                }
            }
        };
        Ok(Self {
            backends: &runtime.backends,
            state: Some(state),
        })
    }
}

impl Drop for RequestKv<'_> {
    fn drop(&mut self) {
        match (self.backends, self.state.take()) {
            (HybridBackends::AllGpu(backend), Some(KvState::AllGpu(kv))) => {
                kv_cache::free(backend, kv)
            }
            (HybridBackends::CpuOnly(backend), Some(KvState::CpuOnly(kv))) => {
                kv_cache::free(backend, kv)
            }
            (
                HybridBackends::Mixed { cpu, gpu },
                Some(KvState::Mixed {
                    cpu: cpu_kv,
                    gpu: gpu_kv,
                }),
            ) => {
                kv_cache::free(cpu, cpu_kv);
                kv_cache::free(gpu, gpu_kv);
            }
            _ => unreachable!("hybrid KV state matches immutable backend owners"),
        }
    }
}

pub(crate) fn token(kv: &RequestKv<'_>, cfg: &MistralConfig, token: u32, pos: usize) -> Result<()> {
    match (kv.backends, kv.state.as_ref()) {
        (HybridBackends::AllGpu(backend), Some(KvState::AllGpu(state))) => {
            graph::token(backend, cfg, state, token, pos)
        }
        (HybridBackends::CpuOnly(backend), Some(KvState::CpuOnly(state))) => {
            graph::token(backend, cfg, state, token, pos)
        }
        (
            HybridBackends::Mixed { cpu, gpu },
            Some(KvState::Mixed {
                cpu: cpu_kv,
                gpu: gpu_kv,
            }),
        ) => mixed(cpu, gpu, cfg, cpu_kv, gpu_kv, token, pos),
        _ => unreachable!("hybrid KV state matches immutable backend owners"),
    }
}

pub(crate) fn read_logits(kv: &RequestKv<'_>, vocab: usize) -> Result<Vec<f32>> {
    match kv.backends {
        HybridBackends::AllGpu(backend) => backend.read_logits(&backend.buffers().logits, vocab),
        HybridBackends::Mixed { gpu, .. } => gpu.read_logits(&gpu.buffers().logits, vocab),
        HybridBackends::CpuOnly(backend) => backend.read_logits(&backend.buffers().logits, vocab),
    }
}

pub(crate) fn read_argmax(kv: &RequestKv<'_>, vocab: usize) -> Result<u32> {
    match kv.backends {
        HybridBackends::AllGpu(backend) => backend.read_argmax(&backend.buffers().logits, vocab),
        HybridBackends::Mixed { gpu, .. } => gpu.read_argmax(&gpu.buffers().logits, vocab),
        HybridBackends::CpuOnly(backend) => backend.read_argmax(&backend.buffers().logits, vocab),
    }
}

pub(crate) fn read_topk(kv: &RequestKv<'_>, vocab: usize, k: usize) -> Result<Vec<(u32, f32)>> {
    match kv.backends {
        HybridBackends::AllGpu(backend) => backend.read_topk(&backend.buffers().logits, vocab, k),
        HybridBackends::Mixed { gpu, .. } => gpu.read_topk(&gpu.buffers().logits, vocab, k),
        HybridBackends::CpuOnly(backend) => backend.read_topk(&backend.buffers().logits, vocab, k),
    }
}

fn alloc<B: Backend>(
    backend: &B,
    layers: usize,
    cfg: &MistralConfig,
    context: usize,
    scheme: KvQuant,
) -> Result<Kv<B::Buffer>> {
    kv_cache::alloc_shape(
        backend,
        layers,
        context,
        cfg.kv_head_count,
        cfg.key_length,
        cfg.value_length,
        scheme,
    )
}

#[allow(clippy::too_many_arguments)]
fn mixed(
    cpu: &CpuBackend,
    gpu: &VulkanBackend,
    cfg: &MistralConfig,
    cpu_kv: &Kv<CpuBuffer>,
    gpu_kv: &Kv<GpuBuffer>,
    token: u32,
    pos: usize,
) -> Result<()> {
    if pos >= cpu_kv.context || pos >= cfg.context_length {
        bail!("mistral graph: position beyond context");
    }
    let cpu_enc = cpu.begin()?;
    graph::embedding(cpu, &cpu_enc, cfg, token)?;
    graph::range(cpu, &cpu_enc, cfg, cpu_kv, 0..cpu_kv.block_count, pos)?;
    cpu.submit(cpu_enc)?;
    crossing::copy(
        &cpu.buffers().scratch.x,
        gpu,
        &gpu.buffers().scratch.x,
        cfg.embedding_length,
    )?;
    let gpu_enc = gpu.begin()?;
    graph::range(gpu, &gpu_enc, cfg, gpu_kv, 0..gpu_kv.block_count, pos)?;
    tail::record(gpu, &gpu_enc, cfg);
    gpu.submit(gpu_enc)
}
