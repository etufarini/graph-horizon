/*
 * gh_zero_engine — partitioned token traversal
 * Executes plain or greedy decode for immutable all-device, CPU-only, and mixed
 * ownership. Mixed traversal records one CPU prefix, crosses the residual once,
 * and completes the device suffix; session/KV ownership and sampling stay outside.
 */

use color_eyre::eyre::{Result, bail};

use super::{KvState, PartitionedSession};
use crate::backend::Backend;
use crate::backend::cpu::{CpuBackend, CpuBuffer};
use crate::backend::hybrid::contract::HybridDevice;
use crate::backend::hybrid::{HybridBackends, weights::runtime::RuntimeShape};
use crate::kv_cache::Kv;
use crate::runtime::contract::LayeredGraph;

pub(super) fn token<D: HybridDevice, G: LayeredGraph>(
    session: &PartitionedSession<'_, D, G>,
    token: u32,
    position: usize,
) -> Result<()> {
    match (session.backends, session.state.as_ref()) {
        (HybridBackends::AllGpu(backend), Some(KvState::AllGpu(kv))) => {
            G::token(backend, session.config, kv, token, position)
        }
        (HybridBackends::CpuOnly(backend), Some(KvState::CpuOnly(kv))) => {
            G::token(backend, session.config, kv, token, position)
        }
        (
            HybridBackends::Mixed { cpu, gpu },
            Some(KvState::Mixed {
                cpu: cpu_kv,
                gpu: gpu_kv,
            }),
        ) => {
            prefix::<D, G>(
                cpu,
                gpu,
                session.config,
                cpu_kv,
                gpu_kv,
                session.shape,
                token,
                position,
            )?;
            let encoder = suffix::<D, G>(gpu, session.config, gpu_kv, position)?;
            gpu.submit(encoder)
        }
        _ => unreachable!("partition KV matches immutable owners"),
    }
}

pub(super) fn token_argmax<D: HybridDevice, G: LayeredGraph>(
    session: &PartitionedSession<'_, D, G>,
    token: u32,
    position: usize,
    vocab: usize,
) -> Result<u32> {
    match (session.backends, session.state.as_ref()) {
        (HybridBackends::AllGpu(backend), Some(KvState::AllGpu(kv))) => {
            G::token_argmax(backend, session.config, kv, token, position, vocab)
        }
        (HybridBackends::CpuOnly(backend), Some(KvState::CpuOnly(kv))) => {
            G::token_argmax(backend, session.config, kv, token, position, vocab)
        }
        (
            HybridBackends::Mixed { cpu, gpu },
            Some(KvState::Mixed {
                cpu: cpu_kv,
                gpu: gpu_kv,
            }),
        ) => {
            prefix::<D, G>(
                cpu,
                gpu,
                session.config,
                cpu_kv,
                gpu_kv,
                session.shape,
                token,
                position,
            )?;
            let encoder = suffix::<D, G>(gpu, session.config, gpu_kv, position)?;
            gpu.submit_argmax(encoder, &gpu.buffers().logits, vocab)
        }
        _ => unreachable!("partition KV matches immutable owners"),
    }
}

#[allow(clippy::too_many_arguments)]
fn prefix<D: HybridDevice, G: LayeredGraph>(
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
    let encoder = cpu.begin()?;
    G::embedding(cpu, &encoder, config, token)?;
    G::range(
        cpu,
        &encoder,
        config,
        cpu_kv,
        0..cpu_kv.block_count,
        position,
    )?;
    cpu.submit(encoder)?;
    crate::backend::hybrid::crossing::copy(
        &cpu.buffers().scratch.x,
        gpu,
        &gpu.buffers().scratch.x,
        shape.embedding,
    )
}

fn suffix<D: HybridDevice, G: LayeredGraph>(
    gpu: &D,
    config: &G::Config,
    gpu_kv: &Kv<D::Buffer>,
    position: usize,
) -> Result<D::Encoder> {
    let encoder = gpu.begin()?;
    if let Err(error) = G::range(
        gpu,
        &encoder,
        config,
        gpu_kv,
        0..gpu_kv.block_count,
        position,
    ) {
        let _ = gpu.submit(encoder);
        return Err(error);
    }
    G::tail(gpu, &encoder, config);
    Ok(encoder)
}
