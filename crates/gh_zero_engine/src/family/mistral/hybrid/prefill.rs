/*
 * gh_zero_engine — bounded hybrid prefill execution
 * Runs homogeneous prefill on one backend or, for each bounded mixed batch,
 * records the CPU prefix, crosses one FP32 residual matrix, and records the
 * Vulkan suffix/tail. It owns no placement, weights or sampling.
 */

use color_eyre::eyre::{Result, bail, eyre};

use super::HybridBackends;
use super::crossing;
use super::forward::{KvState, RequestKv};
use crate::family::mistral::MistralConfig;
use crate::family::mistral::graph::prefill as graph;

pub(crate) fn run<F: FnMut() -> Result<()>>(
    kv: &RequestKv<'_>,
    cfg: &MistralConfig,
    prompt: &[u32],
    mut before_batch: F,
) -> Result<()> {
    match (kv.backends, kv.state.as_ref()) {
        (HybridBackends::AllGpu(backend), Some(KvState::AllGpu(state))) => {
            graph::prefill_with(backend, cfg, state, prompt, before_batch)
        }
        (HybridBackends::CpuOnly(backend), Some(KvState::CpuOnly(state))) => {
            graph::prefill_with(backend, cfg, state, prompt, before_batch)
        }
        (
            HybridBackends::Mixed { cpu, gpu },
            Some(KvState::Mixed {
                cpu: cpu_kv,
                gpu: gpu_kv,
            }),
        ) => {
            if prompt.is_empty() {
                bail!("mistral graph: empty prompt");
            }
            let cpu_batch = graph::BatchBuffers::new(cpu, cfg)?;
            let gpu_batch = graph::BatchBuffers::new(gpu, cfg)?;
            for (batch_index, tokens) in prompt.chunks(graph::BATCH_ROWS).enumerate() {
                before_batch()?;
                let base = batch_index * graph::BATCH_ROWS;
                graph::record_batch(cpu, cfg, cpu_kv, &cpu_batch, tokens, base, true, false)?;
                let elements = tokens
                    .len()
                    .checked_mul(cfg.embedding_length)
                    .ok_or_else(|| eyre!("hybrid residual crossing overflow"))?;
                crossing::copy(
                    cpu_batch.all(graph::X),
                    gpu,
                    gpu_batch.all(graph::X),
                    elements,
                )?;
                graph::record_batch(gpu, cfg, gpu_kv, &gpu_batch, tokens, base, false, true)?;
            }
            Ok(())
        }
        _ => unreachable!("hybrid KV state matches immutable backend owners"),
    }
}
