/*
 * graph_horizon_engine — partitioned bounded prefill
 * Runs each prompt batch on immutable owners and performs exactly one checked
 * CPU-to-GPU residual crossing only for mixed batches. It owns no allocation.
 */

use color_eyre::eyre::{Result, bail, eyre};

use super::{KvState, PartitionedSession};
use crate::backend::hybrid::HybridBackends;
use crate::backend::hybrid::contract::HybridDevice;
use crate::runtime::contract::LayeredGraph;

pub(super) fn run<D: HybridDevice, G: LayeredGraph>(
    session: &PartitionedSession<'_, D, G>,
    prompt: &[u32],
    before_batch: &mut dyn FnMut() -> Result<()>,
) -> Result<()> {
    match (session.backends, session.state.as_ref()) {
        (HybridBackends::AllGpu(backend), Some(KvState::AllGpu(kv))) => G::prefill(
            backend,
            session.config,
            kv,
            prompt,
            session.shape.gpu_prefill_rows,
            before_batch,
        ),
        (HybridBackends::CpuOnly(backend), Some(KvState::CpuOnly(kv))) => G::prefill(
            backend,
            session.config,
            kv,
            prompt,
            session.shape.cpu_prefill_rows,
            before_batch,
        ),
        (
            HybridBackends::Mixed { cpu, gpu },
            Some(KvState::Mixed {
                cpu: cpu_kv,
                gpu: gpu_kv,
            }),
        ) => {
            if prompt.is_empty() {
                bail!("runtime: empty prompt");
            }
            let rows = session.shape.mixed_prefill_rows;
            let cpu_batch = G::batch(cpu, session.config, rows)?;
            let gpu_batch = G::batch(gpu, session.config, rows)?;
            for (batch_index, tokens) in prompt.chunks(rows).enumerate() {
                before_batch()?;
                let base = batch_index
                    .checked_mul(rows)
                    .ok_or_else(|| eyre!("hybrid residual crossing overflow"))?;
                G::record_batch(
                    cpu,
                    session.config,
                    cpu_kv,
                    &cpu_batch,
                    tokens,
                    base,
                    true,
                    false,
                )?;
                let elements = tokens
                    .len()
                    .checked_mul(session.shape.embedding)
                    .ok_or_else(|| eyre!("hybrid residual crossing overflow"))?;
                crate::backend::hybrid::crossing::copy(
                    G::batch_residual(&cpu_batch),
                    gpu,
                    G::batch_residual(&gpu_batch),
                    elements,
                )?;
                G::record_batch(
                    gpu,
                    session.config,
                    gpu_kv,
                    &gpu_batch,
                    tokens,
                    base,
                    false,
                    true,
                )?;
            }
            Ok(())
        }
        _ => unreachable!("partition KV matches immutable owners"),
    }
}
