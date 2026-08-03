/*
 * gh_zero_engine — partitioned runtime namespace
 * Defines partitioned request state and its deterministic cleanup, exporting
 * decode and bounded-prefill orchestration. Planning and graph policy stay out.
 */

use std::marker::PhantomData;

use crate::backend::cpu::CpuBuffer;
use crate::backend::hybrid::HybridBackends;
use crate::backend::hybrid::contract::HybridDevice;
use crate::backend::hybrid::weights::runtime::RuntimeShape;
use crate::kv_cache::{self, Kv};
use crate::runtime::contract::LayeredGraph;

mod prefill;
mod session;

pub(super) enum KvState<D: HybridDevice> {
    AllGpu(Kv<D::Buffer>),
    Mixed {
        cpu: Kv<CpuBuffer>,
        gpu: Kv<D::Buffer>,
    },
    CpuOnly(Kv<CpuBuffer>),
}

pub(crate) struct PartitionedSession<'a, D: HybridDevice, G: LayeredGraph> {
    pub(super) backends: &'a HybridBackends<D>,
    pub(super) config: &'a G::Config,
    pub(super) state: Option<KvState<D>>,
    pub(super) shape: RuntimeShape,
    graph: PhantomData<G>,
}

impl<D: HybridDevice, G: LayeredGraph> Drop for PartitionedSession<'_, D, G> {
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
            _ => unreachable!("partition KV matches immutable owners"),
        }
    }
}
