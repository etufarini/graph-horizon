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

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use color_eyre::eyre::{Result, bail};

    use super::*;
    use crate::backend::Backend;
    use crate::backend::buffers::{Buffers, Scratch, WeightSet};
    use crate::backend::cpu::{CpuBackend, CpuBuffer, CpuFormat};
    use crate::backend::hybrid::contract::HybridDevice;
    use crate::backend::hybrid::placement::{BudgetInput, MemoryTopology};
    use crate::backend::hybrid::weights::runtime::DeviceFixedBytes;
    use crate::backend::hybrid::{BackendBytes, HybridPlan, HybridRuntime};
    use crate::backend::source::{WeightSelection, WeightSource};
    use crate::gguf::loader::GgufFile;
    use crate::gguf::metadata::ModelMetadata;
    use crate::kv_cache::scheme::KvQuant;
    use crate::runtime::contract::{LayeredGraph, RuntimeSession};

    thread_local! {
        static RANGE_CALLS: Cell<usize> = const { Cell::new(0) };
        static FAIL_SUFFIX: Cell<bool> = const { Cell::new(false) };
        static KV_LAYERS: std::cell::RefCell<Vec<usize>> = const { std::cell::RefCell::new(Vec::new()) };
    }

    // Test-only adapter: host buffers exercise generic partition traversal.
    impl HybridDevice for CpuBackend {
        type Device = ();

        fn host_available() -> Result<u64> {
            Ok(u64::MAX)
        }

        fn acquire() -> Result<Option<Self::Device>> {
            Ok(Some(()))
        }

        fn budget(_: &Self::Device) -> Result<BudgetInput> {
            Ok(BudgetInput::Unified {
                physical_memory: u64::MAX / 9,
                recommended_working_set: u64::MAX,
                current_allocated: 0,
            })
        }

        fn topology() -> MemoryTopology {
            MemoryTopology::Unified
        }

        fn all_mode_name() -> &'static str {
            "all-test"
        }

        fn invalid_percentage_error() -> &'static str {
            "invalid test percentage"
        }

        fn fixed_bytes(_: &RuntimeShape) -> Result<DeviceFixedBytes> {
            Ok(DeviceFixedBytes {
                host: 0,
                device: 0,
                staging: 0,
            })
        }

        fn load_selected(
            _: Self::Device,
            _: &ModelMetadata,
            _: &dyn WeightSource,
            _: &GgufFile,
            _: &WeightSelection,
        ) -> Result<Self> {
            bail!("unused test loader")
        }

        fn buffer_bytes(buffer: &Self::Buffer) -> u64 {
            buffer.byte_len() as u64
        }

        fn upload_residual(&self, target: &Self::Buffer, bytes: &[u8]) -> Result<()> {
            target.with_bytes_mut(|target| target[..bytes.len()].copy_from_slice(bytes));
            Ok(())
        }
    }

    struct Graph;

    struct Batch<'a, B: Backend> {
        backend: &'a B,
        x: Option<B::Buffer>,
    }

    impl<B: Backend> Drop for Batch<'_, B> {
        fn drop(&mut self) {
            self.backend
                .free_buffer(self.x.take().expect("test batch owns x"));
        }
    }

    impl LayeredGraph for Graph {
        type Config = ();
        type Batch<'a, B: Backend>
            = Batch<'a, B>
        where
            B: 'a;

        const BATCH_ROWS: usize = 32;

        fn shape(_: &Self::Config) -> RuntimeShape {
            shape()
        }

        fn token<B: Backend>(
            _: &B,
            _: &Self::Config,
            _: &Kv<B::Buffer>,
            _: u32,
            _: usize,
        ) -> Result<()> {
            Ok(())
        }

        fn embedding<B: Backend>(
            backend: &B,
            encoder: &B::Encoder,
            _: &Self::Config,
            token: u32,
        ) -> Result<()> {
            backend.embed(
                encoder,
                &backend.buffers().scratch.x,
                backend
                    .buffers()
                    .weights
                    .token_embd
                    .as_ref()
                    .expect("test embedding owner"),
                token,
                2,
            )
        }

        fn range<B: Backend>(
            _: &B,
            _: &B::Encoder,
            _: &Self::Config,
            kv: &Kv<B::Buffer>,
            layers: std::ops::Range<usize>,
            _: usize,
        ) -> Result<()> {
            assert_eq!(layers, 0..1);
            KV_LAYERS.with(|seen| seen.borrow_mut().push(kv.block_count));
            let call = RANGE_CALLS.with(|calls| {
                let call = calls.get();
                calls.set(call + 1);
                call
            });
            if call % 2 == 1 && FAIL_SUFFIX.with(Cell::get) {
                bail!("synthetic suffix failure");
            }
            Ok(())
        }

        fn tail<B: Backend>(_: &B, _: &B::Encoder, _: &Self::Config) {}

        fn prefill<B: Backend>(
            _: &B,
            _: &Self::Config,
            _: &Kv<B::Buffer>,
            _: &[u32],
            _: &mut dyn FnMut() -> Result<()>,
        ) -> Result<()> {
            Ok(())
        }

        fn batch<'a, B: Backend>(
            backend: &'a B,
            _: &Self::Config,
            capacity: usize,
        ) -> Result<Batch<'a, B>> {
            Ok(Batch {
                backend,
                x: Some(backend.alloc_buffer((capacity * 8) as u64)?),
            })
        }

        fn batch_residual<'a, B: Backend>(batch: &'a Batch<'_, B>) -> &'a B::Buffer {
            batch.x.as_ref().expect("test batch owns x")
        }

        fn record_batch<B: Backend>(
            backend: &B,
            _: &Self::Config,
            kv: &Kv<B::Buffer>,
            batch: &Batch<'_, B>,
            tokens: &[u32],
            _: usize,
            embedding: bool,
            _: bool,
        ) -> Result<()> {
            KV_LAYERS.with(|seen| seen.borrow_mut().push(kv.block_count));
            if embedding {
                let encoder = backend.begin()?;
                for (row, token) in tokens.iter().copied().enumerate() {
                    let target = backend.view(
                        batch.x.as_ref().expect("test batch owns x"),
                        (row * 8) as u64,
                        8,
                    );
                    backend.embed(
                        &encoder,
                        &target,
                        backend
                            .buffers()
                            .weights
                            .token_embd
                            .as_ref()
                            .expect("test embedding owner"),
                        token,
                        2,
                    )?;
                }
                backend.submit(encoder)?;
            }
            Ok(())
        }
    }

    fn shape() -> RuntimeShape {
        RuntimeShape {
            block_count: 2,
            embedding: 2,
            q: 1,
            k: 1,
            v: 1,
            attention: 1,
            feed_forward: 1,
            vocab: 2,
            kv_heads: 1,
            key_length: 1,
            value_length: 1,
            prefill_rows: 32,
        }
    }

    fn backend(x_bytes: usize, embedding: bool) -> CpuBackend {
        let buffer = |bytes, format| CpuBuffer::zeroed(bytes, format);
        let token_embd = embedding.then(|| {
            let values = buffer(8, CpuFormat::F16);
            values.write_f16_from_f32(&[1.0, 2.0, 3.0, 4.0]);
            values
        });
        CpuBackend::from_buffers(Buffers {
            weights: WeightSet {
                token_embd,
                output_norm: None,
                output: None,
                layers: Vec::new(),
            },
            scratch: Scratch {
                x: buffer(x_bytes, CpuFormat::F32),
                normed: buffer(2, CpuFormat::F16),
                q: buffer(2, CpuFormat::F16),
                k: buffer(2, CpuFormat::F16),
                v: buffer(2, CpuFormat::F16),
                attn: buffer(2, CpuFormat::F16),
                proj: buffer(2, CpuFormat::F16),
                gate: buffer(2, CpuFormat::F16),
                up: buffer(2, CpuFormat::F16),
                act: buffer(2, CpuFormat::F16),
                ffn_out: buffer(2, CpuFormat::F16),
            },
            logits: buffer(8, CpuFormat::F32),
        })
    }

    fn mixed(gpu_x_bytes: usize) -> HybridRuntime<CpuBackend> {
        HybridRuntime {
            plan: HybridPlan::new(1, 2, BackendBytes::default(), BackendBytes::default()).unwrap(),
            backends: HybridBackends::Mixed {
                cpu: backend(8, true),
                gpu: backend(gpu_x_bytes, false),
            },
        }
    }

    fn reset() {
        crate::backend::hybrid::crossing::reset_count();
        RANGE_CALLS.with(|calls| calls.set(0));
        FAIL_SUFFIX.with(|fail| fail.set(false));
        KV_LAYERS.with(|seen| seen.borrow_mut().clear());
    }

    #[test]
    fn mixed_decode_and_prefill_cross_once_with_local_kv() -> Result<()> {
        reset();
        let runtime = mixed(8);
        let session =
            PartitionedSession::<CpuBackend, Graph>::new(&runtime, &(), shape(), 8, KvQuant::F16)?;
        session.token(0, 0)?;
        assert_eq!(crate::backend::hybrid::crossing::count(), 1);
        let HybridBackends::Mixed { gpu, .. } = &runtime.backends else {
            unreachable!()
        };
        assert_eq!(gpu.buffers().scratch.x.read_f32(), [1.0, 2.0]);
        assert!(KV_LAYERS.with(|seen| seen.borrow().iter().all(|layers| *layers == 1)));

        crate::backend::hybrid::crossing::reset_count();
        session.prefill(&[0, 1, 0], &mut || Ok(()))?;
        assert_eq!(crate::backend::hybrid::crossing::count(), 1);
        Ok(())
    }

    #[test]
    fn mixed_prefill_crosses_once_per_dynamic_chunk() -> Result<()> {
        for (rows, expected) in [(16, 1), (33, 2), (2048, 64)] {
            reset();
            let runtime = mixed(8);
            let session = PartitionedSession::<CpuBackend, Graph>::new(
                &runtime,
                &(),
                shape(),
                4096,
                KvQuant::F16,
            )?;
            session.prefill(&vec![0; rows], &mut || Ok(()))?;
            assert_eq!(crate::backend::hybrid::crossing::count(), expected);
        }
        Ok(())
    }

    #[test]
    fn homogeneous_modes_never_cross() -> Result<()> {
        reset();
        for runtime in [
            HybridRuntime {
                plan: HybridPlan::new(0, 2, BackendBytes::default(), BackendBytes::default())?,
                backends: HybridBackends::AllGpu(backend(8, true)),
            },
            HybridRuntime {
                plan: HybridPlan::new(2, 2, BackendBytes::default(), BackendBytes::default())?,
                backends: HybridBackends::CpuOnly(backend(8, true)),
            },
        ] {
            let session = PartitionedSession::<CpuBackend, Graph>::new(
                &runtime,
                &(),
                shape(),
                4096,
                KvQuant::F16,
            )?;
            session.token(0, 0)?;
            for rows in [16, 33, 2048] {
                session.prefill(&vec![0; rows], &mut || Ok(()))?;
            }
        }
        assert_eq!(crate::backend::hybrid::crossing::count(), 0);
        Ok(())
    }

    #[test]
    fn crossing_and_suffix_failures_preserve_the_plan() -> Result<()> {
        reset();
        let checks = mixed(8);
        let HybridBackends::Mixed { gpu, .. } = &checks.backends else {
            unreachable!()
        };
        let source = CpuBuffer::zeroed(4, CpuFormat::F32);
        assert_eq!(
            crate::backend::hybrid::crossing::copy(&source, gpu, &gpu.buffers().scratch.x, 2,)
                .unwrap_err()
                .to_string(),
            "hybrid residual crossing source is too small"
        );
        assert_eq!(
            crate::backend::hybrid::crossing::copy(
                &source,
                gpu,
                &gpu.buffers().scratch.x,
                usize::MAX,
            )
            .unwrap_err()
            .to_string(),
            "hybrid residual crossing overflow"
        );
        assert_eq!(crate::backend::hybrid::crossing::count(), 0);

        let short = mixed(4);
        let session =
            PartitionedSession::<CpuBackend, Graph>::new(&short, &(), shape(), 8, KvQuant::F16)?;
        assert_eq!(
            session.token(0, 0).unwrap_err().to_string(),
            "hybrid residual crossing destination is too small"
        );
        assert_eq!(short.plan.split, 1);

        reset();
        FAIL_SUFFIX.with(|fail| fail.set(true));
        let suffix = mixed(8);
        let session =
            PartitionedSession::<CpuBackend, Graph>::new(&suffix, &(), shape(), 8, KvQuant::F16)?;
        assert_eq!(
            session.token(0, 0).unwrap_err().to_string(),
            "synthetic suffix failure"
        );
        assert_eq!(suffix.plan.split, 1);
        Ok(())
    }
}
