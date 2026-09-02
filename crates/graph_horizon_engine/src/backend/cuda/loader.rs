/*
 * graph_horizon_engine — standalone CUDA load transaction.
 * Configuration and full-context capacity are fixed before module or model
 * allocation; only a fully constructed backend crosses this boundary.
 */

use color_eyre::eyre::{Result, bail};

#[cfg(any(feature = "cuda", test))]
use super::mem::budget;
use super::mem::{buffers, weights};
use super::{CudaBackend, Device};
#[cfg(any(feature = "cuda", test))]
use crate::backend::hybrid::weights::runtime::RuntimeShape;
use crate::backend::source::{WeightSelection, WeightSource};
use crate::gguf::loader::GgufFile;
use crate::gguf::metadata::ModelMetadata;
#[cfg(any(feature = "cuda", test))]
use crate::kv_cache::scheme::KvQuant;

#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn load(
    file: &GgufFile,
    source: &dyn WeightSource,
    metadata: &ModelMetadata,
    shape: RuntimeShape,
    context: usize,
    scheme: KvQuant,
    weights_percent: Option<u8>,
    reserve_mib: Option<u64>,
) -> Result<CudaBackend> {
    load_inner(
        file,
        source,
        metadata,
        shape,
        context,
        scheme,
        weights_percent,
        reserve_mib,
        Failpoints::default(),
    )
}

#[derive(Default)]
struct Failpoints {
    function: Option<usize>,
    weight: Option<usize>,
    buffer: Option<usize>,
}

#[allow(clippy::too_many_arguments)]
#[cfg(any(feature = "cuda", test))]
fn load_inner(
    file: &GgufFile,
    source: &dyn WeightSource,
    metadata: &ModelMetadata,
    shape: RuntimeShape,
    context: usize,
    scheme: KvQuant,
    weights_percent: Option<u8>,
    reserve_mib: Option<u64>,
    failpoints: Failpoints,
) -> Result<CudaBackend> {
    if source.groups().layers.len() != metadata.block_count {
        bail!("cuda: model allocation failed");
    }
    let percent = budget::validate_percentage(weights_percent)?;
    let plan = budget::MemoryPlan::new(source, shape, context, scheme)?;
    let device = Device::acquire()?;
    let reserve = budget::reserve_bytes(device.total_bytes, reserve_mib)?;
    budget::preflight(device.free_bytes, reserve, percent, &plan)?;
    load_selected_inner(
        device,
        metadata,
        source,
        file,
        &WeightSelection::full(metadata.block_count),
        failpoints,
    )
}

#[cfg(feature = "cuda-hybrid")]
pub(crate) fn load_selected(
    device: Device,
    metadata: &ModelMetadata,
    source: &dyn WeightSource,
    file: &GgufFile,
    selection: &WeightSelection,
) -> Result<CudaBackend> {
    load_selected_inner(
        device,
        metadata,
        source,
        file,
        selection,
        Failpoints::default(),
    )
}

fn load_selected_inner(
    device: Device,
    metadata: &ModelMetadata,
    source: &dyn WeightSource,
    file: &GgufFile,
    selection: &WeightSelection,
    failpoints: Failpoints,
) -> Result<CudaBackend> {
    if source.groups().layers.len() != metadata.block_count {
        bail!("cuda: model allocation failed");
    }
    let module = if failpoints.function.is_some() {
        super::module::Module::load_inner(&device.context, failpoints.function)?
    } else {
        super::module::Module::load(&device.context)?
    };
    let weights = if failpoints.weight.is_some() {
        weights::load_inner(&device, file, source, selection, failpoints.weight)?
    } else {
        weights::load_selected(&device, file, source, selection)?
    };
    let (buffers, reduce) = if failpoints.buffer.is_some() {
        buffers::allocate_inner(&device, metadata, weights, failpoints.buffer)?
    } else {
        buffers::allocate(&device, metadata, weights)?
    };
    Ok(CudaBackend {
        device,
        module,
        buffers,
        reduce,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::cuda::mem::buffer::{reset_test_counts, test_counts};
    use crate::backend::source::WeightGroups;
    use crate::gguf::tensor_index::TensorInfo;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_FILE: AtomicU64 = AtomicU64::new(0);

    struct Source<'a>(&'a [TensorInfo]);

    impl WeightSource for Source<'_> {
        fn groups(&self) -> WeightGroups<'_> {
            WeightGroups::new(&self.0[0], &self.0[1], None, Vec::new())
        }
    }

    fn gguf() -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        bytes.extend_from_slice(&3u32.to_le_bytes());
        bytes.extend_from_slice(&2u64.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        for (name, offset) in [("embedding", 0u64), ("norm", 8u64)] {
            bytes.extend_from_slice(&(name.len() as u64).to_le_bytes());
            bytes.extend_from_slice(name.as_bytes());
            bytes.extend_from_slice(&1u32.to_le_bytes());
            bytes.extend_from_slice(&4u64.to_le_bytes());
            bytes.extend_from_slice(&1u32.to_le_bytes());
            bytes.extend_from_slice(&offset.to_le_bytes());
        }
        while !bytes.len().is_multiple_of(32) {
            bytes.push(0);
        }
        bytes.extend_from_slice(&[0; 16]);
        bytes
    }

    fn with_file(run: impl FnOnce(&GgufFile) -> Result<()>) -> Result<()> {
        let path = std::env::temp_dir().join(format!(
            "graph_horizon_cuda_loader_{}_{}.gguf",
            std::process::id(),
            NEXT_FILE.fetch_add(1, Ordering::Relaxed),
        ));
        std::fs::write(&path, gguf())?;
        let file = GgufFile::open(&path)?;
        let result = run(&file);
        std::fs::remove_file(path)?;
        result
    }

    fn metadata() -> ModelMetadata {
        ModelMetadata {
            block_count: 0,
            embedding_length: 4,
            head_count: 1,
            head_count_kv: 1,
            head_dim: 4,
            feed_forward_length: 8,
            vocab_size: 8,
        }
    }

    fn shape() -> RuntimeShape {
        RuntimeShape {
            block_count: 0,
            embedding: 4,
            q: 4,
            k: 4,
            v: 4,
            attention: 4,
            feed_forward: 8,
            vocab: 8,
            kv_heads: 1,
            key_length: 4,
            value_length: 4,
            cpu_prefill_rows: 4,
            gpu_prefill_rows: 32,
            mixed_prefill_rows: 4,
        }
    }

    fn injected(file: &GgufFile, failpoints: Failpoints) -> Result<CudaBackend> {
        let source = Source(file.tensors());
        load_inner(
            file,
            &source,
            &metadata(),
            shape(),
            4,
            KvQuant::F16,
            None,
            Some(0),
            failpoints,
        )
    }

    fn assert_send_sync<T: Send + Sync>() {}

    #[test]
    fn published_backend_crosses_the_engine_worker_boundary() {
        assert_send_sync::<CudaBackend>();
    }

    #[test]
    fn invalid_percentage_precedes_device_acquisition() {
        assert_eq!(
            budget::validate_percentage(Some(0))
                .unwrap_err()
                .to_string(),
            "invalid CUDA weight percentage"
        );
    }

    #[test]
    fn every_transaction_failpoint_releases_all_completed_allocations() -> Result<()> {
        with_file(|file| {
            for (failpoints, expected) in (0..16)
                .map(|function| {
                    (
                        Failpoints {
                            function: Some(function),
                            ..Failpoints::default()
                        },
                        "cuda: kernel module load failed",
                    )
                })
                .chain((0..=2).map(|weight| {
                    (
                        Failpoints {
                            weight: Some(weight),
                            ..Failpoints::default()
                        },
                        "cuda: weight upload failed",
                    )
                }))
                .chain((0..=13).map(|buffer| {
                    (
                        Failpoints {
                            buffer: Some(buffer),
                            ..Failpoints::default()
                        },
                        "cuda: model allocation failed",
                    )
                }))
            {
                reset_test_counts();
                assert_eq!(
                    injected(file, failpoints)
                        .err()
                        .expect("injected CUDA loader failure")
                        .to_string(),
                    expected,
                    "unexpected failpoint error"
                );
                assert_eq!(test_counts().0, test_counts().1);
            }

            reset_test_counts();
            let backend = injected(file, Failpoints::default())?;
            assert!(test_counts().0 > 0);
            drop(backend);
            assert_eq!(test_counts().0, test_counts().1);
            Ok(())
        })
    }
}
