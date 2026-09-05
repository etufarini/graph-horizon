/*
 * graph_horizon_engine — transactional CUDA weight conversion and upload.
 * Canonical groups are validated before a complete WeightSet is published.
 */

use std::borrow::Cow;

pub(in crate::backend::cuda) mod cache;

use color_eyre::eyre::{Result, bail, eyre};

use super::buffer::{CudaBuffer, CudaFormat};
use crate::backend::buffers::{LayerWeights, WeightSet};
use crate::backend::f16::f32_to_f16_bytes;
use crate::backend::source::{OutputWeight, WeightSelection, WeightSource};
use crate::gguf::loader::GgufFile;
use crate::gguf::tensor_index::{GgmlType, TensorInfo};

use crate::backend::cuda::Device;
use crate::backend::cuda::module::Module;

pub(crate) fn load_selected(
    device: &Device,
    file: &GgufFile,
    source: &dyn WeightSource,
    selection: &WeightSelection,
) -> Result<WeightSet<CudaBuffer>> {
    load_inner(device, file, source, selection, None, None)
}

pub(in crate::backend::cuda) fn load_inner(
    device: &Device,
    file: &GgufFile,
    source: &dyn WeightSource,
    selection: &WeightSelection,
    cache: Option<&Module>,
    fail_at: Option<usize>,
) -> Result<WeightSet<CudaBuffer>> {
    let groups = source.groups();
    if selection.layers.start > selection.layers.end
        || selection.layers.end > groups.layers.len()
        || groups.layers[selection.layers.clone()]
            .iter()
            .any(|group| group.len() != 9)
    {
        bail!("cuda: model allocation failed");
    }
    let mut index = 0;
    let tied = groups.tail.output.is_tied();
    let token_embd = (selection.embedding || (selection.tail && tied))
        .then(|| load_step(device, file, groups.embedding, None, &mut index, fail_at))
        .transpose()?;
    let output_norm = selection
        .tail
        .then(|| load_step(device, file, groups.tail.norm, None, &mut index, fail_at))
        .transpose()?;
    let output = match groups.tail.output {
        OutputWeight::Dedicated(tensor) if selection.tail => {
            Some(load_step(device, file, tensor, None, &mut index, fail_at)?)
        }
        _ => None,
    };
    let mut layers = Vec::with_capacity(selection.layers.len());
    for group in &groups.layers[selection.layers.clone()] {
        let mut loaded = Vec::with_capacity(9);
        for (slot, tensor) in group.iter().enumerate() {
            let cache = cache.filter(|_| !matches!(slot, 0 | 5) && tensor.dims.len() == 2);
            loaded.push(load_step(device, file, tensor, cache, &mut index, fail_at)?);
        }
        let mut loaded = loaded.into_iter();
        layers.push(LayerWeights {
            attn_norm: loaded.next().expect("validated CUDA layer"),
            attn_q: loaded.next().expect("validated CUDA layer"),
            attn_k: loaded.next().expect("validated CUDA layer"),
            attn_v: loaded.next().expect("validated CUDA layer"),
            attn_output: loaded.next().expect("validated CUDA layer"),
            ffn_norm: loaded.next().expect("validated CUDA layer"),
            ffn_gate: loaded.next().expect("validated CUDA layer"),
            ffn_up: loaded.next().expect("validated CUDA layer"),
            ffn_down: loaded.next().expect("validated CUDA layer"),
        });
    }
    if fail_at == Some(index) {
        bail!("cuda: weight upload failed");
    }
    Ok(WeightSet {
        token_embd,
        output_norm,
        output,
        layers,
    })
}

fn load_step(
    device: &Device,
    file: &GgufFile,
    tensor: &TensorInfo,
    cache: Option<&Module>,
    index: &mut usize,
    fail_at: Option<usize>,
) -> Result<CudaBuffer> {
    if fail_at == Some(*index) {
        bail!("cuda: weight upload failed");
    }
    *index += 1;
    load_tensor(device, file, tensor, cache)
}

fn load_tensor(
    device: &Device,
    file: &GgufFile,
    tensor: &TensorInfo,
    cache: Option<&Module>,
) -> Result<CudaBuffer> {
    let raw = file
        .tensor_bytes(tensor)
        .map_err(|_| eyre!("cuda: weight upload failed"))?;
    let (bytes, format): (Cow<'_, [u8]>, CudaFormat) = match tensor.ggml_type {
        GgmlType::F32 => (f32_to_f16_bytes(raw).into(), CudaFormat::F16),
        GgmlType::F16 => (raw.into(), CudaFormat::F16),
        GgmlType::Q4_K => (raw.into(), CudaFormat::Q4K),
        GgmlType::Q5_K => (raw.into(), CudaFormat::Q5K),
        GgmlType::Q6_K => (raw.into(), CudaFormat::Q6K),
        other => bail!("cuda: unsupported weight format '{}'", other.name()),
    };
    cache::upload(device, cache, &bytes, format)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::cuda::mem::buffer::{reset_test_counts, test_counts};
    use crate::backend::source::WeightGroups;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_FILE: AtomicU64 = AtomicU64::new(0);

    struct Source<'a>(&'a [TensorInfo]);

    impl WeightSource for Source<'_> {
        fn groups(&self) -> WeightGroups<'_> {
            let layers = self.0[2..]
                .chunks(9)
                .map(|chunk| chunk.iter().collect())
                .collect();
            WeightGroups::new(&self.0[0], &self.0[1], None, layers)
        }
    }

    struct SelectedSource<'a> {
        tensors: &'a [TensorInfo],
        dedicated: bool,
    }

    impl WeightSource for SelectedSource<'_> {
        fn groups(&self) -> WeightGroups<'_> {
            let output = self.dedicated.then_some(&self.tensors[2]);
            let first_layer = 2 + usize::from(self.dedicated);
            let layers = self.tensors[first_layer..]
                .chunks(9)
                .map(|chunk| chunk.iter().collect())
                .collect();
            WeightGroups::new(&self.tensors[0], &self.tensors[1], output, layers)
        }
    }

    fn type_id(format: GgmlType) -> u32 {
        match format {
            GgmlType::F32 => 0,
            GgmlType::F16 => 1,
            GgmlType::Q4_K => 12,
            GgmlType::Q5_K => 13,
            GgmlType::Q6_K => 14,
            _ => unreachable!(),
        }
    }

    fn tensor(format: GgmlType) -> (u64, Vec<u8>) {
        match format {
            GgmlType::F32 => (4, vec![0; 16]),
            GgmlType::F16 => (4, vec![0; 8]),
            GgmlType::Q4_K => (256, vec![0; 144]),
            GgmlType::Q5_K => (256, vec![0; 176]),
            GgmlType::Q6_K => (256, vec![0; 210]),
            _ => unreachable!(),
        }
    }

    fn gguf(formats: &[GgmlType]) -> Vec<u8> {
        let mut header = Vec::new();
        header.extend_from_slice(b"GGUF");
        header.extend_from_slice(&3u32.to_le_bytes());
        header.extend_from_slice(&(formats.len() as u64).to_le_bytes());
        header.extend_from_slice(&0u64.to_le_bytes());
        let mut offset = 0u64;
        let mut payloads = Vec::new();
        for (index, format) in formats.iter().copied().enumerate() {
            let name = format!("w{index}");
            header.extend_from_slice(&(name.len() as u64).to_le_bytes());
            header.extend_from_slice(name.as_bytes());
            let matrix = format == GgmlType::Q6_K;
            header.extend_from_slice(&(if matrix { 2u32 } else { 1u32 }).to_le_bytes());
            let (dimension, payload) = tensor(format);
            header.extend_from_slice(&dimension.to_le_bytes());
            if matrix {
                header.extend_from_slice(&1u64.to_le_bytes());
            }
            header.extend_from_slice(&type_id(format).to_le_bytes());
            header.extend_from_slice(&offset.to_le_bytes());
            offset += payload.len() as u64;
            payloads.push(payload);
        }
        while !header.len().is_multiple_of(32) {
            header.push(0);
        }
        for payload in payloads {
            header.extend(payload);
        }
        header
    }

    fn with_file<R>(formats: &[GgmlType], run: impl FnOnce(&GgufFile) -> R) -> R {
        let path = std::env::temp_dir().join(format!(
            "graph_horizon_cuda_weights_{}_{}.gguf",
            std::process::id(),
            NEXT_FILE.fetch_add(1, Ordering::Relaxed),
        ));
        std::fs::write(&path, gguf(formats)).expect("write CUDA weight fixture");
        let file = GgufFile::open(&path).expect("open CUDA weight fixture");
        let result = run(&file);
        std::fs::remove_file(path).expect("remove CUDA weight fixture");
        result
    }

    #[test]
    fn formats_are_retained_and_every_upload_failpoint_releases() -> Result<()> {
        let device = Device::acquire()?;
        let formats = [
            GgmlType::F32,
            GgmlType::F16,
            GgmlType::Q4_K,
            GgmlType::Q5_K,
            GgmlType::Q6_K,
        ];
        let expected = [
            CudaFormat::F16,
            CudaFormat::F16,
            CudaFormat::Q4K,
            CudaFormat::Q5K,
            CudaFormat::Q6K,
        ];
        with_file(&formats, |file| -> Result<()> {
            for (tensor, expected) in file.tensors().iter().zip(expected) {
                assert_eq!(load_tensor(&device, file, tensor, None)?.format(), expected);
            }
            Ok(())
        })?;

        with_file(&[GgmlType::F16; 11], |file| -> Result<()> {
            let source = Source(file.tensors());
            let selection = WeightSelection::full(1);
            for fail_at in 0..=11 {
                reset_test_counts();
                assert_eq!(
                    load_inner(&device, file, &source, &selection, None, Some(fail_at))
                        .err()
                        .expect("injected CUDA upload failure")
                        .to_string(),
                    "cuda: weight upload failed"
                );
                assert_eq!(test_counts().0, test_counts().1, "failpoint {fail_at}");
            }
            Ok(())
        })
    }

    #[test]
    fn selected_inventories_preserve_roles_and_local_layer_order() -> Result<()> {
        let device = Device::acquire()?;
        let mut formats = vec![GgmlType::F16; 11];
        formats.extend([GgmlType::Q4_K; 9]);
        with_file(&formats, |file| -> Result<()> {
            let source = SelectedSource {
                tensors: file.tensors(),
                dedicated: false,
            };
            let full = load_selected(&device, file, &source, &WeightSelection::full(2))?;
            assert!(full.token_embd.is_some());
            assert!(full.output_norm.is_some());
            assert!(full.output.is_none());
            assert_eq!(full.layers.len(), 2);

            let prefix = load_selected(
                &device,
                file,
                &source,
                &WeightSelection {
                    layers: 0..1,
                    embedding: true,
                    tail: false,
                },
            )?;
            assert!(prefix.token_embd.is_some());
            assert!(prefix.output_norm.is_none());
            assert_eq!(prefix.layers.len(), 1);
            assert_eq!(prefix.layers[0].attn_norm.format(), CudaFormat::F16);

            let suffix = load_selected(
                &device,
                file,
                &source,
                &WeightSelection {
                    layers: 1..2,
                    embedding: false,
                    tail: true,
                },
            )?;
            assert!(suffix.token_embd.is_some(), "tied tail owns its matrix");
            assert!(suffix.output_norm.is_some());
            assert!(suffix.output.is_none());
            assert_eq!(suffix.layers.len(), 1);
            assert_eq!(suffix.layers[0].attn_norm.format(), CudaFormat::Q4K);
            Ok(())
        })?;

        with_file(&[GgmlType::F16; 12], |file| -> Result<()> {
            let source = SelectedSource {
                tensors: file.tensors(),
                dedicated: true,
            };
            let suffix = load_selected(
                &device,
                file,
                &source,
                &WeightSelection {
                    layers: 0..1,
                    embedding: false,
                    tail: true,
                },
            )?;
            assert!(suffix.token_embd.is_none());
            assert!(suffix.output_norm.is_some());
            assert!(suffix.output.is_some());
            Ok(())
        })
    }

    #[test]
    fn invalid_selected_ranges_and_layer_groups_are_rejected() -> Result<()> {
        let device = Device::acquire()?;
        with_file(&[GgmlType::F16; 10], |file| {
            let source = SelectedSource {
                tensors: file.tensors(),
                dedicated: false,
            };
            for selection in [
                WeightSelection {
                    layers: std::ops::Range { start: 1, end: 0 },
                    embedding: false,
                    tail: false,
                },
                WeightSelection {
                    layers: 0..1,
                    embedding: false,
                    tail: false,
                },
            ] {
                assert_eq!(
                    load_selected(&device, file, &source, &selection)
                        .err()
                        .expect("invalid selection")
                        .to_string(),
                    "cuda: model allocation failed"
                );
            }
        });
        Ok(())
    }

    #[test]
    fn cached_upload_failpoints_release_raw_and_converted_allocations() -> Result<()> {
        let device = Device::acquire()?;
        let module = Module::load(&device.context)?;
        with_file(&[GgmlType::Q6_K; 11], |file| -> Result<()> {
            let source = Source(file.tensors());
            let selection = WeightSelection::full(1);
            for fail_at in 0..=11 {
                reset_test_counts();
                assert!(
                    load_inner(
                        &device,
                        file,
                        &source,
                        &selection,
                        Some(&module),
                        Some(fail_at)
                    )
                    .is_err()
                );
                assert_eq!(
                    test_counts().0,
                    test_counts().1,
                    "cached failpoint {fail_at}"
                );
            }
            reset_test_counts();
            let weights = load_inner(&device, file, &source, &selection, Some(&module), None)?;
            assert_eq!(
                weights.token_embd.as_ref().unwrap().format(),
                CudaFormat::Q6K
            );
            assert_eq!(weights.layers[0].attn_q.format(), CudaFormat::Q6K);
            assert_eq!(
                weights.layers[0].attn_q.prefill().format(),
                CudaFormat::Q6KCached
            );
            drop(weights);
            assert_eq!(test_counts().0, test_counts().1);
            Ok(())
        })
    }
}
