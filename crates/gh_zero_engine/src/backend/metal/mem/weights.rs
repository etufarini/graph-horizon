/*
 * gh_zero_engine — transactional Metal weight loading
 * Walks neutral weight groups, converts only F32 norms to FP16, preserves the
 * four retained formats, and publishes a complete owned WeightSet after upload.
 */

use std::borrow::Cow;

use color_eyre::eyre::{Result, bail};

use super::buffer::{MetalBuffer, MetalFormat};
use crate::backend::buffers::{LayerWeights, WeightSet};
use crate::backend::f16::f32_to_f16_bytes;
use crate::backend::metal::Device;
use crate::backend::source::{OutputWeight, WeightSelection, WeightSource};
use crate::gguf::loader::GgufFile;
use crate::gguf::tensor_index::{GgmlType, TensorInfo};

pub(crate) fn load(
    device: &Device,
    file: &GgufFile,
    source: &dyn WeightSource,
) -> Result<WeightSet<MetalBuffer>> {
    let layers = source.groups().layers.len();
    load_selected(device, file, source, &WeightSelection::full(layers))
}

pub(crate) fn load_selected(
    device: &Device,
    file: &GgufFile,
    source: &dyn WeightSource,
    selection: &WeightSelection,
) -> Result<WeightSet<MetalBuffer>> {
    load_inner(device, file, source, selection, None)
}

fn load_inner(
    device: &Device,
    file: &GgufFile,
    source: &dyn WeightSource,
    selection: &WeightSelection,
    fail_at: Option<usize>,
) -> Result<WeightSet<MetalBuffer>> {
    let groups = source.groups();
    if selection.layers.start > selection.layers.end
        || selection.layers.end > groups.layers.len()
        || groups.layers[selection.layers.clone()]
            .iter()
            .any(|group| group.len() != 9)
    {
        bail!("metal: model allocation failed");
    }
    let mut index = 0;
    let tied = groups.tail.output.is_tied();
    let token_embd = (selection.embedding || (selection.tail && tied))
        .then(|| load_step(device, file, groups.embedding, &mut index, fail_at))
        .transpose()?;
    let output_norm = selection
        .tail
        .then(|| load_step(device, file, groups.tail.norm, &mut index, fail_at))
        .transpose()?;
    let output = match groups.tail.output {
        OutputWeight::Dedicated(tensor) if selection.tail => {
            Some(load_step(device, file, tensor, &mut index, fail_at)?)
        }
        _ => None,
    };
    let mut layers = Vec::with_capacity(selection.layers.len());
    for group in &groups.layers[selection.layers.clone()] {
        let mut loaded = Vec::with_capacity(9);
        for tensor in group {
            loaded.push(load_step(device, file, tensor, &mut index, fail_at)?);
        }
        let mut loaded = loaded.into_iter();
        layers.push(LayerWeights {
            attn_norm: loaded.next().expect("validated layer"),
            attn_q: loaded.next().expect("validated layer"),
            attn_k: loaded.next().expect("validated layer"),
            attn_v: loaded.next().expect("validated layer"),
            attn_output: loaded.next().expect("validated layer"),
            ffn_norm: loaded.next().expect("validated layer"),
            ffn_gate: loaded.next().expect("validated layer"),
            ffn_up: loaded.next().expect("validated layer"),
            ffn_down: loaded.next().expect("validated layer"),
        });
    }
    if fail_at == Some(index) {
        bail!("metal: weight upload failed");
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
    index: &mut usize,
    fail_at: Option<usize>,
) -> Result<MetalBuffer> {
    if fail_at == Some(*index) {
        bail!("metal: weight upload failed");
    }
    *index += 1;
    load_tensor(device, file, tensor)
}

fn load_tensor(device: &Device, file: &GgufFile, tensor: &TensorInfo) -> Result<MetalBuffer> {
    let raw = file
        .tensor_bytes(tensor)
        .map_err(|_| color_eyre::eyre::eyre!("metal: weight upload failed"))?;
    let (bytes, format): (Cow<'_, [u8]>, MetalFormat) = match tensor.ggml_type {
        GgmlType::F32 => (f32_to_f16_bytes(raw).into(), MetalFormat::F16),
        GgmlType::F16 => (raw.into(), MetalFormat::F16),
        GgmlType::Q4_K => (raw.into(), MetalFormat::Q4K),
        GgmlType::Q5_K => (raw.into(), MetalFormat::Q5K),
        GgmlType::Q6_K => (raw.into(), MetalFormat::Q6K),
        other => bail!("metal: unsupported weight format '{}'", other.name()),
    };
    let buffer = MetalBuffer::allocate(device, bytes.len() as u64, format)?;
    buffer.write(&bytes)?;
    Ok(buffer)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::metal::mem::buffer::{reset_test_counts, test_counts};
    use crate::backend::source::WeightGroups;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_FILE: AtomicU64 = AtomicU64::new(0);

    struct Source<'a> {
        tensors: &'a [TensorInfo],
        dedicated: bool,
    }

    impl WeightSource for Source<'_> {
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
            GgmlType::Q4_0 => 2,
            GgmlType::Q4_K => 12,
            GgmlType::Q5_K => 13,
            GgmlType::Q6_K => 14,
            _ => unreachable!(),
        }
    }

    fn tensor(format: GgmlType) -> (Vec<u64>, Vec<u8>) {
        match format {
            GgmlType::F32 => (vec![4], vec![0; 16]),
            GgmlType::F16 => (vec![4], vec![0; 8]),
            GgmlType::Q4_0 => (vec![32], vec![0; 18]),
            GgmlType::Q4_K => (vec![256], vec![0; 144]),
            GgmlType::Q5_K => (vec![256], vec![0; 176]),
            GgmlType::Q6_K => (vec![256], vec![0; 210]),
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
            let (dims, bytes) = tensor(format);
            header.extend_from_slice(&(dims.len() as u32).to_le_bytes());
            for dim in dims {
                header.extend_from_slice(&dim.to_le_bytes());
            }
            header.extend_from_slice(&type_id(format).to_le_bytes());
            header.extend_from_slice(&offset.to_le_bytes());
            offset += bytes.len() as u64;
            payloads.push(bytes);
        }
        while header.len() % 32 != 0 {
            header.push(0);
        }
        for bytes in payloads {
            header.extend(bytes);
        }
        header
    }

    fn with_file<R>(formats: &[GgmlType], run: impl FnOnce(&GgufFile) -> R) -> R {
        let path = std::env::temp_dir().join(format!(
            "gh_zero_metal_weights_{}_{}_{}.gguf",
            std::process::id(),
            NEXT_FILE.fetch_add(1, Ordering::Relaxed),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::write(&path, gguf(formats)).unwrap();
        let file = GgufFile::open(&path).unwrap();
        let result = run(&file);
        std::fs::remove_file(path).unwrap();
        result
    }

    #[test]
    fn every_supported_format_is_retained_or_converted() -> Result<()> {
        let formats = [
            GgmlType::F32,
            GgmlType::F16,
            GgmlType::Q4_K,
            GgmlType::Q5_K,
            GgmlType::Q6_K,
        ];
        let expected = [
            MetalFormat::F16,
            MetalFormat::F16,
            MetalFormat::Q4K,
            MetalFormat::Q5K,
            MetalFormat::Q6K,
        ];
        let device = Device::acquire()?;
        with_file(&formats, |file| -> Result<()> {
            for (info, format) in file.tensors().iter().zip(expected) {
                assert_eq!(load_tensor(&device, file, info)?.format(), format);
            }
            Ok(())
        })
    }

    #[test]
    fn unsupported_and_foreign_spans_are_normalized() -> Result<()> {
        let device = Device::acquire()?;
        with_file(&[GgmlType::Q4_0], |file| {
            assert_eq!(
                load_tensor(&device, file, &file.tensors()[0])
                    .err()
                    .expect("unsupported format")
                    .to_string(),
                "metal: unsupported weight format 'Q4_0'"
            );
        });
        with_file(&[GgmlType::F16], |file| {
            let foreign = TensorInfo {
                name: "foreign".into(),
                dims: vec![4],
                ggml_type: GgmlType::F16,
                offset: u64::MAX,
            };
            assert_eq!(
                load_tensor(&device, file, &foreign)
                    .err()
                    .expect("foreign span")
                    .to_string(),
                "metal: weight upload failed"
            );
        });
        Ok(())
    }

    #[test]
    fn tied_output_is_one_allocation_and_failpoints_roll_back() -> Result<()> {
        let formats = [GgmlType::F16; 11];
        let device = Device::acquire()?;
        with_file(&formats, |file| -> Result<()> {
            let source = Source {
                tensors: file.tensors(),
                dedicated: false,
            };
            let weights = load(&device, file, &source)?;
            assert!(weights.output.is_none());
            drop(weights);
            let full = WeightSelection::full(1);
            for fail_at in 0..=formats.len() {
                reset_test_counts();
                assert_eq!(
                    load_inner(&device, file, &source, &full, Some(fail_at))
                        .err()
                        .expect("injected upload failure")
                        .to_string(),
                    "metal: weight upload failed"
                );
                let (allocations, drops) = test_counts();
                assert_eq!(allocations, drops, "leak at failpoint {fail_at}");
            }
            Ok(())
        })
    }

    #[test]
    fn selected_inventories_match_prefix_and_tail_roles() -> Result<()> {
        let device = Device::acquire()?;
        with_file(&[GgmlType::F16; 11], |file| -> Result<()> {
            let source = Source {
                tensors: file.tensors(),
                dedicated: false,
            };
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
            assert!(prefix.output.is_none());
            assert_eq!(prefix.layers.len(), 1);

            let tail = load_selected(
                &device,
                file,
                &source,
                &WeightSelection {
                    layers: 0..1,
                    embedding: false,
                    tail: true,
                },
            )?;
            assert!(tail.token_embd.is_some(), "tied tail owns its own matrix");
            assert!(tail.output_norm.is_some());
            assert!(tail.output.is_none());
            assert_eq!(tail.layers.len(), 1);
            Ok(())
        })?;
        with_file(&[GgmlType::F16; 12], |file| -> Result<()> {
            let source = Source {
                tensors: file.tensors(),
                dedicated: true,
            };
            let tail = load_selected(
                &device,
                file,
                &source,
                &WeightSelection {
                    layers: 0..1,
                    embedding: false,
                    tail: true,
                },
            )?;
            assert!(tail.token_embd.is_none());
            assert!(tail.output_norm.is_some());
            assert!(tail.output.is_some());
            Ok(())
        })
    }
}
