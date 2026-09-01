/*
 * graph_horizon_engine — transactional CUDA weight conversion and upload.
 * Canonical groups are validated before a complete WeightSet is published.
 */

use std::borrow::Cow;

use color_eyre::eyre::{Result, bail, eyre};

use super::buffer::{CudaBuffer, CudaFormat};
use crate::backend::buffers::{LayerWeights, WeightSet};
use crate::backend::f16::f32_to_f16_bytes;
use crate::backend::source::{OutputWeight, WeightSelection, WeightSource};
use crate::gguf::loader::GgufFile;
use crate::gguf::tensor_index::{GgmlType, TensorInfo};

use super::super::Device;

pub(crate) fn load(
    device: &Device,
    file: &GgufFile,
    source: &dyn WeightSource,
) -> Result<WeightSet<CudaBuffer>> {
    load_inner(device, file, source, None)
}

fn load_inner(
    device: &Device,
    file: &GgufFile,
    source: &dyn WeightSource,
    fail_at: Option<usize>,
) -> Result<WeightSet<CudaBuffer>> {
    let groups = source.groups();
    let selection = WeightSelection::full(groups.layers.len());
    if selection.layers.start > selection.layers.end
        || selection.layers.end > groups.layers.len()
        || groups.layers[selection.layers.clone()]
            .iter()
            .any(|group| group.len() != 9)
    {
        bail!("cuda: model allocation failed");
    }
    let mut index = 0;
    let token_embd = Some(load_step(
        device,
        file,
        groups.embedding,
        &mut index,
        fail_at,
    )?);
    let output_norm = Some(load_step(
        device,
        file,
        groups.tail.norm,
        &mut index,
        fail_at,
    )?);
    let output = match groups.tail.output {
        OutputWeight::Tied => None,
        OutputWeight::Dedicated(tensor) => {
            Some(load_step(device, file, tensor, &mut index, fail_at)?)
        }
    };
    let mut layers = Vec::with_capacity(groups.layers.len());
    debug_assert_eq!(groups.tail.output.is_tied(), output.is_none());
    for group in &groups.layers[selection.layers] {
        let mut loaded = Vec::with_capacity(9);
        for tensor in group {
            loaded.push(load_step(device, file, tensor, &mut index, fail_at)?);
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
    index: &mut usize,
    fail_at: Option<usize>,
) -> Result<CudaBuffer> {
    if fail_at == Some(*index) {
        bail!("cuda: weight upload failed");
    }
    *index += 1;
    load_tensor(device, file, tensor)
}

fn load_tensor(device: &Device, file: &GgufFile, tensor: &TensorInfo) -> Result<CudaBuffer> {
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
    CudaBuffer::upload(device, &bytes, format)
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
            header.extend_from_slice(&1u32.to_le_bytes());
            let (dimension, payload) = tensor(format);
            header.extend_from_slice(&dimension.to_le_bytes());
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
                assert_eq!(load_tensor(&device, file, tensor)?.format(), expected);
            }
            Ok(())
        })?;

        with_file(&[GgmlType::F16; 11], |file| -> Result<()> {
            let source = Source(file.tensors());
            for fail_at in 0..=11 {
                reset_test_counts();
                assert_eq!(
                    load_inner(&device, file, &source, Some(fail_at))
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
}
