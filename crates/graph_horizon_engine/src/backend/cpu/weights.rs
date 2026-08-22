/*
 * graph_horizon_engine — CPU weight load
 * Loads neutral weight groups into CPU storage and allocates graph scratch. It walks
 * embedding/tail/layer groups and copies only selected bytes in their retained
 * format. Family validation admits Q4_K/Q6_K matrices; this generic loader also
 * retains Q5_K/F16 and rejects malformed layer groups before allocation.
 * Quantized weights are validated at
 * load (dequant::validate); F32 norms are converted to FP16 on upload, exactly
 * like the GPU loaders. Scratch and logits follow the shared graph buffer layout.
 * This file knows nothing about a model family.
*/

use color_eyre::eyre::{Result, bail};

use super::buffer::{CpuBuffer, CpuFormat, f32_to_f16_bytes};
use super::dequant;
use crate::backend::buffers::{Buffers, LayerWeights, Scratch, WeightSet};
use crate::backend::source::{OutputWeight, WeightSelection, WeightSource};
use crate::gguf::loader::GgufFile;
use crate::gguf::metadata::ModelMetadata;
use crate::gguf::tensor_index::GgmlType;

// Builds the whole CPU buffer set (weights + scratch + logits). Nothing is kept
// private on the CPU side (no readback mirror as on Vulkan), so only `Buffers`
// is returned.
#[cfg(feature = "cpu")]
pub(super) fn load(
    meta: &ModelMetadata,
    ws: &dyn WeightSource,
    gguf: &GgufFile,
    _context: usize,
) -> Result<Buffers<CpuBuffer>> {
    let layer_count = ws.groups().layers.len();
    load_selected(meta, ws, gguf, &WeightSelection::full(layer_count))
}

pub(super) fn load_selected(
    meta: &ModelMetadata,
    ws: &dyn WeightSource,
    gguf: &GgufFile,
    selection: &WeightSelection,
) -> Result<Buffers<CpuBuffer>> {
    let weights = load_weights(gguf, ws, selection)?;
    let scratch = alloc_scratch(meta);
    let logits = CpuBuffer::zeroed(
        if selection.tail {
            meta.vocab_size * 4
        } else {
            0
        },
        CpuFormat::F32,
    );
    Ok(Buffers {
        weights,
        scratch,
        logits,
    })
}

// Walks tensors in the canonical order declared by the neutral source. The source
// declares optional slots explicitly, so no model type or count heuristic leaks
// into this loader.
fn load_weights(
    gguf: &GgufFile,
    ws: &dyn WeightSource,
    selection: &WeightSelection,
) -> Result<WeightSet<CpuBuffer>> {
    load_weights_inner(gguf, ws, selection, None)
}

fn load_weights_inner(
    gguf: &GgufFile,
    ws: &dyn WeightSource,
    selection: &WeightSelection,
    fail_at: Option<usize>,
) -> Result<WeightSet<CpuBuffer>> {
    let groups = ws.groups();
    if selection.layers.start > selection.layers.end || selection.layers.end > groups.layers.len() {
        bail!("cpu: invalid weight layout");
    }
    let mut index = 0;
    let tied = groups.tail.output.is_tied();
    let token_embd = (selection.embedding || (selection.tail && tied))
        .then(|| load_step(gguf, groups.embedding, &mut index, fail_at))
        .transpose()?;
    let output_norm = selection
        .tail
        .then(|| load_step(gguf, groups.tail.norm, &mut index, fail_at))
        .transpose()?;
    let output = match groups.tail.output {
        OutputWeight::Dedicated(tensor) if selection.tail => {
            Some(load_step(gguf, tensor, &mut index, fail_at)?)
        }
        _ => None,
    };
    let mut layers = Vec::with_capacity(selection.layers.len());
    for group in &groups.layers[selection.layers.clone()] {
        let [
            attn_norm,
            attn_q,
            attn_k,
            attn_v,
            attn_output,
            ffn_norm,
            ffn_gate,
            ffn_up,
            ffn_down,
        ] = group.as_slice()
        else {
            bail!("cpu: invalid weight layout");
        };
        layers.push(LayerWeights {
            attn_norm: load_step(gguf, attn_norm, &mut index, fail_at)?,
            attn_q: load_step(gguf, attn_q, &mut index, fail_at)?,
            attn_k: load_step(gguf, attn_k, &mut index, fail_at)?,
            attn_v: load_step(gguf, attn_v, &mut index, fail_at)?,
            attn_output: load_step(gguf, attn_output, &mut index, fail_at)?,
            ffn_norm: load_step(gguf, ffn_norm, &mut index, fail_at)?,
            ffn_gate: load_step(gguf, ffn_gate, &mut index, fail_at)?,
            ffn_up: load_step(gguf, ffn_up, &mut index, fail_at)?,
            ffn_down: load_step(gguf, ffn_down, &mut index, fail_at)?,
        });
    }
    if fail_at == Some(index) {
        bail!("cpu: weight load failed");
    }
    Ok(WeightSet {
        token_embd,
        output_norm,
        output,
        layers,
    })
}

fn load_step(
    gguf: &GgufFile,
    tensor: &crate::gguf::tensor_index::TensorInfo,
    index: &mut usize,
    fail_at: Option<usize>,
) -> Result<CpuBuffer> {
    if fail_at == Some(*index) {
        bail!("cpu: weight load failed");
    }
    *index += 1;
    load_tensor(gguf, tensor)
}

// Copies one tensor into a CpuBuffer in its original format. F32 norms become
// FP16 (parity with Vulkan); F16 and the quantized matmul weights keep their
// on-disk block layout. Quantized lengths are validated before any compute.
fn load_tensor(gguf: &GgufFile, info: &crate::gguf::tensor_index::TensorInfo) -> Result<CpuBuffer> {
    let raw = gguf.tensor_bytes(info)?;
    Ok(match info.ggml_type {
        GgmlType::F16 => CpuBuffer::from_bytes(raw.to_vec(), CpuFormat::F16),
        GgmlType::F32 => CpuBuffer::from_bytes(f32_to_f16_bytes(raw), CpuFormat::F16),
        GgmlType::Q4_K => {
            dequant::validate(CpuFormat::Q4_K, raw.len())?;
            CpuBuffer::from_bytes(raw.to_vec(), CpuFormat::Q4_K)
        }
        GgmlType::Q5_K => {
            dequant::validate(CpuFormat::Q5_K, raw.len())?;
            CpuBuffer::from_bytes(raw.to_vec(), CpuFormat::Q5_K)
        }
        GgmlType::Q6_K => {
            dequant::validate(CpuFormat::Q6_K, raw.len())?;
            CpuBuffer::from_bytes(raw.to_vec(), CpuFormat::Q6_K)
        }
        other => bail!(
            "cpu: weight '{}' is {} — unsupported quantization",
            info.name,
            other.name()
        ),
    })
}

// Reusable activation buffers following the shared graph layout: all FP16 except
// `x`, the residual stream, which is FP32. Every buffer starts zeroed.
fn alloc_scratch(meta: &ModelMetadata) -> Scratch<CpuBuffer> {
    let f16 = |n: usize| n * 2;
    let embd = f16(meta.embedding_length);
    let qd = f16(meta.head_count * meta.head_dim);
    let kv = f16(meta.head_count_kv * meta.head_dim);
    let ffn = f16(meta.feed_forward_length);
    let a = |bytes: usize| CpuBuffer::zeroed(bytes, CpuFormat::F16);
    Scratch {
        x: CpuBuffer::zeroed(meta.embedding_length * 4, CpuFormat::F32),
        normed: a(embd),
        q: a(qd),
        k: a(kv),
        v: a(kv),
        attn: a(qd),
        proj: a(embd),
        gate: a(ffn),
        up: a(ffn),
        act: a(ffn),
        ffn_out: a(embd),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::source::WeightGroups;
    use std::sync::atomic::{AtomicU64, Ordering};

    const GGUF_MAGIC: &[u8; 4] = b"GGUF";
    const ALIGNMENT: usize = 32;
    static NEXT_FILE: AtomicU64 = AtomicU64::new(0);

    fn push_str(buf: &mut Vec<u8>, s: &str) {
        buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
        buf.extend_from_slice(s.as_bytes());
    }

    fn type_id(ty: GgmlType) -> u32 {
        match ty {
            GgmlType::F32 => 0,
            GgmlType::F16 => 1,
            GgmlType::Q4_K => 12,
            GgmlType::Q5_K => 13,
            GgmlType::Q6_K => 14,
            _ => panic!("test dtype is not a CPU weight format"),
        }
    }

    fn dims_for(ty: GgmlType) -> Vec<u64> {
        match ty {
            GgmlType::F32 | GgmlType::F16 => vec![4],
            GgmlType::Q4_K | GgmlType::Q5_K | GgmlType::Q6_K => vec![256],
            _ => panic!("test dtype is not a CPU weight format"),
        }
    }

    fn tensor_bytes(ty: GgmlType) -> Vec<u8> {
        let bytes = match ty {
            GgmlType::F32 => 4 * 4,
            GgmlType::F16 => 4 * 2,
            GgmlType::Q4_K => 144,
            GgmlType::Q5_K => 176,
            GgmlType::Q6_K => 210,
            _ => panic!("test dtype is not a CPU weight format"),
        };
        vec![0; bytes]
    }

    fn build_gguf(tensors: &[(String, GgmlType, Vec<u8>)]) -> Vec<u8> {
        let mut header = Vec::new();
        header.extend_from_slice(GGUF_MAGIC);
        header.extend_from_slice(&3u32.to_le_bytes());
        header.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
        header.extend_from_slice(&0u64.to_le_bytes());

        let mut offset = 0u64;
        for (name, ty, bytes) in tensors {
            push_str(&mut header, name);
            let dims = dims_for(*ty);
            header.extend_from_slice(&(dims.len() as u32).to_le_bytes());
            for dim in dims {
                header.extend_from_slice(&dim.to_le_bytes());
            }
            header.extend_from_slice(&type_id(*ty).to_le_bytes());
            header.extend_from_slice(&offset.to_le_bytes());
            offset += bytes.len() as u64;
        }

        while header.len() % ALIGNMENT != 0 {
            header.push(0);
        }
        for (_, _, bytes) in tensors {
            header.extend_from_slice(bytes);
        }
        header
    }

    fn with_temp_gguf<R>(bytes: &[u8], f: impl FnOnce(GgufFile) -> R) -> R {
        let path = temp_path();
        std::fs::write(&path, bytes).expect("write temp GGUF");
        let file = GgufFile::open(&path).expect("open temp GGUF");
        let out = f(file);
        let _ = std::fs::remove_file(&path);
        out
    }

    fn temp_path() -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "graph_horizon_cpu_weights_{}_{}.gguf",
            std::process::id(),
            NEXT_FILE.fetch_add(1, Ordering::Relaxed)
        ))
    }

    fn assert_format(buf: &CpuBuffer, expected: CpuFormat) {
        assert!(buf.format == expected);
    }

    struct Source<'a>(&'a [crate::gguf::tensor_index::TensorInfo]);

    impl WeightSource for Source<'_> {
        fn groups(&self) -> WeightGroups<'_> {
            WeightGroups::new(
                &self.0[0],
                &self.0[1],
                None,
                self.0[2..]
                    .chunks(9)
                    .map(|group| group.iter().collect())
                    .collect(),
            )
        }
    }

    #[test]
    fn load_tensor_preserves_every_internal_weight_format() {
        let formats = [
            (GgmlType::F32, CpuFormat::F16),
            (GgmlType::F16, CpuFormat::F16),
            (GgmlType::Q4_K, CpuFormat::Q4_K),
            (GgmlType::Q5_K, CpuFormat::Q5_K),
            (GgmlType::Q6_K, CpuFormat::Q6_K),
        ];
        let tensors: Vec<_> = formats
            .iter()
            .enumerate()
            .map(|(i, (ty, _))| (format!("w{i}"), *ty, tensor_bytes(*ty)))
            .collect();
        let gguf = build_gguf(&tensors);

        with_temp_gguf(&gguf, |file| {
            for (info, (_, expected)) in file.tensors().iter().zip(formats) {
                let buf = load_tensor(&file, info).expect("load tensor");
                assert_format(&buf, expected);
            }
        });
    }

    #[test]
    fn selected_inventory_and_every_load_failpoint_are_transactional() {
        let tensors = (0..11)
            .map(|index| {
                (
                    format!("w{index}"),
                    GgmlType::F16,
                    tensor_bytes(GgmlType::F16),
                )
            })
            .collect::<Vec<_>>();
        let gguf = build_gguf(&tensors);
        with_temp_gguf(&gguf, |file| {
            let source = Source(file.tensors());
            let selection = WeightSelection {
                layers: 0..1,
                embedding: true,
                tail: false,
            };
            let weights = load_weights(&file, &source, &selection).unwrap();
            assert!(weights.token_embd.is_some());
            assert!(weights.output_norm.is_none());
            assert_eq!(weights.layers.len(), 1);
            for fail_at in 0..=10 {
                assert_eq!(
                    load_weights_inner(&file, &source, &selection, Some(fail_at))
                        .err()
                        .expect("injected CPU load failure")
                        .to_string(),
                    "cpu: weight load failed"
                );
            }
        });
    }

    #[test]
    fn gguf_open_rejects_truncated_quantized_blocks_before_cpu_load() {
        let gguf = build_gguf(&[("bad".into(), GgmlType::Q5_K, vec![0; 175])]);
        let path = temp_path();
        std::fs::write(&path, gguf).expect("write temp GGUF");
        let err = match GgufFile::open(&path) {
            Ok(_) => panic!("partial Q5_K block must fail before CPU load"),
            Err(err) => err.to_string(),
        };
        let _ = std::fs::remove_file(&path);
        assert!(err.contains("extends beyond end of file"));
    }
}
