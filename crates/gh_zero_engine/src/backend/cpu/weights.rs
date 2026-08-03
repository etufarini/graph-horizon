/*
 * gh_zero_engine — CPU weight load
 * The CPU counterpart of vulkan::weights plus the alloc_scratch/create part of
 * vulkan::buffers. Walks the WeightSource in its canonical dense layout and
 * copies each retained weight's bytes in its original format. Family validation
 * admits Q4_K/Q6_K matrices; this generic loader also retains Q5_K/F16. Optional output and Q/K
 * norms are explicit, and a mismatched list is rejected before indexing.
 * Quantized weights are validated at
 * load (dequant::validate); F32 norms are converted to FP16 on upload, exactly
 * like Vulkan. Scratch and logits use the same byte sizes as
 * vulkan::buffers::alloc_scratch. This file knows nothing about a model family.
*/

use color_eyre::eyre::{Result, bail};

use super::buffer::{CpuBuffer, CpuFormat, f32_to_f16_bytes};
use super::dequant;
use crate::backend::buffers::{Buffers, LayerWeights, Scratch, WeightSet};
use crate::backend::source::{WeightSelection, WeightSource};
use crate::gguf::loader::GgufFile;
use crate::gguf::metadata::ModelMetadata;
use crate::gguf::tensor_index::GgmlType;

// Builds the whole CPU buffer set (weights + scratch + logits). Nothing is kept
// private on the CPU side (no readback mirror as on Vulkan), so only `Buffers`
// is returned.
#[cfg(any(test, not(feature = "vulcan-hybrid")))]
pub(super) fn load(
    meta: &ModelMetadata,
    ws: &dyn WeightSource,
    gguf: &GgufFile,
    _context: usize,
) -> Result<Buffers<CpuBuffer>> {
    load_selected(
        meta,
        ws,
        gguf,
        &WeightSelection::full(ws.layout().layer_count),
    )
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

// Walks tensors in the same canonical order vulkan::weights uses. The source
// declares optional slots explicitly, so no model type or count heuristic leaks
// into this loader.
fn load_weights(
    gguf: &GgufFile,
    ws: &dyn WeightSource,
    selection: &WeightSelection,
) -> Result<WeightSet<CpuBuffer>> {
    let tensors = ws.tensors();
    let layout = ws.layout();
    let per_layer = 9;
    let expected = layout
        .layer_count
        .checked_mul(per_layer)
        .and_then(|n| n.checked_add(2 + usize::from(layout.has_output)))
        .ok_or_else(|| color_eyre::eyre::eyre!("cpu: invalid weight layout"))?;
    if tensors.len() != expected
        || selection.layers.start > selection.layers.end
        || selection.layers.end > layout.layer_count
    {
        bail!("cpu: invalid weight layout");
    }

    let mut i = 0usize;
    let take = |i: &mut usize, selected: bool| -> Result<Option<CpuBuffer>> {
        let tensor = tensors[*i];
        *i += 1;
        if !selected {
            return Ok(None);
        }
        load_tensor(gguf, tensor).map(Some)
    };

    let token_embd = take(
        &mut i,
        selection.embedding || (selection.tail && !layout.has_output),
    )?;
    let output_norm = take(&mut i, selection.tail)?;
    // `output` only takes a slot when present (absent for an embedding model), so
    // `i` must NOT advance otherwise or the per-layer tensors read off by one.
    let output = layout
        .has_output
        .then(|| take(&mut i, selection.tail))
        .transpose()?
        .flatten();
    let mut layers = Vec::with_capacity(selection.layers.len());
    for layer in 0..layout.layer_count {
        let selected = selection.layers.contains(&layer);
        let values = [
            take(&mut i, selected)?,
            take(&mut i, selected)?,
            take(&mut i, selected)?,
            take(&mut i, selected)?,
        ];
        let tail = [
            take(&mut i, selected)?,
            take(&mut i, selected)?,
            take(&mut i, selected)?,
            take(&mut i, selected)?,
            take(&mut i, selected)?,
        ];
        if selected {
            let [attn_norm, attn_q, attn_k, attn_v] = values.map(Option::unwrap);
            let [attn_output, ffn_norm, ffn_gate, ffn_up, ffn_down] = tail.map(Option::unwrap);
            layers.push(LayerWeights {
                attn_norm,
                attn_q,
                attn_k,
                attn_v,
                attn_output,
                ffn_norm,
                ffn_gate,
                ffn_up,
                ffn_down,
            });
        }
    }
    Ok(WeightSet {
        token_embd,
        output_norm,
        output,
        layers,
    })
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

// Reusable activation buffers, identical byte sizes to vulkan::buffers::
// alloc_scratch: all FP16 except `x`, the residual stream, which is FP32 (late
// layers overflow FP16). Zeroed.
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

    const GGUF_MAGIC: &[u8; 4] = b"GGUF";
    const ALIGNMENT: usize = 32;

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
            "gh_zero_cpu_weights_{}_{}.gguf",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("clock before epoch")
                .as_nanos()
        ))
    }

    fn assert_format(buf: &CpuBuffer, expected: CpuFormat) {
        assert!(buf.format == expected);
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
