/*
 * gh_zero_engine — transactional Vulkan weight upload
 * Validates the canonical dense selection, converts F32 norms through shared
 * FP16, and uploads F16/quantized GGUF blocks with their dispatch format.
 * Partial globals are `None`; the local guard owns cleanup until commit.
*/

use color_eyre::eyre::{Result, bail};

use super::buffers::{GpuBuffer, WeightFormat};
use crate::backend::buffers::{LayerWeights, WeightSet as GpuWeightSet};
use crate::backend::f16::f32_to_f16_bytes;
use crate::backend::source::{WeightSelection, WeightSource};
use crate::backend::vulkan::device::Device;
use crate::gguf::loader::GgufFile;
use crate::gguf::tensor_index::{GgmlType, TensorInfo};

pub(crate) fn upload_weights(
    dev: &Device,
    gguf: &GgufFile,
    ws: &dyn WeightSource,
    host: &[bool],
    selection: Option<&WeightSelection>,
) -> Result<GpuWeightSet<GpuBuffer>> {
    let tensors = ws.tensors();
    let mut layout = ws.layout();
    let per_layer = 9;
    let globals = 2 + usize::from(layout.has_output);
    let expected = layout
        .layer_count
        .checked_mul(per_layer)
        .and_then(|n| n.checked_add(globals))
        .ok_or_else(|| color_eyre::eyre::eyre!("vulkan: invalid weight layout"))?;
    let full = WeightSelection::full(layout.layer_count);
    let selection = selection.unwrap_or(&full);
    if tensors.len() != expected
        || (!host.is_empty() && host.len() != tensors.len())
        || selection.layers.start > selection.layers.end
        || selection.layers.end > layout.layer_count
    {
        bail!("vulkan: invalid weight layout");
    }
    let start = globals + selection.layers.start * per_layer;
    let end = globals + selection.layers.end * per_layer;
    let indices = (0..globals).chain(start..end).collect::<Vec<_>>();
    let slots = indices
        .iter()
        .enumerate()
        .map(|(slot, &index)| {
            let owned = match slot {
                0 => selection.embedding || (selection.tail && !layout.has_output),
                1 => selection.tail,
                2 if layout.has_output => selection.tail,
                _ => true,
            };
            owned.then_some((index, host.get(index).copied().unwrap_or(false)))
        })
        .collect::<Vec<_>>();
    layout.layer_count = selection.layers.len();
    upload_slots(dev, gguf, &tensors, layout, &slots)
}

fn upload_slots(
    dev: &Device,
    gguf: &GgufFile,
    tensors: &[&TensorInfo],
    layout: crate::backend::source::WeightLayout,
    slots: &[Option<(usize, bool)>],
) -> Result<GpuWeightSet<GpuBuffer>> {
    // This guard owns every partial upload until the complete selected set is
    // assembled, so any failure destroys each prior allocation exactly once.
    let mut uploaded = Uploaded {
        dev,
        buffers: Vec::new(),
    };
    for slot in slots {
        uploaded.buffers.push(
            slot.map(|(index, host)| upload_tensor(dev, gguf, tensors[index], host))
                .transpose()?,
        );
    }
    let mut buffers = std::mem::take(&mut uploaded.buffers).into_iter();
    let mut take = || buffers.next().expect("validated weight layout");
    let token_embd = take();
    let output_norm = take();
    let output = layout.has_output.then(&mut take).flatten();
    let mut layers = Vec::with_capacity(layout.layer_count);
    for _ in 0..layout.layer_count {
        let attn_norm = take().expect("selected layer weight");
        let attn_q = take().expect("selected layer weight");
        let attn_k = take().expect("selected layer weight");
        let attn_v = take().expect("selected layer weight");
        layers.push(LayerWeights {
            attn_norm,
            attn_q,
            attn_k,
            attn_v,
            attn_output: take().expect("selected layer weight"),
            ffn_norm: take().expect("selected layer weight"),
            ffn_gate: take().expect("selected layer weight"),
            ffn_up: take().expect("selected layer weight"),
            ffn_down: take().expect("selected layer weight"),
        });
    }
    Ok(GpuWeightSet {
        token_embd,
        output_norm,
        output,
        layers,
    })
}

struct Uploaded<'a> {
    dev: &'a Device,
    buffers: Vec<Option<GpuBuffer>>,
}

impl Drop for Uploaded<'_> {
    fn drop(&mut self) {
        for buffer in self.buffers.iter().flatten() {
            buffer.destroy(self.dev);
        }
    }
}

fn upload_tensor(
    dev: &Device,
    gguf: &GgufFile,
    info: &TensorInfo,
    host: bool,
) -> Result<GpuBuffer> {
    // F32 norms narrow to FP16; all other accepted GGUF blocks stay byte-exact,
    // with `fmt` selecting the later matmul/dequant kernel.
    let (bytes, fmt): (std::borrow::Cow<[u8]>, WeightFormat) = match info.ggml_type {
        GgmlType::F16 => (gguf.tensor_bytes(info)?.into(), WeightFormat::F16),
        GgmlType::F32 => (
            f32_to_f16_bytes(gguf.tensor_bytes(info)?).into(),
            WeightFormat::F16,
        ),
        GgmlType::Q4_K => (gguf.tensor_bytes(info)?.into(), WeightFormat::Q4K),
        GgmlType::Q5_K => (gguf.tensor_bytes(info)?.into(), WeightFormat::Q5K),
        GgmlType::Q6_K => (gguf.tensor_bytes(info)?.into(), WeightFormat::Q6K),
        other => bail!(
            "vulkan: weight '{}' is {} — unsupported quantization",
            info.name,
            other.name()
        ),
    };
    let mut buf = GpuBuffer::alloc(dev, bytes.len() as u64, host)?;
    if let Err(err) = buf.upload(dev, &bytes) {
        // Ownership has not reached `Uploaded`; release this final allocation here.
        buf.destroy(dev);
        return Err(err);
    }
    buf.quant = fmt;
    Ok(buf)
}
