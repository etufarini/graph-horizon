/*
 * graph_horizon_engine — transactional Vulkan weight upload
 * Validates neutral embedding/tail/layer selection, converts F32 norms through
 * shared FP16, and uploads retained GGUF blocks with their dispatch format.
 * Partial globals are `None`; the local guard owns cleanup until commit.
*/

use color_eyre::eyre::{Result, bail};

use super::buffers::{GpuBuffer, WeightFormat};
use crate::backend::buffers::{LayerWeights, WeightSet as GpuWeightSet};
use crate::backend::f16::f32_to_f16_bytes;
use crate::backend::source::{OutputWeight, WeightSelection, WeightSource};
use crate::backend::vulkan::device::Device;
use crate::backend::vulkan::mem::predecode;
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
    let groups = ws.groups();
    let full = WeightSelection::full(groups.layers.len());
    let selection = selection.unwrap_or(&full);
    if (!host.is_empty() && host.len() != tensors.len())
        || selection.layers.start > selection.layers.end
        || selection.layers.end > groups.layers.len()
    {
        bail!("vulkan: invalid weight layout");
    }
    let dedicated = matches!(groups.tail.output, OutputWeight::Dedicated(_));
    let mut index = 0usize;
    let mut slots = vec![
        selected_slot(
            groups.embedding,
            selection.embedding || (selection.tail && !dedicated),
            host,
            &mut index,
        ),
        selected_slot(groups.tail.norm, selection.tail, host, &mut index),
    ];
    if let OutputWeight::Dedicated(output) = groups.tail.output {
        slots.push(selected_slot(output, selection.tail, host, &mut index));
    }
    for (layer, group) in groups.layers.iter().enumerate() {
        if group.len() != 9 {
            bail!("vulkan: invalid weight layout");
        }
        for tensor in group {
            let slot = selected_slot(tensor, selection.layers.contains(&layer), host, &mut index);
            if selection.layers.contains(&layer) {
                slots.push(slot);
            }
        }
    }
    upload_slots(dev, gguf, dedicated, selection.layers.len(), &slots)
}

fn selected_slot<'a>(
    tensor: &'a TensorInfo,
    owned: bool,
    host: &[bool],
    index: &mut usize,
) -> Option<(&'a TensorInfo, bool)> {
    let slot = owned.then_some((tensor, host.get(*index).copied().unwrap_or(false)));
    *index += 1;
    slot
}

fn upload_slots(
    dev: &Device,
    gguf: &GgufFile,
    has_output: bool,
    layer_count: usize,
    slots: &[Option<(&TensorInfo, bool)>],
) -> Result<GpuWeightSet<GpuBuffer>> {
    // This guard owns every partial upload until the complete selected set is
    // assembled, so any failure destroys each prior allocation exactly once.
    let mut uploaded = Uploaded {
        dev,
        buffers: Vec::new(),
    };
    for slot in slots {
        uploaded.buffers.push(
            slot.map(|(tensor, host)| upload_tensor(dev, gguf, tensor, host))
                .transpose()?,
        );
    }
    let mut buffers = std::mem::take(&mut uploaded.buffers).into_iter();
    let mut take = || buffers.next().expect("validated weight layout");
    let token_embd = take();
    let output_norm = take();
    let output = has_output.then(&mut take).flatten();
    let mut layers = Vec::with_capacity(layer_count);
    for _ in 0..layer_count {
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
    let native = if predecoded_bytes(info).is_some() {
        let in_dim = usize::try_from(info.dims[0])?;
        let out_dim = usize::try_from(info.dims[1])?;
        Some(predecode::q4_f16(&bytes, in_dim, out_dim)?)
    } else {
        None
    };
    let native_offset = native.as_ref().map(|_| bytes.len() as u64);
    let total = bytes
        .len()
        .checked_add(native.as_ref().map_or(0, Vec::len))
        .ok_or_else(|| color_eyre::eyre::eyre!("vulkan: weight size overflow"))?;
    let mut buf = GpuBuffer::alloc(dev, total as u64, host)?;
    if let Err(err) = buf.upload(dev, &bytes) {
        // Ownership has not reached `Uploaded`; release this final allocation here.
        buf.destroy(dev);
        return Err(err);
    }
    if let (Some(offset), Some(native)) = (native_offset, native.as_ref()) {
        let view = buf.view(offset, native.len() as u64);
        if let Err(err) = view.upload(dev, native) {
            buf.destroy(dev);
            return Err(err);
        }
    }
    buf.quant = fmt;
    buf.native_offset = native_offset;
    Ok(buf)
}

#[cfg(feature = "vulkan")]
fn predecode_enabled(info: &TensorInfo) -> bool {
    let enabled = |name| {
        matches!(
            std::env::var(name).ok().as_deref(),
            Some("1" | "true" | "yes")
        )
    };
    let mlp = enabled("GRAPH_HORIZON_PREFILL_PREDECODE_MLP");
    (info.name.ends_with(".ffn_gate.weight")
        && (mlp || enabled("GRAPH_HORIZON_PREFILL_PREDECODE_GATE")))
        || (info.name.ends_with(".ffn_up.weight")
            && (mlp || enabled("GRAPH_HORIZON_PREFILL_PREDECODE_UP")))
}

#[cfg(feature = "vulkan")]
pub(super) fn predecoded_bytes(info: &TensorInfo) -> Option<u64> {
    if info.ggml_type != GgmlType::Q4_K || info.dims != [3072, 9216] || !predecode_enabled(info) {
        return None;
    }
    info.element_count()?.checked_mul(2)
}

#[cfg(feature = "vulkan-hybrid")]
pub(super) fn predecoded_bytes(_info: &TensorInfo) -> Option<u64> {
    None
}
