/*
 * graph_orizon_engine — Vulkan persistent buffer transaction
 * Assembles weights, scratch, logits, and the host readback mirror after memory
 * preflight. It owns every partial allocation until the complete buffer set is
 * returned and destroys the same set during backend teardown. It performs no
 * device selection, graph dispatch, or fallback policy.
 */

use color_eyre::eyre::Result;

use crate::backend::buffers::{Buffers, Scratch, WeightSet};
use crate::backend::source::WeightSource;
use crate::backend::vulkan::buffers::GpuBuffer;
use crate::backend::vulkan::device::Device;
#[cfg(any(test, not(feature = "vulkan-hybrid")))]
use crate::backend::vulkan::mem::memory::MemoryPlan;
use crate::backend::vulkan::weights::upload_weights;
use crate::gguf::loader::GgufFile;
use crate::gguf::metadata::ModelMetadata;

#[cfg(any(test, not(feature = "vulkan-hybrid")))]
pub(super) fn create_buffers(
    dev: &Device,
    plan: &MemoryPlan,
    gguf: &GgufFile,
    ws: &dyn WeightSource,
    meta: &ModelMetadata,
) -> Result<(Buffers<GpuBuffer>, GpuBuffer)> {
    let weights = upload_weights(dev, gguf, ws, &plan.host, None)?;
    create_with_weights(dev, weights, meta)
}

#[cfg(feature = "vulkan-hybrid")]
pub(super) fn create_selected_buffers(
    dev: &Device,
    gguf: &GgufFile,
    ws: &dyn WeightSource,
    meta: &ModelMetadata,
    selection: &crate::backend::source::WeightSelection,
) -> Result<(Buffers<GpuBuffer>, GpuBuffer)> {
    let weights = upload_weights(dev, gguf, ws, &[], Some(selection))?;
    create_with_weights(dev, weights, meta)
}

fn create_with_weights(
    dev: &Device,
    weights: WeightSet<GpuBuffer>,
    meta: &ModelMetadata,
) -> Result<(Buffers<GpuBuffer>, GpuBuffer)> {
    let scratch = match alloc_scratch(dev, meta) {
        Ok(scratch) => scratch,
        Err(err) => {
            destroy_weights(dev, &weights);
            return Err(err);
        }
    };
    let logits_bytes = meta.vocab_size as u64 * 4;
    let logits = match GpuBuffer::alloc(dev, logits_bytes, false) {
        Ok(logits) => logits,
        Err(err) => {
            destroy_scratch(dev, &scratch);
            destroy_weights(dev, &weights);
            return Err(err);
        }
    };
    let logits_host = match GpuBuffer::alloc(dev, logits_bytes, true) {
        Ok(host) => host,
        Err(err) => {
            logits.destroy(dev);
            destroy_scratch(dev, &scratch);
            destroy_weights(dev, &weights);
            return Err(err);
        }
    };
    Ok((
        Buffers {
            weights,
            scratch,
            logits,
        },
        logits_host,
    ))
}

pub(super) fn destroy_buffers(dev: &Device, buffers: &Buffers<GpuBuffer>) {
    buffers.logits.destroy(dev);
    destroy_scratch(dev, &buffers.scratch);
    destroy_weights(dev, &buffers.weights);
}

fn alloc_scratch(dev: &Device, meta: &ModelMetadata) -> Result<Scratch<GpuBuffer>> {
    let f16 = |n: usize| n as u64 * 2;
    let embd = f16(meta.embedding_length);
    let qd = f16(meta.head_count * meta.head_dim);
    let kv = f16(meta.head_count_kv * meta.head_dim);
    let ffn = f16(meta.feed_forward_length);
    let mut owned = Allocations {
        dev,
        buffers: Vec::with_capacity(11),
    };
    for size in [
        meta.embedding_length as u64 * 4,
        embd,
        qd,
        kv,
        kv,
        qd,
        embd,
        ffn,
        ffn,
        ffn,
        embd,
    ] {
        owned.buffers.push(GpuBuffer::alloc(dev, size, false)?);
    }
    let mut buffers = owned.finish().into_iter();
    // The residual stream is FP32; the remaining scratch buffers are FP16.
    Ok(Scratch {
        x: buffers.next().unwrap(),
        normed: buffers.next().unwrap(),
        q: buffers.next().unwrap(),
        k: buffers.next().unwrap(),
        v: buffers.next().unwrap(),
        attn: buffers.next().unwrap(),
        proj: buffers.next().unwrap(),
        gate: buffers.next().unwrap(),
        up: buffers.next().unwrap(),
        act: buffers.next().unwrap(),
        ffn_out: buffers.next().unwrap(),
    })
}

fn destroy_scratch(dev: &Device, scratch: &Scratch<GpuBuffer>) {
    for buffer in [
        &scratch.x,
        &scratch.normed,
        &scratch.q,
        &scratch.k,
        &scratch.v,
        &scratch.attn,
        &scratch.proj,
        &scratch.gate,
        &scratch.up,
        &scratch.act,
        &scratch.ffn_out,
    ] {
        buffer.destroy(dev);
    }
}

fn destroy_weights(dev: &Device, weights: &WeightSet<GpuBuffer>) {
    if let Some(token_embd) = &weights.token_embd {
        token_embd.destroy(dev);
    }
    if let Some(output_norm) = &weights.output_norm {
        output_norm.destroy(dev);
    }
    if let Some(output) = &weights.output {
        output.destroy(dev);
    }
    for layer in &weights.layers {
        for buffer in [
            &layer.attn_norm,
            &layer.attn_q,
            &layer.attn_k,
            &layer.attn_v,
            &layer.attn_output,
            &layer.ffn_norm,
            &layer.ffn_gate,
            &layer.ffn_up,
            &layer.ffn_down,
        ] {
            buffer.destroy(dev);
        }
    }
}

struct Allocations<'a> {
    dev: &'a Device,
    buffers: Vec<GpuBuffer>,
}

impl Allocations<'_> {
    fn finish(mut self) -> Vec<GpuBuffer> {
        std::mem::take(&mut self.buffers)
    }
}

impl Drop for Allocations<'_> {
    fn drop(&mut self) {
        for buffer in &self.buffers {
            buffer.destroy(self.dev);
        }
    }
}
