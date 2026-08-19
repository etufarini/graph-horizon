/*
 * graph_horizon_engine — Vulkan backend construction
 * Builds a `VulkanBackend` from a weight source and placement: the shared
 * `load_inner` bootstrap (budget/plan → buffer creation → pipeline build → scratch
 * allocation), the MMVQ per-8 Q8 scratch, and the test-only `bare` device/pipeline
 * entry. Bodies moved 1:1 from the former monolithic `mod.rs`.
*/

use color_eyre::eyre::Result;

use super::device::{AMD_VENDOR_ID, Device};
use crate::backend::source::WeightSource;
use crate::backend::vulkan::buffers::GpuBuffer;
use crate::backend::vulkan::kernels::{attention, reduce};
use crate::backend::vulkan::loader;
#[cfg(feature = "vulkan")]
use crate::backend::vulkan::mem::budget::device_budget;
#[cfg(feature = "vulkan")]
use crate::backend::vulkan::mem::memory::plan;
use crate::backend::vulkan::pipeline::PipelineRegistry;
use crate::backend::vulkan::{MMVQ_SCRATCH_ELEMENTS, VulkanBackend};
use crate::gguf::loader::GgufFile;
use crate::gguf::metadata::ModelMetadata;

#[cfg(feature = "vulkan")]
pub(crate) fn load(
    meta: &ModelMetadata,
    source: &dyn WeightSource,
    file: &GgufFile,
    context: usize,
    weights_percent: Option<u8>,
    reserve_mib: Option<u64>,
) -> Result<VulkanBackend> {
    let device = Device::init().map_err(super::pure_loader_unavailable)?;
    VulkanBackend::load_inner(
        device,
        meta,
        source,
        file,
        context,
        weights_percent,
        reserve_mib,
    )
}

// Shared full-backend bootstrap. Partial hybrid loads use `load_selected`,
// which bypasses this all-weight memory plan after immutable placement.
impl VulkanBackend {
    #[cfg(feature = "vulkan")]
    pub(in crate::backend::vulkan) fn load_inner(
        dev: Device,
        meta: &ModelMetadata,
        ws: &dyn WeightSource,
        gguf: &GgufFile,
        context: usize,
        weights_percent: Option<u8>,
        reserve_mib: Option<u64>,
    ) -> Result<Self> {
        let budget = placement_budget(&dev);
        let plan = plan(meta, ws, context, &budget, weights_percent, reserve_mib)?;
        let (buf, logits_host) = loader::create_buffers(&dev, &plan, gguf, ws, meta)?;
        let reg = match PipelineRegistry::build(&dev) {
            Ok(reg) => reg,
            Err(err) => {
                logits_host.destroy(&dev);
                loader::destroy_buffers(&dev, &buf);
                return Err(err);
            }
        };
        Self::finish(dev, reg, buf, logits_host)
    }

    #[cfg(feature = "vulkan-hybrid")]
    pub(crate) fn load_selected(
        dev: Device,
        meta: &ModelMetadata,
        ws: &dyn WeightSource,
        gguf: &GgufFile,
        selection: &crate::backend::source::WeightSelection,
    ) -> Result<Self> {
        let (buf, logits_host) = loader::create_selected_buffers(&dev, gguf, ws, meta, selection)?;
        let reg = match PipelineRegistry::build(&dev) {
            Ok(reg) => reg,
            Err(err) => {
                logits_host.destroy(&dev);
                loader::destroy_buffers(&dev, &buf);
                return Err(err);
            }
        };
        Self::finish(dev, reg, buf, logits_host)
    }

    fn finish(
        dev: Device,
        reg: PipelineRegistry,
        buf: crate::backend::buffers::Buffers<GpuBuffer>,
        logits_host: GpuBuffer,
    ) -> Result<Self> {
        let reduce_bytes = (reduce::TOPK_GROUPS as u64 * reduce::MAX_K as u64 * 8)
            .max(attention::GQA_DECODE_PARTIAL_BYTES);
        let reduce = match GpuBuffer::alloc(&dev, reduce_bytes, false) {
            Ok(reduce) => reduce,
            Err(err) => {
                reg.destroy(&dev);
                logits_host.destroy(&dev);
                loader::destroy_buffers(&dev, &buf);
                return Err(err);
            }
        };
        let (mmvq_qs, mmvq_ds) = match Self::alloc_mmvq_scratch(&dev) {
            Ok(scratch) => scratch,
            Err(err) => {
                reduce.destroy(&dev);
                reg.destroy(&dev);
                logits_host.destroy(&dev);
                loader::destroy_buffers(&dev, &buf);
                return Err(err);
            }
        };
        Ok(VulkanBackend {
            dev,
            reg,
            buf,
            logits_host,
            reduce,
            mmvq_qs,
            mmvq_ds,
        })
    }

    // Packed Q8 quants and per-8-block (d, s) pairs each consume one byte per
    // activation element and cover the largest supported prefill batch.
    fn alloc_mmvq_scratch(dev: &Device) -> Result<(GpuBuffer, GpuBuffer)> {
        let elements = if dev.vendor_id == AMD_VENDOR_ID && dev.dp4a {
            MMVQ_SCRATCH_ELEMENTS
        } else {
            crate::backend::vulkan::MMVQ_SCRATCH_IN_DIM
        };
        let qs = GpuBuffer::alloc(dev, elements, false)?;
        let ds_bytes = elements.max(attention::GQA_DECODE_STATE_BYTES);
        let ds = match GpuBuffer::alloc(dev, ds_bytes, false) {
            Ok(ds) => ds,
            Err(err) => {
                qs.destroy(dev);
                return Err(err);
            }
        };
        Ok((qs, ds))
    }
}

#[cfg(feature = "vulkan")]
fn placement_budget(dev: &Device) -> crate::backend::vulkan::mem::memory::Budget {
    let mut budget = device_budget(dev);
    budget.vram = selected_vram_budget(budget.vram, dev.free_vram());
    budget
}

#[cfg(feature = "vulkan")]
fn selected_vram_budget(total: u64, current: Option<u64>) -> u64 {
    current.unwrap_or(total)
}

#[cfg(test)]
impl VulkanBackend {
    // A bare backend: device + compute pipelines only, with placeholder buffer
    // holders. The family-dedicated split owns its
    // weights itself and only uses the device/kernels (matmul, upload, readback),
    // never `self.buf`/`logits_host`, so those are 1-byte placeholders. Returned to
    // the family SplitModel, which uploads its own GPU-side weights afterwards.
    pub(crate) fn bare() -> Result<Self> {
        let dev = Device::init()?;
        #[cfg(test)]
        {
            let _tracked = crate::backend::vulkan::fault::track(
                crate::backend::vulkan::fault::Point::Initialization,
            );
            crate::backend::vulkan::fault::hit(
                crate::backend::vulkan::fault::Point::Initialization,
            )?;
        }
        let reg = PipelineRegistry::build(&dev)?;
        #[cfg(test)]
        {
            let _tracked = crate::backend::vulkan::fault::track(
                crate::backend::vulkan::fault::Point::Pipeline,
            );
            if let Err(error) =
                crate::backend::vulkan::fault::hit(crate::backend::vulkan::fault::Point::Pipeline)
            {
                reg.destroy(&dev);
                return Err(error);
            }
        }
        let buf = match placeholder_buffers(&dev) {
            Ok(buf) => buf,
            Err(err) => {
                reg.destroy(&dev);
                return Err(err);
            }
        };
        #[cfg(test)]
        {
            let _tracked = crate::backend::vulkan::fault::track(
                crate::backend::vulkan::fault::Point::Allocation,
            );
            if let Err(error) =
                crate::backend::vulkan::fault::hit(crate::backend::vulkan::fault::Point::Allocation)
            {
                loader::destroy_buffers(&dev, &buf);
                reg.destroy(&dev);
                return Err(error);
            }
        }
        let logits_host = match GpuBuffer::alloc(&dev, 1, true) {
            Ok(host) => host,
            Err(err) => {
                loader::destroy_buffers(&dev, &buf);
                reg.destroy(&dev);
                return Err(err);
            }
        };
        Self::finish(dev, reg, buf, logits_host)
    }
}

#[cfg(test)]
fn placeholder_buffers(dev: &Device) -> Result<crate::backend::buffers::Buffers<GpuBuffer>> {
    use crate::backend::buffers::{Buffers, Scratch, WeightSet};

    let mut owned = Vec::with_capacity(14);
    for _ in 0..14 {
        match GpuBuffer::alloc(dev, 1, false) {
            Ok(buffer) => owned.push(buffer),
            Err(err) => {
                for buffer in &owned {
                    buffer.destroy(dev);
                }
                return Err(err);
            }
        }
    }
    let mut b = owned.into_iter();
    let mut take = || b.next().unwrap();
    Ok(Buffers {
        weights: WeightSet {
            token_embd: Some(take()),
            output_norm: Some(take()),
            output: None,
            layers: Vec::new(),
        },
        scratch: Scratch {
            x: take(),
            normed: take(),
            q: take(),
            k: take(),
            v: take(),
            attn: take(),
            proj: take(),
            gate: take(),
            up: take(),
            act: take(),
            ffn_out: take(),
        },
        logits: take(),
    })
}

impl Drop for VulkanBackend {
    fn drop(&mut self) {
        // No command may reference persistent storage while ownership is released.
        // SAFETY: this backend exclusively owns the logical device and queue.
        unsafe {
            let _ = self.dev.device.device_wait_idle();
        }
        self.mmvq_ds.destroy(&self.dev);
        self.mmvq_qs.destroy(&self.dev);
        self.reduce.destroy(&self.dev);
        self.logits_host.destroy(&self.dev);
        loader::destroy_buffers(&self.dev, &self.buf);
        self.reg.destroy(&self.dev);
    }
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "vulkan")]
    #[test]
    fn pure_vulkan_prefers_current_memory_budget_when_available() {
        assert_eq!(super::selected_vram_budget(1000, Some(400)), 400);
        assert_eq!(super::selected_vram_budget(1000, None), 1000);
    }
}
