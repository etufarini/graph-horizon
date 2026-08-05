/*
 * graph_orizon_engine — Vulkan backend construction
 * Builds a `VulkanBackend` from a weight source and placement: the shared
 * `load_inner` bootstrap (budget → memory plan → buffer creation → pipeline build →
 * scratch alloc), the mmvq Q8_1 scratch allocation, and the test-only `bare`
 * device/pipeline entry. Bodies moved 1:1 from the former monolithic `mod.rs`.
*/

use color_eyre::eyre::Result;

use super::device::Device;
use crate::backend::source::WeightSource;
use crate::backend::vulkan::buffers::GpuBuffer;
use crate::backend::vulkan::kernels::reduce;
use crate::backend::vulkan::loader;
#[cfg(any(test, not(feature = "vulkan-hybrid")))]
use crate::backend::vulkan::mem::budget::device_budget;
#[cfg(any(test, not(feature = "vulkan-hybrid")))]
use crate::backend::vulkan::mem::memory::plan;
use crate::backend::vulkan::pipeline::PipelineRegistry;
use crate::backend::vulkan::{MMVQ_SCRATCH_IN_DIM, VulkanBackend};
use crate::gguf::loader::GgufFile;
use crate::gguf::metadata::ModelMetadata;

// Shared full-backend bootstrap. Partial hybrid loads use `load_selected`,
// which bypasses this all-weight memory plan after immutable placement.
impl VulkanBackend {
    #[cfg(any(test, not(feature = "vulkan-hybrid")))]
    pub(in crate::backend::vulkan) fn load_inner(
        dev: Device,
        meta: &ModelMetadata,
        ws: &dyn WeightSource,
        gguf: &GgufFile,
        context: usize,
    ) -> Result<Self> {
        let budget = placement_budget(&dev);
        let plan = plan(meta, ws, context, &budget)?;
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
        let reduce_bytes = reduce::TOPK_GROUPS as u64 * reduce::MAX_K as u64 * 8;
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

    // The mmvq Q8_1 scratch: packed int8 quants (in_dim/4 uints = in_dim bytes)
    // plus per-32-block (d, s) f32 pairs, sized to MMVQ_SCRATCH_IN_DIM.
    fn alloc_mmvq_scratch(dev: &Device) -> Result<(GpuBuffer, GpuBuffer)> {
        let qs = GpuBuffer::alloc(dev, MMVQ_SCRATCH_IN_DIM, false)?;
        let ds = match GpuBuffer::alloc(dev, MMVQ_SCRATCH_IN_DIM / 32 * 2 * 4, false) {
            Ok(ds) => ds,
            Err(err) => {
                qs.destroy(dev);
                return Err(err);
            }
        };
        Ok((qs, ds))
    }
}

#[cfg(not(feature = "vulkan-hybrid"))]
fn placement_budget(dev: &Device) -> crate::backend::vulkan::mem::memory::Budget {
    let mut budget = device_budget(dev);
    budget.vram = selected_vram_budget(budget.vram, dev.free_vram());
    budget
}

#[cfg(all(feature = "vulkan-hybrid", test))]
fn placement_budget(dev: &Device) -> crate::backend::vulkan::mem::memory::Budget {
    device_budget(dev)
}

#[cfg(not(feature = "vulkan-hybrid"))]
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

    #[cfg(test)]
    pub(crate) fn replace_test_buffers(
        &mut self,
        buffers: crate::backend::buffers::Buffers<GpuBuffer>,
        vocab: usize,
    ) -> Result<()> {
        let host = GpuBuffer::alloc(&self.dev, vocab as u64 * 4, true)?;
        loader::destroy_buffers(&self.dev, &self.buf);
        self.logits_host.destroy(&self.dev);
        self.buf = buffers;
        self.logits_host = host;
        Ok(())
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
    #[cfg(not(feature = "vulkan-hybrid"))]
    #[test]
    fn pure_vulkan_prefers_current_memory_budget_when_available() {
        assert_eq!(super::selected_vram_budget(1000, Some(400)), 400);
        assert_eq!(super::selected_vram_budget(1000, None), 1000);
    }
}
