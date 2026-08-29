/*
 * graph_horizon_engine — compute pipeline registry
 * Owns transactional construction, lookup, and destruction of the reachable
 * compute pipelines. The registry builds its unconditional set plus the
 * capability-gated 512/1024-thread attention, FP16 coopmat, and MMVQ variants.
*/

use std::collections::HashMap;

use ash::vk;
use color_eyre::eyre::{Result, eyre};

use super::device::Device;

mod caps;
mod compiler;
mod kernel;
mod record;

pub(crate) use kernel::Kernel;
pub(crate) use record::{dispatch, dispatch_2d};

pub(crate) struct Pipeline {
    pub pipeline: vk::Pipeline,
    pub layout: vk::PipelineLayout,
    pub set_layout: vk::DescriptorSetLayout,
}

pub(crate) struct PipelineRegistry {
    map: HashMap<Kernel, Pipeline>,
    cache: vk::PipelineCache,
}

impl PipelineRegistry {
    pub(crate) fn build(dev: &Device) -> Result<PipelineRegistry> {
        let caps::PipelineCaps {
            wide_attention,
            tiled_attention,
            coop_qk_attention,
            matrix2,
            matrix2_attention_q64,
            attention_1024,
            gqa_prefill_required_wave32,
            gqa_decode,
            gqa_decode_required_wave32,
            gqa_decode_wave64,
            q4_metadata,
        } = caps::check(dev)?;
        // SAFETY: `dev.device` is alive; the default cache-create info is valid.
        let cache = unsafe {
            dev.device
                .create_pipeline_cache(&vk::PipelineCacheCreateInfo::default(), None)
        }
        .map_err(|_| eyre!("vulkan: cannot create pipeline cache"))?;

        let mut map = HashMap::new();
        let built = (|| -> Result<()> {
            let fixed_wave32 = dev.profile.fixed_wave32();
            for &k in kernel::BASE {
                let pipeline = if fixed_wave32 && matches!(k, Kernel::MatmulQ6K | Kernel::LogitsQ6K)
                {
                    compiler::build_wave32(dev, cache, k)?
                } else {
                    compiler::build(dev, cache, k)?
                };
                map.insert(k, pipeline);
            }
            if wide_attention {
                for k in [Kernel::AttentionDecodeWide, Kernel::AttentionPrefillWide] {
                    map.insert(k, compiler::build(dev, cache, k)?);
                }
            }
            if tiled_attention {
                map.insert(
                    Kernel::AttentionPrefillTiled,
                    compiler::build(dev, cache, Kernel::AttentionPrefillTiled)?,
                );
            }
            if coop_qk_attention {
                map.insert(
                    Kernel::AttentionPrefillTiledCoopQk,
                    compiler::build(dev, cache, Kernel::AttentionPrefillTiledCoopQk)?,
                );
            }
            if matrix2 {
                // A driver may advertise NV2 but reject an exact SPIR-V shape.
                // Pipeline absence is therefore a capability fallback, not fatal.
                if let Ok(pipeline) = compiler::build(dev, cache, Kernel::AttentionPrefillMatrix2) {
                    map.insert(Kernel::AttentionPrefillMatrix2, pipeline);
                }
                if let Ok(pipeline) = compiler::build(dev, cache, Kernel::MatmulQ4KMatrix2F16Out) {
                    map.insert(Kernel::MatmulQ4KMatrix2F16Out, pipeline);
                }
                if let Ok(pipeline) = compiler::build(dev, cache, Kernel::MatmulQ6KMatrix2F16Out) {
                    map.insert(Kernel::MatmulQ6KMatrix2F16Out, pipeline);
                }
            }
            if matrix2_attention_q64
                && let Ok(pipeline) =
                    compiler::build(dev, cache, Kernel::AttentionPrefillMatrix2Q64)
            {
                map.insert(Kernel::AttentionPrefillMatrix2Q64, pipeline);
            }
            if attention_1024 {
                map.insert(
                    Kernel::AttentionDecode1024,
                    compiler::build(dev, cache, Kernel::AttentionDecode1024)?,
                );
            }
            if gqa_prefill_required_wave32 {
                for k in [
                    Kernel::AttentionPrefillGqaSplit,
                    Kernel::AttentionPrefillGqaReduce,
                ] {
                    map.insert(k, compiler::build_wave32(dev, cache, k)?);
                }
            }
            if gqa_decode {
                for k in [
                    Kernel::AttentionDecodeGqaSplit,
                    Kernel::AttentionDecodeGqaInt8Split,
                ] {
                    map.insert(k, compiler::build(dev, cache, k)?);
                }
                map.insert(
                    Kernel::AttentionDecodeGqaReduce,
                    compiler::build(dev, cache, Kernel::AttentionDecodeGqaReduce)?,
                );
            }
            if gqa_decode_required_wave32 {
                for k in [
                    Kernel::AttentionDecodeGqaSplit,
                    Kernel::AttentionDecodeGqaInt8Split,
                ] {
                    map.insert(k, compiler::build_wave32(dev, cache, k)?);
                }
                map.insert(
                    Kernel::AttentionDecodeGqaReduce,
                    compiler::build_wave32(dev, cache, Kernel::AttentionDecodeGqaReduce)?,
                );
            }
            if gqa_decode_wave64 && !gqa_decode_required_wave32 {
                map.insert(
                    Kernel::AttentionDecodeGqaWave64Split,
                    compiler::build(dev, cache, Kernel::AttentionDecodeGqaWave64Split)?,
                );
                map.insert(
                    Kernel::AttentionDecodeGqaWave64Reduce,
                    compiler::build(dev, cache, Kernel::AttentionDecodeGqaWave64Reduce)?,
                );
            }
            // Capability-gated SPIR-V is built only when the device can execute it.
            if dev.coopmat.available {
                for k in [
                    Kernel::MatmulQ4KCoopmatF16Out,
                    Kernel::MatmulQ6KCoopmatF16Out,
                ] {
                    map.insert(k, compiler::build(dev, cache, k)?);
                }
                if q4_metadata {
                    map.insert(
                        Kernel::MatmulQ4KCoopmatMetadataF16Out,
                        compiler::build(dev, cache, Kernel::MatmulQ4KCoopmatMetadataF16Out)?,
                    );
                }
            }
            if dev.dp4a {
                for k in [Kernel::QuantAQ8F16, Kernel::MatmulQ4KMmvqF16Out] {
                    map.insert(k, compiler::build(dev, cache, k)?);
                }
                if dev.profile.integer_q4_batch() {
                    let k = Kernel::MatmulQ4KMmqBatchF16Out;
                    map.insert(k, compiler::build(dev, cache, k)?);
                }
            }
            Ok(())
        })();
        if let Err(err) = built {
            // Transactional build: every successfully-created pipeline and the shared
            // cache are released before the backend load returns the fixed public error.
            PipelineRegistry { map, cache }.destroy(dev);
            return Err(err);
        }
        Ok(PipelineRegistry { map, cache })
    }

    pub(crate) fn get(&self, k: Kernel) -> &Pipeline {
        // Dispatch selects capability-gated variants only after `contains` succeeds.
        self.map.get(&k).expect("pipeline built for every kernel")
    }

    pub(crate) fn contains(&self, k: Kernel) -> bool {
        self.map.contains_key(&k)
    }

    pub(crate) fn destroy(&self, dev: &Device) {
        // SAFETY: caller ensures the device is idle and the registry is dropped once; each
        // pipeline's pipeline/layout/set-layout are destroyed before the shared cache.
        unsafe {
            for p in self.map.values() {
                dev.device.destroy_pipeline(p.pipeline, None);
                dev.device.destroy_pipeline_layout(p.layout, None);
                dev.device.destroy_descriptor_set_layout(p.set_layout, None);
            }
            dev.device.destroy_pipeline_cache(self.cache, None);
        }
    }
}
