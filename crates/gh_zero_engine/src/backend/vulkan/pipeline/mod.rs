/*
 * gh_zero_engine — compute pipeline registry
 * Owns transactional construction, lookup, and destruction of the reachable
 * compute pipelines. The registry builds its unconditional set plus the
 * capability-gated FP16 coopmat and MMVQ pairs.
*/

use std::collections::HashMap;

use ash::vk;
use color_eyre::eyre::{Result, eyre};

use super::device::Device;

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

// Capability-gated variants are added separately during construction.
const KERNELS: [Kernel; 28] = [
    Kernel::MatmulF16,
    Kernel::MatmulQ4KTiled,
    Kernel::MatmulQ5K,
    Kernel::MatmulQ6K,
    Kernel::MatmulQ8,
    Kernel::Logits,
    Kernel::LogitsQ4K,
    Kernel::LogitsQ5K,
    Kernel::LogitsQ6K,
    Kernel::LogitsQ8,
    Kernel::EmbedF16,
    Kernel::EmbedQ4K,
    Kernel::EmbedQ5K,
    Kernel::EmbedQ6K,
    Kernel::EmbedQ8,
    Kernel::RmsNormX,
    Kernel::Rope,
    Kernel::Residual,
    Kernel::KvWrite,
    Kernel::AttentionDecode,
    Kernel::AttentionPrefill,
    Kernel::KvWriteInt8,
    Kernel::AttentionDecodeInt8,
    Kernel::AttentionPrefillInt8,
    Kernel::Argmax,
    Kernel::TopkPartial,
    Kernel::SiluMul,
    Kernel::MatmulQ4KBatchF16Out,
];

impl PipelineRegistry {
    pub(crate) fn build(dev: &Device) -> Result<PipelineRegistry> {
        Self::check_limits(dev)?;
        // SAFETY: `dev.device` is alive; the default cache-create info is valid.
        let cache = unsafe {
            dev.device
                .create_pipeline_cache(&vk::PipelineCacheCreateInfo::default(), None)
        }
        .map_err(|_| eyre!("vulkan: cannot create pipeline cache"))?;

        let mut map = HashMap::new();
        let built = (|| -> Result<()> {
            for &k in &KERNELS {
                map.insert(k, record::build_one(dev, cache, k)?);
            }
            // Capability-gated SPIR-V is built only when the device can execute it.
            if dev.coopmat.available {
                map.insert(
                    Kernel::MatmulQ4KCoopmatF16Out,
                    record::build_one(dev, cache, Kernel::MatmulQ4KCoopmatF16Out)?,
                );
            }
            if dev.dp4a {
                map.insert(
                    Kernel::QuantAQ8F16,
                    record::build_one(dev, cache, Kernel::QuantAQ8F16)?,
                );
                map.insert(
                    Kernel::MatmulQ4KMmvqF16Out,
                    record::build_one(dev, cache, Kernel::MatmulQ4KMmvqF16Out)?,
                );
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
        // KERNELS covers every variant, so the lookup is always present.
        self.map.get(&k).expect("pipeline built for every kernel")
    }

    fn check_limits(dev: &Device) -> Result<()> {
        // The kernels use up to 256-invocation workgroups and 36-byte push blocks
        // (attention_prefill_int8); Vulkan guarantees maxPushConstantsSize >= 128.
        // SAFETY: `dev.instance` is live and `dev.physical` is one of its enumerated devices.
        let l = unsafe {
            dev.instance
                .get_physical_device_properties(dev.physical)
                .limits
        };
        if l.max_compute_work_group_invocations < 256 || l.max_push_constants_size < 36 {
            return Err(eyre!(
                "vulkan: device workgroup/push-constant limits too small"
            ));
        }
        Ok(())
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
