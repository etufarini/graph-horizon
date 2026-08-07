/*
 * graph_horizon_engine — compute pipeline registry
 * Owns transactional construction, lookup, and destruction of the reachable
 * compute pipelines. The registry builds its unconditional set plus the
 * capability-gated wide-attention, FP16 coopmat, and MMVQ variants.
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
const KERNELS: [Kernel; 26] = [
    Kernel::MatmulF16,
    Kernel::MatmulQ4KTiled,
    Kernel::MatmulQ5K,
    Kernel::MatmulQ6K,
    Kernel::Logits,
    Kernel::LogitsQ4K,
    Kernel::LogitsQ5K,
    Kernel::LogitsQ6K,
    Kernel::EmbedF16,
    Kernel::EmbedQ4K,
    Kernel::EmbedQ5K,
    Kernel::EmbedQ6K,
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
    Kernel::MatmulQ6KBatchF16Out,
];

const WIDE_ATTENTION_SHARED_BYTES: u32 = 32 * 128 * 4 + 32 * 4 * 2;

fn supports_wide_attention(invocations: u32, size_x: u32, shared_bytes: u32) -> bool {
    invocations >= 512 && size_x >= 512 && shared_bytes >= WIDE_ATTENTION_SHARED_BYTES
}

impl PipelineRegistry {
    pub(crate) fn build(dev: &Device) -> Result<PipelineRegistry> {
        let wide_attention = Self::check_limits(dev)?;
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
            if wide_attention {
                for k in [Kernel::AttentionDecodeWide, Kernel::AttentionPrefillWide] {
                    map.insert(k, record::build_one(dev, cache, k)?);
                }
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
        // Dispatch selects capability-gated variants only after `contains` succeeds.
        self.map.get(&k).expect("pipeline built for every kernel")
    }

    pub(crate) fn contains(&self, k: Kernel) -> bool {
        self.map.contains_key(&k)
    }

    fn check_limits(dev: &Device) -> Result<bool> {
        // Required kernels use up to 256 invocations; the optional wide-attention
        // pair is built only at 512+. Vulkan guarantees maxPushConstantsSize >= 128.
        // SAFETY: `dev.instance` is live and `dev.physical` is one of its enumerated devices.
        let l = unsafe {
            dev.instance
                .get_physical_device_properties(dev.physical)
                .limits
        };
        if l.max_compute_work_group_invocations < 256
            || l.max_compute_work_group_size[0] < 256
            || l.max_push_constants_size < 36
        {
            return Err(eyre!(
                "vulkan: device workgroup/push-constant limits too small"
            ));
        }
        Ok(supports_wide_attention(
            l.max_compute_work_group_invocations,
            l.max_compute_work_group_size[0],
            l.max_compute_shared_memory_size,
        ))
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

#[cfg(test)]
mod tests {
    use super::{WIDE_ATTENTION_SHARED_BYTES, supports_wide_attention};

    #[test]
    fn wide_attention_requires_every_resource_limit() {
        assert!(supports_wide_attention(
            512,
            512,
            WIDE_ATTENTION_SHARED_BYTES
        ));
        assert!(!supports_wide_attention(
            511,
            512,
            WIDE_ATTENTION_SHARED_BYTES
        ));
        assert!(!supports_wide_attention(
            512,
            511,
            WIDE_ATTENTION_SHARED_BYTES
        ));
        assert!(!supports_wide_attention(
            512,
            512,
            WIDE_ATTENTION_SHARED_BYTES - 1
        ));
    }
}
