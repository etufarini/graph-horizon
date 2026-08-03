/*
 * gh_zero_engine — compile-time backend selection
 * Maps one explicit public profile to its resource owner, request session, and
 * setup hooks. It performs no target inference, runtime detection, or graph work.
 */

use color_eyre::eyre::Result;

#[cfg(any(feature = "cpu", feature = "vulcan"))]
use super::Backend;
use super::hybrid::HybridPlan;
use super::hybrid::weights::runtime::RuntimeShape;
use super::source::WeightSource;
use crate::gguf::loader::GgufFile;
use crate::gguf::metadata::ModelMetadata;
use crate::kv_cache::scheme::KvQuant;
use crate::runtime::contract::LayeredGraph;

#[cfg(feature = "cpu")]
pub(crate) type SelectedBackend = super::cpu::CpuBackend;
#[cfg(feature = "vulcan")]
pub(crate) type SelectedBackend = super::vulkan::VulkanBackend;
#[cfg(feature = "vulcan-hybrid")]
pub(crate) type SelectedBackend = super::hybrid::HybridRuntime<super::vulkan::VulkanBackend>;

#[cfg(feature = "cpu")]
pub(crate) type SelectedSession<'a, G> =
    crate::runtime::homogeneous::HomogeneousSession<'a, super::cpu::CpuBackend, G>;
#[cfg(feature = "vulcan")]
pub(crate) type SelectedSession<'a, G> =
    crate::runtime::homogeneous::HomogeneousSession<'a, super::vulkan::VulkanBackend, G>;
#[cfg(feature = "vulcan-hybrid")]
pub(crate) type SelectedSession<'a, G> =
    crate::runtime::partitioned::PartitionedSession<'a, super::vulkan::VulkanBackend, G>;

#[allow(clippy::too_many_arguments)]
pub(crate) fn load(
    file: &GgufFile,
    source: &dyn WeightSource,
    metadata: &ModelMetadata,
    shape: RuntimeShape,
    context: usize,
    scheme: KvQuant,
    weights_percent: Option<u8>,
    reserve_mib: Option<u64>,
) -> Result<SelectedBackend> {
    #[cfg(any(feature = "cpu", feature = "vulcan"))]
    {
        let _ = (shape, scheme, weights_percent, reserve_mib);
        SelectedBackend::load(metadata, source, file, context)
    }
    #[cfg(feature = "vulcan-hybrid")]
    {
        super::hybrid::loader::load(
            file,
            source,
            metadata,
            shape,
            context,
            scheme,
            weights_percent,
            reserve_mib,
        )
    }
}

pub(crate) fn session<'a, G: LayeredGraph>(
    backend: &'a SelectedBackend,
    config: &'a G::Config,
    shape: RuntimeShape,
    context: usize,
    scheme: KvQuant,
) -> Result<SelectedSession<'a, G>> {
    #[cfg(any(feature = "cpu", feature = "vulcan"))]
    {
        crate::runtime::homogeneous::HomogeneousSession::new(
            backend, config, shape, context, scheme,
        )
    }
    #[cfg(feature = "vulcan-hybrid")]
    {
        crate::runtime::partitioned::PartitionedSession::new(
            backend, config, shape, context, scheme,
        )
    }
}

pub(crate) fn configure(
    cpu_threads: Option<usize>,
    no_attn_simd: bool,
    weights_percent: Option<u8>,
    reserve_mib: Option<u64>,
) {
    #[cfg(any(feature = "cpu", feature = "vulcan-hybrid"))]
    {
        super::cpu::parallel::set_threads(cpu_threads);
        super::cpu::set_no_simd(no_attn_simd);
    }
    #[cfg(feature = "vulcan")]
    {
        super::vulkan::set_weights_percent(weights_percent);
        super::vulkan::set_reserve_mib(reserve_mib);
    }
    #[cfg(not(feature = "vulcan"))]
    let _ = (weights_percent, reserve_mib);
    #[cfg(not(any(feature = "cpu", feature = "vulcan-hybrid")))]
    let _ = (cpu_threads, no_attn_simd);
}

pub(crate) fn placement(backend: &SelectedBackend) -> Option<&HybridPlan> {
    #[cfg(feature = "vulcan-hybrid")]
    {
        Some(&backend.plan)
    }
    #[cfg(not(feature = "vulcan-hybrid"))]
    {
        let _ = backend;
        None
    }
}
