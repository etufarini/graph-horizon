/*
 * graph_horizon_engine — compile-time backend selection
 * Maps one explicit public profile to its resource owner, request session, and
 * setup hooks. It performs no target inference, runtime detection, or graph work.
 */

use color_eyre::eyre::Result;

#[cfg(any(feature = "vulkan", feature = "metal"))]
use super::Backend;
#[cfg(any(feature = "vulkan-hybrid", feature = "metal-hybrid"))]
use super::hybrid::HybridPlan;
use super::hybrid::weights::runtime::RuntimeShape;
use super::source::WeightSource;
use crate::gguf::loader::GgufFile;
use crate::gguf::metadata::ModelMetadata;
use crate::kv_cache::scheme::KvQuant;
use crate::runtime::contract::LayeredGraph;

#[cfg(feature = "cpu")]
pub(crate) type SelectedBackend = super::cpu::CpuBackend;
#[cfg(feature = "vulkan")]
pub(crate) type SelectedBackend = super::vulkan::VulkanBackend;
#[cfg(feature = "vulkan-hybrid")]
pub(crate) type SelectedBackend = super::hybrid::HybridRuntime<super::vulkan::VulkanBackend>;
#[cfg(feature = "metal")]
pub(crate) type SelectedBackend = super::metal::MetalBackend;
#[cfg(feature = "metal-hybrid")]
pub(crate) type SelectedBackend = super::hybrid::HybridRuntime<super::metal::MetalBackend>;

#[cfg(feature = "cpu")]
pub(crate) type SelectedSession<'a, G> =
    crate::runtime::homogeneous::HomogeneousSession<'a, super::cpu::CpuBackend, G>;
#[cfg(feature = "vulkan")]
pub(crate) type SelectedSession<'a, G> =
    crate::runtime::homogeneous::HomogeneousSession<'a, super::vulkan::VulkanBackend, G>;
#[cfg(any(feature = "vulkan", feature = "metal"))]
pub(crate) type CachedState = crate::kv_cache::Kv<<SelectedBackend as Backend>::Buffer>;
#[cfg(feature = "vulkan-hybrid")]
pub(crate) type SelectedSession<'a, G> =
    crate::runtime::partitioned::PartitionedSession<'a, super::vulkan::VulkanBackend, G>;
#[cfg(feature = "metal")]
pub(crate) type SelectedSession<'a, G> =
    crate::runtime::homogeneous::HomogeneousSession<'a, super::metal::MetalBackend, G>;
#[cfg(feature = "metal-hybrid")]
pub(crate) type SelectedSession<'a, G> =
    crate::runtime::partitioned::PartitionedSession<'a, super::metal::MetalBackend, G>;

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
    #[cfg(feature = "cpu")]
    {
        let _ = (shape, scheme, weights_percent, reserve_mib);
        super::cpu::load(metadata, source, file, context)
    }
    #[cfg(feature = "vulkan")]
    {
        let _ = (shape, scheme);
        super::vulkan::load(
            metadata,
            source,
            file,
            context,
            weights_percent,
            reserve_mib,
        )
    }
    #[cfg(feature = "metal")]
    {
        super::metal::load(
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
    #[cfg(any(feature = "vulkan-hybrid", feature = "metal-hybrid"))]
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
    #[cfg(any(feature = "cpu", feature = "vulkan", feature = "metal"))]
    {
        #[cfg(feature = "cpu")]
        let row_capacity = shape.cpu_prefill_rows;
        #[cfg(feature = "vulkan")]
        let row_capacity = backend.prefill_rows(shape.block_count, context);
        #[cfg(feature = "metal")]
        let row_capacity = super::metal::PREFILL_ROWS;
        crate::runtime::homogeneous::HomogeneousSession::new(
            backend,
            config,
            shape,
            row_capacity,
            context,
            scheme,
        )
    }
    #[cfg(any(feature = "vulkan-hybrid", feature = "metal-hybrid"))]
    {
        crate::runtime::partitioned::PartitionedSession::new(
            backend, config, shape, context, scheme,
        )
    }
}

#[cfg(any(feature = "vulkan", feature = "metal"))]
pub(crate) fn cached_session<'a, G: LayeredGraph>(
    backend: &'a SelectedBackend,
    config: &'a G::Config,
    shape: RuntimeShape,
    context: usize,
    scheme: KvQuant,
    state: Option<CachedState>,
) -> Result<SelectedSession<'a, G>> {
    #[cfg(feature = "vulkan")]
    let row_capacity = backend.prefill_rows(shape.block_count, context);
    #[cfg(feature = "metal")]
    let row_capacity = super::metal::PREFILL_ROWS;
    crate::runtime::homogeneous::HomogeneousSession::with_state(
        backend,
        config,
        shape,
        row_capacity,
        context,
        scheme,
        state,
    )
}

#[cfg(any(feature = "vulkan", feature = "metal"))]
pub(crate) fn free_cached_state(backend: &SelectedBackend, state: CachedState) {
    crate::kv_cache::free(backend, state);
}

pub(crate) fn configure(cpu_threads: Option<usize>) {
    #[cfg(any(feature = "cpu", feature = "vulkan-hybrid", feature = "metal-hybrid"))]
    {
        super::cpu::parallel::set_threads(cpu_threads);
    }
    #[cfg(not(any(feature = "cpu", feature = "vulkan-hybrid", feature = "metal-hybrid")))]
    let _ = cpu_threads;
}

#[cfg(any(feature = "vulkan-hybrid", feature = "metal-hybrid"))]
pub(crate) fn placement(backend: &SelectedBackend) -> Option<&HybridPlan> {
    Some(&backend.plan)
}

#[cfg(any(feature = "vulkan-hybrid", feature = "metal-hybrid"))]
pub(crate) fn placement_mode(mode: super::hybrid::HybridMode) -> &'static str {
    #[cfg(feature = "vulkan-hybrid")]
    {
        mode.name_for("all-gpu")
    }
    #[cfg(feature = "metal-hybrid")]
    {
        mode.name_for("all-metal")
    }
}

#[cfg(all(test, any(feature = "vulkan-hybrid", feature = "metal-hybrid")))]
mod tests {
    use super::*;
    use crate::backend::hybrid::HybridMode;

    #[test]
    fn selected_hybrid_uses_its_public_homogeneous_label() {
        #[cfg(feature = "vulkan-hybrid")]
        assert_eq!(placement_mode(HybridMode::AllGpu), "all-gpu");
        #[cfg(feature = "metal-hybrid")]
        assert_eq!(placement_mode(HybridMode::AllGpu), "all-metal");
        assert_eq!(placement_mode(HybridMode::Mixed), "mixed");
        assert_eq!(placement_mode(HybridMode::CpuOnly), "cpu-only");
    }
}
