/*
 * graph_orizon_engine — transactional hybrid loader
 * Acquires an optional generic GPU, computes one immutable plan, and constructs
 * exactly its CPU/GPU owners from neutral weights and shape. It owns no family
 * graph, runtime traversal, post-plan fallback, or retry.
 */

use color_eyre::eyre::{Result, bail, eyre};

use super::contract::HybridDevice;
use super::placement::{self, BudgetInput, PlacementInput};
use super::weights::model::WeightBytes;
use super::weights::runtime::{RuntimeBytes, RuntimeShape};
use super::{HybridBackends, HybridMode, HybridPlan, HybridRuntime};
use crate::backend::cpu::CpuBackend;
use crate::backend::source::{WeightSelection, WeightSource};
use crate::gguf::loader::GgufFile;
use crate::gguf::metadata::ModelMetadata;
use crate::kv_cache::scheme::KvQuant;

const MIB: u64 = 1024 * 1024;

pub(crate) fn load<G: HybridDevice>(
    file: &GgufFile,
    source: &dyn WeightSource,
    metadata: &ModelMetadata,
    shape: RuntimeShape,
    context: usize,
    scheme: KvQuant,
    weights_percent: Option<u8>,
    reserve_mib: Option<u64>,
) -> Result<HybridRuntime<G>> {
    let weights_percent = weight_percentage::<G>(weights_percent)?;
    if metadata.block_count != shape.block_count {
        bail!("hybrid placement layer count mismatch");
    }
    let mut device = acquire_device::<G>(weights_percent)?;
    let plan = select_plan::<G>(
        source,
        shape,
        context,
        scheme,
        weights_percent,
        reserve_mib,
        G::host_available()?,
        device.as_ref().map(G::budget).transpose()?,
    )?;
    let backends = match plan.mode {
        HybridMode::AllGpu => HybridBackends::AllGpu(G::load_selected(
            device.take().expect("validated all-GPU plan has a device"),
            metadata,
            source,
            file,
            &WeightSelection::full(plan.block_count),
        )?),
        HybridMode::Mixed => {
            let cpu = CpuBackend::load_selected(
                metadata,
                source,
                file,
                &WeightSelection {
                    layers: 0..plan.split,
                    embedding: true,
                    tail: false,
                },
            )?;
            let gpu = G::load_selected(
                device.take().expect("validated mixed plan has a device"),
                metadata,
                source,
                file,
                &WeightSelection {
                    layers: plan.split..plan.block_count,
                    embedding: false,
                    tail: true,
                },
            )?;
            HybridBackends::Mixed { cpu, gpu }
        }
        HybridMode::CpuOnly => HybridBackends::CpuOnly(CpuBackend::load_selected(
            metadata,
            source,
            file,
            &WeightSelection::full(plan.block_count),
        )?),
    };
    eprintln!(
        "hybrid: mode={} cpu_layers={} gpu_layers={} cpu_bytes={} gpu_bytes={}",
        plan.mode.name_for(G::all_mode_name()),
        plan.cpu_layers,
        plan.gpu_layers,
        plan.cpu.total,
        plan.gpu.total
    );
    Ok(HybridRuntime { plan, backends })
}

fn weight_percentage<G: HybridDevice>(value: Option<u8>) -> Result<u8> {
    match value.unwrap_or(100) {
        value @ 0..=100 => Ok(value),
        _ => bail!(G::invalid_percentage_error()),
    }
}

fn acquire_device<G: HybridDevice>(weights_percent: u8) -> Result<Option<G::Device>> {
    if weights_percent == 0 {
        return Ok(None);
    }
    G::acquire()
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn select_plan<G: HybridDevice>(
    source: &dyn WeightSource,
    shape: RuntimeShape,
    context: usize,
    scheme: KvQuant,
    weights_percent: u8,
    reserve_mib: Option<u64>,
    cpu_available: u64,
    budget: Option<BudgetInput>,
) -> Result<HybridPlan> {
    if weights_percent > 100 {
        bail!(G::invalid_percentage_error());
    }
    if shape.block_count != source.groups().layers.len() {
        bail!("hybrid placement layer count mismatch");
    }
    let weights = WeightBytes::from_source(source)?;
    let topology = G::topology();
    let gpu_enabled = weights_percent > 0 && budget.is_some();
    let (cpu_available, gpu_available, reserve) = match (topology, budget) {
        (placement::MemoryTopology::Separate, Some(BudgetInput::Separate { gpu_available })) => (
            cpu_available,
            gpu_available,
            reserve_bytes(gpu_available, reserve_mib)?,
        ),
        (
            placement::MemoryTopology::Unified,
            Some(BudgetInput::Unified {
                physical_memory,
                recommended_working_set,
                current_allocated,
            }),
        ) => {
            let gross = placement::unified_gross(physical_memory, recommended_working_set)
                .ok_or_else(|| eyre!("hybrid placement arithmetic overflow"))?;
            let reserve = reserve_bytes(gross, reserve_mib)?;
            let available = placement::unified_capacity(gross, current_allocated);
            (available, available, reserve)
        }
        (_, None) => (cpu_available, 0, 0),
        _ => return Err(eyre!("hybrid placement topology mismatch")),
    };
    let weight_total = weights
        .globals
        .all()
        .and_then(|globals| {
            weights
                .layer_range(0..weights.layers.len())?
                .checked_add(globals)
        })
        .ok_or_else(|| eyre!("hybrid placement arithmetic overflow"))?;
    let gpu_weight_available = if gpu_enabled {
        ((weight_total as u128 * weights_percent as u128) / 100)
            .min(gpu_available.saturating_sub(reserve) as u128) as u64
    } else {
        0
    };
    let runtime = RuntimeBytes::new(shape, context, scheme)?;
    let fixed = G::fixed_bytes(&shape)?;
    placement::select(
        topology,
        &weights,
        PlacementInput {
            cpu_available,
            gpu_available,
            gpu_weight_available,
            gpu_enabled,
            context,
            cpu_kv_per_layer: runtime.kv_per_layer,
            gpu_kv_per_layer: runtime.kv_per_layer,
            cpu_scratch: runtime.scratch,
            cpu_fixed: runtime.logits,
            gpu_host_fixed: fixed.host,
            gpu_scratch: runtime.scratch,
            gpu_fixed: fixed.device,
            gpu_staging: fixed.staging,
            gpu_reserve: reserve,
            crossing: runtime.crossing,
        },
    )
}

fn reserve_bytes(gpu_available: u64, override_mib: Option<u64>) -> Result<u64> {
    match override_mib {
        Some(mib) => mib
            .checked_mul(MIB)
            .ok_or_else(|| eyre!("hybrid placement arithmetic overflow")),
        None => Ok((256 * MIB).max(gpu_available / 20)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reserve_override_and_default_are_checked() {
        assert_eq!(reserve_bytes(16 << 30, None).unwrap(), (16 << 30) / 20);
        assert_eq!(reserve_bytes(1 << 30, None).unwrap(), 256 * MIB);
        assert!(reserve_bytes(u64::MAX, Some(u64::MAX)).is_err());
    }

    #[test]
    fn unified_gross_is_checked_and_uses_the_smaller_limit() {
        assert_eq!(placement::unified_gross(100, 95), Some(90));
        assert_eq!(placement::unified_gross(100, 80), Some(80));
        assert_eq!(placement::unified_gross(u64::MAX, u64::MAX), None);
        // Current allocation reduces the shared capacity; reserve remains one
        // explicit GPU report category and is therefore not subtracted here.
        assert_eq!(placement::unified_capacity(90, 20), 70);
        assert_eq!(placement::unified_capacity(90, 91), 0);
    }

    #[cfg(feature = "vulkan-hybrid")]
    #[test]
    fn explicit_zero_skips_the_vulkan_probe() {
        use crate::backend::vulkan::VulkanBackend;

        crate::backend::vulkan::reset_probe_count();
        assert!(acquire_device::<VulkanBackend>(0).unwrap().is_none());
        assert_eq!(crate::backend::vulkan::probe_count(), 0);
    }

    #[cfg(feature = "metal-hybrid")]
    #[test]
    fn explicit_zero_skips_the_metal_probe() {
        use crate::backend::metal::MetalBackend;

        crate::backend::metal::reset_probe_count();
        assert!(acquire_device::<MetalBackend>(0).unwrap().is_none());
        assert_eq!(crate::backend::metal::probe_count(), 0);
    }

    #[cfg(feature = "metal-hybrid")]
    #[test]
    fn invalid_metal_percentage_precedes_the_probe() {
        use crate::backend::metal::MetalBackend;

        crate::backend::metal::reset_probe_count();
        assert_eq!(weight_percentage::<MetalBackend>(None).unwrap(), 100);
        assert_eq!(
            weight_percentage::<MetalBackend>(Some(101))
                .unwrap_err()
                .to_string(),
            "invalid Metal weight percentage"
        );
        assert_eq!(crate::backend::metal::probe_count(), 0);
    }
}
