/*
 * gh_zero_engine — transactional hybrid loader
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
    if metadata.block_count != shape.block_count {
        bail!("hybrid placement layer count mismatch");
    }
    let weights_percent = weights_percent.unwrap_or(100);
    if weights_percent > 100 {
        bail!(G::invalid_percentage_error());
    }
    let mut device = acquire_device::<G>(weights_percent)?;
    let plan = select_plan::<G>(
        source,
        shape,
        context,
        scheme,
        weights_percent,
        reserve_mib,
        available_ram(),
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
    let gpu_available = match budget {
        Some(BudgetInput::Separate { gpu_available }) => gpu_available,
        Some(BudgetInput::Unified { .. }) => {
            return Err(eyre!("unified hybrid placement is unavailable"));
        }
        None => 0,
    };
    let gpu_enabled = weights_percent > 0 && budget.is_some();
    let reserve = if gpu_enabled {
        reserve_bytes(gpu_available, reserve_mib)?
    } else {
        0
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
        G::topology(),
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
            gpu_reserve: reserve,
            crossing: runtime.crossing,
        },
    )
}

fn available_ram() -> u64 {
    std::fs::read_to_string("/proc/meminfo")
        .ok()
        .and_then(|text| parse_mem_available(&text))
        .map(|available| ((available as u128 * 90) / 100) as u64)
        .unwrap_or(0)
}

fn parse_mem_available(text: &str) -> Option<u64> {
    let mut found = None;
    for line in text.lines() {
        let Some((key, raw)) = line.split_once(':') else {
            continue;
        };
        if key != "MemAvailable" {
            continue;
        }
        if found.is_some() {
            return None;
        }
        let mut fields = raw.split_whitespace();
        let kib = fields.next()?.parse::<u64>().ok()?;
        if fields.next()? != "kB" || fields.next().is_some() {
            return None;
        }
        found = kib.checked_mul(1024);
    }
    found
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
    fn mem_available_parser_is_strict_and_checked() {
        assert_eq!(
            parse_mem_available("MemAvailable: 1234 kB\n"),
            Some(1_263_616)
        );
        assert_eq!(parse_mem_available("MemTotal: 1234 kB\n"), None);
        assert_eq!(parse_mem_available("MemAvailable: 1 MB\n"), None);
        assert_eq!(
            parse_mem_available("MemAvailable: 1 kB\nMemAvailable: 2 kB\n"),
            None
        );
    }

    #[test]
    fn reserve_override_and_default_are_checked() {
        assert_eq!(reserve_bytes(16 << 30, None).unwrap(), (16 << 30) / 20);
        assert_eq!(reserve_bytes(1 << 30, None).unwrap(), 256 * MIB);
        assert!(reserve_bytes(u64::MAX, Some(u64::MAX)).is_err());
    }

    #[cfg(feature = "vulcan-hybrid")]
    #[test]
    fn explicit_zero_skips_the_vulkan_probe() {
        use crate::backend::vulkan::VulkanBackend;

        crate::backend::vulkan::reset_probe_count();
        assert!(acquire_device::<VulkanBackend>(0).unwrap().is_none());
        assert_eq!(crate::backend::vulkan::probe_count(), 0);
    }
}
