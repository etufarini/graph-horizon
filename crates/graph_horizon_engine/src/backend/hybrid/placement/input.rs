/*
 * graph_horizon_engine — checked hybrid placement input
 * Converts acquired host/device budgets and one runtime shape into immutable,
 * candidate-specific byte facts. It performs no probing, allocation, loading,
 * graph traversal, or placement selection.
 */

use color_eyre::eyre::{Result, bail, eyre};

use super::{BudgetInput, MemoryTopology};
use crate::backend::hybrid::contract::HybridDevice;
use crate::backend::hybrid::weights::model::WeightBytes;
use crate::backend::hybrid::weights::runtime::{RuntimeBytes, RuntimeShape};
use crate::backend::source::WeightSource;
use crate::kv_cache::scheme::KvQuant;

const MIB: u64 = 1024 * 1024;

#[derive(Clone, Copy, Debug)]
pub(crate) struct PlacementInput {
    pub(crate) cpu_available: u64,
    pub(crate) gpu_available: u64,
    pub(crate) gpu_weight_available: u64,
    pub(crate) gpu_enabled: bool,
    pub(crate) context: usize,
    pub(crate) cpu_kv_per_layer: u64,
    pub(crate) gpu_kv_per_layer: u64,
    pub(crate) cpu_scratch: u64,
    pub(crate) cpu_fixed: u64,
    pub(crate) gpu_host_fixed: u64,
    pub(crate) gpu_all_scratch: u64,
    pub(crate) gpu_mixed_scratch: u64,
    pub(crate) gpu_fixed: u64,
    pub(crate) gpu_staging: u64,
    pub(crate) gpu_reserve: u64,
    pub(crate) crossing: u64,
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn build<G: HybridDevice>(
    source: &dyn WeightSource,
    shape: RuntimeShape,
    context: usize,
    scheme: KvQuant,
    weights_percent: u8,
    reserve_mib: Option<u64>,
    host_available: u64,
    budget: Option<BudgetInput>,
) -> Result<(MemoryTopology, WeightBytes, PlacementInput)> {
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
        (MemoryTopology::Separate, Some(BudgetInput::Separate { gpu_available })) => (
            host_available,
            gpu_available,
            reserve_bytes(gpu_available, reserve_mib)?,
        ),
        (
            MemoryTopology::Unified,
            Some(BudgetInput::Unified {
                physical_memory,
                recommended_working_set,
                current_allocated,
            }),
        ) => {
            let gross = super::unified_gross(physical_memory, recommended_working_set)
                .ok_or_else(overflow)?;
            let available = super::unified_capacity(gross, current_allocated);
            (available, available, reserve_bytes(gross, reserve_mib)?)
        }
        (_, None) => (host_available, 0, 0),
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
        .ok_or_else(overflow)?;
    let gpu_weight_available = if gpu_enabled {
        ((weight_total as u128 * weights_percent as u128) / 100)
            .min(gpu_available.saturating_sub(reserve) as u128) as u64
    } else {
        0
    };
    let cpu = RuntimeBytes::new(shape, context, scheme, shape.cpu_prefill_rows)?;
    let gpu = RuntimeBytes::new(shape, context, scheme, shape.gpu_prefill_rows)?;
    let mixed = RuntimeBytes::new(shape, context, scheme, shape.mixed_prefill_rows)?;
    let fixed = G::fixed_bytes(&shape)?;
    Ok((
        topology,
        weights,
        PlacementInput {
            cpu_available,
            gpu_available,
            gpu_weight_available,
            gpu_enabled,
            context,
            cpu_kv_per_layer: cpu.kv_per_layer,
            gpu_kv_per_layer: gpu.kv_per_layer,
            cpu_scratch: cpu.scratch,
            cpu_fixed: cpu.logits,
            gpu_host_fixed: fixed.host,
            gpu_all_scratch: gpu.scratch,
            gpu_mixed_scratch: mixed.scratch,
            gpu_fixed: fixed.device,
            gpu_staging: fixed.staging,
            gpu_reserve: reserve,
            crossing: mixed.crossing,
        },
    ))
}

fn reserve_bytes(available: u64, override_mib: Option<u64>) -> Result<u64> {
    override_mib
        .map(|mib| mib.checked_mul(MIB).ok_or_else(overflow))
        .unwrap_or_else(|| Ok((256 * MIB).max(available / 20)))
}

fn overflow() -> color_eyre::Report {
    eyre!("hybrid placement arithmetic overflow")
}

#[cfg(all(test, feature = "vulkan-hybrid"))]
mod tests {
    use super::*;
    use crate::backend::source::{WeightGroups, WeightSource};
    use crate::backend::vulkan::VulkanBackend;
    use crate::gguf::tensor_index::{GgmlType, TensorInfo};

    struct Source(Vec<TensorInfo>);

    impl WeightSource for Source {
        fn groups(&self) -> WeightGroups<'_> {
            WeightGroups::new(
                &self.0[0],
                &self.0[1],
                None,
                vec![vec![&self.0[2]], vec![&self.0[3]]],
            )
        }
    }

    fn tensor(name: &str, ty: GgmlType, dims: &[u64]) -> TensorInfo {
        TensorInfo {
            name: name.into(),
            dims: dims.into(),
            ggml_type: ty,
            offset: 0,
        }
    }

    fn shape() -> RuntimeShape {
        RuntimeShape {
            block_count: 2,
            embedding: 8,
            q: 8,
            k: 4,
            v: 4,
            attention: 8,
            feed_forward: 16,
            vocab: 32,
            kv_heads: 1,
            key_length: 4,
            value_length: 4,
            cpu_prefill_rows: 4,
            gpu_prefill_rows: 32,
            mixed_prefill_rows: 4,
        }
    }

    #[test]
    fn placement_input_preserves_probe_independent_arithmetic() {
        let source = Source(vec![
            tensor("embedding", GgmlType::Q4_K, &[256]),
            tensor("norm", GgmlType::F32, &[8]),
            tensor("layer.0", GgmlType::Q6_K, &[256]),
            tensor("layer.1", GgmlType::F16, &[16]),
        ]);
        crate::backend::vulkan::reset_probe_count();
        let (topology, weights, input) = build::<VulkanBackend>(
            &source,
            shape(),
            16,
            KvQuant::F16,
            100,
            Some(0),
            10_000,
            Some(BudgetInput::Separate {
                gpu_available: 20_000,
            }),
        )
        .unwrap();
        assert_eq!(topology, MemoryTopology::Separate);
        assert_eq!(weights.layers.len(), 2);
        assert_eq!(input.cpu_available, 10_000);
        assert_eq!(input.gpu_available, 20_000);
        assert_eq!(input.gpu_reserve, 0);
        assert!(input.gpu_all_scratch > input.gpu_mixed_scratch);
        assert!(input.crossing > 0);
        assert_eq!(crate::backend::vulkan::probe_count(), 0);
        assert!(reserve_bytes(u64::MAX, Some(u64::MAX)).is_err());
    }
}
