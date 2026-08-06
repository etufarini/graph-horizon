/*
 * graph_horizon_engine — unified-memory placement
 * Enumerates immutable CPU-prefix/Metal-suffix candidates against one shared
 * budget. It performs checked accounting only: no probing, I/O, or allocation.
 */

use color_eyre::eyre::{Result, bail, eyre};

use super::PlacementInput;
use crate::backend::hybrid::weights::model::WeightBytes;
use crate::backend::hybrid::{BackendBytes, HybridPlan};

pub(super) fn select(weights: &WeightBytes, input: PlacementInput) -> Result<HybridPlan> {
    let blocks = weights.layers.len();
    let first_split = if input.gpu_enabled { 0 } else { blocks };
    let mut base_fit = false;
    for split in first_split..=blocks {
        let (cpu_base, gpu_base) = candidate(weights, input, split, false)?;
        if add(cpu_base.total, gpu_base.total)? <= input.cpu_available
            && gpu_base.weights <= input.gpu_weight_available
        {
            base_fit = true;
            let (cpu, gpu) = candidate(weights, input, split, true)?;
            if add(cpu.total, gpu.total)? <= input.cpu_available {
                return HybridPlan::new(split, blocks, cpu, gpu);
            }
        }
    }
    if base_fit {
        bail!(
            "context {} does not fit the selected backend; context was not reduced",
            input.context
        );
    }
    bail!("model does not fit available unified memory")
}

fn candidate(
    weights: &WeightBytes,
    input: PlacementInput,
    split: usize,
    include_kv: bool,
) -> Result<(BackendBytes, BackendBytes)> {
    let blocks = weights.layers.len();
    if split > blocks {
        bail!("invalid hybrid split");
    }
    let mixed = split > 0 && split < blocks;
    let cpu_globals = if split == blocks {
        required(weights.globals.all())?
    } else if mixed {
        weights.globals.embedding
    } else {
        0
    };
    let gpu_globals = if split == 0 {
        required(weights.globals.all())?
    } else if mixed {
        required(weights.globals.tail())?
    } else {
        0
    };
    let crossing = if mixed { input.crossing } else { 0 };
    let gpu_scratch = if mixed {
        input.gpu_mixed_scratch
    } else if split < blocks {
        input.gpu_all_scratch
    } else {
        0
    };
    let cpu = breakdown(
        add(cpu_globals, required(weights.layer_range(0..split))?)?,
        if include_kv {
            mul(input.cpu_kv_per_layer, split)?
        } else {
            0
        },
        if split > 0 { input.cpu_scratch } else { 0 },
        if split == blocks { input.cpu_fixed } else { 0 },
        0,
        0,
        0,
    )?;
    let gpu = breakdown(
        add(gpu_globals, required(weights.layer_range(split..blocks))?)?,
        if include_kv {
            mul(input.gpu_kv_per_layer, blocks - split)?
        } else {
            0
        },
        gpu_scratch,
        if split < blocks {
            add(input.gpu_fixed, input.gpu_host_fixed)?
        } else {
            0
        },
        if split < blocks { input.gpu_staging } else { 0 },
        crossing,
        if split < blocks { input.gpu_reserve } else { 0 },
    )?;
    Ok((cpu, gpu))
}

#[allow(clippy::too_many_arguments)]
fn breakdown(
    weights: u64,
    kv: u64,
    scratch: u64,
    fixed: u64,
    staging: u64,
    crossing: u64,
    reserve: u64,
) -> Result<BackendBytes> {
    Ok(BackendBytes {
        weights,
        kv,
        scratch,
        fixed,
        staging,
        crossing,
        reserve,
        total: sum([weights, kv, scratch, fixed, staging, crossing, reserve])?,
    })
}

fn required(value: Option<u64>) -> Result<u64> {
    value.ok_or_else(overflow)
}

fn sum<const N: usize>(values: [u64; N]) -> Result<u64> {
    values.into_iter().try_fold(0, add)
}

fn add(left: u64, right: u64) -> Result<u64> {
    left.checked_add(right).ok_or_else(overflow)
}

fn mul(left: u64, right: usize) -> Result<u64> {
    left.checked_mul(right as u64).ok_or_else(overflow)
}

fn overflow() -> color_eyre::Report {
    eyre!("hybrid placement arithmetic overflow")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::hybrid::HybridMode;
    use crate::backend::hybrid::weights::model::{GlobalBytes, LayerBytes};

    fn weights() -> WeightBytes {
        WeightBytes {
            globals: GlobalBytes {
                embedding: 20,
                output_norm: 5,
                output: Some(25),
            },
            layers: vec![
                LayerBytes { total: 10, peak: 8 },
                LayerBytes {
                    total: 20,
                    peak: 16,
                },
                LayerBytes {
                    total: 30,
                    peak: 24,
                },
            ],
        }
    }

    fn input(available: u64, weight_cap: u64) -> PlacementInput {
        PlacementInput {
            cpu_available: available,
            gpu_available: available,
            gpu_weight_available: weight_cap,
            gpu_enabled: true,
            context: 4096,
            cpu_kv_per_layer: 2,
            gpu_kv_per_layer: 2,
            cpu_scratch: 20,
            cpu_fixed: 4,
            gpu_host_fixed: 0,
            gpu_all_scratch: 20,
            gpu_mixed_scratch: 20,
            gpu_fixed: 4,
            gpu_staging: 3,
            gpu_reserve: 1,
            crossing: 8,
        }
    }

    #[test]
    fn maximum_metal_order_covers_every_mode() {
        let all = select(&weights(), input(200, 200)).unwrap();
        assert_eq!(all.mode, HybridMode::AllGpu);
        assert_eq!(all.mode.name_for("all-metal"), "all-metal");
        assert_eq!(select(&weights(), input(200, 100)).unwrap().split, 1);
        assert_eq!(select(&weights(), input(200, 60)).unwrap().split, 2);

        let mut disabled = input(200, 0);
        disabled.gpu_enabled = false;
        let cpu = select(&weights(), disabled).unwrap();
        assert_eq!(cpu.mode, HybridMode::CpuOnly);
        assert_eq!(cpu.gpu, BackendBytes::default());

        disabled.cpu_available = 1;
        assert_eq!(
            select(&weights(), disabled).unwrap_err().to_string(),
            "model does not fit available unified memory"
        );
    }

    #[test]
    fn exact_fit_and_single_budget_sum_are_invariants() {
        let plan = select(&weights(), input(u64::MAX, u64::MAX)).unwrap();
        let total = plan.cpu.total.checked_add(plan.gpu.total).unwrap();
        assert_eq!(select(&weights(), input(total, u64::MAX)).unwrap(), plan);
        assert_eq!(plan.gpu.reserve, 1);
        assert_eq!(plan.gpu.staging, 3);
        assert_eq!(plan.cpu.reserve, 0);
    }

    #[test]
    fn base_context_and_overflow_failures_are_distinct() {
        assert_eq!(
            select(&weights(), input(1, u64::MAX))
                .unwrap_err()
                .to_string(),
            "model does not fit available unified memory"
        );
        assert_eq!(
            select(&weights(), input(138, u64::MAX))
                .unwrap_err()
                .to_string(),
            "context 4096 does not fit the selected backend; context was not reduced"
        );
        let mut overflow = input(u64::MAX, u64::MAX);
        overflow.gpu_all_scratch = u64::MAX;
        assert_eq!(
            select(&weights(), overflow).unwrap_err().to_string(),
            "hybrid placement arithmetic overflow"
        );
    }

    #[test]
    fn unified_candidates_use_mode_specific_prefill_bytes() {
        let mut facts = input(u64::MAX, u64::MAX);
        facts.cpu_scratch = 4;
        facts.gpu_all_scratch = 32;
        facts.gpu_mixed_scratch = 4;
        facts.crossing = 7;

        let (all_cpu, all_gpu) = candidate(&weights(), facts, 0, true).unwrap();
        assert_eq!(all_cpu.scratch, 0);
        assert_eq!(all_gpu.scratch, 32);
        assert_eq!(all_gpu.crossing, 0);
        let (mixed_cpu, mixed_gpu) = candidate(&weights(), facts, 1, true).unwrap();
        assert_eq!(mixed_cpu.scratch, 4);
        assert_eq!(mixed_gpu.scratch, 4);
        assert_eq!(mixed_gpu.crossing, 7);
        let (cpu, no_gpu) = candidate(&weights(), facts, 3, true).unwrap();
        assert_eq!(cpu.scratch, 4);
        assert_eq!(no_gpu.total, 0);

        let mut exact = facts;
        exact.cpu_available = add(all_cpu.total, all_gpu.total).unwrap();
        exact.gpu_available = exact.cpu_available;
        assert_eq!(select(&weights(), exact).unwrap().mode, HybridMode::AllGpu);
        exact.cpu_available -= 1;
        exact.gpu_available -= 1;
        assert_ne!(select(&weights(), exact).unwrap().mode, HybridMode::AllGpu);

        let mut cpu_exact = facts;
        cpu_exact.gpu_enabled = false;
        cpu_exact.gpu_available = 0;
        cpu_exact.cpu_available = cpu.total;
        assert_eq!(
            select(&weights(), cpu_exact).unwrap().mode,
            HybridMode::CpuOnly
        );
        cpu_exact.cpu_available -= 1;
        assert!(select(&weights(), cpu_exact).is_err());
    }
}
