/*
 * gh_zero_engine — deterministic Ministral hybrid placement
 * Evaluates every legal contiguous split in maximum-GPU order using pure,
 * checked byte arithmetic. It distinguishes model/base failure (E16) from
 * requested-context failure (E17) and performs no probing, I/O, or allocation.
 */

use color_eyre::eyre::{Result, bail, eyre};

use super::weights::WeightBytes;
use super::{BackendBytes, HybridPlan};

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
    pub(crate) gpu_scratch: u64,
    pub(crate) gpu_fixed: u64,
    pub(crate) gpu_reserve: u64,
    pub(crate) crossing: u64,
}

#[derive(Clone, Copy)]
struct Candidate {
    cpu_base: BackendBytes,
    gpu_base: BackendBytes,
    cpu_full: BackendBytes,
    gpu_full: BackendBytes,
}

pub(crate) fn select(weights: &WeightBytes, input: PlacementInput) -> Result<HybridPlan> {
    let block_count = weights.layers.len();
    let splits: Box<dyn Iterator<Item = usize>> = if input.gpu_enabled {
        Box::new(0..=block_count)
    } else {
        Box::new(std::iter::once(block_count))
    };
    let mut base_fit = false;
    for split in splits {
        let candidate = candidate(weights, input, split)?;
        if candidate.cpu_base.total <= input.cpu_available
            && candidate.gpu_base.total <= input.gpu_available
            && candidate.gpu_base.weights <= input.gpu_weight_available
        {
            base_fit = true;
            if candidate.cpu_full.total <= input.cpu_available
                && candidate.gpu_full.total <= input.gpu_available
            {
                // Iteration order is a load-bearing invariant: the first fit
                // owns the maximum possible contiguous Vulkan suffix.
                return Ok(HybridPlan::new(
                    split,
                    block_count,
                    candidate.cpu_full,
                    candidate.gpu_full,
                ));
            }
        }
    }
    if base_fit {
        bail!(
            "context {} does not fit the selected backend; context was not reduced",
            input.context
        );
    }
    bail!("model does not fit available RAM and VRAM")
}

fn candidate(weights: &WeightBytes, input: PlacementInput, split: usize) -> Result<Candidate> {
    let block_count = weights.layers.len();
    if split > block_count {
        return Err(eyre!("invalid hybrid split"));
    }
    let mixed = split > 0 && split < block_count;
    let has_cpu = split > 0;
    let has_gpu = split < block_count;
    let cpu_globals = if split == block_count {
        weights.globals.all()
    } else if mixed {
        Some(weights.globals.embedding)
    } else {
        Some(0)
    };
    let gpu_globals = if split == 0 {
        weights.globals.all()
    } else if mixed {
        weights.globals.tail()
    } else {
        Some(0)
    };
    let cpu_layers = weights.layer_range(0..split);
    let gpu_layers = weights.layer_range(split..block_count);
    let host_staging = required(weights.gpu_peak(split))?;
    let cpu_base = breakdown(
        add(required(cpu_globals)?, required(cpu_layers)?)?,
        0,
        if has_cpu { input.cpu_scratch } else { 0 },
        add(
            if split == block_count {
                input.cpu_fixed
            } else {
                0
            },
            if has_gpu { input.gpu_host_fixed } else { 0 },
        )?,
        host_staging,
        if mixed { input.crossing } else { 0 },
        0,
    )?;
    let gpu_base = breakdown(
        add(required(gpu_globals)?, required(gpu_layers)?)?,
        0,
        if has_gpu { input.gpu_scratch } else { 0 },
        if has_gpu { input.gpu_fixed } else { 0 },
        0,
        if mixed { input.crossing } else { 0 },
        if has_gpu { input.gpu_reserve } else { 0 },
    )?;
    let cpu_kv = mul(input.cpu_kv_per_layer, split)?;
    let gpu_kv = mul(input.gpu_kv_per_layer, block_count - split)?;
    Ok(Candidate {
        cpu_base,
        gpu_base,
        cpu_full: with_kv(cpu_base, cpu_kv)?,
        gpu_full: with_kv(gpu_base, gpu_kv)?,
    })
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

fn with_kv(mut bytes: BackendBytes, kv: u64) -> Result<BackendBytes> {
    bytes.kv = kv;
    bytes.total = add(bytes.total, kv)?;
    Ok(bytes)
}

fn required(value: Option<u64>) -> Result<u64> {
    value.ok_or_else(|| eyre!("hybrid placement arithmetic overflow"))
}

fn sum<const N: usize>(values: [u64; N]) -> Result<u64> {
    values.into_iter().try_fold(0u64, add)
}

fn add(left: u64, right: u64) -> Result<u64> {
    left.checked_add(right)
        .ok_or_else(|| eyre!("hybrid placement arithmetic overflow"))
}

fn mul(left: u64, right: usize) -> Result<u64> {
    left.checked_mul(
        u64::try_from(right).map_err(|_| eyre!("hybrid placement arithmetic overflow"))?,
    )
    .ok_or_else(|| eyre!("hybrid placement arithmetic overflow"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::family::mistral::hybrid::HybridMode;
    use crate::family::mistral::hybrid::weights::{GlobalBytes, LayerBytes};

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

    fn input(cpu: u64, gpu: u64) -> PlacementInput {
        PlacementInput {
            cpu_available: cpu,
            gpu_available: gpu,
            gpu_weight_available: gpu,
            gpu_enabled: true,
            context: 4096,
            cpu_kv_per_layer: 2,
            gpu_kv_per_layer: 2,
            cpu_scratch: 3,
            cpu_fixed: 4,
            gpu_host_fixed: 0,
            gpu_scratch: 3,
            gpu_fixed: 4,
            gpu_reserve: 1,
            crossing: 8,
        }
    }

    #[test]
    fn selects_all_gpu_then_first_fitting_mixed_split() {
        let all = select(&weights(), input(30, 130)).unwrap();
        assert_eq!(all.mode, HybridMode::AllGpu);
        assert_eq!((all.cpu_layers, all.gpu_layers), (0, 3));

        let mixed = select(&weights(), input(80, 100)).unwrap();
        assert_eq!(mixed.mode, HybridMode::Mixed);
        assert_eq!(mixed.split, 1);
        assert_eq!((mixed.cpu_layers, mixed.gpu_layers), (1, 2));
    }

    #[test]
    fn supports_one_layer_suffix_and_heterogeneous_costs() {
        let plan = select(&weights(), input(120, 78)).unwrap();
        assert_eq!(plan.split, 2);
        assert_eq!(plan.gpu_layers, 1);
    }

    #[test]
    fn unavailable_or_explicit_zero_gpu_selects_cpu_only() {
        let mut unavailable = input(123, 0);
        unavailable.gpu_enabled = false;
        let plan = select(&weights(), unavailable).unwrap();
        assert_eq!(plan.mode, HybridMode::CpuOnly);
        assert_eq!(plan.gpu.total, 0);
    }

    #[test]
    fn exact_fit_is_accepted() {
        let first = select(&weights(), input(u64::MAX, u64::MAX)).unwrap();
        let mut exact_input = input(first.cpu.total, first.gpu.total);
        exact_input.gpu_weight_available = first.gpu.weights;
        let exact = select(&weights(), exact_input).unwrap();
        assert_eq!(exact, first);
    }

    #[test]
    fn weight_cap_does_not_reduce_kv_or_fixed_vram_budget() {
        let mut capped = input(123, 130);
        capped.gpu_weight_available = 80;
        let plan = select(&weights(), capped).unwrap();
        assert_eq!(plan.split, 1);
        assert_eq!(plan.gpu.weights, 80);
        assert_eq!(plan.gpu.total, 100);
    }

    #[test]
    fn reports_exact_backend_breakdown() {
        let plan = select(&weights(), input(80, 100)).unwrap();
        assert_eq!(plan.split, 1);
        assert_eq!(
            plan.cpu,
            BackendBytes {
                weights: 30,
                kv: 2,
                scratch: 3,
                staging: 25,
                crossing: 8,
                total: 68,
                ..BackendBytes::default()
            }
        );
        assert_eq!(
            plan.gpu,
            BackendBytes {
                weights: 80,
                kv: 4,
                scratch: 3,
                fixed: 4,
                crossing: 8,
                reserve: 1,
                total: 100,
                ..BackendBytes::default()
            }
        );
    }

    #[test]
    fn error_matrix_e16_e17_distinguishes_base_and_context_failure() {
        let e16 = select(&weights(), input(1, 1)).unwrap_err().to_string();
        assert_eq!(e16, "model does not fit available RAM and VRAM");

        let mut context = input(120, 120);
        context.cpu_kv_per_layer = 1_000;
        context.gpu_kv_per_layer = 1_000;
        let e17 = select(&weights(), context).unwrap_err().to_string();
        assert_eq!(
            e17,
            "context 4096 does not fit the selected backend; context was not reduced"
        );
    }

    #[test]
    fn overflow_is_rejected_without_wrapping() {
        let mut overflow = input(u64::MAX, u64::MAX);
        overflow.gpu_fixed = u64::MAX;
        assert!(
            select(&weights(), overflow)
                .unwrap_err()
                .to_string()
                .contains("overflow")
        );
    }
}
