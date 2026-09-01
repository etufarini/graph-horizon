/*
 * graph_horizon_engine — checked standalone CUDA memory planning and preflight.
 * The calculation is pure: it cannot allocate, reduce context, or change KV.
 */

use color_eyre::eyre::{Result, bail, eyre};

use crate::backend::hybrid::weights::runtime::{RuntimeBytes, RuntimeShape};
use crate::backend::source::WeightSource;
use crate::gguf::tensor_index::GgmlType;
use crate::kv_cache::scheme::KvQuant;

const MIB: u64 = 1024 * 1024;
const REDUCTION_BYTES: u64 = 16 * 1024;

pub(crate) struct MemoryPlan {
    pub(crate) weights: u64,
    pub(crate) fixed: u64,
    pub(crate) staging: u64,
    pub(crate) kv: u64,
    pub(crate) scratch: u64,
}

impl MemoryPlan {
    pub(crate) fn new(
        source: &dyn WeightSource,
        shape: RuntimeShape,
        context: usize,
        scheme: KvQuant,
    ) -> Result<Self> {
        let runtime = RuntimeBytes::new(shape, context, scheme, super::super::PREFILL_ROWS)
            .map_err(|_| arithmetic())?;
        let mut weights = 0u64;
        let mut staging = 0u64;
        for tensor in source.tensors() {
            let raw = tensor.byte_len().ok_or_else(arithmetic)?;
            let retained = match tensor.ggml_type {
                GgmlType::F32 => raw.checked_div(2).ok_or_else(arithmetic)?,
                GgmlType::F16 | GgmlType::Q4_K | GgmlType::Q5_K | GgmlType::Q6_K => raw,
                other => bail!("cuda: unsupported weight format '{}'", other.name()),
            };
            weights = weights.checked_add(retained).ok_or_else(arithmetic)?;
            staging = staging.max(retained);
        }
        Ok(Self {
            weights,
            fixed: runtime
                .logits
                .checked_add(REDUCTION_BYTES)
                .ok_or_else(arithmetic)?,
            staging,
            kv: runtime
                .kv_per_layer
                .checked_mul(u64::try_from(shape.block_count).map_err(|_| arithmetic())?)
                .ok_or_else(arithmetic)?,
            scratch: runtime.scratch,
        })
    }
}

pub(crate) fn validate_percentage(percent: Option<u8>) -> Result<u8> {
    match percent.unwrap_or(100) {
        value @ 1..=100 => Ok(value),
        _ => bail!("invalid CUDA weight percentage"),
    }
}

pub(crate) fn reserve_bytes(total: u64, reserve_mib: Option<u64>) -> Result<u64> {
    reserve_mib
        .map(|value| value.checked_mul(MIB).ok_or_else(arithmetic))
        .unwrap_or_else(|| Ok((256 * MIB).max(total / 20)))
}

pub(crate) fn preflight(free: u64, reserve: u64, percent: u8, plan: &MemoryPlan) -> Result<()> {
    if !(1..=100).contains(&percent) {
        bail!("invalid CUDA weight percentage");
    }
    let available = free
        .checked_sub(reserve)
        .ok_or_else(|| insufficient(reserve, free))?;
    let weight_cap = ((plan.weights as u128 * percent as u128) / 100) as u64;
    if plan.weights > weight_cap {
        return Err(insufficient(plan.weights, weight_cap));
    }
    let required = sum([
        plan.weights,
        plan.fixed,
        plan.staging,
        plan.kv,
        plan.scratch,
    ])?;
    if required > available {
        return Err(insufficient(required, available));
    }
    Ok(())
}

fn sum<const N: usize>(values: [u64; N]) -> Result<u64> {
    values.into_iter().try_fold(0u64, |total, value| {
        total.checked_add(value).ok_or_else(arithmetic)
    })
}

fn insufficient(required: u64, available: u64) -> color_eyre::Report {
    eyre!("CUDA memory is insufficient: required {required} bytes, available {available} bytes")
}

fn arithmetic() -> color_eyre::Report {
    eyre!("cuda: buffer arithmetic overflow")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plan(weights: u64, fixed: u64, staging: u64, kv: u64, scratch: u64) -> MemoryPlan {
        MemoryPlan {
            weights,
            fixed,
            staging,
            kv,
            scratch,
        }
    }

    #[test]
    fn exact_fit_and_one_byte_failure_are_distinct() {
        let exact = plan(50, 10, 5, 20, 15);
        preflight(110, 10, 100, &exact).expect("100-byte exact fit");
        assert_eq!(
            preflight(109, 10, 100, &exact).unwrap_err().to_string(),
            "CUDA memory is insufficient: required 100 bytes, available 99 bytes"
        );
    }

    #[test]
    fn reserve_precedence_and_percentage_boundaries_are_checked() {
        assert_eq!(reserve_bytes(20 * MIB, None).unwrap(), 256 * MIB);
        assert_eq!(reserve_bytes(20 * MIB, Some(1)).unwrap(), MIB);
        assert_eq!(validate_percentage(None).unwrap(), 100);
        assert_eq!(validate_percentage(Some(1)).unwrap(), 1);
        assert_eq!(validate_percentage(Some(100)).unwrap(), 100);
        assert_eq!(
            validate_percentage(Some(0)).unwrap_err().to_string(),
            "invalid CUDA weight percentage"
        );
        assert!(reserve_bytes(1, Some(u64::MAX)).is_err());
    }

    #[test]
    fn every_sum_overflow_is_rejected() {
        for overflow in [
            plan(u64::MAX, 1, 0, 0, 0),
            plan(1, u64::MAX, 0, 0, 0),
            plan(1, 0, u64::MAX, 0, 0),
            plan(1, 0, 0, u64::MAX, 0),
            plan(1, 0, 0, 0, u64::MAX),
        ] {
            assert_eq!(
                preflight(u64::MAX, 0, 100, &overflow)
                    .unwrap_err()
                    .to_string(),
                "cuda: buffer arithmetic overflow"
            );
        }
    }

    #[test]
    fn weight_percentage_never_allows_partial_placement() {
        let weights = plan(100, 0, 0, 0, 0);
        assert_eq!(
            preflight(1000, 0, 1, &weights).unwrap_err().to_string(),
            "CUDA memory is insufficient: required 100 bytes, available 1 bytes"
        );
    }
}
