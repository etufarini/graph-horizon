/*
 * graph_horizon_engine — checked standalone Metal memory budget
 * Computes unique weight, runtime, reserve, and unified-memory totals before
 * allocation. It performs no device probing, GGUF I/O, placement, or fallback.
 */

use color_eyre::eyre::{Result, bail, eyre};

use crate::backend::hybrid::weights::runtime::{RuntimeBytes, RuntimeShape};
use crate::backend::source::WeightSource;
use crate::gguf::tensor_index::GgmlType;
use crate::kv_cache::scheme::KvQuant;

const MIB: u64 = 1024 * 1024;
const REDUCE_BYTES: u64 = 16 * 1024;
const STAGING_BYTES: u64 = 16 * 1024;

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
        let runtime = RuntimeBytes::new(shape, context, scheme, shape.gpu_prefill_rows)
            .map_err(|_| arithmetic())?;
        let mut weights = 0u64;
        let mut staging = STAGING_BYTES;
        for tensor in source.tensors() {
            let raw = tensor.byte_len().ok_or_else(arithmetic)?;
            let retained = match tensor.ggml_type {
                GgmlType::F32 => raw / 2,
                GgmlType::F16 | GgmlType::Q4_K | GgmlType::Q5_K | GgmlType::Q6_K => raw,
                other => bail!("metal: unsupported weight format '{}'", other.name()),
            };
            weights = weights
                .checked_add(align(retained)?)
                .ok_or_else(arithmetic)?;
            staging = staging.max(retained);
        }
        let fixed = runtime
            .logits
            .checked_add(REDUCE_BYTES)
            .and_then(|value| value.checked_add(STAGING_BYTES))
            .ok_or_else(arithmetic)?;
        let kv = runtime
            .kv_per_layer
            .checked_mul(shape.block_count as u64)
            .ok_or_else(arithmetic)?;
        Ok(Self {
            weights,
            fixed,
            staging,
            kv,
            scratch: runtime.scratch,
        })
    }
}

pub(crate) fn validate_percentage(percent: Option<u8>) -> Result<u8> {
    match percent.unwrap_or(100) {
        1..=100 => Ok(percent.unwrap_or(100)),
        _ => bail!("invalid Metal weight percentage"),
    }
}

pub(crate) fn reserve_bytes(gross: u64, reserve_mib: Option<u64>) -> Result<u64> {
    reserve_mib
        .map(|value| value.checked_mul(MIB).ok_or_else(arithmetic))
        .unwrap_or_else(|| Ok((256 * MIB).max(gross / 20)))
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn preflight(
    physical_memory: u64,
    recommended_max: u64,
    current_allocated: u64,
    reserve: u64,
    percent: u8,
    plan: &MemoryPlan,
    context: usize,
) -> Result<()> {
    if !(1..=100).contains(&percent) {
        bail!("invalid Metal weight percentage");
    }
    let gross = crate::backend::hybrid::placement::unified_gross(physical_memory, recommended_max)
        .ok_or_else(arithmetic)?;
    if reserve > gross || current_allocated > gross.saturating_sub(reserve) {
        return Err(insufficient(
            reserve.saturating_add(current_allocated),
            gross,
        ));
    }
    let available =
        crate::backend::hybrid::placement::unified_capacity(gross - reserve, current_allocated);
    let weight_cap = ((plan.weights as u128 * percent as u128) / 100) as u64;
    if plan.weights > weight_cap {
        return Err(insufficient(plan.weights, weight_cap));
    }
    let model = sum([plan.weights, plan.fixed, plan.staging])?;
    if model > available {
        return Err(insufficient(model, available));
    }
    let total = sum([model, plan.kv, plan.scratch])?;
    if total > available {
        bail!("context {context} does not fit the selected backend; context was not reduced");
    }
    Ok(())
}

fn align(value: u64) -> Result<u64> {
    value
        .checked_add(31)
        .map(|value| value / 32 * 32)
        .ok_or_else(arithmetic)
}

fn sum<const N: usize>(values: [u64; N]) -> Result<u64> {
    values.into_iter().try_fold(0u64, |total, value| {
        total.checked_add(value).ok_or_else(arithmetic)
    })
}

fn insufficient(required: u64, available: u64) -> color_eyre::Report {
    eyre!("Metal memory is insufficient: required {required} bytes, available {available} bytes")
}

fn arithmetic() -> color_eyre::Report {
    eyre!("metal: buffer arithmetic overflow")
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
    fn exact_fit_and_percentage_boundaries_are_checked() {
        let exact = plan(50, 10, 0, 20, 10);
        preflight(100, 100, 0, 0, 100, &exact, 32).expect("90-byte exact fit");
        assert_eq!(validate_percentage(None).unwrap(), 100);
        assert_eq!(validate_percentage(Some(100)).unwrap(), 100);
        for value in [Some(0), Some(101)] {
            assert_eq!(
                validate_percentage(value).unwrap_err().to_string(),
                "invalid Metal weight percentage"
            );
        }
    }

    #[test]
    fn reserve_and_current_snapshot_are_subtracted_once() {
        let exact = plan(40, 10, 0, 10, 10);
        preflight(100, 100, 10, 10, 100, &exact, 8).expect("70-byte exact fit");
        let error = preflight(100, 100, 11, 10, 100, &exact, 8)
            .unwrap_err()
            .to_string();
        assert_eq!(
            error,
            "context 8 does not fit the selected backend; context was not reduced"
        );
        assert!(
            preflight(100, 100, 91, 0, 100, &exact, 8)
                .unwrap_err()
                .to_string()
                .starts_with("Metal memory is insufficient")
        );
    }

    #[test]
    fn reserve_larger_than_gross_and_partial_weights_fail_as_model_errors() {
        let tiny = plan(1, 0, 0, 0, 0);
        assert!(
            preflight(100, 100, 0, 91, 100, &tiny, 1)
                .unwrap_err()
                .to_string()
                .starts_with("Metal memory is insufficient")
        );
        assert_eq!(
            preflight(100, 100, 0, 0, 50, &tiny, 1)
                .unwrap_err()
                .to_string(),
            "Metal memory is insufficient: required 1 bytes, available 0 bytes"
        );
    }
}
