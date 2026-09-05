/*
 * graph_horizon_engine — checked standalone CUDA memory planning and preflight.
 * The calculation is pure: it cannot allocate, reduce context, or change KV.
 */

#[cfg(any(feature = "cuda", test))]
use color_eyre::eyre::{Result, bail, eyre};

#[cfg(any(feature = "cuda", test))]
use crate::backend::hybrid::weights::runtime::{RuntimeBytes, RuntimeShape};
#[cfg(any(feature = "cuda", test))]
use crate::backend::source::WeightSource;
#[cfg(any(feature = "cuda", test))]
use crate::gguf::tensor_index::GgmlType;
#[cfg(any(feature = "cuda", test))]
use crate::kv_cache::scheme::KvQuant;

#[cfg(any(feature = "cuda", test))]
const MIB: u64 = 1024 * 1024;
#[cfg(any(feature = "cuda", test))]
const REDUCTION_BYTES: u64 = 16 * 1024;

#[cfg(feature = "cuda-hybrid")]
pub(crate) fn host_available() -> u64 {
    std::fs::read_to_string("/proc/meminfo")
        .ok()
        .map_or(0, |input| parse_host_available(&input))
}

#[cfg(any(feature = "cuda-hybrid", test))]
fn parse_host_available(input: &str) -> u64 {
    let mut value = None;
    for line in input.lines() {
        let Some(rest) = line.strip_prefix("MemAvailable:") else {
            continue;
        };
        if value.is_some() {
            return 0;
        }
        let mut fields = rest.split_ascii_whitespace();
        let Some(kib) = fields.next().and_then(|field| field.parse::<u64>().ok()) else {
            return 0;
        };
        if fields.next() != Some("kB") || fields.next().is_some() {
            return 0;
        }
        value = kib
            .checked_mul(1024)
            .and_then(|bytes| bytes.checked_mul(9))
            .and_then(|bytes| bytes.checked_div(10));
        if value.is_none() {
            return 0;
        }
    }
    value.unwrap_or(0)
}

#[cfg(any(feature = "cuda", test))]
pub(crate) struct MemoryPlan {
    pub(crate) weights: u64,
    pub(crate) fixed: u64,
    pub(crate) staging: u64,
    pub(crate) kv: u64,
    pub(crate) scratch: u64,
}

#[cfg(any(feature = "cuda", test))]
impl MemoryPlan {
    pub(crate) fn new(
        source: &dyn WeightSource,
        shape: RuntimeShape,
        context: usize,
        scheme: KvQuant,
        cached: bool,
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
        if cached {
            // Only layer matrices own companions. Raw originals remain counted above.
            for (slot, tensor) in source
                .groups()
                .layers
                .into_iter()
                .flat_map(|group| group.into_iter().enumerate())
            {
                if !matches!(slot, 0 | 5)
                    && tensor.ggml_type == GgmlType::Q6_K
                    && tensor.dims.len() == 2
                {
                    let extra = tensor
                        .element_count()
                        .filter(|count| count.is_multiple_of(256))
                        .and_then(|count| (count / 256).checked_mul(320))
                        .ok_or_else(arithmetic)?;
                    weights = weights.checked_add(extra).ok_or_else(arithmetic)?;
                }
            }
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

#[cfg(any(feature = "cuda", test))]
pub(crate) fn validate_percentage(percent: Option<u8>) -> Result<u8> {
    match percent.unwrap_or(100) {
        value @ 1..=100 => Ok(value),
        _ => bail!("invalid CUDA weight percentage"),
    }
}

#[cfg(any(feature = "cuda", test))]
pub(crate) fn reserve_bytes(total: u64, reserve_mib: Option<u64>) -> Result<u64> {
    reserve_mib
        .map(|value| value.checked_mul(MIB).ok_or_else(arithmetic))
        .unwrap_or_else(|| Ok((256 * MIB).max(total / 20)))
}

#[cfg(any(feature = "cuda", test))]
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

#[cfg(any(feature = "cuda", test))]
fn sum<const N: usize>(values: [u64; N]) -> Result<u64> {
    values.into_iter().try_fold(0u64, |total, value| {
        total.checked_add(value).ok_or_else(arithmetic)
    })
}

#[cfg(any(feature = "cuda", test))]
fn insufficient(required: u64, available: u64) -> color_eyre::Report {
    eyre!("CUDA memory is insufficient: required {required} bytes, available {available} bytes")
}

#[cfg(any(feature = "cuda", test))]
fn arithmetic() -> color_eyre::Report {
    eyre!("cuda: buffer arithmetic overflow")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::source::WeightGroups;
    use crate::gguf::tensor_index::TensorInfo;
    use crate::kv_cache::scheme::KvRole;

    struct Source {
        embedding: TensorInfo,
        norm: TensorInfo,
    }

    impl WeightSource for Source {
        fn groups(&self) -> WeightGroups<'_> {
            WeightGroups::new(&self.embedding, &self.norm, None, Vec::new())
        }
    }

    fn tensor(name: &str, dims: &[u64], ggml_type: GgmlType) -> TensorInfo {
        TensorInfo {
            name: name.into(),
            dims: dims.into(),
            ggml_type,
            offset: 0,
        }
    }

    fn shape() -> RuntimeShape {
        RuntimeShape {
            block_count: 2,
            embedding: 8,
            q: 8,
            k: 8,
            v: 4,
            attention: 8,
            feed_forward: 16,
            vocab: 32,
            kv_heads: 2,
            key_length: 8,
            value_length: 4,
            cpu_prefill_rows: 4,
            gpu_prefill_rows: 32,
            mixed_prefill_rows: 4,
        }
    }

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
    fn cache_capacity_covers_conversion_and_preserves_raw_fallback() -> Result<()> {
        struct LayerSource {
            globals: Source,
            projection: TensorInfo,
        }
        impl WeightSource for LayerSource {
            fn groups(&self) -> WeightGroups<'_> {
                WeightGroups::new(
                    &self.globals.embedding,
                    &self.globals.norm,
                    None,
                    vec![vec![&self.globals.norm, &self.projection]],
                )
            }
        }
        let source = LayerSource {
            globals: Source {
                embedding: tensor("embedding", &[256, 1], GgmlType::Q6_K),
                norm: tensor("norm", &[8], GgmlType::F32),
            },
            projection: tensor("projection", &[256, 1], GgmlType::Q6_K),
        };
        let raw = MemoryPlan::new(&source, shape(), 16, KvQuant::F16, false)?;
        let cached = MemoryPlan::new(&source, shape(), 16, KvQuant::F16, true)?;
        assert_eq!((raw.weights, raw.staging), (452, 210));
        assert_eq!((cached.weights, cached.staging), (772, 210));
        let runtime =
            RuntimeBytes::new(shape(), 16, KvQuant::F16, super::super::super::PREFILL_ROWS)?;
        assert_eq!(raw.scratch, runtime.scratch);
        assert_eq!(cached.scratch, runtime.scratch);
        let required = sum([
            cached.weights,
            cached.fixed,
            cached.staging,
            cached.kv,
            cached.scratch,
        ])?;
        preflight(required + 10, 10, 100, &cached)?;
        assert!(preflight(required + 9, 10, 100, &cached).is_err());
        preflight(required + 9, 10, 100, &raw)?;
        let overflow = Source {
            embedding: tensor("embedding", &[256, u64::MAX], GgmlType::Q4_K),
            norm: tensor("norm", &[8], GgmlType::F32),
        };
        assert!(MemoryPlan::new(&overflow, shape(), 16, KvQuant::F16, true).is_err());
        Ok(())
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
    fn host_available_requires_one_exact_checked_field() {
        assert_eq!(parse_host_available("MemAvailable: 100 kB\n"), 92_160);
        for invalid in [
            "",
            "MemFree: 100 kB\n",
            "MemAvailable: nope kB\n",
            "MemAvailable: 100 MB\n",
            "MemAvailable: 100 kB trailing\n",
            "MemAvailable: 100 kB\nMemAvailable: 200 kB\n",
            "MemAvailable: 18014398509481984 kB\n",
        ] {
            assert_eq!(parse_host_available(invalid), 0, "{invalid:?}");
        }
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

    #[test]
    fn f16_and_int8_kv_totals_match_the_shared_layout() {
        let source = Source {
            embedding: tensor("embedding", &[256], GgmlType::Q4_K),
            norm: tensor("norm", &[8], GgmlType::F32),
        };
        let context = 16;
        for scheme in [KvQuant::F16, KvQuant::Int8] {
            let actual = MemoryPlan::new(&source, shape(), context, scheme, false)
                .expect("valid CUDA memory plan")
                .kv;
            let expected = layout_bytes(scheme, KvRole::Key, context, 8)
                + layout_bytes(scheme, KvRole::Value, context, 4);
            assert_eq!(actual, expected, "{} KV total", scheme.name());
        }
    }

    fn layout_bytes(scheme: KvQuant, role: KvRole, context: usize, width: usize) -> u64 {
        crate::kv_cache::layout::buffer_bytes(scheme, role, 2, context, 2, width)
    }
}
