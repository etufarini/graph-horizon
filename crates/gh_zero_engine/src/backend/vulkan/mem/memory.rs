/*
 * gh_zero_engine — memory estimation and weight placement plan
 * Estimates the engine's memory footprint (weights, FP16 KV cache for the real
 * context, scratch activations, FP32 logits) and decides each weight tensor's
 * placement. Pure Vulkan runs all-GPU-or-error after checked preflight; the
 * hybrid build keeps the legacy host-placement plan for tensors delegated away
 * from GPU residency. This is a pre-flight check: impossible footprints fail
 * before persistent model upload.
 *
 * Quantization: per-weight sizing is already exact for quantized dtypes —
 * TensorInfo::byte_len follows the ggml block layout (Q4_K/Q6_K), so the plan
 * uses the real on-disk (and on-GPU) size of each quantized tensor. KV cache,
 * scratch and logits stay FP16/FP32 regardless of weight quantization. The
 * pre-flight therefore naturally covers the expected case of a model that fits in
 * VRAM only when quantized.
 * The KV-cache byte size comes from `kv_cache::layout::cache_bytes`, the single
 * source of truth for that arithmetic — it is no longer computed here.
 *
 * Knobs: the placement dials (`--vram-weights-percent`, `--vram-reserve-mib`) and
 * the byte-exact budget arithmetic live in `budget`; `plan` consumes them via
 * `budget::weight_vram_percent`/`weight_vram_budget`. `scratch_bytes` is
 * `pub(crate)` for the hybrid auto cap (it charges scratch against free VRAM).
 *
 * Partial hybrid ownership is handled by the separate immutable placement and
 * selected-loader path; this full-backend planner always counts every weight.
*/

#[cfg(any(test, not(feature = "vulcan-hybrid")))]
use color_eyre::eyre::{Result, eyre};

#[cfg(any(test, not(feature = "vulcan-hybrid")))]
use crate::backend::source::WeightSource;
#[cfg(any(test, not(feature = "vulcan-hybrid")))]
use crate::gguf::metadata::ModelMetadata;

// Device-local byte budget the all-Vulkan plan must fit.
pub(crate) struct Budget {
    pub vram: u64,
}

// Placement decision per weight tensor, parallel to `WeightSet::tensors()`.
#[cfg(any(test, not(feature = "vulcan-hybrid")))]
pub(crate) struct MemoryPlan {
    pub host: Vec<bool>,
}

// Builds an all-device placement. Hybrid range placement uses its independent
// pure planner and selected loader; this path never spills individual tensors.
#[cfg(any(test, not(feature = "vulcan-hybrid")))]
pub(crate) fn plan(
    meta: &ModelMetadata,
    ws: &dyn WeightSource,
    context_len: usize,
    budget: &Budget,
) -> Result<MemoryPlan> {
    let kv_bytes = checked_product(&[
        4,
        meta.block_count as u64,
        context_len as u64,
        meta.head_count_kv as u64,
        meta.head_dim as u64,
    ])
    .ok_or_else(|| {
        eyre!("context {context_len} does not fit the selected backend; context was not reduced")
    })?;
    let logits_bytes = (meta.vocab_size as u64).checked_mul(4).ok_or_else(|| {
        eyre!(
            "Vulkan memory is insufficient: required overflow bytes, available {} bytes",
            budget.vram
        )
    })?;
    let scratch_bytes = checked_scratch_bytes(meta).ok_or_else(|| {
        eyre!("context {context_len} does not fit the selected backend; context was not reduced")
    })?;
    let tensors = ws.tensors();
    let mut weight_bytes = Vec::with_capacity(tensors.len());
    let mut weights_total = 0u64;
    for t in &tensors {
        let raw = t
            .byte_len()
            .ok_or_else(|| eyre!("memory: cannot size tensor '{}'", t.name))?;
        // GGUF tensors are independently aligned to 32 bytes. Tied weights occur
        // once in WeightSource, so this is both aligned and unique accounting.
        let b = aligned_weight_bytes(raw).ok_or_else(|| {
            eyre!(
                "Vulkan memory is insufficient: required overflow bytes, available {} bytes",
                budget.vram
            )
        })?;
        weights_total = weights_total.checked_add(b).ok_or_else(|| {
            eyre!(
                "Vulkan memory is insufficient: required overflow bytes, available {} bytes",
                budget.vram
            )
        })?;
        weight_bytes.push(b);
    }
    let peak_staging_bytes = weight_bytes.iter().copied().max().unwrap_or(0);
    super::budget::pure_preflight(
        budget.vram,
        super::budget::reserve_bytes(budget.vram, super::budget::configured_reserve_mib()),
        super::budget::weight_vram_percent(),
        weights_total,
        logits_bytes,
        peak_staging_bytes,
        kv_bytes,
        scratch_bytes,
        context_len,
    )?;
    Ok(MemoryPlan {
        host: vec![false; tensors.len()],
    })
}

#[cfg(any(test, not(feature = "vulcan-hybrid")))]
fn checked_product(values: &[u64]) -> Option<u64> {
    values
        .iter()
        .try_fold(1u64, |acc, value| acc.checked_mul(*value))
}

#[cfg(any(test, not(feature = "vulcan-hybrid")))]
fn aligned_weight_bytes(bytes: u64) -> Option<u64> {
    bytes.checked_add(31).map(|n| n / 32 * 32)
}

#[cfg(any(test, not(feature = "vulcan-hybrid")))]
fn checked_scratch_bytes(meta: &ModelMetadata) -> Option<u64> {
    let embd = meta.embedding_length as u64;
    let q = (meta.head_count as u64).checked_mul(meta.head_dim as u64)?;
    let kv = (meta.head_count_kv as u64).checked_mul(meta.head_dim as u64)?;
    let ffn = meta.feed_forward_length as u64;
    3u64.checked_mul(embd)?
        .checked_add(2u64.checked_mul(q)?)?
        .checked_add(2u64.checked_mul(kv)?)?
        .checked_add(3u64.checked_mul(ffn)?)?
        .checked_mul(2)?
        .checked_add(embd.checked_mul(4)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::tensor_index::{GgmlType, TensorInfo};

    // A WeightSource over an owned tensor list for checked accounting tests.
    struct Vecs(Vec<TensorInfo>);
    impl WeightSource for Vecs {
        fn groups(&self) -> crate::backend::source::WeightGroups<'_> {
            crate::backend::source::WeightGroups::new(&self.0[0], &self.0[1], None, Vec::new())
        }
    }

    fn ti(name: &str, n: u64) -> TensorInfo {
        TensorInfo {
            name: name.into(),
            dims: vec![n],
            ggml_type: GgmlType::F16,
            offset: 0,
        }
    }

    // Minimal metadata: zero layers (no KV), tiny scratch/logits, so the budget is
    // dominated by the two globals.
    fn tiny_meta() -> ModelMetadata {
        ModelMetadata {
            block_count: 0,
            embedding_length: 4,
            head_count: 1,
            head_count_kv: 1,
            head_dim: 4,
            feed_forward_length: 4,
            vocab_size: 2,
        }
    }

    #[test]
    fn weight_total_overflow_is_rejected_before_preflight() {
        let ws = Vecs(vec![ti("token_embd", u64::MAX / 2), ti("output_norm", 2)]);
        let meta = tiny_meta();
        let budget = Budget { vram: u64::MAX };
        let err = match plan(&meta, &ws, 8, &budget) {
            Ok(_) => panic!("overflowing weight total must fail"),
            Err(err) => err.to_string(),
        };
        assert!(err.starts_with("Vulkan memory is insufficient: required overflow bytes"));
    }

    #[cfg(not(feature = "vulcan-hybrid"))]
    #[test]
    fn pure_weight_accounting_is_aligned_and_checked() {
        assert_eq!(aligned_weight_bytes(0), Some(0));
        assert_eq!(aligned_weight_bytes(1), Some(32));
        assert_eq!(aligned_weight_bytes(32), Some(32));
        assert_eq!(aligned_weight_bytes(33), Some(64));
        assert_eq!(aligned_weight_bytes(u64::MAX), None);
    }
}
