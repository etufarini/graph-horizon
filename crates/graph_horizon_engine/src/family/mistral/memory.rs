/*
 * graph_horizon_engine — Ministral immutable memory summary
 * Computes retained model weights and full-context KV capacity from validated
 * family data, or folds an immutable hybrid plan. It performs no allocation,
 * probing, placement selection, or runtime observation.
 */

use color_eyre::eyre::{Result, eyre};

use crate::api::engine::ModelMemory;
#[cfg(any(
    feature = "vulkan-hybrid",
    feature = "metal-hybrid",
    feature = "cuda-hybrid"
))]
use crate::backend::hybrid::HybridPlan;
#[cfg(not(any(
    feature = "vulkan-hybrid",
    feature = "metal-hybrid",
    feature = "cuda-hybrid"
)))]
use crate::backend::hybrid::weights::model::WeightBytes;
#[cfg(not(any(
    feature = "vulkan-hybrid",
    feature = "metal-hybrid",
    feature = "cuda-hybrid"
)))]
use crate::backend::source::WeightSource;
#[cfg(not(any(
    feature = "vulkan-hybrid",
    feature = "metal-hybrid",
    feature = "cuda-hybrid"
)))]
use crate::kv_cache::layout;
#[cfg(not(any(
    feature = "vulkan-hybrid",
    feature = "metal-hybrid",
    feature = "cuda-hybrid"
)))]
use crate::kv_cache::scheme::KvQuant;
#[cfg(not(any(
    feature = "vulkan-hybrid",
    feature = "metal-hybrid",
    feature = "cuda-hybrid"
)))]
use crate::kv_cache::scheme::KvRole;

#[cfg(not(any(
    feature = "vulkan-hybrid",
    feature = "metal-hybrid",
    feature = "cuda-hybrid"
)))]
use super::MistralConfig;

#[cfg(not(any(
    feature = "vulkan-hybrid",
    feature = "metal-hybrid",
    feature = "cuda-hybrid"
)))]
pub(super) fn homogeneous(
    _tensors: &dyn WeightSource,
    config: &MistralConfig,
    context: usize,
    scheme: KvQuant,
    _backend: &crate::backend::selection::SelectedBackend,
) -> Result<ModelMemory> {
    let weights = WeightBytes::from_source(_tensors)?
        .total()
        .ok_or_else(overflow)?;
    #[cfg(feature = "cuda")]
    let weights = {
        let retained = _backend.weight_bytes()?;
        // Companions add storage; the original aligned representation cannot disappear.
        if retained < weights {
            return Err(overflow());
        }
        retained
    };
    let key = layout::buffer_bytes(
        scheme,
        KvRole::Key,
        config.block_count,
        context,
        config.kv_head_count,
        config.key_length,
    );
    let value = layout::buffer_bytes(
        scheme,
        KvRole::Value,
        config.block_count,
        context,
        config.kv_head_count,
        config.value_length,
    );
    Ok(ModelMemory {
        weights,
        kv: key.checked_add(value).ok_or_else(overflow)?,
    })
}

#[cfg(any(
    feature = "vulkan-hybrid",
    feature = "metal-hybrid",
    feature = "cuda-hybrid"
))]
pub(super) fn hybrid(plan: Option<&HybridPlan>) -> Result<ModelMemory> {
    let plan = plan.ok_or_else(|| eyre!("hybrid placement unavailable"))?;
    Ok(ModelMemory {
        // A split may retain a tied embedding on both owners, so the physical
        // plan is authoritative over the homogeneous unique representation.
        weights: plan
            .cpu
            .weights
            .checked_add(plan.gpu.weights)
            .ok_or_else(overflow)?,
        kv: plan.cpu.kv.checked_add(plan.gpu.kv).ok_or_else(overflow)?,
    })
}

fn overflow() -> color_eyre::Report {
    eyre!("model memory accounting overflow")
}
