/*
 * gh_zero_engine — temporary Mistral hybrid shape adapter
 * Converts the validated family configuration into neutral runtime dimensions
 * and delegates resource planning/loading to backend::hybrid. It owns no budget,
 * placement, device probe, allocation policy, graph traversal, or fallback.
 */

use color_eyre::eyre::Result;

use super::LoadedHybrid;
use crate::backend::hybrid::weights::runtime::RuntimeShape;
use crate::backend::vulkan::VulkanBackend;
use crate::family::mistral::MistralContract;
use crate::gguf::loader::GgufFile;
use crate::gguf::metadata::ModelMetadata;
use crate::kv_cache::scheme::KvQuant;

pub(crate) fn load(
    file: &GgufFile,
    contract: &MistralContract<'_>,
    context: usize,
    scheme: KvQuant,
    weights_percent: Option<u8>,
    reserve_mib: Option<u64>,
) -> Result<LoadedHybrid> {
    let config = &contract.config;
    let metadata = ModelMetadata::from_gguf(file)?;
    crate::backend::hybrid::loader::load::<VulkanBackend>(
        file,
        &contract.tensors,
        &metadata,
        RuntimeShape {
            block_count: config.block_count,
            embedding: config.embedding_length,
            q: config.q_width,
            k: config.k_width,
            v: config.v_width,
            attention: config.attention_width,
            feed_forward: config.feed_forward_length,
            vocab: config.vocab_size,
            kv_heads: config.kv_head_count,
            key_length: config.key_length,
            value_length: config.value_length,
            prefill_rows: super::super::graph::prefill::BATCH_ROWS,
        },
        context,
        scheme,
        weights_percent,
        reserve_mib,
    )
}
