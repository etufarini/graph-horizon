/*
 * gh_zero_engine — model metadata
 * Extracts only the family-neutral dimensions required to allocate backend
 * weights and scratch buffers. Family validation reads its richer contract
 * directly from the GGUF metadata.
*/

use color_eyre::eyre::{Result, eyre};

use super::loader::GgufFile;

// `Clone` lets the hybrid split runtime derive per-side metadata with a reduced
// block count while preserving all allocation dimensions.
#[derive(Clone)]
pub(crate) struct ModelMetadata {
    #[cfg(any(
        all(test, any(feature = "vulcan", feature = "vulcan-hybrid")),
        feature = "vulcan"
    ))]
    pub block_count: usize,
    pub embedding_length: usize,
    pub head_count: usize,
    pub head_count_kv: usize,
    pub head_dim: usize,
    pub feed_forward_length: usize,
    pub vocab_size: usize,
}

impl ModelMetadata {
    pub(crate) fn from_gguf(f: &GgufFile) -> Result<ModelMetadata> {
        let md = f.metadata();
        let architecture = md
            .get("general.architecture")
            .and_then(|v| v.as_str())
            .ok_or_else(|| eyre!("gguf: missing 'general.architecture'"))?
            .to_string();

        // Hyper-parameter keys are namespaced by the architecture name.
        let key = |suffix: &str| format!("{architecture}.{suffix}");
        let req_u = |suffix: &str| -> Result<u64> {
            let k = key(suffix);
            md.get(&k)
                .and_then(|v| v.as_u64())
                .ok_or_else(|| eyre!("gguf: missing or invalid '{k}'"))
        };
        let head_count = req_u("attention.head_count")? as usize;
        let embedding_length = req_u("embedding_length")? as usize;

        // head_dim: prefer the explicit key/value length; otherwise derive it
        // from embedding_length / head_count (the family-neutral GGUF fallback).
        let head_dim = md
            .get(&key("attention.key_length"))
            .or_else(|| md.get(&key("attention.value_length")))
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .or_else(|| (head_count != 0).then(|| embedding_length / head_count))
            .ok_or_else(|| eyre!("gguf: cannot resolve head_dim"))?;

        // GQA: head_count_kv defaults to head_count when the key is absent.
        let head_count_kv = md
            .get(&key("attention.head_count_kv"))
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(head_count);

        // Vocabulary size is the embedded token table length. Family validation
        // owns the fixed error for missing tokenizer metadata.
        let tokens = md.get("tokenizer.ggml.tokens").and_then(|v| v.as_array());
        let vocab_size = tokens.map(|a| a.len()).unwrap_or(0);

        #[cfg(any(
            all(test, any(feature = "vulcan", feature = "vulcan-hybrid")),
            feature = "vulcan"
        ))]
        let block_count = req_u("block_count")? as usize;
        let feed_forward_length = req_u("feed_forward_length")? as usize;

        Ok(ModelMetadata {
            #[cfg(any(
                all(test, any(feature = "vulcan", feature = "vulcan-hybrid")),
                feature = "vulcan"
            ))]
            block_count,
            embedding_length,
            head_count,
            head_count_kv,
            head_dim,
            feed_forward_length,
            vocab_size,
        })
    }
}
