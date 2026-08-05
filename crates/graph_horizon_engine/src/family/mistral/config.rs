/*
 * graph_horizon_engine — Ministral dense configuration
 * Builds the immutable dense `mistral3` shape contract from untrusted GGUF
 * metadata plus tokenizer-provided vocabulary/special ids. It performs checked
 * arithmetic for Q/K/V widths and context dimensions only; no tensor bytes are
 * read and no backend allocation is performed.
*/

use std::collections::HashMap;

use color_eyre::eyre::{Result, bail, eyre};

use crate::gguf::loader::GgufValue;

#[derive(Clone)]
pub struct MistralConfig {
    pub block_count: usize,
    pub context_length: usize,
    pub embedding_length: usize,
    pub feed_forward_length: usize,
    pub head_count: usize,
    pub kv_head_count: usize,
    pub key_length: usize,
    pub value_length: usize,
    pub q_width: usize,
    pub k_width: usize,
    pub v_width: usize,
    pub attention_width: usize,
    pub rope_dimension: usize,
    pub rope_freq_base: f32,
    pub rms_epsilon: f32,
    pub yarn_factor: f32,
    pub yarn_beta_fast: f32,
    pub yarn_beta_slow: f32,
    pub yarn_log_multiplier: f32,
    pub yarn_original_context: usize,
    pub attention_temperature_scale: f32,
    pub vocab_size: usize,
    pub bos_id: u32,
    pub eos_id: u32,
}

impl MistralConfig {
    pub fn from_metadata(
        md: &HashMap<String, GgufValue>,
        tokenizer_vocab_size: usize,
        bos_id: u32,
        eos_id: u32,
    ) -> Result<Self> {
        let u = |k: &str| positive_usize(md, k);
        let f = |k: &str| positive_f32(md, k);
        let scaling = md
            .get("mistral3.rope.scaling.type")
            .and_then(|v| v.as_str())
            .ok_or_else(|| {
                eyre!("E06 missing or invalid GGUF metadata 'mistral3.rope.scaling.type'")
            })?;
        if scaling != "yarn" {
            bail!("E06 missing or invalid GGUF metadata 'mistral3.rope.scaling.type'");
        }

        let block_count = u("mistral3.block_count")?;
        let context_length = u("mistral3.context_length")?;
        let embedding_length = u("mistral3.embedding_length")?;
        let feed_forward_length = u("mistral3.feed_forward_length")?;
        let vocab_size = u("mistral3.vocab_size")?;
        if vocab_size != tokenizer_vocab_size {
            bail!("E06 missing or invalid GGUF metadata 'mistral3.vocab_size'");
        }
        let head_count = u("mistral3.attention.head_count")?;
        let kv_head_count = u("mistral3.attention.head_count_kv")?;
        let key_length = u("mistral3.attention.key_length")?;
        let value_length = u("mistral3.attention.value_length")?;
        if kv_head_count > head_count || embedding_length % head_count != 0 {
            bail!("E06 missing or invalid GGUF metadata 'mistral3.attention.head_count'");
        }
        let rope_dimension = u("mistral3.rope.dimension_count")?;
        if rope_dimension == 0 || rope_dimension > key_length || rope_dimension % 2 != 0 {
            bail!("E06 missing or invalid GGUF metadata 'mistral3.rope.dimension_count'");
        }

        // These widths are load-bearing allocation dimensions. Overflow or a
        // zero factor would make later tensor spans ambiguous, so both fail E06.
        let q_width = checked_mul(head_count, key_length, "mistral3.attention.key_length")?;
        let k_width = checked_mul(kv_head_count, key_length, "mistral3.attention.key_length")?;
        let v_width = checked_mul(
            kv_head_count,
            value_length,
            "mistral3.attention.value_length",
        )?;
        let attention_width =
            checked_mul(head_count, value_length, "mistral3.attention.value_length")?;

        Ok(Self {
            block_count,
            context_length,
            embedding_length,
            feed_forward_length,
            head_count,
            kv_head_count,
            key_length,
            value_length,
            q_width,
            k_width,
            v_width,
            attention_width,
            rope_dimension,
            rope_freq_base: f("mistral3.rope.freq_base")?,
            rms_epsilon: f("mistral3.attention.layer_norm_rms_epsilon")?,
            yarn_factor: f("mistral3.rope.scaling.factor")?,
            yarn_beta_fast: f("mistral3.rope.scaling.yarn_beta_fast")?,
            yarn_beta_slow: f("mistral3.rope.scaling.yarn_beta_slow")?,
            yarn_log_multiplier: f("mistral3.rope.scaling.yarn_log_multiplier")?,
            yarn_original_context: u("mistral3.rope.scaling.original_context_length")?,
            attention_temperature_scale: f("mistral3.attention.temperature_scale")?,
            vocab_size,
            bos_id,
            eos_id,
        })
    }
}

fn positive_usize(md: &HashMap<String, GgufValue>, key: &str) -> Result<usize> {
    let value = md
        .get(key)
        .and_then(unsigned_value)
        .ok_or_else(|| eyre!("E06 missing or invalid GGUF metadata '{key}'"))?;
    let value = usize::try_from(value)
        .map_err(|_| eyre!("E06 missing or invalid GGUF metadata '{key}'"))?;
    if value == 0 {
        bail!("E06 missing or invalid GGUF metadata '{key}'");
    }
    Ok(value)
}

fn positive_f32(md: &HashMap<String, GgufValue>, key: &str) -> Result<f32> {
    let value = md
        .get(key)
        .and_then(|v| match v {
            GgufValue::F32(v) => Some(*v),
            _ => None,
        })
        .filter(|v| v.is_finite() && *v > 0.0)
        .ok_or_else(|| eyre!("E06 missing or invalid GGUF metadata '{key}'"))?;
    Ok(value)
}

fn unsigned_value(value: &GgufValue) -> Option<u64> {
    match *value {
        GgufValue::U8(v) => Some(v as u64),
        GgufValue::U16(v) => Some(v as u64),
        GgufValue::U32(v) => Some(v as u64),
        GgufValue::U64(v) => Some(v),
        _ => None,
    }
}

fn checked_mul(a: usize, b: usize, key: &str) -> Result<usize> {
    a.checked_mul(b)
        .filter(|v| *v > 0)
        .ok_or_else(|| eyre!("E06 missing or invalid GGUF metadata '{key}'"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::family::mistral::version::{
        ATTENTION_WIDTH, HEAD_COUNT, K_WIDTH, KEY_LENGTH, KV_HEAD_COUNT, MAX_CONTEXT, Q_WIDTH,
        REFERENCE_ROWS, ROPE_DIMENSION, ReferenceRow, V_WIDTH, VALUE_LENGTH,
    };

    fn md(
        block_count: u32,
        embedding: u32,
        heads: u32,
        kv_heads: u32,
    ) -> HashMap<String, GgufValue> {
        HashMap::from([
            ("mistral3.block_count".into(), GgufValue::U32(block_count)),
            ("mistral3.context_length".into(), GgufValue::U32(4096)),
            (
                "mistral3.embedding_length".into(),
                GgufValue::U32(embedding),
            ),
            ("mistral3.feed_forward_length".into(), GgufValue::U32(64)),
            ("mistral3.vocab_size".into(), GgufValue::U32(32)),
            (
                "mistral3.attention.head_count".into(),
                GgufValue::U32(heads),
            ),
            (
                "mistral3.attention.head_count_kv".into(),
                GgufValue::U32(kv_heads),
            ),
            ("mistral3.attention.key_length".into(), GgufValue::U32(8)),
            ("mistral3.attention.value_length".into(), GgufValue::U32(8)),
            (
                "mistral3.attention.layer_norm_rms_epsilon".into(),
                GgufValue::F32(0.00001),
            ),
            (
                "mistral3.attention.temperature_scale".into(),
                GgufValue::F32(1.0),
            ),
            ("mistral3.rope.dimension_count".into(), GgufValue::U32(8)),
            (
                "mistral3.rope.freq_base".into(),
                GgufValue::F32(1_000_000.0),
            ),
            (
                "mistral3.rope.scaling.type".into(),
                GgufValue::String("yarn".into()),
            ),
            ("mistral3.rope.scaling.factor".into(), GgufValue::F32(8.0)),
            (
                "mistral3.rope.scaling.original_context_length".into(),
                GgufValue::U32(4096),
            ),
            (
                "mistral3.rope.scaling.yarn_beta_fast".into(),
                GgufValue::F32(32.0),
            ),
            (
                "mistral3.rope.scaling.yarn_beta_slow".into(),
                GgufValue::F32(1.0),
            ),
            (
                "mistral3.rope.scaling.yarn_log_multiplier".into(),
                GgufValue::F32(0.1),
            ),
        ])
    }

    fn exact_md(row: ReferenceRow) -> HashMap<String, GgufValue> {
        let (blocks, embedding, ffn, _) = row;
        let mut values = md(
            blocks as u32,
            embedding as u32,
            HEAD_COUNT as u32,
            KV_HEAD_COUNT as u32,
        );
        for (key, value) in [
            ("mistral3.context_length", MAX_CONTEXT as u32),
            ("mistral3.feed_forward_length", ffn as u32),
            ("mistral3.attention.key_length", KEY_LENGTH as u32),
            ("mistral3.attention.value_length", VALUE_LENGTH as u32),
            ("mistral3.rope.dimension_count", ROPE_DIMENSION as u32),
        ] {
            values.insert(key.into(), GgufValue::U32(value));
        }
        values
    }

    #[test]
    fn derives_distinct_q_k_v_and_attention_widths() {
        let cfg = MistralConfig::from_metadata(&md(1, 32, 4, 2), 32, 0, 1).unwrap();
        assert_eq!(cfg.q_width, 32);
        assert_eq!(cfg.k_width, 16);
        assert_eq!(cfg.v_width, 16);
        assert_eq!(cfg.attention_width, 32);
    }

    #[test]
    fn versioned_rows_use_the_same_constructor() {
        for row in REFERENCE_ROWS {
            let (blocks, hidden, ffn, dedicated) = row;
            let cfg = MistralConfig::from_metadata(&exact_md(row), 32, 0, 1).unwrap();
            assert_eq!(cfg.block_count, blocks);
            assert_eq!(cfg.context_length, MAX_CONTEXT);
            assert_eq!(cfg.embedding_length, hidden);
            assert_eq!(cfg.feed_forward_length, ffn);
            assert_eq!(cfg.q_width, Q_WIDTH);
            assert_eq!(cfg.k_width, K_WIDTH);
            assert_eq!(cfg.v_width, V_WIDTH);
            assert_eq!(cfg.attention_width, ATTENTION_WIDTH);
            println!(
                "blocks={blocks} hidden={hidden} q={Q_WIDTH} k={K_WIDTH} v={V_WIDTH} attention={ATTENTION_WIDTH} ffn={ffn} output={}",
                if dedicated { "dedicated" } else { "tied" }
            );
        }
    }

    #[test]
    fn error_matrix_e06_rejects_invalid_dense_invariant() {
        let err = match MistralConfig::from_metadata(&md(1, 32, 2, 4), 32, 0, 1) {
            Ok(_) => panic!("invalid KV head invariant must fail"),
            Err(err) => err.to_string(),
        };
        assert!(err.contains("E06 missing or invalid GGUF metadata"));
    }

    #[test]
    fn error_matrix_e06_rejects_wrong_type_missing_scaling_and_bad_rope() {
        let mut md = md(1, 32, 4, 2);
        md.insert(
            "mistral3.context_length".into(),
            GgufValue::String("4096".into()),
        );
        let err = match MistralConfig::from_metadata(&md, 32, 0, 1) {
            Ok(_) => panic!("wrong-typed context_length must fail"),
            Err(err) => err.to_string(),
        };
        assert!(err.contains("E06 missing or invalid GGUF metadata 'mistral3.context_length'"));

        let mut md = self::md(1, 32, 4, 2);
        md.insert(
            "mistral3.rope.scaling.type".into(),
            GgufValue::String("linear".into()),
        );
        let err = match MistralConfig::from_metadata(&md, 32, 0, 1) {
            Ok(_) => panic!("non-yarn scaling must fail"),
            Err(err) => err.to_string(),
        };
        assert!(err.contains("mistral3.rope.scaling.type"));

        let mut md = self::md(1, 32, 4, 2);
        md.insert("mistral3.rope.dimension_count".into(), GgufValue::U32(7));
        let err = match MistralConfig::from_metadata(&md, 32, 0, 1) {
            Ok(_) => panic!("odd rope dimension must fail"),
            Err(err) => err.to_string(),
        };
        assert!(err.contains("mistral3.rope.dimension_count"));
    }
}
