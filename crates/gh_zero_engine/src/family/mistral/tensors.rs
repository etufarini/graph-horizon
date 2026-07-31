/*
 * gh_zero_engine — Ministral dense tensor contract
 * Resolves the required `mistral3` logical tensors from an untrusted GGUF tensor
 * table, validates rank, shapes, tied output and profile-specific dtypes, then
 * exposes borrowed descriptors only. No tensor bytes are copied and no backend
 * allocation or dtype widening is performed here.
*/

use color_eyre::eyre::{Result, bail};

use crate::backend::source::{WeightLayout, WeightSource};
use crate::gguf::tensor_index::{GgmlType, TensorIndex, TensorInfo};

use super::config::MistralConfig;
use super::detect::WeightProfile;

pub(crate) struct MistralTensors<'a> {
    pub(crate) token_embd: &'a TensorInfo,
    pub(crate) output_norm: &'a TensorInfo,
    pub(crate) output: OutputTensor<'a>,
    pub(crate) layers: Vec<MistralLayer<'a>>,
}

pub(crate) enum OutputTensor<'a> {
    Tied,
    Dedicated(&'a TensorInfo),
}

pub(crate) struct MistralLayer<'a> {
    pub(crate) attn_norm: &'a TensorInfo,
    pub(crate) attn_q: &'a TensorInfo,
    pub(crate) attn_k: &'a TensorInfo,
    pub(crate) attn_v: &'a TensorInfo,
    pub(crate) attn_output: &'a TensorInfo,
    pub(crate) ffn_norm: &'a TensorInfo,
    pub(crate) ffn_gate: &'a TensorInfo,
    pub(crate) ffn_up: &'a TensorInfo,
    pub(crate) ffn_down: &'a TensorInfo,
}

impl<'a> MistralTensors<'a> {
    pub(crate) fn build(
        cfg: &MistralConfig,
        profile: WeightProfile,
        idx: &TensorIndex<'a>,
    ) -> Result<Self> {
        let token_embd = matrix(
            idx,
            "token_embd.weight",
            cfg.embedding_length,
            cfg.vocab_size,
            profile,
        )?;
        let output_norm = norm(idx, "output_norm.weight", cfg.embedding_length)?;
        let output = match idx.get("output.weight") {
            Some(t) => OutputTensor::Dedicated(check_matrix(
                t,
                "output.weight",
                cfg.embedding_length,
                cfg.vocab_size,
                profile,
            )?),
            None => OutputTensor::Tied,
        };

        let mut layers = Vec::with_capacity(cfg.block_count);
        for i in 0..cfg.block_count {
            let p = |name: &str| format!("blk.{i}.{name}");
            layers.push(MistralLayer {
                attn_norm: norm(idx, &p("attn_norm.weight"), cfg.embedding_length)?,
                attn_q: matrix(
                    idx,
                    &p("attn_q.weight"),
                    cfg.embedding_length,
                    cfg.q_width,
                    profile,
                )?,
                attn_k: matrix(
                    idx,
                    &p("attn_k.weight"),
                    cfg.embedding_length,
                    cfg.k_width,
                    profile,
                )?,
                attn_v: matrix(
                    idx,
                    &p("attn_v.weight"),
                    cfg.embedding_length,
                    cfg.v_width,
                    profile,
                )?,
                attn_output: matrix(
                    idx,
                    &p("attn_output.weight"),
                    cfg.attention_width,
                    cfg.embedding_length,
                    profile,
                )?,
                ffn_norm: norm(idx, &p("ffn_norm.weight"), cfg.embedding_length)?,
                ffn_gate: matrix(
                    idx,
                    &p("ffn_gate.weight"),
                    cfg.embedding_length,
                    cfg.feed_forward_length,
                    profile,
                )?,
                ffn_up: matrix(
                    idx,
                    &p("ffn_up.weight"),
                    cfg.embedding_length,
                    cfg.feed_forward_length,
                    profile,
                )?,
                ffn_down: matrix(
                    idx,
                    &p("ffn_down.weight"),
                    cfg.feed_forward_length,
                    cfg.embedding_length,
                    profile,
                )?,
            });
        }

        Ok(Self {
            token_embd,
            output_norm,
            output,
            layers,
        })
    }
}

impl WeightSource for MistralTensors<'_> {
    fn tensors(&self) -> Vec<&TensorInfo> {
        let mut out = Vec::with_capacity(2 + self.layers.len() * 9 + 1);
        out.push(self.token_embd);
        out.push(self.output_norm);
        if let OutputTensor::Dedicated(t) = self.output {
            out.push(t);
        }
        for layer in &self.layers {
            out.extend([
                layer.attn_norm,
                layer.attn_q,
                layer.attn_k,
                layer.attn_v,
                layer.attn_output,
                layer.ffn_norm,
                layer.ffn_gate,
                layer.ffn_up,
                layer.ffn_down,
            ]);
        }
        out
    }

    fn layout(&self) -> WeightLayout {
        WeightLayout {
            has_output: matches!(self.output, OutputTensor::Dedicated(_)),
            layer_count: self.layers.len(),
        }
    }
}

fn norm<'a>(idx: &TensorIndex<'a>, name: &str, len: usize) -> Result<&'a TensorInfo> {
    let t = idx
        .get(name)
        .ok_or_else(|| color_eyre::eyre::eyre!("E07 missing or invalid tensor '{name}'"))?;
    if t.dims != [len as u64] {
        bail!("E07 missing or invalid tensor '{name}'");
    }
    if t.ggml_type != GgmlType::F32 {
        bail!("E08 unsupported tensor type '{}'", t.ggml_type.name());
    }
    Ok(t)
}

fn matrix<'a>(
    idx: &TensorIndex<'a>,
    name: &str,
    input: usize,
    output: usize,
    profile: WeightProfile,
) -> Result<&'a TensorInfo> {
    let t = idx
        .get(name)
        .ok_or_else(|| color_eyre::eyre::eyre!("E07 missing or invalid tensor '{name}'"))?;
    check_matrix(t, name, input, output, profile)
}

fn check_matrix<'a>(
    t: &'a TensorInfo,
    name: &str,
    input: usize,
    output: usize,
    profile: WeightProfile,
) -> Result<&'a TensorInfo> {
    if t.dims != [input as u64, output as u64] || t.byte_len().is_none() {
        bail!("E07 missing or invalid tensor '{name}'");
    }
    let ok = match profile {
        WeightProfile::Q8_0 => t.ggml_type == GgmlType::Q8_0,
        WeightProfile::Q4_K_M => matches!(t.ggml_type, GgmlType::Q4_K | GgmlType::Q6_K),
    };
    if !ok {
        bail!("E08 unsupported tensor type '{}'", t.ggml_type.name());
    }
    Ok(t)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::family::mistral::version::{
        ATTENTION_WIDTH, HEAD_COUNT, K_WIDTH, KEY_LENGTH, KV_HEAD_COUNT, MAX_CONTEXT, Q_WIDTH,
        REFERENCE_ROWS, ROPE_DIMENSION, ReferenceRow, V_WIDTH, VALUE_LENGTH,
    };

    fn cfg_shape(
        block_count: usize,
        embedding: usize,
        heads: usize,
        kv_heads: usize,
    ) -> MistralConfig {
        MistralConfig {
            block_count,
            context_length: 128,
            embedding_length: embedding,
            feed_forward_length: embedding * 2,
            head_count: heads,
            kv_head_count: kv_heads,
            key_length: 8,
            value_length: 8,
            q_width: heads * 8,
            k_width: kv_heads * 8,
            v_width: kv_heads * 8,
            attention_width: heads * 8,
            rope_dimension: 8,
            rope_freq_base: 1_000_000.0,
            rms_epsilon: 0.00001,
            yarn_factor: 8.0,
            yarn_beta_fast: 32.0,
            yarn_beta_slow: 1.0,
            yarn_log_multiplier: 0.1,
            yarn_original_context: 128,
            attention_temperature_scale: 1.0,
            vocab_size: 32,
            bos_id: 0,
            eos_id: 1,
        }
    }

    fn cfg() -> MistralConfig {
        cfg_shape(1, 32, 4, 2)
    }

    fn exact_cfg(row: ReferenceRow) -> MistralConfig {
        let (blocks, hidden, ffn, _) = row;
        let mut cfg = cfg_shape(blocks, hidden, HEAD_COUNT, KV_HEAD_COUNT);
        cfg.context_length = MAX_CONTEXT;
        cfg.feed_forward_length = ffn;
        cfg.key_length = KEY_LENGTH;
        cfg.value_length = VALUE_LENGTH;
        cfg.q_width = Q_WIDTH;
        cfg.k_width = K_WIDTH;
        cfg.v_width = V_WIDTH;
        cfg.attention_width = ATTENTION_WIDTH;
        cfg.rope_dimension = ROPE_DIMENSION;
        cfg
    }

    fn t(name: &str, dims: &[u64], ty: GgmlType) -> TensorInfo {
        TensorInfo {
            name: name.into(),
            dims: dims.to_vec(),
            ggml_type: ty,
            offset: 0,
        }
    }

    fn table_for(cfg: &MistralConfig, ty: GgmlType) -> Vec<TensorInfo> {
        let mut tensors = vec![
            t("token_embd.weight", &[cfg.embedding_length as u64, 32], ty),
            t(
                "output_norm.weight",
                &[cfg.embedding_length as u64],
                GgmlType::F32,
            ),
        ];
        for i in 0..cfg.block_count {
            let p = |name: &str| format!("blk.{i}.{name}");
            tensors.extend([
                t(
                    &p("attn_norm.weight"),
                    &[cfg.embedding_length as u64],
                    GgmlType::F32,
                ),
                t(
                    &p("attn_q.weight"),
                    &[cfg.embedding_length as u64, cfg.q_width as u64],
                    ty,
                ),
                t(
                    &p("attn_k.weight"),
                    &[cfg.embedding_length as u64, cfg.k_width as u64],
                    ty,
                ),
                t(
                    &p("attn_v.weight"),
                    &[cfg.embedding_length as u64, cfg.v_width as u64],
                    ty,
                ),
                t(
                    &p("attn_output.weight"),
                    &[cfg.attention_width as u64, cfg.embedding_length as u64],
                    ty,
                ),
                t(
                    &p("ffn_norm.weight"),
                    &[cfg.embedding_length as u64],
                    GgmlType::F32,
                ),
                t(
                    &p("ffn_gate.weight"),
                    &[cfg.embedding_length as u64, cfg.feed_forward_length as u64],
                    ty,
                ),
                t(
                    &p("ffn_up.weight"),
                    &[cfg.embedding_length as u64, cfg.feed_forward_length as u64],
                    ty,
                ),
                t(
                    &p("ffn_down.weight"),
                    &[cfg.feed_forward_length as u64, cfg.embedding_length as u64],
                    ty,
                ),
            ]);
        }
        tensors
    }

    fn table(ty: GgmlType) -> Vec<TensorInfo> {
        table_for(&cfg(), ty)
    }

    #[test]
    fn accepts_q8_and_ties_missing_output() {
        let tensors = table(GgmlType::Q8_0);
        let idx = TensorIndex::new(&tensors);
        let set = MistralTensors::build(&cfg(), WeightProfile::Q8_0, &idx).unwrap();
        assert!(matches!(set.output, OutputTensor::Tied));
        assert_eq!(set.layers.len(), 1);
    }

    #[test]
    fn weight_source_uses_canonical_deduplicated_order() {
        let cfg = cfg();
        let tensors = table_for(&cfg, GgmlType::Q8_0);
        let idx = TensorIndex::new(&tensors);
        let set = MistralTensors::build(&cfg, WeightProfile::Q8_0, &idx).unwrap();
        let names: Vec<&str> = set.tensors().iter().map(|t| t.name.as_str()).collect();
        assert_eq!(
            names,
            [
                "token_embd.weight",
                "output_norm.weight",
                "blk.0.attn_norm.weight",
                "blk.0.attn_q.weight",
                "blk.0.attn_k.weight",
                "blk.0.attn_v.weight",
                "blk.0.attn_output.weight",
                "blk.0.ffn_norm.weight",
                "blk.0.ffn_gate.weight",
                "blk.0.ffn_up.weight",
                "blk.0.ffn_down.weight",
            ]
        );
    }

    #[test]
    fn accepts_q4_profile_mixed_q4_and_q6_matrices() {
        let mut tensors = table(GgmlType::Q4_K);
        tensors[4].ggml_type = GgmlType::Q6_K;
        let idx = TensorIndex::new(&tensors);
        assert!(MistralTensors::build(&cfg(), WeightProfile::Q4_K_M, &idx).is_ok());
    }

    #[test]
    fn error_matrix_e07_e08_rejects_bad_shape_and_bad_profile_dtype() {
        let mut tensors = table(GgmlType::Q8_0);
        tensors[3].dims = vec![32, 64];
        let idx = TensorIndex::new(&tensors);
        let err = match MistralTensors::build(&cfg(), WeightProfile::Q8_0, &idx) {
            Ok(_) => panic!("bad Q shape must fail"),
            Err(err) => err.to_string(),
        };
        assert!(err.contains("E07 missing or invalid tensor 'blk.0.attn_q.weight'"));

        let tensors = table(GgmlType::Q5_K);
        let idx = TensorIndex::new(&tensors);
        let err = match MistralTensors::build(&cfg(), WeightProfile::Q4_K_M, &idx) {
            Ok(_) => panic!("Q5_K must not pass the Q4_K_M contract"),
            Err(err) => err.to_string(),
        };
        assert!(err.contains("E08 unsupported tensor type 'Q5_K'"));
    }

    #[test]
    fn versioned_rows_accept_both_profiles() {
        for row in REFERENCE_ROWS {
            let dedicated = row.3;
            let cfg = exact_cfg(row);
            let q8 = table_for(&cfg, GgmlType::Q8_0);
            let mut q8 = q8;
            if dedicated {
                q8.push(t(
                    "output.weight",
                    &[cfg.embedding_length as u64, cfg.vocab_size as u64],
                    GgmlType::Q8_0,
                ));
            }
            let q8_set =
                MistralTensors::build(&cfg, WeightProfile::Q8_0, &TensorIndex::new(&q8)).unwrap();

            let mut q4 = table_for(&cfg, GgmlType::Q4_K);
            q4.iter_mut()
                .find(|tensor| tensor.name == "blk.0.ffn_down.weight")
                .unwrap()
                .ggml_type = GgmlType::Q6_K;
            if dedicated {
                q4.push(t(
                    "output.weight",
                    &[cfg.embedding_length as u64, cfg.vocab_size as u64],
                    GgmlType::Q6_K,
                ));
            }
            let q4_set =
                MistralTensors::build(&cfg, WeightProfile::Q4_K_M, &TensorIndex::new(&q4)).unwrap();

            assert_eq!(q8_set.layers.len(), cfg.block_count);
            assert_eq!(q4_set.layers.len(), cfg.block_count);
            assert_eq!(q8_set.layers[0].attn_q.dims[1], Q_WIDTH as u64);
            assert_eq!(q8_set.layers[0].attn_k.dims[1], K_WIDTH as u64);
            assert_eq!(q8_set.layers[0].attn_v.dims[1], V_WIDTH as u64);
            assert_eq!(q8_set.layers[0].attn_output.dims[0], ATTENTION_WIDTH as u64);
            assert_eq!(
                matches!(q8_set.output, OutputTensor::Dedicated(_)),
                dedicated
            );
            assert_eq!(
                matches!(q4_set.output, OutputTensor::Dedicated(_)),
                dedicated
            );
            println!(
                "blocks={} hidden={} q={Q_WIDTH} k={K_WIDTH} v={V_WIDTH} attention={ATTENTION_WIDTH} ffn={} output={}",
                cfg.block_count,
                cfg.embedding_length,
                cfg.feed_forward_length,
                if dedicated { "dedicated" } else { "tied" }
            );
        }
    }

    #[test]
    fn error_matrix_e07_e08_rejects_missing_tensor_dtype_and_bad_output() {
        let mut tensors = table(GgmlType::Q8_0);
        tensors.retain(|t| t.name != "blk.0.attn_v.weight");
        let err =
            match MistralTensors::build(&cfg(), WeightProfile::Q8_0, &TensorIndex::new(&tensors)) {
                Ok(_) => panic!("missing V tensor must fail"),
                Err(err) => err.to_string(),
            };
        assert!(err.contains("E07 missing or invalid tensor 'blk.0.attn_v.weight'"));

        let mut tensors = table(GgmlType::Q8_0);
        tensors[1].ggml_type = GgmlType::F16;
        let err =
            match MistralTensors::build(&cfg(), WeightProfile::Q8_0, &TensorIndex::new(&tensors)) {
                Ok(_) => panic!("F16 norm must fail"),
                Err(err) => err.to_string(),
            };
        assert!(err.contains("E08 unsupported tensor type 'F16'"));

        let tensors = table(GgmlType::F16);
        let err =
            match MistralTensors::build(&cfg(), WeightProfile::Q8_0, &TensorIndex::new(&tensors)) {
                Ok(_) => panic!("F16 matrix must fail"),
                Err(err) => err.to_string(),
            };
        assert!(err.contains("E08 unsupported tensor type 'F16'"));

        let mut tensors = table(GgmlType::Q8_0);
        tensors.push(t("output.weight", &[64, 32], GgmlType::Q8_0));
        let err =
            match MistralTensors::build(&cfg(), WeightProfile::Q8_0, &TensorIndex::new(&tensors)) {
                Ok(_) => panic!("bad dedicated output must fail"),
                Err(err) => err.to_string(),
            };
        assert!(err.contains("E07 missing or invalid tensor 'output.weight'"));
    }
}
