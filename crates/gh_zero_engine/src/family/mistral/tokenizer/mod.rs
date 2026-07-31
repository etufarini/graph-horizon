/*
 * gh_zero_engine — Tekken tokenizer metadata contract
 * Loads the GGUF byte-level BPE tokenizer metadata required by Ministral 3:
 * Tekken pre-tokenizer marker, unique vocabulary, merge ranks, token types and
 * structural control ids and the immutable Reasoning 2512 policy shared by
 * 3B/8B/14B. Ordinary `encode` treats caller text as untrusted bytes and never
 * emits structural ids merely because their spelling appears.
*/

use std::collections::HashMap;

use color_eyre::eyre::{Result, bail, eyre};

use crate::gguf::loader::GgufValue;

mod bpe;
mod profile;
mod reasoning;
mod tekken;

pub struct TekkenTokenizer {
    id_to_token: Vec<String>,
    token_to_id: HashMap<String, u32>,
    merge_rank: HashMap<String, u32>,
    byte_encoder: [char; 256],
    byte_decoder: HashMap<char, u8>,
    special: TekkenSpecial,
    profile: profile::ChatProfile,
}

#[derive(Clone, Copy)]
struct TekkenSpecial {
    bos: u32,
    eos: u32,
    inst_open: u32,
    inst_close: u32,
    system_open: u32,
    system_close: u32,
}

impl TekkenTokenizer {
    pub fn from_metadata(md: &HashMap<String, GgufValue>) -> Result<Self> {
        let profile = profile::classify(md)?;
        required_str(md, "tokenizer.ggml.model", "gpt2")?;
        required_str(md, "tokenizer.ggml.pre", "tekken")?;
        if !matches!(
            md.get("tokenizer.ggml.add_bos_token"),
            Some(GgufValue::Bool(true))
        ) {
            bail!("E09 invalid Tekken tokenizer");
        }
        let tokens = required_array(md, "tokenizer.ggml.tokens")?;
        let token_type = required_array(md, "tokenizer.ggml.token_type")?;
        if tokens.is_empty() || tokens.len() > u32::MAX as usize || token_type.len() != tokens.len()
        {
            bail!("E09 invalid Tekken tokenizer");
        }
        let mut id_to_token = Vec::with_capacity(tokens.len());
        let mut token_to_id = HashMap::with_capacity(tokens.len());
        for (i, value) in tokens.iter().enumerate() {
            let token = value
                .as_str()
                .ok_or_else(|| eyre!("E09 invalid Tekken tokenizer"))?;
            if token_to_id.insert(token.to_string(), i as u32).is_some() {
                bail!("E09 invalid Tekken tokenizer");
            }
            id_to_token.push(token.to_string());
        }
        reasoning::validate(profile, &token_to_id)?;
        let (byte_encoder, byte_decoder) = bpe::byte_maps();
        if token_type
            .iter()
            .any(|value| !matches!(value.as_u64(), Some(1..=6)))
        {
            bail!("E09 invalid Tekken tokenizer");
        }
        let merge_rank = bpe::load_merges(required_array(md, "tokenizer.ggml.merges")?)?;
        for merge in merge_rank.keys() {
            let (left, right) = merge.split_once(' ').expect("validated merge");
            let joined = format!("{left}{right}");
            if !token_to_id.contains_key(left)
                || !token_to_id.contains_key(right)
                || !token_to_id.contains_key(&joined)
            {
                bail!("E09 invalid Tekken tokenizer");
            }
        }
        let id = |s: &str| {
            token_to_id
                .get(s)
                .copied()
                .ok_or_else(|| eyre!("E09 invalid Tekken tokenizer"))
        };
        let special = TekkenSpecial {
            bos: required_id(md, "tokenizer.ggml.bos_token_id", tokens.len())?,
            eos: required_id(md, "tokenizer.ggml.eos_token_id", tokens.len())?,
            inst_open: id("[INST]")?,
            inst_close: id("[/INST]")?,
            system_open: id("[SYSTEM_PROMPT]")?,
            system_close: id("[/SYSTEM_PROMPT]")?,
        };
        if special.bos == special.eos
            || is_structural(&id_to_token[special.bos as usize])
            || is_structural(&id_to_token[special.eos as usize])
        {
            bail!("E09 invalid Tekken tokenizer");
        }
        Ok(Self {
            id_to_token,
            token_to_id,
            merge_rank,
            byte_encoder,
            byte_decoder,
            special,
            profile,
        })
    }

    pub fn vocab_size(&self) -> usize {
        self.id_to_token.len()
    }
    pub fn bos_id(&self) -> u32 {
        self.special.bos
    }
    pub fn eos_id(&self) -> u32 {
        self.special.eos
    }
    pub fn inst_open_id(&self) -> u32 {
        self.special.inst_open
    }
    pub fn inst_close_id(&self) -> u32 {
        self.special.inst_close
    }
    pub fn system_open_id(&self) -> u32 {
        self.special.system_open
    }
    pub fn system_close_id(&self) -> u32 {
        self.special.system_close
    }
    pub(crate) fn uses_reasoning_profile(&self) -> bool {
        self.profile == profile::ChatProfile::Reasoning2512
    }
    pub(crate) fn encode_reasoning_system(&self, prompt: &str) -> Vec<u32> {
        reasoning::encode(self, prompt)
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut out = Vec::new();
        for piece in tekken::segments(text) {
            let word: String = piece
                .bytes()
                .map(|b| self.byte_encoder[b as usize])
                .collect();
            if let Some(&id) = self.token_to_id.get(&word)
                && !is_structural(&word)
            {
                // `ignore_merges=true`: a complete segment wins before BPE.
                out.push(id);
                continue;
            }
            let symbols = bpe::encode_piece(&self.merge_rank, &word);
            for sym in symbols {
                if let Some(&id) = self.token_to_id.get(&sym) {
                    // Literal structural spellings must fall back to byte tokens.
                    if is_structural(&sym) {
                        out.extend(
                            sym.chars()
                                .filter_map(|c| self.token_to_id.get(&c.to_string()).copied()),
                        );
                    } else {
                        out.push(id);
                    }
                } else {
                    out.extend(
                        sym.chars()
                            .filter_map(|c| self.token_to_id.get(&c.to_string()).copied()),
                    );
                }
            }
        }
        out
    }

    pub fn decode_bytes(&self, ids: &[u32]) -> Vec<u8> {
        ids.iter()
            .filter_map(|id| self.id_to_token.get(*id as usize))
            .flat_map(|token| token.chars())
            .filter_map(|ch| self.byte_decoder.get(&ch).copied())
            .collect()
    }
}

fn required_array<'a>(md: &'a HashMap<String, GgufValue>, key: &str) -> Result<&'a [GgufValue]> {
    md.get(key)
        .and_then(|v| v.as_array())
        .ok_or_else(|| eyre!("E09 invalid Tekken tokenizer"))
}

fn required_str(md: &HashMap<String, GgufValue>, key: &str, expected: &str) -> Result<()> {
    (md.get(key).and_then(|v| v.as_str()) == Some(expected))
        .then_some(())
        .ok_or_else(|| eyre!("E09 invalid Tekken tokenizer"))
}

fn required_id(md: &HashMap<String, GgufValue>, key: &str, vocab: usize) -> Result<u32> {
    md.get(key)
        .and_then(|v| v.as_u64())
        .filter(|id| *id < vocab as u64)
        .map(|id| id as u32)
        .ok_or_else(|| eyre!("E09 invalid Tekken tokenizer"))
}

fn is_structural(token: &str) -> bool {
    matches!(
        token,
        "[INST]" | "[/INST]" | "[SYSTEM_PROMPT]" | "[/SYSTEM_PROMPT]"
    )
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    pub(crate) fn mini_tokenizer() -> TekkenTokenizer {
        TekkenTokenizer::from_metadata(&mini_md()).unwrap()
    }

    pub(crate) fn mini_reasoning_tokenizer() -> TekkenTokenizer {
        mini_reasoning_tokenizer_for("ministral-3B-Reasoning-2512")
    }

    fn mini_reasoning_tokenizer_for(name: &str) -> TekkenTokenizer {
        let mut md = mini_md();
        md.insert("general.name".into(), GgufValue::String(name.into()));
        if let Some(GgufValue::Array(tokens)) = md.get_mut("tokenizer.ggml.tokens") {
            tokens.extend([
                GgufValue::String("[THINK]".into()),
                GgufValue::String("[/THINK]".into()),
            ]);
        }
        if let Some(GgufValue::Array(types)) = md.get_mut("tokenizer.ggml.token_type") {
            types.extend([GgufValue::U32(4), GgufValue::U32(4)]);
        }
        TekkenTokenizer::from_metadata(&md).unwrap()
    }

    fn mini_md() -> HashMap<String, GgufValue> {
        let (byte_encoder, _) = bpe::byte_maps();
        let mut tokens = byte_encoder
            .iter()
            .map(|c| c.to_string())
            .collect::<Vec<_>>();
        tokens.extend([
            "<s>".into(),
            "</s>".into(),
            "[INST]".into(),
            "[/INST]".into(),
            "[SYSTEM_PROMPT]".into(),
            "[/SYSTEM_PROMPT]".into(),
        ]);
        let values = tokens
            .into_iter()
            .map(GgufValue::String)
            .collect::<Vec<_>>();
        let len = values.len();
        HashMap::from([
            (
                "tokenizer.ggml.model".into(),
                GgufValue::String("gpt2".into()),
            ),
            (
                "tokenizer.ggml.pre".into(),
                GgufValue::String("tekken".into()),
            ),
            ("tokenizer.ggml.tokens".into(), GgufValue::Array(values)),
            (
                "tokenizer.ggml.token_type".into(),
                GgufValue::Array((0..len).map(|_| GgufValue::U32(1)).collect()),
            ),
            ("tokenizer.ggml.merges".into(), GgufValue::Array(Vec::new())),
            ("tokenizer.ggml.bos_token_id".into(), GgufValue::U32(256)),
            ("tokenizer.ggml.eos_token_id".into(), GgufValue::U32(257)),
            ("tokenizer.ggml.add_bos_token".into(), GgufValue::Bool(true)),
        ])
    }

    #[test]
    fn validates_required_metadata_and_control_tokens() {
        let tok = mini_tokenizer();
        assert!(!tok.uses_reasoning_profile());
        assert_eq!(tok.vocab_size(), 262);
        assert_eq!(tok.inst_open_id(), 258);
        let mut bad = mini_md();
        bad.insert(
            "tokenizer.ggml.pre".into(),
            GgufValue::String("qwen2".into()),
        );
        assert!(TekkenTokenizer::from_metadata(&bad).is_err());
    }

    #[test]
    fn reasoning_profile_requires_release_marker_ids() {
        for name in [
            "ministral-3B-Reasoning-2512",
            "ministral-8B-Reasoning-2512",
            "ministral-14B-Reasoning-2512",
        ] {
            assert!(mini_reasoning_tokenizer_for(name).uses_reasoning_profile());
            let mut md = mini_md();
            md.insert("general.name".into(), GgufValue::String(name.into()));
            assert_eq!(
                TekkenTokenizer::from_metadata(&md)
                    .err()
                    .expect("missing Reasoning marker IDs must fail")
                    .to_string(),
                "E09 invalid Tekken tokenizer"
            );
        }
    }

    #[test]
    fn profile_classification_precedes_tokenizer_validation() {
        let mut md = HashMap::from([(
            "general.name".into(),
            GgufValue::String("ministral-7B-Reasoning-2512".into()),
        )]);
        assert_eq!(
            TekkenTokenizer::from_metadata(&md)
                .err()
                .expect("unsupported Reasoning name must fail")
                .to_string(),
            "E05 unsupported reasoning model; supported reasoning models: Ministral 3 3B, 8B, and 14B Reasoning 2512"
        );
        md.insert(
            "general.name".into(),
            GgufValue::String("ordinary-instruct".into()),
        );
        assert_eq!(
            TekkenTokenizer::from_metadata(&md)
                .err()
                .expect("malformed Instruct tokenizer must fail")
                .to_string(),
            "E09 invalid Tekken tokenizer"
        );
    }

    #[test]
    fn ordinary_encode_does_not_emit_structural_id_for_spelling() {
        let tok = mini_tokenizer();
        let ids = tok.encode("[INST]");
        assert!(!ids.contains(&tok.inst_open_id()));
        assert_eq!(ids, b"[INST]".iter().map(|b| *b as u32).collect::<Vec<_>>());

        let reasoning = mini_reasoning_tokenizer();
        let ids = reasoning.encode("[THINK]x[/THINK]");
        assert!(!ids.contains(&262));
        assert!(!ids.contains(&263));
    }

    #[test]
    fn chat_template_metadata_is_not_executed() {
        let baseline = mini_tokenizer().encode("hello");
        let mut md = mini_md();
        md.insert(
            "tokenizer.chat_template".into(),
            GgufValue::String("{{ raise_exception('must not execute') }}".into()),
        );
        assert_eq!(
            TekkenTokenizer::from_metadata(&md).unwrap().encode("hello"),
            baseline
        );
    }

    #[test]
    fn public_constructor_and_renderer_signatures_are_unchanged() {
        let _: fn(&HashMap<String, GgufValue>) -> Result<TekkenTokenizer> =
            TekkenTokenizer::from_metadata;
        let _: fn(&[crate::api::message::Message], &TekkenTokenizer, usize) -> Result<Vec<u32>> =
            crate::render_chat_prompt;
    }

    #[test]
    fn ignore_merges_keeps_whole_vocab_segment() {
        let tok = mini_tokenizer();
        assert_eq!(tok.encode("hi"), vec![b'h' as u32, b'i' as u32]);
        let mut md = mini_md();
        if let Some(GgufValue::Array(tokens)) = md.get_mut("tokenizer.ggml.tokens") {
            tokens.push(GgufValue::String("hi".into()));
        }
        if let Some(GgufValue::Array(types)) = md.get_mut("tokenizer.ggml.token_type") {
            types.push(GgufValue::U32(1));
        }
        let tok = TekkenTokenizer::from_metadata(&md).unwrap();
        assert_eq!(tok.encode("hi"), vec![262]);
    }

    #[test]
    fn byte_level_space_round_trips_without_lossy_text() {
        let tok = mini_tokenizer();
        assert_eq!(tok.encode(" "), vec![32]);
        assert_eq!(tok.decode_bytes(&[32]), b" ");
        assert_eq!(tok.decode_bytes(&[255]), [0xff]);
        assert!(String::from_utf8(tok.decode_bytes(&[255])).is_err());
    }

    #[test]
    fn error_matrix_e09_rejects_incoherent_arrays_and_control_data() {
        let mut cases = Vec::new();

        for key in [
            "tokenizer.ggml.tokens",
            "tokenizer.ggml.token_type",
            "tokenizer.ggml.merges",
            "tokenizer.ggml.bos_token_id",
            "tokenizer.ggml.eos_token_id",
        ] {
            let mut missing = mini_md();
            missing.remove(key);
            cases.push(missing);
        }

        let mut duplicate = mini_md();
        if let Some(GgufValue::Array(tokens)) = duplicate.get_mut("tokenizer.ggml.tokens") {
            tokens[1] = GgufValue::String("Ā".into());
        }
        cases.push(duplicate);

        let mut wrong_vocab_type = mini_md();
        if let Some(GgufValue::Array(tokens)) = wrong_vocab_type.get_mut("tokenizer.ggml.tokens") {
            tokens[0] = GgufValue::U32(0);
        }
        cases.push(wrong_vocab_type);

        for token_type in [GgufValue::String("normal".into()), GgufValue::U32(7)] {
            let mut bad_type = mini_md();
            if let Some(GgufValue::Array(types)) = bad_type.get_mut("tokenizer.ggml.token_type") {
                types[0] = token_type;
            }
            cases.push(bad_type);
        }

        let mut bad_merge = mini_md();
        bad_merge.insert(
            "tokenizer.ggml.merges".into(),
            GgufValue::Array(vec![GgufValue::String("a absent".into())]),
        );
        cases.push(bad_merge);

        let mut wrong_merge_type = mini_md();
        wrong_merge_type.insert(
            "tokenizer.ggml.merges".into(),
            GgufValue::Array(vec![GgufValue::U32(0)]),
        );
        cases.push(wrong_merge_type);

        let mut duplicate_merge = mini_md();
        if let Some(GgufValue::Array(tokens)) = duplicate_merge.get_mut("tokenizer.ggml.tokens") {
            tokens.push(GgufValue::String("ab".into()));
        }
        if let Some(GgufValue::Array(values)) = duplicate_merge.get_mut("tokenizer.ggml.token_type")
        {
            values.push(GgufValue::U32(1));
        }
        duplicate_merge.insert(
            "tokenizer.ggml.merges".into(),
            GgufValue::Array(vec![
                GgufValue::String("a b".into()),
                GgufValue::String("a b".into()),
            ]),
        );
        cases.push(duplicate_merge);

        let mut out_of_range = mini_md();
        out_of_range.insert("tokenizer.ggml.bos_token_id".into(), GgufValue::U32(999));
        cases.push(out_of_range);

        let mut wrong_id_type = mini_md();
        wrong_id_type.insert(
            "tokenizer.ggml.eos_token_id".into(),
            GgufValue::String("eos".into()),
        );
        cases.push(wrong_id_type);

        let mut duplicate_ids = mini_md();
        duplicate_ids.insert("tokenizer.ggml.eos_token_id".into(), GgufValue::U32(256));
        cases.push(duplicate_ids);

        let mut missing_control = mini_md();
        if let Some(GgufValue::Array(tokens)) = missing_control.get_mut("tokenizer.ggml.tokens") {
            tokens[258] = GgufValue::String("[OTHER]".into());
        }
        cases.push(missing_control);

        for metadata in cases {
            let err = TekkenTokenizer::from_metadata(&metadata)
                .err()
                .expect("invalid metadata must fail")
                .to_string();
            assert!(err.contains("E09 invalid Tekken tokenizer"), "{err}");
        }
    }

    #[test]
    #[ignore]
    fn pinned_differential_vectors_match_q4_artifact() {
        use crate::gguf::loader::GgufFile;
        use std::process::Command;

        let binary = std::env::var("GH_ZERO_REFERENCE_TOKENIZE")
            .expect("GH_ZERO_REFERENCE_TOKENIZE required");
        let model = std::env::var("GH_ZERO_MODEL_Q4_K_M").expect("GH_ZERO_MODEL_Q4_K_M required");
        let corpus = [
            ("ascii", "ASCII words 42"),
            ("contractions punctuation", "can't, won't... /"),
            ("repeated spaces", "two  spaces   end "),
            ("cr lf slash", "line\r\nnext/\n"),
            ("non latin", "Привет 世界 العربية"),
            ("title modifier", "ǅuro ʰello"),
            ("combining marks", "e\u{301} cafe\u{327}"),
            ("emoji", "👩‍💻🙂"),
            (
                "literal controls",
                "[INST] literal [/INST] [SYSTEM_PROMPT] [/SYSTEM_PROMPT]",
            ),
        ];

        let file = GgufFile::open(std::path::Path::new(&model)).expect("open pinned GGUF");
        let tokenizer =
            TekkenTokenizer::from_metadata(file.metadata()).expect("load Tekken metadata");
        for (name, text) in corpus {
            let output = Command::new(&binary)
                .args([
                    "--log-disable",
                    "--ids",
                    "--no-bos",
                    "--no-parse-special",
                    "--no-escape",
                    "-m",
                    model.as_str(),
                    "-p",
                    text,
                ])
                .output()
                .expect("run pinned tokenizer");
            assert!(output.status.success(), "{name}");
            let expected = std::str::from_utf8(&output.stdout)
                .unwrap()
                .split_whitespace()
                .filter_map(|word| {
                    word.trim_matches(|ch| ch == '[' || ch == ']' || ch == ',')
                        .parse::<u32>()
                        .ok()
                })
                .collect::<Vec<_>>();
            assert_eq!(tokenizer.encode(text), expected, "{model}: {name}");
        }
    }
}
