/*
 * gh_zero_engine — release-owned Reasoning marker encoding
 * Validates the two existing GGUF marker IDs for the single Reasoning2512
 * policy shared by 3B/8B/14B and inserts them only while encoding the fixed
 * implicit system prompt. Caller-owned text uses the ordinary tokenizer path.
 */

use std::collections::HashMap;

use color_eyre::eyre::{Result, bail};

use super::{TekkenTokenizer, profile::ChatProfile};

const OPEN: &str = "[THINK]";
const CLOSE: &str = "[/THINK]";

pub(super) fn validate(profile: ChatProfile, tokens: &HashMap<String, u32>) -> Result<()> {
    if profile == ChatProfile::Reasoning2512
        && (!tokens.contains_key(OPEN) || !tokens.contains_key(CLOSE))
    {
        bail!("E09 invalid Tekken tokenizer");
    }
    Ok(())
}

pub(super) fn encode(tokenizer: &TekkenTokenizer, prompt: &str) -> Vec<u32> {
    let (before, rest) = prompt.split_once(OPEN).expect("fixed prompt has [THINK]");
    let (inside, after) = rest.split_once(CLOSE).expect("fixed prompt has [/THINK]");
    let mut out = tokenizer.encode(before);
    out.push(tokenizer.token_to_id[OPEN]);
    out.extend(tokenizer.encode(inside));
    out.push(tokenizer.token_to_id[CLOSE]);
    out.extend(tokenizer.encode(after));
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::loader::GgufValue;

    fn profile(name: &str) -> ChatProfile {
        super::super::profile::classify(&HashMap::from([(
            "general.name".into(),
            GgufValue::String(name.into()),
        )]))
        .unwrap()
    }

    #[test]
    fn shared_policy_requires_both_release_markers_for_every_size() {
        let names = [
            "ministral-3B-Reasoning-2512",
            "ministral-8B-Reasoning-2512",
            "ministral-14B-Reasoning-2512",
        ];
        for name in names {
            let chat = profile(name);
            let both = HashMap::from([(OPEN.into(), 1), (CLOSE.into(), 2)]);
            assert!(validate(chat, &both).is_ok());
            for tokens in [
                HashMap::from([(OPEN.into(), 1)]),
                HashMap::from([(CLOSE.into(), 2)]),
                HashMap::new(),
            ] {
                assert_eq!(
                    validate(chat, &tokens).unwrap_err().to_string(),
                    "E09 invalid Tekken tokenizer"
                );
            }
        }
    }

    #[test]
    fn instruct_policy_does_not_require_reasoning_markers() {
        assert!(validate(ChatProfile::Instruct, &HashMap::new()).is_ok());
    }
}
