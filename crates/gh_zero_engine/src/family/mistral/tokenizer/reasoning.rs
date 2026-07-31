/*
 * gh_zero_engine — release-owned Reasoning marker encoding
 * Validates the two existing GGUF marker IDs for the Reasoning profile and
 * inserts them only while encoding the fixed implicit system prompt. All
 * caller-owned text continues through the ordinary tokenizer path.
 */

use std::collections::HashMap;

use color_eyre::eyre::{Result, bail};

use super::{TekkenTokenizer, profile::ChatProfile};

const OPEN: &str = "[THINK]";
const CLOSE: &str = "[/THINK]";

pub(super) fn validate(profile: ChatProfile, tokens: &HashMap<String, u32>) -> Result<()> {
    if profile == ChatProfile::Reasoning3B2512
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
