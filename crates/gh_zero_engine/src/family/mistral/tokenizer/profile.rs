/*
 * gh_zero_engine — private Ministral chat-profile classification
 * Selects only the fixed chat policy from untrusted `general.name`; it never
 * authenticates a model. Unsupported Reasoning variants are rejected before
 * backend allocation, without inspecting any other artifact property.
 */

use std::collections::HashMap;

use color_eyre::eyre::{Result, bail};

use crate::gguf::loader::GgufValue;

const REASONING_3B_2512: &str = "ministral-3B-Reasoning-2512";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ChatProfile {
    Instruct,
    Reasoning3B2512,
}

pub(super) fn classify(md: &HashMap<String, GgufValue>) -> Result<ChatProfile> {
    let Some(name) = md.get("general.name").and_then(GgufValue::as_str) else {
        return Ok(ChatProfile::Instruct);
    };
    if name == REASONING_3B_2512 {
        Ok(ChatProfile::Reasoning3B2512)
    } else if name.contains("Reasoning") {
        bail!(
            "E05 unsupported reasoning model; supported reasoning model: Ministral 3 3B Reasoning 2512"
        )
    } else {
        Ok(ChatProfile::Instruct)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn metadata(name: GgufValue) -> HashMap<String, GgufValue> {
        HashMap::from([("general.name".into(), name)])
    }

    #[test]
    fn chat_profile_selects_exact_reasoning_name() {
        let md = metadata(GgufValue::String(REASONING_3B_2512.into()));
        assert_eq!(classify(&md).unwrap(), ChatProfile::Reasoning3B2512);
    }

    #[test]
    fn chat_profile_preserves_instruct_for_non_reasoning_names() {
        assert_eq!(classify(&HashMap::new()).unwrap(), ChatProfile::Instruct);
        assert_eq!(
            classify(&metadata(GgufValue::U32(3))).unwrap(),
            ChatProfile::Instruct
        );
        for name in [
            "Ministral-3-3B-Instruct-2512",
            "ministral-3B-reasoning-2512",
        ] {
            assert_eq!(
                classify(&metadata(GgufValue::String(name.into()))).unwrap(),
                ChatProfile::Instruct
            );
        }
    }

    #[test]
    fn chat_profile_rejects_every_other_reasoning_variant() {
        for name in [
            "ministral-3B-Reasoning-2512-modified",
            "ministral-8B-Reasoning-2512",
            "ministral-14B-Reasoning-2512",
            "Ministral-3B-Reasoning-2512",
        ] {
            assert_eq!(
                classify(&metadata(GgufValue::String(name.into())))
                    .unwrap_err()
                    .to_string(),
                "E05 unsupported reasoning model; supported reasoning model: Ministral 3 3B Reasoning 2512"
            );
        }
    }
}
