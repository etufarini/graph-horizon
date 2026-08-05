/*
 * graph_horizon_engine — private Ministral chat-profile classification
 * Maps the exact 3B, 8B, and 14B names to one fixed Reasoning 2512 policy from
 * untrusted `general.name`; name metadata never authenticates model weights.
 * Unsupported Reasoning variants fail before backend allocation.
 */

use std::collections::HashMap;

use color_eyre::eyre::{Result, bail};

use crate::gguf::loader::GgufValue;

const REASONING_2512: [&str; 3] = [
    "ministral-3B-Reasoning-2512",
    "ministral-8B-Reasoning-2512",
    "ministral-14B-Reasoning-2512",
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ChatProfile {
    Instruct,
    Reasoning2512,
}

pub(super) fn classify(md: &HashMap<String, GgufValue>) -> Result<ChatProfile> {
    let Some(name) = md.get("general.name").and_then(GgufValue::as_str) else {
        return Ok(ChatProfile::Instruct);
    };
    if REASONING_2512.contains(&name) {
        Ok(ChatProfile::Reasoning2512)
    } else if name.contains("Reasoning") {
        bail!(
            "E05 unsupported reasoning model; supported reasoning models: Ministral 3 3B, 8B, and 14B Reasoning 2512"
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
    fn chat_profile_selects_all_exact_reasoning_names() {
        for name in REASONING_2512 {
            let md = metadata(GgufValue::String(name.into()));
            assert_eq!(classify(&md).unwrap(), ChatProfile::Reasoning2512);
        }
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
            "ordinary-instruct",
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
            "Ministral-3B-Reasoning-2512",
            "ministral-7B-Reasoning-2512",
            "ministral-32B-Reasoning-2512",
        ] {
            assert_eq!(
                classify(&metadata(GgufValue::String(name.into())))
                    .unwrap_err()
                    .to_string(),
                "E05 unsupported reasoning model; supported reasoning models: Ministral 3 3B, 8B, and 14B Reasoning 2512"
            );
        }
    }
}
