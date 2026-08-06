/*
 * Graph Horizon headless server properties vocabulary
 * Single responsibility: serialize resolved positive chat capacity settings
 * into the public `/props` response without exposing engine internals.
 */

use serde::Serialize;

#[derive(Serialize)]
pub(crate) struct Properties {
    default_generation_settings: GenerationSettings,
}

#[derive(Serialize)]
struct GenerationSettings {
    n_ctx: u32,
    max_tokens: usize,
}

pub(crate) fn payload(context_limit: u32, max_tokens: usize) -> Properties {
    Properties {
        default_generation_settings: GenerationSettings {
            n_ctx: context_limit,
            max_tokens,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn props_payload_reports_only_capacity_settings() {
        let payload = serde_json::to_value(payload(u32::MAX, 4096)).unwrap();

        assert_eq!(
            payload,
            serde_json::json!({
                "default_generation_settings": {
                    "n_ctx": u32::MAX,
                    "max_tokens": 4096
                }
            })
        );
        assert!(payload["default_generation_settings"]["n_ctx"].is_u64());
        assert!(payload["default_generation_settings"]["max_tokens"].is_u64());

        let serialized = payload.to_string();
        for prohibited in ["path", "metadata", "backend", "memory", "stack"] {
            assert!(!serialized.contains(prohibited));
        }
    }
}
