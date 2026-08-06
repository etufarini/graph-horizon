/*
 * Graph Horizon headless server properties vocabulary
 * Single responsibility: serialize the resolved positive context limit into
 * the public `/props` response without exposing engine internals.
 */

use serde::Serialize;

#[derive(Serialize)]
pub(crate) struct Properties {
    default_generation_settings: GenerationSettings,
}

#[derive(Serialize)]
struct GenerationSettings {
    n_ctx: u32,
}

pub(crate) fn payload(context_limit: u32) -> Properties {
    Properties {
        default_generation_settings: GenerationSettings {
            n_ctx: context_limit,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn props_payload_reports_only_context_limit() {
        let payload = serde_json::to_value(payload(u32::MAX)).unwrap();

        assert_eq!(
            payload,
            serde_json::json!({
                "default_generation_settings": { "n_ctx": u32::MAX }
            })
        );
        assert!(payload["default_generation_settings"]["n_ctx"].is_u64());

        let serialized = payload.to_string();
        for prohibited in ["path", "metadata", "backend", "memory", "stack"] {
            assert!(!serialized.contains(prohibited));
        }
    }
}
