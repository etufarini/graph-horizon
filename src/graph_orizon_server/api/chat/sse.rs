/*
 * Graph Orizon headless server - API - Chat - SSE
 * Server-facing imports and fixture tests for the shared OpenAI-compatible SSE
 * chunk envelope. The actual serialization lives in app::sse so the headless
 * server and web surface use one chat-chunk shape.
 */

pub(super) use crate::app::sse::{Delta, delta_line};
pub(crate) use crate::app::sse::{done_line, final_line, usage_line};

#[cfg(test)]
mod tests {
    use super::*;
    use graph_orizon_engine::GenerationStats;

    #[test]
    fn final_line_carries_stop() {
        let line = final_line();
        let json: serde_json::Value =
            serde_json::from_str(line.trim_start_matches("data: ").trim_end()).unwrap();
        assert_eq!(json["choices"][0]["finish_reason"], "stop");
    }

    #[test]
    fn done_line_is_exact() {
        assert_eq!(done_line(), "data: [DONE]\n\n");
    }

    #[test]
    fn usage_line_carries_exact_stats() {
        let line = usage_line(&GenerationStats {
            prompt_tokens: 128,
            completion_tokens: 42,
            prefill_ms: 400,
            decode_ms: 875,
        });
        assert!(line.starts_with("data: "));
        assert!(line.ends_with("\n\n"));
        let json: serde_json::Value =
            serde_json::from_str(line.trim_start_matches("data: ").trim_end()).unwrap();
        assert_eq!(json["usage"]["prompt_tokens"], 128);
        assert_eq!(json["usage"]["completion_tokens"], 42);
        assert_eq!(json["usage"]["prefill_ms"], 400);
        assert_eq!(json["usage"]["decode_ms"], 875);
        assert_eq!(json["usage"].as_object().unwrap().len(), 4);
        assert!(json["choices"][0]["finish_reason"].is_null());
    }

    #[test]
    fn delta_and_final_lines_have_no_usage_key() {
        let delta = delta_line(Delta {
            content: Some("ciao".to_string()),
        });
        assert!(!delta.contains("\"usage\""));
        assert!(!final_line().contains("\"usage\""));
    }
}
