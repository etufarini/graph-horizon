/*
 * Graph Horizon headless server - API - Chat (OpenAI chat wire format)
 * Single responsibility: map validated text-only chat JSON to engine requests
 * and engine text events to SSE lines. It depends on graph_horizon_engine API types
 * and does not include tools, tool_choice, workspace, or reasoning controls.
*/

use graph_horizon_engine::{Event, GenerationPhase, Message, Request, SamplingParams};
use serde::Deserialize;

use super::{ApiError, role_from_str};
use crate::app::sse::{Delta, data_line, delta_line};
use crate::graph_horizon_server::ServerConfig;

pub(crate) use crate::app::sse::{done_line, final_line, usage_line};

pub(crate) fn phase_line(phase: GenerationPhase) -> String {
    let phase = match phase {
        GenerationPhase::Prefill => "prefill",
        GenerationPhase::Decode => "decode",
    };
    data_line(&serde_json::json!({ "graph_horizon": { "phase": phase } }))
}

// The OpenAI subset we accept. Unknown fields (`model`, `temperature`, `stream`,
// …) are ignored by design — no `deny_unknown_fields` — so clients sending the
// full OpenAI body still parse. Only `messages` and `max_tokens` are read.
#[derive(Deserialize)]
pub(crate) struct ChatCompletionRequest {
    #[serde(default)]
    messages: Vec<WireMessage>,
    #[serde(default)]
    max_tokens: Option<i64>,
}

// One wire message `{ role, content }`. `role` is validated into `Role` during
// conversion, not here, so the error surfaces as a typed ApiError.
#[derive(Deserialize)]
struct WireMessage {
    role: String,
    #[serde(default)]
    content: String,
}

// Builds the text-only engine request from the validated body, server limit, and
// loaded model's immutable sampling policy.
pub(crate) fn to_request(
    req: ChatCompletionRequest,
    cfg: &ServerConfig,
    sampling: SamplingParams,
) -> Result<Request, ApiError> {
    if req.messages.is_empty() {
        return Err(ApiError::EmptyMessages);
    }

    let messages = req
        .messages
        .into_iter()
        .map(|m| {
            Ok(Message {
                role: role_from_str(&m.role)?,
                content: m.content,
            })
        })
        .collect::<Result<Vec<_>, ApiError>>()?;

    // A present max_tokens must be a positive integer. Absent uses the server
    // default; the provided value is otherwise the sole generation cap.
    let max_tokens = match req.max_tokens {
        Some(n) if n <= 0 => return Err(ApiError::InvalidMaxTokens),
        Some(n) => usize::try_from(n).map_err(|_| ApiError::InvalidMaxTokens)?,
        None => cfg.max_tokens,
    };

    Ok(Request {
        messages,
        sampling,
        max_tokens,
    })
}

// Maps an engine event to its SSE line. Empty deltas are skipped and `Finished`
// produces nothing because the terminal frames belong to `stream.rs`.
pub(crate) fn event_line(event: &Event) -> Option<String> {
    match event {
        Event::TextDelta(s) if !s.is_empty() => Some(delta_line(Delta {
            content: Some(s.clone()),
        })),
        // An error must not leak the Report's internals: a generic error chunk.
        Event::Error(_) => {
            let payload = serde_json::json!({
                "error": { "message": "generation failed", "type": "server_error" }
            });
            Some(format!("data: {payload}\n\n"))
        }
        // Empty deltas and Finished produce no content line.
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use graph_horizon_engine::Role;

    fn cfg() -> ServerConfig {
        ServerConfig {
            host: "127.0.0.1".to_string(),
            port: 8080,
            max_tokens: 256,
        }
    }

    fn parse(body: &str) -> Result<Request, ApiError> {
        let req: ChatCompletionRequest = serde_json::from_str(body).expect("valid json");
        to_request(req, &cfg(), SamplingParams::greedy())
    }

    // LocalGenerateRequest implements neither Debug nor PartialEq, so the error
    // tests assert on the ApiError alone rather than on the whole Result.
    fn parse_err(body: &str) -> ApiError {
        parse(body).err().expect("expected an ApiError")
    }

    #[test]
    fn valid_request_without_max_tokens_uses_default() {
        let req = parse(r#"{"messages":[{"role":"user","content":"hi"}]}"#).unwrap();
        assert_eq!(req.max_tokens, 256);
        assert_eq!(req.sampling.temperature, 0.0); // greedy
        assert_eq!(req.messages.len(), 1);
        assert!(req.messages[0].role == Role::User);
    }

    #[test]
    fn unknown_fields_are_ignored() {
        let req = parse(
            r#"{"model":"gpt-4","temperature":0.7,"stream":true,"messages":[{"role":"system","content":"s"}]}"#,
        )
        .unwrap();
        assert!(req.messages[0].role == Role::System);
    }

    #[test]
    fn loaded_model_sampling_is_preserved() {
        let req: ChatCompletionRequest = serde_json::from_str(
            r#"{"temperature":0,"messages":[{"role":"user","content":"hi"}]}"#,
        )
        .unwrap();
        let sampling = SamplingParams {
            temperature: 0.7,
            top_p: 1.0,
            top_k: 0,
            min_p: 0.0,
            repeat_penalty: 1.0,
            seed: 0,
        };
        let request = to_request(req, &cfg(), sampling).unwrap();
        assert_eq!(request.sampling.temperature, 0.7);
    }

    #[test]
    fn max_tokens_above_default_is_kept() {
        let req =
            parse(r#"{"messages":[{"role":"user","content":"hi"}],"max_tokens":99999}"#).unwrap();
        assert_eq!(req.max_tokens, 99999);
    }

    #[test]
    fn max_tokens_below_default_is_kept() {
        let req =
            parse(r#"{"messages":[{"role":"user","content":"hi"}],"max_tokens":10}"#).unwrap();
        assert_eq!(req.max_tokens, 10);
    }

    #[test]
    fn empty_messages_is_error() {
        assert_eq!(parse_err(r#"{"messages":[]}"#), ApiError::EmptyMessages);
        assert_eq!(parse_err(r#"{}"#), ApiError::EmptyMessages);
    }

    #[test]
    fn unknown_role_is_error() {
        assert_eq!(
            parse_err(r#"{"messages":[{"role":"other","content":"x"}]}"#),
            ApiError::UnknownRole
        );
    }

    #[test]
    fn non_positive_max_tokens_is_error() {
        assert_eq!(
            parse_err(r#"{"messages":[{"role":"user","content":"hi"}],"max_tokens":0}"#),
            ApiError::InvalidMaxTokens
        );
        assert_eq!(
            parse_err(r#"{"messages":[{"role":"user","content":"hi"}],"max_tokens":-5}"#),
            ApiError::InvalidMaxTokens
        );
    }

    #[test]
    fn text_delta_produces_content_line() {
        let line = event_line(&Event::TextDelta("ciao".to_string())).unwrap();
        assert!(line.starts_with("data: "));
        assert!(line.ends_with("\n\n"));
        let json: serde_json::Value =
            serde_json::from_str(line.trim_start_matches("data: ").trim_end()).unwrap();
        assert_eq!(json["choices"][0]["delta"]["content"], "ciao");
        assert_eq!(json["object"], "chat.completion.chunk");
        assert_eq!(json["model"], "graph-horizon");
        assert!(json["choices"][0]["finish_reason"].is_null());
    }

    #[test]
    fn empty_deltas_and_finished_are_skipped() {
        assert!(event_line(&Event::TextDelta(String::new())).is_none());
        assert!(
            event_line(&Event::Finished(graph_horizon_engine::GenerationStats {
                prompt_tokens: 128,
                prefill_tokens: 96,
                completion_tokens: 42,
                prefill_ms: 400,
                decode_ms: 875,
            }))
            .is_none()
        );
    }

    #[test]
    fn phase_extension_is_namespaced() {
        assert_eq!(
            phase_line(GenerationPhase::Prefill),
            "data: {\"graph_horizon\":{\"phase\":\"prefill\"}}\n\n"
        );
        assert_eq!(
            phase_line(GenerationPhase::Decode),
            "data: {\"graph_horizon\":{\"phase\":\"decode\"}}\n\n"
        );
    }

    #[test]
    fn multibyte_content_is_escaped_by_serde() {
        let line = event_line(&Event::TextDelta("a\"b\nc→d".to_string())).unwrap();
        // Round-trips back to the original string, proving correct escaping.
        let json: serde_json::Value =
            serde_json::from_str(line.trim_start_matches("data: ").trim_end()).unwrap();
        assert_eq!(json["choices"][0]["delta"]["content"], "a\"b\nc→d");
    }
}
