/*
 * Graph Horizon CLI Modules - Runtime - SSE
 * Single responsibility: parse provider SSE lines into assistant text, optional
 * Graph Horizon phase boundaries, and exact terminal metrics. Unknown provider
 * fields are tolerated, so `reasoning_content` is ignored without state;
 * flattened maps reject explicit errors and tool calls.
*/

use color_eyre::eyre::{Result, eyre};
use serde::Deserialize;
use std::collections::HashMap;

use super::{GenerationPhase, GenerationStats, StreamEvent};

// Internal structs for deserializing provider SSE lines.
// Fields are tolerant (defaulted) so usage-only or finish chunks that omit
// `choices`/`delta` parse cleanly instead of aborting the stream.
#[derive(Deserialize, Default)]
struct SseChunk {
    #[serde(default)]
    choices: Vec<Choice>,
    #[serde(default)]
    usage: Option<Usage>,
    #[serde(default)]
    graph_horizon: Option<GraphHorizon>,
    #[serde(flatten)]
    extra: HashMap<String, serde_json::Value>,
}

#[derive(Deserialize)]
struct GraphHorizon {
    phase: String,
}

#[derive(Deserialize)]
struct Usage {
    prompt_tokens: Option<usize>,
    prefill_tokens: Option<usize>,
    completion_tokens: Option<usize>,
    prefill_ms: Option<u64>,
    decode_ms: Option<u64>,
}

// Represents one entry in the top-level `choices` array of an SSE chunk.
#[derive(Deserialize)]
struct Choice {
    #[serde(default)]
    delta: Delta,
    #[serde(default)]
    finish_reason: Option<String>,
}

// Only assistant text is modeled. Flattening preserves forbidden-member presence
// while Serde continues to tolerate every other provider extension.
#[derive(Deserialize, Default)]
struct Delta {
    content: Option<String>,
    #[serde(flatten)]
    extra: HashMap<String, serde_json::Value>,
}

// Handles one decoded SSE line: skips non-data lines, parses data frames into
// Chunks and forwards them over the channel. Returns true when the stream should
// stop — either a [DONE] sentinel or a dropped receiver.
pub(super) async fn handle_sse_line(
    line: &str,
    tx: &tokio::sync::mpsc::Sender<Result<StreamEvent>>,
) -> bool {
    let Some(json_str) = line.strip_prefix("data: ") else {
        return false;
    };

    if json_str == "[DONE]" {
        return true;
    }

    match parse_sse_line(json_str) {
        Ok(Some(chunk)) => tx.send(Ok(chunk)).await.is_err(),
        Ok(None) => false,
        Err(err) => tx.send(Err(err)).await.is_err(),
    }
}

// Parses a single SSE line (without the "data: " prefix) into a stream event.
// Returns None for keepalive and standard provider usage frames. Tool calls are
// protocol errors:
// the chat-only UI must not silently display or continue after structured tools.
fn parse_sse_line(json_str: &str) -> Result<Option<StreamEvent>> {
    let chunk: SseChunk = serde_json::from_str(json_str)?;
    // A provider error must never be mistaken for an empty successful frame.
    // Keep the local error generic so remote internals are not exposed.
    if chunk.extra.contains_key("error") {
        return Err(eyre!("provider returned an error response"));
    }
    if let Some(extension) = chunk.graph_horizon {
        let phase = match extension.phase.as_str() {
            "prefill" => GenerationPhase::Prefill,
            "decode" => GenerationPhase::Decode,
            _ => return Err(eyre!("provider returned an invalid generation phase")),
        };
        return Ok(Some(StreamEvent::Phase(phase)));
    }
    if let Some(usage) = chunk.usage {
        let custom = usage.prefill_tokens.is_some()
            || usage.prefill_ms.is_some()
            || usage.decode_ms.is_some();
        if !custom {
            return Ok(None);
        }
        let stats = GenerationStats {
            prompt_tokens: usage
                .prompt_tokens
                .ok_or_else(|| eyre!("provider returned incomplete generation usage"))?,
            prefill_tokens: usage
                .prefill_tokens
                .ok_or_else(|| eyre!("provider returned incomplete generation usage"))?,
            completion_tokens: usage
                .completion_tokens
                .ok_or_else(|| eyre!("provider returned incomplete generation usage"))?,
            prefill_ms: usage
                .prefill_ms
                .ok_or_else(|| eyre!("provider returned incomplete generation usage"))?,
            decode_ms: usage
                .decode_ms
                .ok_or_else(|| eyre!("provider returned incomplete generation usage"))?,
        };
        if stats.prefill_tokens > stats.prompt_tokens {
            return Err(eyre!("provider returned invalid generation usage"));
        }
        return Ok(Some(StreamEvent::Finished(stats)));
    }
    let choice = match chunk.choices.into_iter().next() {
        Some(choice) => choice,
        None => return Ok(None),
    };
    if choice.delta.extra.contains_key(TOOL_FINISH)
        || choice.finish_reason.as_deref() == Some(TOOL_FINISH)
    {
        return Err(eyre!("provider emitted unsupported tool call"));
    }
    let response = choice.delta.content.unwrap_or_default();
    if response.is_empty() {
        return Ok(None);
    }
    Ok(Some(StreamEvent::Text(response)))
}

const TOOL_FINISH: &str = concat!("tool_", "calls");

#[cfg(test)]
mod tests {
    use super::*;

    // Feeds one raw SSE data payload through the parser, returning the Chunk.
    fn parse(json: &str) -> Result<Option<StreamEvent>> {
        parse_sse_line(json)
    }

    #[test]
    fn plain_content_frames_still_parse() {
        let frame = r#"{"choices":[{"delta":{"reasoning_content":"hm","content":"hi"}}]}"#;
        let StreamEvent::Text(text) = parse(frame).unwrap().unwrap() else {
            panic!("content did not produce text");
        };
        assert_eq!(text, "hi");
    }

    #[test]
    fn absent_or_empty_content_emits_nothing() {
        assert!(parse(r#"{"choices":[]}"#).unwrap().is_none());
        assert!(parse(r#"{"choices":[{"delta":{}}]}"#).unwrap().is_none());
        assert!(
            parse(r#"{"choices":[{"delta":{"content":""}}]}"#)
                .unwrap()
                .is_none()
        );
        assert!(
            parse(r#"{"usage":{"completion_tokens":1}}"#)
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn graph_horizon_phase_and_exact_usage_are_typed() {
        assert!(matches!(
            parse(r#"{"graph_horizon":{"phase":"prefill"}}"#).unwrap(),
            Some(StreamEvent::Phase(GenerationPhase::Prefill))
        ));
        let event = parse(
            r#"{"usage":{"prompt_tokens":128,"prefill_tokens":32,"completion_tokens":42,"prefill_ms":400,"decode_ms":875}}"#,
        )
        .unwrap()
        .unwrap();
        let StreamEvent::Finished(stats) = event else {
            panic!("usage did not produce terminal stats");
        };
        assert_eq!(stats.prompt_tokens, 128);
        assert_eq!(stats.prefill_tokens, 32);
        assert_eq!(stats.completion_tokens, 42);
    }

    #[test]
    fn standard_usage_is_optional_but_partial_extension_is_rejected() {
        assert!(
            parse(r#"{"usage":{"prompt_tokens":2,"completion_tokens":1}}"#)
                .unwrap()
                .is_none()
        );
        assert!(
            parse(r#"{"usage":{"prompt_tokens":2,"completion_tokens":1,"prefill_ms":3}}"#).is_err()
        );
        assert!(parse(r#"{"graph_horizon":{"phase":"other"}}"#).is_err());
    }

    #[test]
    fn every_present_provider_error_is_a_protocol_error() {
        for value in [r#"{"message":"failed"}"#, r#""failed""#, "null"] {
            let frame = format!(r#"{{"error":{value}}}"#);
            assert!(parse(&frame).is_err(), "{value}");
        }
    }

    #[test]
    fn every_present_tool_member_is_a_protocol_error() {
        for value in [r#"[{"index":0}]"#, "[]", "null"] {
            let frame = format!(
                r#"{{"choices":[{{"delta":{{"{}":{value}}}}}]}}"#,
                TOOL_FINISH
            );
            assert!(parse(&frame).is_err(), "{value}");
        }
        let finish = concat!(
            r#"{"choices":[{"delta":{},"finish_reason":"#,
            "\"tool_",
            "calls\"",
            r#"}]}"#
        );
        assert!(parse(finish).is_err());
    }

    #[tokio::test]
    async fn done_and_dropped_consumer_stop_the_stream() {
        let (tx, mut rx) = tokio::sync::mpsc::channel(1);
        assert!(handle_sse_line("data: [DONE]", &tx).await);
        assert!(rx.try_recv().is_err());
        drop(rx);
        assert!(handle_sse_line(r#"data: {"choices":[{"delta":{"content":"hi"}}]}"#, &tx).await);
    }
}
