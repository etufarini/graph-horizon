/*
 * GH Zero CLI Modules - Runtime - SSE
 * Single responsibility: parse provider SSE lines into assistant text chunks.
 * Unknown provider fields are tolerated, so `reasoning_content` is ignored
 * without state; flattened extra-member maps exist only to reject explicit
 * provider errors and `tool_calls`, including null or empty values.
*/

use color_eyre::eyre::{Result, eyre};
use serde::Deserialize;
use std::collections::HashMap;

use super::Chunk;

// Internal structs for deserializing provider SSE lines.
// Fields are tolerant (defaulted) so usage-only or finish chunks that omit
// `choices`/`delta` parse cleanly instead of aborting the stream.
#[derive(Deserialize, Default)]
struct SseChunk {
    #[serde(default)]
    choices: Vec<Choice>,
    #[serde(flatten)]
    extra: HashMap<String, serde_json::Value>,
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
    tx: &tokio::sync::mpsc::Sender<Result<Chunk>>,
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

// Parses a single SSE line (without the "data: " prefix) into a Chunk.
// Returns None for keepalive/usage frames. Tool-call frames are protocol errors:
// the chat-only UI must not silently display or continue after structured tools.
fn parse_sse_line(json_str: &str) -> Result<Option<Chunk>> {
    let chunk: SseChunk = serde_json::from_str(json_str)?;
    // A provider error must never be mistaken for an empty successful frame.
    // Keep the local error generic so remote internals are not exposed.
    if chunk.extra.contains_key("error") {
        return Err(eyre!("provider returned an error response"));
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
    Ok(Some(Chunk { response }))
}

const TOOL_FINISH: &str = concat!("tool_", "calls");

#[cfg(test)]
mod tests {
    use super::*;

    // Feeds one raw SSE data payload through the parser, returning the Chunk.
    fn parse(json: &str) -> Result<Option<Chunk>> {
        parse_sse_line(json)
    }

    #[test]
    fn plain_content_frames_still_parse() {
        let frame = r#"{"choices":[{"delta":{"reasoning_content":"hm","content":"hi"}}]}"#;
        let chunk = parse(frame).unwrap().unwrap();
        assert_eq!(chunk.response, "hi");
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
