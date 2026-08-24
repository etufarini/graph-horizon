/*
 * Graph Horizon Web chat request
 * Reads one bounded private request and separates optional validated search
 * metadata from messages converted for the engine. Compatibility fields are rejected.
 */

use bytes::Bytes;
use graph_horizon_engine::{Message, Request as EngineRequest, Role, SamplingParams};
use http_body_util::{BodyExt, Limited};
use hyper::Request;
use hyper::body::Incoming;
use serde::Deserialize;

use super::super::search;

const MAX_BODY_BYTES: usize = 4 * 1024 * 1024;

#[derive(Debug, PartialEq, Eq)]
pub(super) enum Error {
    TooLarge,
    Invalid,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ChatRequest {
    messages: Vec<WireMessage>,
    #[serde(default, deserialize_with = "optional_search")]
    search: Option<search::Request>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct WireMessage {
    role: WireRole,
    content: String,
}

#[derive(Deserialize)]
#[serde(rename_all = "lowercase")]
enum WireRole {
    System,
    User,
    Assistant,
}

pub(super) struct ParsedRequest {
    pub(super) engine: EngineRequest,
    pub(super) search: Option<search::Request>,
}

pub(super) async fn parse(
    request: Request<Incoming>,
    max_tokens: usize,
    sampling: SamplingParams,
) -> Result<ParsedRequest, Error> {
    let body = Limited::new(request.into_body(), MAX_BODY_BYTES);
    let bytes = body.collect().await.map_err(|error| {
        if error
            .downcast_ref::<http_body_util::LengthLimitError>()
            .is_some()
        {
            Error::TooLarge
        } else {
            Error::Invalid
        }
    })?;
    parse_bytes(bytes.to_bytes(), max_tokens, sampling)
}

fn parse_bytes(
    bytes: Bytes,
    max_tokens: usize,
    sampling: SamplingParams,
) -> Result<ParsedRequest, Error> {
    let request: ChatRequest = serde_json::from_slice(&bytes).map_err(|_| Error::Invalid)?;
    if request.messages.is_empty() {
        return Err(Error::Invalid);
    }
    let messages: Vec<Message> = request
        .messages
        .into_iter()
        .map(|message| Message {
            role: match message.role {
                WireRole::System => Role::System,
                WireRole::User => Role::User,
                WireRole::Assistant => Role::Assistant,
            },
            content: message.content,
        })
        .collect();
    let search = request
        .search
        .map(search::Request::validated)
        .transpose()
        .map_err(|_| Error::Invalid)?;
    if search.is_some() && !matches!(messages.last(), Some(message) if message.role == Role::User) {
        return Err(Error::Invalid);
    }
    Ok(ParsedRequest {
        engine: EngineRequest {
            messages,
            sampling,
            max_tokens,
        },
        search,
    })
}

fn optional_search<'de, D>(deserializer: D) -> Result<Option<search::Request>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    search::Request::deserialize(deserializer).map(Some)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(body: &str) -> Result<ParsedRequest, Error> {
        parse_bytes(
            Bytes::copy_from_slice(body.as_bytes()),
            4096,
            SamplingParams::greedy(),
        )
    }

    #[test]
    fn private_shape_maps_text_messages() {
        let request = parse(
            r#"{"messages":[{"role":"system","content":"s"},{"role":"user","content":"hi"}]}"#,
        )
        .unwrap();
        assert_eq!(request.engine.messages.len(), 2);
        assert!(request.engine.messages[0].role == Role::System);
        assert_eq!(request.engine.max_tokens, 4096);
        assert!(request.search.is_none());
    }

    #[test]
    fn compatibility_and_unknown_fields_are_rejected() {
        for body in [
            r#"{"messages":[],"model":"legacy"}"#,
            r#"{"messages":[],"stream":true}"#,
            r#"{"messages":[],"max_tokens":12}"#,
            r#"{"messages":[{"role":"tool","content":"x"}]}"#,
            r#"{"messages":[{"role":"user","content":"x","tool_calls":[]}]}"#,
            r#"{"messages":[{"role":"user","content":"x"}],"search":null}"#,
            r#"{"messages":[{"role":"user","content":"x"}],"search_query":"x"}"#,
            r#"{"messages":[{"role":"user","content":"x"}],"search_date":"2026-08-23"}"#,
            r#"{"messages":[{"role":"assistant","content":"x"}],"search":{"terms":"x","category":"web","language":"it-IT","reference_date":"2026-08-23","published":null}}"#,
        ] {
            assert_eq!(parse(body).err(), Some(Error::Invalid));
        }
    }

    #[test]
    fn structured_search_is_separate_from_engine_messages() {
        let request = parse(
            r#"{"messages":[{"role":"user","content":"framed input"}],"search":{"terms":"  visible query  ","category":"news","language":"it-IT","reference_date":"2024-02-29","published":{"from":"2024-02-19","to":"2024-02-20"}}}"#,
        )
        .unwrap();
        assert!(request.search.is_some());
        assert_eq!(request.engine.messages[0].content, "framed input");

        let long = "x".repeat(513);
        let body = format!(
            r#"{{"messages":[{{"role":"user","content":"x"}}],"search":{{"terms":"{long}","category":"web","language":"en-US","reference_date":"2026-08-23","published":null}}}}"#
        );
        assert_eq!(parse(&body).err(), Some(Error::Invalid));
    }
}
