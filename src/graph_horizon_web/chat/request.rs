/*
 * Graph Horizon Web chat request
 * Reads one size-bounded request from the bundled frontend and converts its
 * strict private JSON shape into an engine request. No compatibility fields are
 * accepted and no HTTP response policy is defined here.
 */

use bytes::Bytes;
use graph_horizon_engine::{Message, Request as EngineRequest, Role, SamplingParams};
use http_body_util::{BodyExt, Limited};
use hyper::Request;
use hyper::body::Incoming;
use serde::Deserialize;

const MAX_BODY_BYTES: usize = 4 * 1024 * 1024;
const MAX_SEARCH_QUERY_BYTES: usize = 2 * 1024;
const MAX_SEARCH_QUERY_CHARACTERS: usize = 512;

#[derive(Debug, PartialEq, Eq)]
pub(super) enum Error {
    TooLarge,
    Invalid,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ChatRequest {
    messages: Vec<WireMessage>,
    #[serde(default, deserialize_with = "search_query")]
    search_query: Option<String>,
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
    pub(super) search_query: Option<String>,
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
    let search_query = request.search_query.map(|query| query.trim().to_string());
    if let Some(query) = &search_query
        && (query.is_empty()
            || query.len() > MAX_SEARCH_QUERY_BYTES
            || query.chars().count() > MAX_SEARCH_QUERY_CHARACTERS
            || !matches!(messages.last(), Some(message) if message.role == Role::User))
    {
        return Err(Error::Invalid);
    }
    Ok(ParsedRequest {
        engine: EngineRequest {
            messages,
            sampling,
            max_tokens,
        },
        search_query,
    })
}

fn search_query<'de, D>(deserializer: D) -> Result<Option<String>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    String::deserialize(deserializer).map(Some)
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
        assert_eq!(request.search_query, None);
    }

    #[test]
    fn compatibility_and_unknown_fields_are_rejected() {
        for body in [
            r#"{"messages":[],"model":"legacy"}"#,
            r#"{"messages":[],"stream":true}"#,
            r#"{"messages":[],"max_tokens":12}"#,
            r#"{"messages":[{"role":"tool","content":"x"}]}"#,
            r#"{"messages":[{"role":"user","content":"x","tool_calls":[]}]}"#,
            r#"{"messages":[{"role":"user","content":"x"}],"search_query":null}"#,
            r#"{"messages":[{"role":"assistant","content":"x"}],"search_query":"x"}"#,
        ] {
            assert_eq!(parse(body).err(), Some(Error::Invalid));
        }
    }

    #[test]
    fn search_query_is_separate_trimmed_and_bounded() {
        let request = parse(
            r#"{"messages":[{"role":"user","content":"framed input"}],"search_query":"  visible query  "}"#,
        )
        .unwrap();
        assert_eq!(request.search_query.as_deref(), Some("visible query"));
        assert_eq!(request.engine.messages[0].content, "framed input");

        let long = "x".repeat(MAX_SEARCH_QUERY_CHARACTERS + 1);
        let body =
            format!(r#"{{"messages":[{{"role":"user","content":"x"}}],"search_query":"{long}"}}"#);
        assert_eq!(parse(&body).err(), Some(Error::Invalid));
    }
}
