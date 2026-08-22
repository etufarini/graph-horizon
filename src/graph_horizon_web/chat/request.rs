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

#[derive(Debug, PartialEq, Eq)]
pub(super) enum Error {
    TooLarge,
    Invalid,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ChatRequest {
    messages: Vec<WireMessage>,
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

pub(super) async fn parse(
    request: Request<Incoming>,
    max_tokens: usize,
    sampling: SamplingParams,
) -> Result<EngineRequest, Error> {
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
) -> Result<EngineRequest, Error> {
    let request: ChatRequest = serde_json::from_slice(&bytes).map_err(|_| Error::Invalid)?;
    if request.messages.is_empty() {
        return Err(Error::Invalid);
    }
    let messages = request
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
    Ok(EngineRequest {
        messages,
        sampling,
        max_tokens,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(body: &str) -> Result<EngineRequest, Error> {
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
        assert_eq!(request.messages.len(), 2);
        assert!(request.messages[0].role == Role::System);
        assert_eq!(request.max_tokens, 4096);
    }

    #[test]
    fn compatibility_and_unknown_fields_are_rejected() {
        for body in [
            r#"{"messages":[],"model":"legacy"}"#,
            r#"{"messages":[],"stream":true}"#,
            r#"{"messages":[],"max_tokens":12}"#,
            r#"{"messages":[{"role":"tool","content":"x"}]}"#,
            r#"{"messages":[{"role":"user","content":"x","tool_calls":[]}]}"#,
        ] {
            assert_eq!(parse(body).err(), Some(Error::Invalid));
        }
    }
}
