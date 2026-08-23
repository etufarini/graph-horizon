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
    #[serde(default, deserialize_with = "optional_string")]
    search_query: Option<String>,
    #[serde(default, deserialize_with = "optional_string")]
    search_date: Option<String>,
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
    pub(super) search: Option<Search>,
}

pub(super) struct Search {
    pub(super) query: String,
    pub(super) date: String,
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
    let search = match (request.search_query, request.search_date) {
        (None, None) => None,
        (Some(query), Some(date)) => {
            let query = query.trim().to_string();
            if query.is_empty()
                || query.len() > MAX_SEARCH_QUERY_BYTES
                || query.chars().count() > MAX_SEARCH_QUERY_CHARACTERS
                || !valid_date(&date)
                || !matches!(messages.last(), Some(message) if message.role == Role::User)
            {
                return Err(Error::Invalid);
            }
            Some(Search { query, date })
        }
        _ => return Err(Error::Invalid),
    };
    Ok(ParsedRequest {
        engine: EngineRequest {
            messages,
            sampling,
            max_tokens,
        },
        search,
    })
}

fn optional_string<'de, D>(deserializer: D) -> Result<Option<String>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    String::deserialize(deserializer).map(Some)
}

fn valid_date(value: &str) -> bool {
    let bytes = value.as_bytes();
    if bytes.len() != 10
        || bytes[4] != b'-'
        || bytes[7] != b'-'
        || !bytes
            .iter()
            .enumerate()
            .all(|(index, byte)| matches!(index, 4 | 7) || byte.is_ascii_digit())
    {
        return false;
    }
    let year = value[0..4].parse::<u16>().unwrap_or(0);
    let month = value[5..7].parse::<u8>().unwrap_or(0);
    let day = value[8..10].parse::<u8>().unwrap_or(0);
    let leap = year % 4 == 0 && (year % 100 != 0 || year % 400 == 0);
    let days = match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if leap => 29,
        2 => 28,
        _ => return false,
    };
    year != 0 && day != 0 && day <= days
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
            r#"{"messages":[{"role":"user","content":"x"}],"search_query":null}"#,
            r#"{"messages":[{"role":"user","content":"x"}],"search_query":"x"}"#,
            r#"{"messages":[{"role":"user","content":"x"}],"search_date":"2026-08-23"}"#,
            r#"{"messages":[{"role":"assistant","content":"x"}],"search_query":"x","search_date":"2026-08-23"}"#,
        ] {
            assert_eq!(parse(body).err(), Some(Error::Invalid));
        }
    }

    #[test]
    fn search_query_is_separate_trimmed_and_bounded() {
        let request = parse(
            r#"{"messages":[{"role":"user","content":"framed input"}],"search_query":"  visible query  ","search_date":"2024-02-29"}"#,
        )
        .unwrap();
        let search = request.search.unwrap();
        assert_eq!(search.query, "visible query");
        assert_eq!(search.date, "2024-02-29");
        assert_eq!(request.engine.messages[0].content, "framed input");

        let long = "x".repeat(MAX_SEARCH_QUERY_CHARACTERS + 1);
        let body = format!(
            r#"{{"messages":[{{"role":"user","content":"x"}}],"search_query":"{long}","search_date":"2026-08-23"}}"#
        );
        assert_eq!(parse(&body).err(), Some(Error::Invalid));
    }

    #[test]
    fn search_date_is_a_real_gregorian_date() {
        for date in ["", "2026-8-23", "0000-01-01", "2023-02-29", "2026-13-01"] {
            assert!(!valid_date(date), "accepted {date}");
        }
        for date in ["2024-02-29", "2026-08-23", "2000-02-29"] {
            assert!(valid_date(date), "rejected {date}");
        }
    }
}
