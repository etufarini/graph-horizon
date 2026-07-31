/*
 * GH Zero headless server - Handler - Body
 * Single responsibility: read and pre-validate untrusted chat request bodies.
 * It depends on hyper body collection and serde_json raw inspection, and
 * rejects tool/reasoning-incompatible message shapes before engine conversion.
 */

use bytes::Bytes;
use http_body_util::{BodyExt, Limited};
use hyper::body::Incoming;
use hyper::{Request, Response, StatusCode};

use super::ResponseBody;
use super::error::{coded_error_response, error_response};

// Hard cap on the request body: untrusted input must not exhaust memory. Limited
// stops reading past this many bytes, so an oversized body never buffers in full.
const MAX_BODY_BYTES: usize = 4 * 1024 * 1024;

// Reads the request body under the hard size cap, shared by both paths. On
// success returns the collected bytes; on failure returns the error Response so
// the caller just propagates it. LengthLimitError → 413 (oversized); any other
// read error → broken/closed connection → 400. Neither leaks internals.
pub(crate) async fn read_body(req: Request<Incoming>) -> Result<Bytes, Response<ResponseBody>> {
    let body = Limited::new(req.into_body(), MAX_BODY_BYTES);
    match body.collect().await {
        Ok(collected) => Ok(collected.to_bytes()),
        Err(e) => Err(
            if e.downcast_ref::<http_body_util::LengthLimitError>()
                .is_some()
            {
                error_response(
                    StatusCode::PAYLOAD_TOO_LARGE,
                    "request body too large",
                    "invalid_request_error",
                )
            } else {
                error_response(
                    StatusCode::BAD_REQUEST,
                    "could not read request body",
                    "invalid_request_error",
                )
            },
        ),
    }
}

#[allow(clippy::result_large_err)]
pub(crate) fn validate_chat_body(bytes: &[u8]) -> Result<(), Response<ResponseBody>> {
    let value: serde_json::Value = serde_json::from_slice(bytes).map_err(|_| {
        error_response(
            StatusCode::BAD_REQUEST,
            "invalid request body",
            "invalid_request_error",
        )
    })?;
    let Some(object) = value.as_object() else {
        return Err(error_response(
            StatusCode::BAD_REQUEST,
            "invalid request body",
            "invalid_request_error",
        ));
    };
    if object.contains_key("tools") || object.contains_key("tool_choice") {
        return Err(unsupported_tools());
    }
    let Some(messages) = object.get("messages").and_then(|v| v.as_array()) else {
        return Ok(());
    };
    for message in messages {
        let Some(message) = message.as_object() else {
            return Err(invalid_request("text messages are required"));
        };
        if message.contains_key("tool_calls") || message.contains_key("function_call") {
            return Err(unsupported_tools());
        }
        match message.get("role").and_then(|role| role.as_str()) {
            Some("tool" | "function") => return Err(unsupported_tools()),
            Some("system" | "user" | "assistant") => {}
            Some(_) => return Err(invalid_request("unsupported message role")),
            None => return Err(invalid_request("unsupported message role")),
        }
        if !message
            .get("content")
            .is_some_and(serde_json::Value::is_string)
        {
            return Err(invalid_request("text messages are required"));
        }
    }
    Ok(())
}

fn unsupported_tools() -> Response<ResponseBody> {
    coded_error_response(
        StatusCode::BAD_REQUEST,
        "unsupported_feature",
        "tool calling is not supported",
        "invalid_request_error",
    )
}

fn invalid_request(message: &str) -> Response<ResponseBody> {
    coded_error_response(
        StatusCode::BAD_REQUEST,
        "invalid_request",
        message,
        "invalid_request_error",
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use http_body_util::BodyExt;

    async fn response_bytes(response: Response<ResponseBody>) -> (StatusCode, Vec<u8>) {
        let status = response.status();
        let bytes = response
            .into_body()
            .collect()
            .await
            .unwrap()
            .to_bytes()
            .to_vec();
        (status, bytes)
    }

    #[tokio::test]
    async fn error_matrix_e13_forbidden_tool_members_use_exact_body() {
        for body in [
            br#"{"tools":null,"messages":[{"role":"user","content":"x"}]}"#.as_slice(),
            br#"{"tool_choice":false,"messages":[{"role":"user","content":"x"}]}"#.as_slice(),
            br#"{"messages":[{"role":"assistant","content":"","tool_calls":[]}]}"#.as_slice(),
            br#"{"messages":[{"role":"assistant","content":"","function_call":{}}]}"#.as_slice(),
            br#"{"messages":[{"role":"tool","content":"x"}]}"#.as_slice(),
            br#"{"messages":[{"role":"function","content":"x"}]}"#.as_slice(),
        ] {
            let (status, bytes) = response_bytes(validate_chat_body(body).unwrap_err()).await;
            assert_eq!(status, StatusCode::BAD_REQUEST);
            assert_eq!(
                bytes,
                br#"{"error":{"code":"unsupported_feature","message":"tool calling is not supported","type":"invalid_request_error"}}"#
            );
        }
    }

    #[tokio::test]
    async fn non_text_content_and_unknown_roles_use_invalid_request_body() {
        let cases = [
            (
                br#"{"messages":[{"role":"user","content":["x"]}]}"#.as_slice(),
                br#"{"error":{"code":"invalid_request","message":"text messages are required","type":"invalid_request_error"}}"#.as_slice(),
            ),
            (
                br#"{"messages":[{"role":"developer","content":"x"}]}"#.as_slice(),
                br#"{"error":{"code":"invalid_request","message":"unsupported message role","type":"invalid_request_error"}}"#.as_slice(),
            ),
        ];
        for (body, expected) in cases {
            let (status, bytes) = response_bytes(validate_chat_body(body).unwrap_err()).await;
            assert_eq!(status, StatusCode::BAD_REQUEST);
            assert_eq!(bytes, expected);
        }
    }
}
