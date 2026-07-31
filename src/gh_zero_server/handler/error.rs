/*
 * GH Zero headless server - Handler - Error
 * Single responsibility: build stable JSON error responses for chat requests.
 * It depends only on hyper response types and never exposes paths, stack
 * traces, tools, workspace data, or reasoning internals.
 */

use hyper::{Response, StatusCode, header};

use super::{ResponseBody, full};

// Builds a generic OpenAI-style error response. The body carries only a message
// and a type — never a path, stack trace or other internal. Status and headers
// are static, so the builder cannot fail on untrusted input.
pub(super) fn error_response(
    status: StatusCode,
    message: &str,
    kind: &str,
) -> Response<ResponseBody> {
    let payload = serde_json::json!({ "error": { "message": message, "type": kind } });
    json_response(status, payload)
}

pub(super) fn coded_error_response(
    status: StatusCode,
    code: &str,
    message: &str,
    kind: &str,
) -> Response<ResponseBody> {
    let payload = serde_json::json!({
        "error": { "code": code, "message": message, "type": kind }
    });
    json_response(status, payload)
}

fn json_response(status: StatusCode, payload: serde_json::Value) -> Response<ResponseBody> {
    let bytes = serde_json::to_vec(&payload).unwrap_or_else(|_| b"{}".to_vec());
    Response::builder()
        .status(status)
        .header(header::CONTENT_TYPE, "application/json")
        .body(full(bytes))
        .expect("a response from a static status and header is always valid")
}
