/*
 * GH Zero web router
 * Single responsibility: route static assets and delegate chat requests to the
 * headless server pipeline. It depends on assets and gh_zero_server state, and
 * exposes no tools, confirmations, workspace, or reasoning endpoints.
 */

use std::convert::Infallible;
use std::sync::Arc;

use bytes::Bytes;
use http_body_util::{BodyExt, Full};
use hyper::body::Incoming;
use hyper::{Method, Request, Response, StatusCode, header};

use crate::gh_zero_server::{ResponseBody, ServerState};

use super::assets::{AssetFile, Assets};

pub(super) async fn handle(
    req: Request<Incoming>,
    assets: Arc<Assets>,
    chat: ServerState,
) -> Result<Response<ResponseBody>, Infallible> {
    let response = match (req.method(), req.uri().path()) {
        (&Method::GET, "/") | (&Method::GET, "/index.html") => asset_response(assets.index()),
        (&Method::GET, path) if path.starts_with("/assets/") => asset_response(assets.asset(path)),
        (&Method::POST, "/v1/chat/completions") => {
            crate::gh_zero_server::handler::chat_completions(req, chat).await
        }
        (_, "/v1/chat/completions") => {
            error_response(StatusCode::METHOD_NOT_ALLOWED, "method not allowed")
        }
        _ => error_response(StatusCode::NOT_FOUND, "not found"),
    };
    Ok(response)
}

fn asset_response(file: Option<AssetFile>) -> Response<ResponseBody> {
    match file {
        Some(file) => Response::builder()
            .status(StatusCode::OK)
            .header(header::CONTENT_TYPE, file.mime)
            .body(full(file.bytes))
            .expect("a response from static headers and a valid body is always valid"),
        None => error_response(StatusCode::NOT_FOUND, "not found"),
    }
}

pub(super) fn error_response(status: StatusCode, message: &str) -> Response<ResponseBody> {
    let payload = serde_json::json!({
        "error": { "message": message, "type": "invalid_request_error" }
    });
    let bytes = serde_json::to_vec(&payload).unwrap_or_else(|_| b"{}".to_vec());
    Response::builder()
        .status(status)
        .header(header::CONTENT_TYPE, "application/json")
        .body(full(bytes))
        .expect("a response from a static status and header is always valid")
}

pub(super) fn full(bytes: impl Into<Bytes>) -> ResponseBody {
    Full::new(bytes.into())
        .map_err(|never| match never {})
        .boxed()
}
