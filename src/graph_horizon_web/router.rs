/*
 * Graph Horizon web router
 * Single responsibility: route bundled assets and private browser requests.
 * The internal paths are implementation details of the packaged Web UI and do
 * not form a supported integration contract.
 */

use std::convert::Infallible;
use std::sync::Arc;

use bytes::Bytes;
use http_body_util::{BodyExt, Full, combinators::BoxBody};
use hyper::body::Incoming;
use hyper::{Method, Request, Response, StatusCode, header};

use super::assets::{AssetFile, Assets};
use super::chat::State;

#[derive(Debug, PartialEq, Eq)]
enum InternalRoute {
    Chat,
    Context,
    Runtime,
    MethodNotAllowed,
    None,
}

pub(super) async fn handle(
    req: Request<Incoming>,
    assets: Arc<Assets>,
    chat: State,
) -> Result<Response<ResponseBody>, Infallible> {
    let response = match internal_route(req.method(), req.uri().path()) {
        InternalRoute::Chat => super::chat::handle(req, chat).await,
        InternalRoute::Context => json_response(&serde_json::json!({
            "context_limit": chat.engine.context_limit(),
        })),
        InternalRoute::Runtime => {
            let bytes = serde_json::to_vec(&super::runtime::payload(&chat.engine))
                .unwrap_or_else(|_| b"{}".to_vec());
            Response::builder()
                .status(StatusCode::OK)
                .header(header::CONTENT_TYPE, "application/json")
                .body(full(bytes))
                .expect("runtime properties use fixed response metadata")
        }
        InternalRoute::MethodNotAllowed => {
            error_response(StatusCode::METHOD_NOT_ALLOWED, "method not allowed")
        }
        InternalRoute::None => match (req.method(), req.uri().path()) {
            (&Method::GET, "/") | (&Method::GET, "/index.html") => asset_response(assets.index()),
            (&Method::GET, path) if path.starts_with("/assets/") => {
                asset_response(assets.asset(path))
            }
            _ => error_response(StatusCode::NOT_FOUND, "not found"),
        },
    };
    Ok(response)
}

fn internal_route(method: &Method, path: &str) -> InternalRoute {
    match (method, path) {
        (&Method::POST, "/internal/chat") => InternalRoute::Chat,
        (_, "/internal/chat") => InternalRoute::MethodNotAllowed,
        (&Method::GET, "/internal/context") => InternalRoute::Context,
        (_, "/internal/context") => InternalRoute::MethodNotAllowed,
        (&Method::GET, "/internal/runtime") => InternalRoute::Runtime,
        (_, "/internal/runtime") => InternalRoute::MethodNotAllowed,
        _ => InternalRoute::None,
    }
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
    json_response_with_status(status, &serde_json::json!({ "error": message }))
}

fn json_response(value: &serde_json::Value) -> Response<ResponseBody> {
    json_response_with_status(StatusCode::OK, value)
}

fn json_response_with_status(
    status: StatusCode,
    value: &serde_json::Value,
) -> Response<ResponseBody> {
    let bytes = serde_json::to_vec(value).unwrap_or_else(|_| b"{}".to_vec());
    Response::builder()
        .status(status)
        .header(header::CONTENT_TYPE, "application/json")
        .body(full(bytes))
        .expect("a response from a static status and header is always valid")
}

pub(super) type ResponseBody = BoxBody<Bytes, Box<dyn std::error::Error + Send + Sync>>;

fn full(bytes: impl Into<Bytes>) -> ResponseBody {
    Full::new(bytes.into())
        .map_err(|never| match never {})
        .boxed()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn private_context_route_is_exact_and_get_only() {
        assert_eq!(
            internal_route(&Method::GET, "/internal/context"),
            InternalRoute::Context
        );
        assert_eq!(
            internal_route(&Method::POST, "/internal/context"),
            InternalRoute::MethodNotAllowed
        );
        assert_eq!(
            internal_route(&Method::GET, "/internal/context/"),
            InternalRoute::None
        );
    }

    #[test]
    fn legacy_public_paths_have_no_route() {
        for (method, path) in [
            (&Method::POST, "/v1/chat/completions"),
            (&Method::GET, "/props"),
        ] {
            assert_eq!(internal_route(method, path), InternalRoute::None);
        }
    }

    #[test]
    fn runtime_route_is_web_only_and_get_only() {
        assert_eq!(
            internal_route(&Method::GET, "/internal/runtime"),
            InternalRoute::Runtime
        );
        assert_eq!(
            internal_route(&Method::POST, "/internal/runtime"),
            InternalRoute::MethodNotAllowed
        );
        assert_eq!(
            internal_route(&Method::GET, "/internal/runtime/"),
            InternalRoute::None
        );
    }
}
