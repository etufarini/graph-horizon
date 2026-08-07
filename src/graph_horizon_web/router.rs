/*
 * Graph Horizon web router
 * Single responsibility: route static assets and delegate exact chat and
 * `/props` requests to shared headless-server boundaries.
 */

use std::convert::Infallible;
use std::sync::Arc;

use bytes::Bytes;
use http_body_util::{BodyExt, Full};
use hyper::body::Incoming;
use hyper::{Method, Request, Response, StatusCode, header};

use crate::graph_horizon_server::{ResponseBody, ServerState};

use super::assets::{AssetFile, Assets};

#[derive(Debug, PartialEq, Eq)]
enum SharedRoute {
    ChatCompletions,
    Properties,
    MethodNotAllowed,
    None,
}

pub(super) async fn handle(
    req: Request<Incoming>,
    assets: Arc<Assets>,
    chat: ServerState,
) -> Result<Response<ResponseBody>, Infallible> {
    let response = match shared_route(req.method(), req.uri().path()) {
        SharedRoute::ChatCompletions => {
            crate::graph_horizon_server::handler::chat_completions(req, chat).await
        }
        SharedRoute::Properties => crate::graph_horizon_server::handler::properties_response(
            chat.chat.context_limit(),
            chat.config.max_tokens,
        ),
        SharedRoute::MethodNotAllowed => {
            error_response(StatusCode::METHOD_NOT_ALLOWED, "method not allowed")
        }
        SharedRoute::None => match (req.method(), req.uri().path()) {
            (&Method::GET, "/") | (&Method::GET, "/index.html") => asset_response(assets.index()),
            (&Method::GET, path) if path.starts_with("/assets/") => {
                asset_response(assets.asset(path))
            }
            _ => error_response(StatusCode::NOT_FOUND, "not found"),
        },
    };
    Ok(response)
}

fn shared_route(method: &Method, path: &str) -> SharedRoute {
    match (method, path) {
        (&Method::POST, "/v1/chat/completions") => SharedRoute::ChatCompletions,
        (_, "/v1/chat/completions") => SharedRoute::MethodNotAllowed,
        (&Method::GET, "/props") => SharedRoute::Properties,
        (_, "/props") => SharedRoute::MethodNotAllowed,
        _ => SharedRoute::None,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn web_router_delegates_exact_props_get() {
        assert_eq!(
            shared_route(&Method::GET, "/props"),
            SharedRoute::Properties
        );
        assert_eq!(
            shared_route(&Method::POST, "/props"),
            SharedRoute::MethodNotAllowed
        );
        assert_eq!(shared_route(&Method::GET, "/props/"), SharedRoute::None);
        assert_eq!(
            shared_route(&Method::GET, "/assets/props.js"),
            SharedRoute::None
        );
    }
}
