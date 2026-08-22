/*
 * Graph Horizon headless server handler
 * Receives raw Hyper requests, routes chat and exact `/props` requests, and
 * builds the fixed properties response without entering chat admission.
 */

use std::convert::Infallible;

use bytes::Bytes;
use http_body_util::{BodyExt, Full, combinators::BoxBody};
use hyper::body::Incoming;
use hyper::{Method, Request, Response, StatusCode, header};

use super::{ServerState, api};
use error::error_response;

mod body;
mod error;
mod pipeline;

#[derive(Debug, PartialEq, Eq)]
enum Route {
    ChatCompletions,
    Properties,
    MethodNotAllowed,
    NotFound,
}

// Single response body type for the whole handler. Errors (Full) and the SSE
// stream are boxed into this one type so the Response signature stays fixed.
pub(crate) type ResponseBody = BoxBody<Bytes, Box<dyn std::error::Error + Send + Sync>>;

// Routes the request. The shared resources (chat engine, mutex, admission
// semaphore, config) are threaded through to the chat path. The Result error is
// Infallible: every branch produces a Response.
pub(crate) async fn handle(
    req: Request<Incoming>,
    state: ServerState,
) -> Result<Response<ResponseBody>, Infallible> {
    let response = match route(req.method(), req.uri().path()) {
        // The chat path caps and prevalidates raw JSON before typed parsing,
        // engine validation, admission, serialization, and SSE generation.
        Route::ChatCompletions => chat_completions(req, state).await,
        // Properties are immutable engine data and never consume chat permits.
        Route::Properties => {
            properties_response(state.chat.context_limit(), state.config.max_tokens)
        }
        // Known paths, wrong method.
        Route::MethodNotAllowed => error_response(
            StatusCode::METHOD_NOT_ALLOWED,
            "method not allowed",
            "invalid_request_error",
        ),
        // Anything else.
        Route::NotFound => {
            error_response(StatusCode::NOT_FOUND, "not found", "invalid_request_error")
        }
    };
    Ok(response)
}

fn route(method: &Method, path: &str) -> Route {
    match (method, path) {
        (&Method::POST, "/v1/chat/completions") => Route::ChatCompletions,
        (_, "/v1/chat/completions") => Route::MethodNotAllowed,
        (&Method::GET, "/props") => Route::Properties,
        (_, "/props") => Route::MethodNotAllowed,
        _ => Route::NotFound,
    }
}

pub(crate) async fn chat_completions(
    req: Request<Incoming>,
    state: ServerState,
) -> Response<ResponseBody> {
    pipeline::chat_completions(req, state).await
}

pub(crate) fn properties_response(context_limit: u32, max_tokens: usize) -> Response<ResponseBody> {
    let bytes = serde_json::to_vec(&api::props::payload(context_limit, max_tokens))
        .unwrap_or_else(|_| b"{}".to_vec());
    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "application/json")
        .body(full(bytes))
        .expect("a response from static headers and a valid body is always valid")
}

// Wraps fixed bytes into the unified boxed body. Full is infallible, so its
// Infallible error is widened to the body's boxed error type.
fn full(bytes: impl Into<Bytes>) -> ResponseBody {
    Full::new(bytes.into())
        .map_err(|never| match never {})
        .boxed()
}

#[cfg(test)]
mod tests {
    use super::*;
    use http_body_util::BodyExt;

    #[test]
    fn removed_embeddings_route_is_standard_not_found() {
        assert_eq!(route(&Method::POST, "/v1/embeddings"), Route::NotFound);
        assert_eq!(route(&Method::GET, "/v1/embeddings"), Route::NotFound);
    }

    #[test]
    fn chat_route_still_distinguishes_wrong_method() {
        assert_eq!(
            route(&Method::POST, "/v1/chat/completions"),
            Route::ChatCompletions
        );
        assert_eq!(
            route(&Method::GET, "/v1/chat/completions"),
            Route::MethodNotAllowed
        );
    }

    #[test]
    fn props_route_is_get_only() {
        assert_eq!(route(&Method::GET, "/props"), Route::Properties);
        assert_eq!(route(&Method::POST, "/props"), Route::MethodNotAllowed);
        assert_eq!(route(&Method::GET, "/props/"), Route::NotFound);
    }

    #[tokio::test]
    async fn props_response_preserves_resolved_capacity() {
        let response = properties_response(u32::MAX, 4096);
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            response.headers().get(header::CONTENT_TYPE).unwrap(),
            "application/json"
        );

        let bytes = response.into_body().collect().await.unwrap().to_bytes();
        assert_eq!(
            serde_json::from_slice::<serde_json::Value>(&bytes).unwrap(),
            serde_json::json!({
                "default_generation_settings": {
                    "n_ctx": u32::MAX,
                    "max_tokens": 4096
                }
            })
        );
    }
}
