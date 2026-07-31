/*
 * GH Zero headless server handler
 * Receives raw hyper requests and routes them to the chat pipeline.
 * It owns the shared response body type and delegates validation/inference work.
 */

use std::convert::Infallible;

use bytes::Bytes;
use http_body_util::{BodyExt, Full, combinators::BoxBody};
use hyper::body::Incoming;
use hyper::{Method, Request, Response, StatusCode};

use super::ServerState;
use error::error_response;

mod body;
mod error;
mod pipeline;

#[derive(Debug, PartialEq, Eq)]
enum Route {
    ChatCompletions,
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
        // The real chat path: body cap → parse → validate → admission → lock →
        // streaming SSE generation.
        Route::ChatCompletions => chat_completions(req, state).await,
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
        _ => Route::NotFound,
    }
}

pub(crate) async fn chat_completions(
    req: Request<Incoming>,
    state: ServerState,
) -> Response<ResponseBody> {
    pipeline::chat_completions(req, state).await
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
}
