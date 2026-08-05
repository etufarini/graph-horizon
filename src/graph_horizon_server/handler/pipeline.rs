/*
 * Graph Horizon headless server handler pipeline
 * Single responsibility: validate text-chat HTTP requests and hand accepted
 * requests to one engine stream. It depends on body validation, chat API
 * mapping, admission gates, and SSE streaming; it does not run tools/reasoning.
 */

use std::sync::Arc;

use hyper::body::Incoming;
use hyper::{Request, Response, StatusCode, header};
use tokio::sync::{Mutex, OwnedMutexGuard, OwnedSemaphorePermit, Semaphore};

use super::super::{ServerState, api, stream};
use super::ResponseBody;
use super::body::{read_body, validate_chat_body};
use super::error::error_response;

// The full chat path. Order is deliberate: read the body under the size cap →
// parse JSON → validate into an engine request → only THEN take an admission slot
// and the serialization lock. The permit and lock are acquired strictly after
// validation succeeds, so an invalid request never occupies a queue slot nor
// waits on the mutex. Both are then handed to `stream::sse_body`, which holds
// them for exactly the lifetime of the generation.
pub(super) async fn chat_completions(
    req: Request<Incoming>,
    state: ServerState,
) -> Response<ResponseBody> {
    // 1. Read the body under the hard size cap (413 if oversized, 400 if broken).
    let bytes = match read_body(req).await {
        Ok(bytes) => bytes,
        Err(response) => return response,
    };

    if let Err(response) = validate_chat_body(&bytes) {
        return response;
    }

    // 2. Parse JSON. Malformed body → 400 with a generic message (no parser detail).
    let parsed: api::chat::ChatCompletionRequest = match serde_json::from_slice(&bytes) {
        Ok(req) => req,
        Err(_) => {
            return error_response(
                StatusCode::BAD_REQUEST,
                "invalid request body",
                "invalid_request_error",
            );
        }
    };

    // 3. Validate + build the engine request. Every ApiError → 400, generic message.
    let request = match api::chat::to_request(parsed, &state.config) {
        Ok(request) => request,
        Err(err) => {
            return error_response(
                StatusCode::BAD_REQUEST,
                err.message(),
                "invalid_request_error",
            );
        }
    };

    // 4 + 5. Admission slot then serialization lock; 429 if the queue is full.
    let (guard, permit) = match acquire_permits(state.admission, state.serialize).await {
        Ok(pair) => pair,
        Err(response) => return response,
    };

    // 6 + 7. Build the streaming body (it owns guard + permit for the generation's
    // lifetime) and return 200 with the SSE headers. Without these headers many
    // clients will not interpret the stream.
    let sse = stream::sse_body(state.chat, request, guard, permit);
    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "text/event-stream")
        .header(header::CACHE_CONTROL, "no-cache")
        .header(header::CONNECTION, "keep-alive")
        .body(sse)
        .expect("a response from static headers and a valid body is always valid")
}

// Takes an admission slot then the serialization lock, in that order — the dedup
// of steps 4 and 5, identical in both pipelines. Admission is non-blocking: a
// full queue returns 429 immediately instead of piling up on the mutex; the
// permit lives for the whole request. The serialization lock then queues admitted
// requests asynchronously (the runtime is never blocked), at most MAX_INFLIGHT - 1
// waiting, bounded by the permit above. On rejection the caller returns the 429.
pub(crate) async fn acquire_permits(
    admission: Arc<Semaphore>,
    serialize: Arc<Mutex<()>>,
) -> Result<(OwnedMutexGuard<()>, OwnedSemaphorePermit), Response<ResponseBody>> {
    let permit = match admission.try_acquire_owned() {
        Ok(permit) => permit,
        Err(_) => {
            return Err(error_response(
                StatusCode::TOO_MANY_REQUESTS,
                "server is busy, retry later",
                "rate_limit_error",
            ));
        }
    };
    let guard = serialize.lock_owned().await;
    Ok((guard, permit))
}
