/*
 * Graph Horizon headless server handler pipeline
 * Single responsibility: validate text-chat HTTP requests and hand accepted
 * requests to one engine stream. It depends on body validation, chat API
 * mapping, admission gates, and SSE streaming; it does not run tools/reasoning.
 */

use std::sync::Arc;

use hyper::body::Incoming;
use hyper::{HeaderMap, Request, Response, StatusCode, header};
use tokio::sync::{Mutex, OwnedMutexGuard, OwnedSemaphorePermit, Semaphore};

use super::super::{ServerState, api, stream};
use super::ResponseBody;
use super::body::{read_body, validate_chat_body};
use super::error::error_response;

const CACHE_KEY_HEADER: &str = "x-graph-horizon-cache";

// The full chat path. Order is deliberate: read the body under the size cap →
// prevalidate raw JSON → deserialize the typed body → build an engine request →
// only THEN take an admission slot and the serialization lock. Both are acquired
// strictly after validation succeeds, so an invalid request never occupies a
// queue slot nor waits on the mutex. Both are then handed to `stream::sse_body`,
// which holds them for exactly the lifetime of the generation.
pub(super) async fn chat_completions(
    req: Request<Incoming>,
    state: ServerState,
) -> Response<ResponseBody> {
    let cache_key = match cache_key(req.headers()) {
        Ok(key) => key,
        Err(()) => {
            return error_response(
                StatusCode::BAD_REQUEST,
                "invalid cache key",
                "invalid_request_error",
            );
        }
    };
    // 1. Read the body under the hard size cap (413 if oversized, 400 if broken).
    let bytes = match read_body(req).await {
        Ok(bytes) => bytes,
        Err(response) => return response,
    };

    // 2. Reject unsupported JSON shapes and fields before typed deserialization.
    if let Err(response) = validate_chat_body(&bytes) {
        return response;
    }

    // 3. Parse JSON. Malformed body → 400 with a generic message (no parser detail).
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

    // 4. Validate + build the engine request. Every ApiError → 400, generic message.
    let sampling = state.chat.default_sampling();
    let request = match api::chat::to_request(parsed, &state.config, sampling) {
        Ok(request) => request,
        Err(err) => {
            return error_response(
                StatusCode::BAD_REQUEST,
                err.message(),
                "invalid_request_error",
            );
        }
    };

    // 5 + 6. Admission slot then serialization lock; 429 if the queue is full.
    let (guard, permit) = match acquire_permits(state.admission, state.serialize).await {
        Ok(pair) => pair,
        Err(response) => return response,
    };

    // 7. Build the streaming body (it owns guard + permit for the generation's
    // lifetime) and return 200 with the SSE headers. Without these headers many
    // clients will not interpret the stream.
    let sse = stream::sse_body(state.chat, request, cache_key, guard, permit);
    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "text/event-stream")
        .header(header::CACHE_CONTROL, "no-cache")
        .header(header::CONNECTION, "keep-alive")
        .body(sse)
        .expect("a response from static headers and a valid body is always valid")
}

fn cache_key(headers: &HeaderMap) -> Result<Option<[u8; 16]>, ()> {
    let Some(value) = headers.get(CACHE_KEY_HEADER) else {
        return Ok(None);
    };
    let text = value.to_str().map_err(|_| ())?;
    if text.len() != 32 {
        return Err(());
    }
    let mut key = [0; 16];
    for (index, pair) in text.as_bytes().chunks_exact(2).enumerate() {
        let nibble = |byte| match byte {
            b'0'..=b'9' => Ok(byte - b'0'),
            b'a'..=b'f' => Ok(byte - b'a' + 10),
            _ => Err(()),
        };
        key[index] = (nibble(pair[0])? << 4) | nibble(pair[1])?;
    }
    Ok(Some(key))
}

// Takes an admission slot then the serialization lock, in that order. Admission
// is non-blocking: a full queue returns 429 immediately instead of piling up on the mutex; the
// permit lives for the whole request. The serialization lock then queues admitted
// requests asynchronously (the runtime is never blocked); at most the semaphore
// capacity minus the active holder can wait. On rejection the caller returns 429.
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_header_is_optional_and_strict_lowercase_hex() {
        let mut headers = HeaderMap::new();
        assert_eq!(cache_key(&headers), Ok(None));

        headers.insert(
            CACHE_KEY_HEADER,
            "00112233445566778899aabbccddeeff".parse().unwrap(),
        );
        assert_eq!(
            cache_key(&headers),
            Ok(Some([
                0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd,
                0xee, 0xff,
            ]))
        );

        for invalid in [
            "00112233445566778899aabbccddeef",
            "00112233445566778899AABBCCDDEEFF",
            "00112233445566778899aabbccddeefg",
        ] {
            headers.insert(CACHE_KEY_HEADER, invalid.parse().unwrap());
            assert_eq!(cache_key(&headers), Err(()));
        }
    }
}
