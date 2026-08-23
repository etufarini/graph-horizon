/*
 * Graph Horizon Web chat pipeline
 * Validates one bundled-frontend request, optionally adds bounded Web-search
 * context before acquiring the engine mutex, and returns the private stream.
 */

use std::sync::Arc;

use hyper::body::Incoming;
use hyper::{HeaderMap, Request, Response, StatusCode, header};

use super::router::{ResponseBody, error_response};
use super::search;

mod request;
mod state;
mod stream;

pub(super) use state::State;

const CACHE_KEY_HEADER: &str = "x-graph-horizon-cache";

pub(super) async fn handle(request: Request<Incoming>, state: State) -> Response<ResponseBody> {
    let cache_key = match cache_key(request.headers()) {
        Ok(key) => key,
        Err(()) => return error_response(StatusCode::BAD_REQUEST, "invalid request"),
    };
    let mut parsed = match request::parse(
        request,
        state.engine.context_limit() as usize,
        state.engine.default_sampling(),
    )
    .await
    {
        Ok(request) => request,
        Err(request::Error::TooLarge) => {
            return error_response(StatusCode::PAYLOAD_TOO_LARGE, "request too large");
        }
        Err(request::Error::Invalid) => {
            return error_response(StatusCode::BAD_REQUEST, "invalid request");
        }
    };
    // Admission covers both remote search and generation, but the engine mutex
    // remains free while an explicitly requested search waits on the network.
    let permit = match Arc::clone(&state.admission).try_acquire_owned() {
        Ok(permit) => permit,
        Err(_) => return error_response(StatusCode::TOO_MANY_REQUESTS, "busy"),
    };
    let mut source_footer = None;
    if let Some(search) = parsed.search.take() {
        let context = match state.search.context(&search.query, &search.date).await {
            Ok(context) => context,
            Err(search::Error::Busy) => {
                return error_response(StatusCode::TOO_MANY_REQUESTS, "busy");
            }
            Err(search::Error::Unavailable) => {
                return error_response(StatusCode::BAD_GATEWAY, "web search unavailable");
            }
        };
        // Search queries are accepted only when the final engine message is User.
        let message = parsed
            .engine
            .messages
            .last_mut()
            .expect("validated Web search has a final user message");
        message.content.insert_str(0, &context.prompt);
        source_footer = Some(context.sources);
    }
    let guard = Arc::clone(&state.serialize).lock_owned().await;
    let body = stream::body(
        state.engine,
        parsed.engine,
        source_footer,
        cache_key,
        guard,
        permit,
    );
    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "text/event-stream")
        .header(header::CACHE_CONTROL, "no-cache")
        .header(header::CONNECTION, "keep-alive")
        .body(body)
        .expect("fixed Web chat response metadata is valid")
}

fn cache_key(headers: &HeaderMap) -> Result<[u8; 16], ()> {
    let text = headers
        .get(CACHE_KEY_HEADER)
        .ok_or(())?
        .to_str()
        .map_err(|_| ())?;
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
    Ok(key)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_header_is_required_and_strict_lowercase_hex() {
        let mut headers = HeaderMap::new();
        assert_eq!(cache_key(&headers), Err(()));
        headers.insert(
            CACHE_KEY_HEADER,
            "00112233445566778899aabbccddeeff".parse().unwrap(),
        );
        assert_eq!(
            cache_key(&headers),
            Ok([
                0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd,
                0xee, 0xff,
            ])
        );
    }
}
