/*
 * Graph Horizon CLI Modules - Runtime - Client
 * Single responsibility: stream text-only HTTP chat completions into runtime
 * chunks. It depends on reqwest, SSE parsing, and ClientConfig, and never
 * serializes tool selection, workspace data, or internal-channel controls.
*/

use color_eyre::eyre::{Result, eyre};
use reqwest::Client;
use serde::Deserialize;
use serde_json::json;
use std::time::Duration;
use tokio_stream::StreamExt;
use tokio_stream::wrappers::ReceiverStream;

use super::{ChatMessage, Chunk, ChunkStream, ClientConfig, sse};

pub(super) const DEFAULT_BASE_URL: &str = "http://127.0.0.1:8080/v1";

// Subset of the HTTP provider's /props response we depend on. Tolerant by omission:
// only the context size is deserialized and the rest of the payload is ignored.
// A missing or non-integer `n_ctx` makes the whole parse fail into None, which
// the caller treats as "no limit" rather than a fatal error.
#[derive(Deserialize)]
struct Props {
    default_generation_settings: GenSettings,
}

// The HTTP provider reports the per-slot context window here. With N parallel slots
// the total context is split N ways, so this is the size a single request may
// actually use — exactly the figure the pruning threshold needs.
#[derive(Deserialize)]
struct GenSettings {
    n_ctx: usize,
}

// Derives the /props URL from the chat base URL. `/props` is served at the
// server root, not under the OpenAI-compat `/v1` prefix, so that suffix
// (and any trailing slash) is stripped before appending.
fn props_url(base_url: &str) -> String {
    let root = base_url.trim_end_matches('/');
    let root = root.strip_suffix("/v1").unwrap_or(root);
    format!("{root}/props")
}

// Asks the configured HTTP provider for its context window via GET /props.
// Returns the per-slot `n_ctx` (when > 0), or None on any failure — server
// down, timeout, non-success status, or unexpected JSON — so the caller falls
// back to pruning disabled instead of aborting startup. The short timeout keeps
// a slow or unreachable server from stalling the launch.
pub(super) async fn fetch_context_limit(base_url: &str) -> Option<usize> {
    let client = Client::builder()
        .timeout(Duration::from_secs(3))
        .build()
        .ok()?;
    let props: Props = client
        .get(props_url(base_url))
        .send()
        .await
        .ok()?
        .error_for_status()
        .ok()?
        .json()
        .await
        .ok()?;
    Some(props.default_generation_settings.n_ctx).filter(|&n| n > 0)
}

// Streams a chat completion from the configured OpenAI-compatible endpoint, parsing SSE lines into Chunks.
pub(super) async fn stream_completion(
    messages: Vec<ChatMessage>,
    config: ClientConfig,
) -> Result<ChunkStream> {
    let url = format!("{}/chat/completions", config.base_url.trim_end_matches('/'));

    let mut body = json!({
        "messages": messages,
        "stream": true,
    });
    // The external server owns the loaded model, so no model name is
    // sent. `max_tokens` is the only generation cap and pruning reserve.
    body["max_tokens"] = json!(config.max_tokens);

    let response = Client::new().post(&url).json(&body).send().await?;

    if !response.status().is_success() {
        return Err(eyre!("provider returned {}", response.status()));
    }

    let (tx, rx) = tokio::sync::mpsc::channel::<Result<Chunk>>(32);

    // Spawn a background task so the caller can start polling the channel
    // while the response body streams in, without blocking the async executor.
    tokio::spawn(async move {
        let mut bytes_stream = response.bytes_stream();
        // Accumulate raw bytes: a multi-byte UTF-8 character may straddle two
        // HTTP chunks, so we must not decode until we have a complete line.
        let mut buffer: Vec<u8> = Vec::new();
        while let Some(item) = bytes_stream.next().await {
            let bytes = match item {
                Ok(b) => b,
                Err(e) => {
                    let _ = tx.send(Err(e.into())).await;
                    return;
                }
            };

            buffer.extend_from_slice(&bytes);

            while let Some(pos) = buffer.iter().position(|&b| b == b'\n') {
                let line = match std::str::from_utf8(&buffer[..pos]) {
                    Ok(line) => line.trim_end_matches('\r').to_string(),
                    Err(_) => {
                        let _ = tx.send(Err(eyre!("provider returned invalid UTF-8"))).await;
                        return;
                    }
                };
                buffer.drain(..=pos);

                if sse::handle_sse_line(&line, &tx).await {
                    return;
                }
            }
        }

        // The body ended without a trailing newline on the final line (e.g. a
        // provider that closes the connection without a [DONE] sentinel). Flush
        // whatever remains so the last content token isn't silently dropped.
        if !buffer.is_empty() {
            let line = match std::str::from_utf8(&buffer) {
                Ok(line) => line.trim_end_matches('\r').to_string(),
                Err(_) => {
                    let _ = tx.send(Err(eyre!("provider returned invalid UTF-8"))).await;
                    return;
                }
            };
            let _ = sse::handle_sse_line(&line, &tx).await;
        }
    });

    Ok(Box::pin(ReceiverStream::new(rx)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    #[test]
    fn props_url_strips_v1_suffix_and_trailing_slash() {
        // /props sits at the server root, so the /v1 suffix and any trailing
        // slash must be dropped; a root-only base URL is left as-is.
        assert_eq!(
            props_url("http://127.0.0.1:8080/v1"),
            "http://127.0.0.1:8080/props"
        );
        assert_eq!(
            props_url("http://127.0.0.1:8080/v1/"),
            "http://127.0.0.1:8080/props"
        );
        assert_eq!(
            props_url("http://127.0.0.1:8080"),
            "http://127.0.0.1:8080/props"
        );
    }

    #[tokio::test]
    async fn flushes_unterminated_final_sse_line() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut request = [0_u8; 4096];
            let _ = socket.read(&mut request).await.unwrap();

            let body = r#"data: {"choices":[{"delta":{"content":"last"}}]}"#;
            let response = format!(
                "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
                body.len()
            );
            socket.write_all(response.as_bytes()).await.unwrap();
        });

        let config = ClientConfig {
            base_url: format!("http://{address}/v1"),
            system: None,
            context_limit: None,
            max_tokens: 1,
        };
        let mut stream = stream_completion(Vec::new(), config).await.unwrap();
        let chunk = stream.next().await.unwrap().unwrap();
        assert_eq!(chunk.response, "last");
        assert!(stream.next().await.is_none());
        server.await.unwrap();
    }

    #[tokio::test]
    async fn rejects_invalid_utf8_sse_lines() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut request = [0_u8; 4096];
            let _ = socket.read(&mut request).await.unwrap();

            let body = b"data: \xff\n";
            let headers = format!(
                "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ncontent-length: {}\r\nconnection: close\r\n\r\n",
                body.len()
            );
            socket.write_all(headers.as_bytes()).await.unwrap();
            socket.write_all(body).await.unwrap();
        });

        let config = ClientConfig {
            base_url: format!("http://{address}/v1"),
            system: None,
            context_limit: None,
            max_tokens: 1,
        };
        let mut stream = stream_completion(Vec::new(), config).await.unwrap();
        let error = match stream.next().await.unwrap() {
            Ok(_) => panic!("invalid UTF-8 was accepted"),
            Err(error) => error.to_string(),
        };
        assert_eq!(error, "provider returned invalid UTF-8");
        assert!(stream.next().await.is_none());
        server.await.unwrap();
    }
}
