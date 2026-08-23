/*
 * Graph Horizon Web chat stream
 * Bridges synchronous engine generation into the bundled frontend's private
 * SSE event shape while retaining cancellation and gate ownership. It contains
 * no routing or public compatibility vocabulary.
 */

use std::convert::Infallible;
use std::sync::Arc;

use bytes::Bytes;
use graph_horizon_engine::{Engine, Event, EventSink, GenerationPhase, Request};
use http_body_util::{BodyExt, StreamBody};
use hyper::body::Frame;
use serde::Serialize;
use tokio::sync::mpsc;
use tokio::sync::{OwnedMutexGuard, OwnedSemaphorePermit};
use tokio_stream::StreamExt;
use tokio_stream::wrappers::ReceiverStream;

use super::super::router::ResponseBody;

pub(super) fn body(
    engine: Arc<Engine>,
    request: Request,
    source_footer: Option<String>,
    cache_key: [u8; 16],
    guard: OwnedMutexGuard<()>,
    permit: OwnedSemaphorePermit,
) -> ResponseBody {
    let (tx, rx) = mpsc::channel::<String>(32);
    tokio::task::spawn_blocking(move || {
        // Both gates live exactly as long as generation, including cancellation.
        let _guard = guard;
        let _permit = permit;
        engine.generate_cached(cache_key, request, &mut WebSink { tx, source_footer });
    });

    let stream =
        ReceiverStream::new(rx).map(|line| Ok::<_, Infallible>(Frame::data(Bytes::from(line))));
    StreamBody::new(stream)
        .map_err(|never: Infallible| -> Box<dyn std::error::Error + Send + Sync> { match never {} })
        .boxed()
}

struct WebSink {
    tx: mpsc::Sender<String>,
    source_footer: Option<String>,
}

impl EventSink for WebSink {
    fn cancelled(&self) -> bool {
        self.tx.is_closed()
    }

    fn emit(&mut self, event: Event) -> bool {
        match event {
            Event::Phase(phase) => self.send(&PhaseFrame {
                phase: match phase {
                    GenerationPhase::Prefill => "prefill",
                    GenerationPhase::Decode => "decode",
                },
            }),
            Event::TextDelta(content) if content.is_empty() => true,
            Event::TextDelta(content) => self.send(&ContentFrame { content: &content }),
            Event::Error(_) => {
                let _ = self.send(&ErrorFrame {
                    error: "generation failed",
                });
                false
            }
            Event::Finished(stats) => {
                if let Some(content) = self.source_footer.take()
                    && !self.send(&ContentFrame { content: &content })
                {
                    return false;
                }
                let _ = self.send(&StatsFrame {
                    stats: Stats {
                        prompt_tokens: stats.prompt_tokens,
                        prefill_tokens: stats.prefill_tokens,
                        completion_tokens: stats.completion_tokens,
                        prefill_ms: stats.prefill_ms,
                        decode_ms: stats.decode_ms,
                    },
                });
                let _ = self.send(&DoneFrame { done: true });
                false
            }
        }
    }
}

impl WebSink {
    fn send<T: Serialize>(&self, frame: &T) -> bool {
        let json = serde_json::to_string(frame).unwrap_or_else(|_| "{}".to_string());
        self.tx.blocking_send(format!("data: {json}\n\n")).is_ok()
    }
}

#[derive(Serialize)]
struct ContentFrame<'a> {
    content: &'a str,
}

#[derive(Serialize)]
struct PhaseFrame<'a> {
    phase: &'a str,
}

#[derive(Serialize)]
struct ErrorFrame<'a> {
    error: &'a str,
}

#[derive(Serialize)]
struct StatsFrame {
    stats: Stats,
}

#[derive(Serialize)]
struct Stats {
    prompt_tokens: usize,
    prefill_tokens: usize,
    completion_tokens: usize,
    prefill_ms: u64,
    decode_ms: u64,
}

#[derive(Serialize)]
struct DoneFrame {
    done: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disconnected_receiver_cancels_generation() {
        let (tx, rx) = mpsc::channel(1);
        drop(rx);
        let mut sink = WebSink {
            tx,
            source_footer: None,
        };
        assert!(!sink.emit(Event::TextDelta("ignored".into())));
        assert!(sink.cancelled());
    }

    #[test]
    fn source_footer_precedes_terminal_frames() {
        let (tx, mut rx) = mpsc::channel(4);
        let mut sink = WebSink {
            tx,
            source_footer: Some("\n\n### Sources\n- [S1](<https://example.com/>)\n".into()),
        };
        assert!(
            !sink.emit(Event::Finished(graph_horizon_engine::GenerationStats {
                prompt_tokens: 2,
                prefill_tokens: 2,
                completion_tokens: 1,
                prefill_ms: 10,
                decode_ms: 20,
            }))
        );
        drop(sink);

        assert!(rx.blocking_recv().unwrap().contains("### Sources"));
        assert!(rx.blocking_recv().unwrap().contains("\"stats\""));
        assert!(rx.blocking_recv().unwrap().contains("\"done\":true"));
        assert!(rx.blocking_recv().is_none());
    }
}
