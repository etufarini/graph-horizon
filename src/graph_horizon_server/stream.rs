/*
 * Graph Horizon headless server - Stream (sync→async bridge for chat)
 * Owns the server's sync→async inference bridge, analogous to
 * `runtime/local.rs`. The engine decodes synchronously,
 * so generation runs on `spawn_blocking`, pushing ready-made SSE lines down an
 * mpsc channel; the receiver is adapted to a `hyper` SSE body. This file does not
 * know the OpenAI wire format (delegated to `api`): it only moves bytes and owns
 * the lifecycle.
 *
 * The channel holds at most 32 frames. A slow client therefore applies
 * backpressure to generation instead of growing process memory without bound.
 * The serialization guard and admission permit remain held while generation is
 * blocked on that backpressure; disconnecting the client closes the receiver,
 * makes `blocking_send` fail, and cancels generation.
 *
 * Three concerns are owned here:
 *  - Serialization: `guard` and `permit` are moved into the blocking task and
 *    dropped exactly when `generate` returns (or on cancellation), never earlier.
 *  - Cancellation: dropping the receiver (client disconnect / stream abandoned)
 *    makes `send` return Err → the sink returns false → the decode loop stops →
 *    the KV cache is freed. No orphaned generation, no lock held for nothing.
 *  - Engine error: `Event::Error` becomes a generic SSE error line (no
 *    Report internals exposed) followed by `[DONE]`, then a clean close. We send
 *    `[DONE]` even on error because the natural consumer is this repo's inbound
 *    parser (`runtime/sse.rs`) and the OpenAI SDKs, which terminate cleanly on the
 *    sentinel / stream close.
*/

use std::convert::Infallible;
use std::sync::Arc;

use bytes::Bytes;
use graph_horizon_engine::{Engine, Event, EventSink, Request};
use http_body_util::{BodyExt, StreamBody};
use hyper::body::Frame;
use tokio::sync::OwnedSemaphorePermit;
use tokio::sync::mpsc;
use tokio_stream::StreamExt;
use tokio_stream::wrappers::ReceiverStream;

use super::api;
use super::handler::ResponseBody;

// Owned guard of the serialization turnstile. Acquired by the handler before this
// is called and moved into the blocking task so it lives exactly as long as the
// generation. Aliased for the (verbose) tokio type.
pub(super) type OwnedGuard = tokio::sync::OwnedMutexGuard<()>;

// Builds the streaming SSE body for one chat generation, ready to drop into a
// `Response`. Receives the shared engine, the already-built request, and the
// serialization guard + admission permit (both already acquired by the handler):
// it owns their lifetime from here until generation ends or is cancelled. See the
// file-level comment for the bounded-channel lifecycle.
pub(super) fn sse_body(
    engine: Arc<Engine>,
    req: Request,
    cache_key: Option<[u8; 16]>,
    guard: OwnedGuard,
    permit: OwnedSemaphorePermit,
) -> ResponseBody {
    let (tx, rx) = mpsc::channel::<String>(32);

    tokio::task::spawn_blocking(move || {
        // Moving and binding both guards here holds admission and serialization
        // until generation returns, including cancellation exits.
        let _guard = guard;
        let _permit = permit;

        let mut sink = ServerSink { tx };
        if let Some(key) = cache_key {
            engine.generate_cached(key, req, &mut sink);
        } else {
            engine.generate(req, &mut sink);
        }
    });

    let stream =
        ReceiverStream::new(rx).map(|line| Ok::<_, Infallible>(Frame::data(Bytes::from(line))));
    StreamBody::new(stream)
        .map_err(|never: Infallible| -> Box<dyn std::error::Error + Send + Sync> { match never {} })
        .boxed()
}

struct ServerSink {
    tx: mpsc::Sender<String>,
}

impl EventSink for ServerSink {
    fn cancelled(&self) -> bool {
        self.tx.is_closed()
    }

    fn emit(&mut self, event: Event) -> bool {
        match &event {
            Event::Phase(phase) => self.tx.blocking_send(api::chat::phase_line(*phase)).is_ok(),
            // A terminal event closes this sink after its final SSE frames.
            Event::Error(_) => {
                if let Some(line) = api::chat::event_line(&event) {
                    let _ = self.tx.blocking_send(line);
                }
                let _ = self.tx.blocking_send(api::chat::done_line());
                false
            }
            Event::Finished(stats) => {
                let _ = self.tx.blocking_send(api::chat::usage_line(stats));
                let _ = self.tx.blocking_send(api::chat::final_line());
                let _ = self.tx.blocking_send(api::chat::done_line());
                false
            }
            Event::TextDelta(_) => match api::chat::event_line(&event) {
                Some(line) => self.tx.blocking_send(line).is_ok(),
                None => true,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disconnected_receiver_cancels_generation() {
        let (tx, rx) = mpsc::channel(1);
        drop(rx);
        let mut sink = ServerSink { tx };

        assert!(!sink.emit(Event::TextDelta("ignored".into())));
        assert!(sink.cancelled());
    }
}
