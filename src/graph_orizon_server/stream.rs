/*
 * Graph Orizon headless server - Stream (sync→async bridge for chat)
 * The ONLY place on the server side that knows `tokio` for inference: the
 * server-side analogue of `runtime/local.rs`. The engine decodes synchronously,
 * so generation runs on `spawn_blocking`, pushing ready-made SSE lines down an
 * mpsc channel; the receiver is adapted to a `hyper` SSE body. This file does not
 * know the OpenAI wire format (delegated to `api`): it only moves bytes and owns
 * the lifecycle.
 *
 * The channel is unbounded BY DESIGN. Unlike local.rs (bounded-32 + blocking_send)
 * the consumer here is a network client we do not trust to read promptly: `send`
 * is synchronous and never blocks, so generation runs at the engine's full speed,
 * DECOUPLED from the consumer. The guard (serialization) and the admission permit
 * are released the instant `generate` returns — not when the client finishes
 * reading — so a slow client never holds the global mutex nor a queue slot hostage.
 * The buffer is bounded in practice by `max_tokens` (already clamped to
 * the `--max-tokens` ceiling), so no explicit capacity cap is needed.
 *
 * Three concerns are owned here:
 *  - Serialization: `guard` and `permit` are moved into the blocking task and
 *    dropped exactly when `generate` returns (or on cancellation), never earlier.
 *  - Cancellation: dropping the receiver (client disconnect / stream abandoned)
 *    makes `send` return Err → the sink returns false → the decode loop stops →
 *    the KV cache is freed. No orphaned generation, no lock held for nothing.
 *  - Engine error: `EngineEvent::Error` becomes a generic SSE error line (no
 *    Report internals exposed) followed by `[DONE]`, then a clean close. We send
 *    `[DONE]` even on error because the natural consumer is this repo's inbound
 *    parser (`runtime/sse.rs`) and the OpenAI SDKs, which terminate cleanly on the
 *    sentinel / stream close.
*/

use std::convert::Infallible;
use std::sync::Arc;

use bytes::Bytes;
use graph_orizon_engine::{Engine, Event, EventSink, Request};
use http_body_util::{BodyExt, StreamBody};
use hyper::body::Frame;
use tokio::sync::OwnedSemaphorePermit;
use tokio::sync::mpsc::unbounded_channel;
use tokio_stream::StreamExt;
use tokio_stream::wrappers::UnboundedReceiverStream;

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
// file-level comment for the unbounded-channel rationale.
pub(super) fn sse_body(
    engine: Arc<Engine>,
    req: Request,
    guard: OwnedGuard,
    permit: OwnedSemaphorePermit,
) -> ResponseBody {
    // Unbounded so `send` never blocks: generation is decoupled from the consumer
    // (see file-level comment). The bound is `max_tokens` in practice.
    let (tx, rx) = unbounded_channel::<String>();

    tokio::task::spawn_blocking(move || {
        // guard and permit are MOVED in here: they are dropped when this closure
        // returns (normal end OR cancellation), releasing the global mutex and
        // returning the admission slot. Bind them so they are not dropped early.
        let _guard = guard;
        let _permit = permit;

        let mut sink = ServerSink { tx };
        engine.generate(req, &mut sink);
        // Returning here drops _guard and _permit: lock released, slot returned.
    });

    let stream = UnboundedReceiverStream::new(rx)
        .map(|line| Ok::<_, Infallible>(Frame::data(Bytes::from(line))));
    StreamBody::new(stream)
        .map_err(|never: Infallible| -> Box<dyn std::error::Error + Send + Sync> { match never {} })
        .boxed()
}

struct ServerSink {
    tx: tokio::sync::mpsc::UnboundedSender<String>,
}

impl EventSink for ServerSink {
    fn cancelled(&self) -> bool {
        self.tx.is_closed()
    }

    fn emit(&mut self, event: Event) -> bool {
        match &event {
            // A terminal event closes this sink after its final SSE frames.
            Event::Error(_) => {
                if let Some(line) = api::chat::event_line(&event) {
                    let _ = self.tx.send(line);
                }
                let _ = self.tx.send(api::chat::done_line());
                false
            }
            Event::Finished(stats) => {
                let _ = self.tx.send(api::chat::usage_line(stats));
                let _ = self.tx.send(api::chat::final_line());
                let _ = self.tx.send(api::chat::done_line());
                false
            }
            Event::TextDelta(_) => match api::chat::event_line(&event) {
                Some(line) => self.tx.send(line).is_ok(),
                None => true,
            },
        }
    }
}
