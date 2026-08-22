/*
 * Graph Horizon CLI Modules - Runtime - Local
 * Single responsibility: bridge CLI text-chat requests to the local engine. It
 * converts messages, maps engine events to assistant text chunks, and never
 * supplies tools, workspace data, or reasoning controls.
 *
 * This local sync→async bridge runs synchronous engine decoding on a
 * blocking task and pushes events down an mpsc channel; the receiver is adapted
 * to a ChunkStream. Dropping the stream (UI cancellation)
 * makes blocking_send fail, the sink return false, and the task stop — freeing the
 * request's KV cache, with no orphaned generation left occupying VRAM.
*/

use std::sync::Arc;

use color_eyre::eyre::{Result, eyre};
use graph_horizon_engine::{Engine, Event, EventSink, Request, SamplingParams};
use tokio_stream::wrappers::ReceiverStream;

use super::{ChatMessage, ChunkStream, StreamEvent};

// Builds the console generation closure. The shared Engine is loaded once;
// every request contains only the text-chat fields the engine owns.
pub(crate) fn provider(
    engine: Arc<Engine>,
    max_tokens: usize,
) -> impl Fn(Vec<ChatMessage>) -> std::future::Ready<Result<ChunkStream>> {
    move |messages| std::future::ready(start(engine.clone(), messages, max_tokens))
}

// Spawns the blocking decode and returns the stream immediately. The channel is
// the sync→async boundary; everything downstream sees only a ChunkStream.
fn start(
    engine: Arc<Engine>,
    messages: Vec<ChatMessage>,
    max_tokens: usize,
) -> Result<ChunkStream> {
    let req = Request {
        messages: messages.into_iter().map(ChatMessage::into_engine).collect(),
        sampling: SamplingParams::greedy(),
        max_tokens,
    };

    let (tx, rx) = tokio::sync::mpsc::channel::<Result<StreamEvent>>(32);
    tokio::task::spawn_blocking(move || {
        let mut sink = ChannelSink { tx };
        engine.generate(req, &mut sink);
    });

    Ok(Box::pin(ReceiverStream::new(rx)))
}

// Phase, text, and terminal statistics are forwarded as stream events. Engine
// errors become display-safe errors; the channel closes after generation
// returns or after the receiver cancels it.
struct ChannelSink {
    tx: tokio::sync::mpsc::Sender<Result<StreamEvent>>,
}

impl EventSink for ChannelSink {
    fn cancelled(&self) -> bool {
        self.tx.is_closed()
    }

    fn emit(&mut self, event: Event) -> bool {
        match to_item(event) {
            Some(item) => self.tx.blocking_send(item).is_ok(),
            None => true,
        }
    }
}

fn to_item(event: Event) -> Option<Result<StreamEvent>> {
    match event {
        Event::Phase(phase) => Some(Ok(StreamEvent::Phase(phase))),
        Event::TextDelta(text) => Some(Ok(StreamEvent::Text(text))),
        Event::Finished(stats) => Some(Ok(StreamEvent::Finished(stats))),
        Event::Error(message) => Some(Err(eyre!(message))),
    }
}
