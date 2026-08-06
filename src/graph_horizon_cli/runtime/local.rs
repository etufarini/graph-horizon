/*
 * Graph Horizon CLI Modules - Runtime - Local
 * Single responsibility: bridge CLI text-chat requests to the local engine. It
 * converts messages, maps engine events to assistant text chunks, and never
 * supplies tools, workspace data, or reasoning controls.
 *
 * Sync→async bridge (this is the ONLY place that knows tokio): the engine decodes
 * synchronously on a spawn_blocking task that pushes events down an mpsc channel;
 * the receiver is adapted to a ChunkStream. Dropping the stream (UI cancellation)
 * makes blocking_send fail, the sink return false, and the task stop — freeing the
 * request's KV cache, with no orphaned generation left occupying VRAM.
*/

use std::sync::Arc;

use color_eyre::eyre::{Result, eyre};
use graph_horizon_engine::{Engine, Event, EventSink, Request, SamplingParams};
use tokio_stream::wrappers::ReceiverStream;

use super::{ChatMessage, Chunk, ChunkStream};

// Builds the local provider closure, shaped like the HTTP path. The shared Engine
// is loaded once; every request contains only the text-chat fields the engine owns.
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

    let (tx, rx) = tokio::sync::mpsc::channel::<Result<Chunk>>(32);
    tokio::task::spawn_blocking(move || {
        let mut sink = ChannelSink { tx };
        engine.generate(req, &mut sink);
    });

    Ok(Box::pin(ReceiverStream::new(rx)))
}

// EngineEvent → stream item (spec sez. 2). Finished maps to None: it is the Ok
// end of the stream, carried by the channel closing, not by a Chunk. A context
// overflow during the turn is an accepted, expected failure (the user wants a
// correct answer, not corrupted output); it is translated here into a readable
// CLI message instead of surfacing the raw engine error.
struct ChannelSink {
    tx: tokio::sync::mpsc::Sender<Result<Chunk>>,
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

fn to_item(event: Event) -> Option<Result<Chunk>> {
    match event {
        Event::TextDelta(s) => Some(Ok(Chunk { response: s })),
        Event::Finished(_) => None,
        Event::Error(message) => Some(Err(eyre!(message))),
    }
}
