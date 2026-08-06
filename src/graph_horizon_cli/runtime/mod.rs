/*
 * Graph Horizon CLI Modules - Runtime
 * Single responsibility: expose chat-only streamed generation to the CLI. It
 * depends on HTTP/local providers and token helpers, and does not expose tool
 * modes, workspace state, or reasoning configuration.
*/

use color_eyre::eyre::Result;
use std::pin::Pin;
use tokio_stream::Stream;

mod client;
mod config;
pub(crate) mod local;
mod message;
mod sse;
mod tokens;

pub(crate) use client::stream_completion as generation_stream;
pub(crate) use config::ClientConfig;
pub(crate) use message::ChatMessage;
pub(crate) use tokens::{Throughput, estimate_tokens, prune_to_limit, pruning_threshold, rate};

// A single assistant text chunk.
pub(crate) struct Chunk {
    pub response: String,
}

// Public API for the runtime module: a stream of Chunks representing the model's response as it arrives.
pub(crate) type ChunkStream = Pin<Box<dyn Stream<Item = Result<Chunk>> + Send>>;

// Returns the response text from a chunk, or None if it is absent or empty.
pub(crate) fn response(chunk: &Chunk) -> Option<&str> {
    (!chunk.response.is_empty()).then_some(chunk.response.as_str())
}
