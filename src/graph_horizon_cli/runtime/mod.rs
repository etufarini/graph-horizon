/*
 * Graph Horizon CLI Modules - Runtime
 * Single responsibility: expose provider-neutral chat messages, context
 * admission values, and streamed generation to the CLI console.
*/

use color_eyre::eyre::Result;
use std::pin::Pin;
use tokio_stream::Stream;

mod config;
mod context;
pub(crate) mod local;
mod message;

pub(crate) use config::RuntimeConfig;
pub(crate) use context::{CapacityError, ContextBudget, ContextUsage};
pub(crate) use graph_horizon_engine::{
    GenerationPhase, GenerationStats, ModelMemory, PlacementReport,
};
pub(crate) use message::ChatMessage;

#[derive(Clone)]
pub(crate) struct RuntimeInfo {
    pub(crate) model_name: String,
    pub(crate) backend: &'static str,
    pub(crate) memory: ModelMemory,
    pub(crate) placement: Option<PlacementReport>,
}

pub(crate) enum StreamEvent {
    Text(String),
    Phase(GenerationPhase),
    Finished(GenerationStats),
}

pub(crate) type ChunkStream = Pin<Box<dyn Stream<Item = Result<StreamEvent>> + Send>>;
