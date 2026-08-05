/*
 * Graph Horizon app engine boundary
 * Loads and validates the chat model engine before startup hands it to a process
 * surface.
 */

use std::path::Path;
use std::sync::Arc;

use color_eyre::eyre::Result;
use graph_horizon_engine::Engine;

pub(crate) mod config;

pub(crate) fn load_chat_engine(path: &str, context_limit: Option<usize>) -> Result<Arc<Engine>> {
    Ok(Arc::new(Engine::new(
        Path::new(path),
        config::engine_config(context_limit),
    )?))
}
