/*
 * GH Zero headless server startup
 * Starts only the OpenAI-compatible server mode: it loads required engines,
 * creates shared server state, then hands ownership to the TCP server loop.
 */

use color_eyre::eyre::{Result, eyre};

use crate::app::engine;

use super::{ServerConfig, ServerState, server};

pub(crate) async fn run(model_path: Option<String>) -> Result<()> {
    let model = model_path
        .clone()
        .ok_or_else(|| eyre!("--mode server requires --model to point to a mistral3 GGUF"))?;
    let chat = engine::load_chat_engine(&model, engine::config::context_tokens_from_args())?;
    let state = ServerState::new(chat, ServerConfig::from_args());
    server::serve(state).await
}
