/*
 * Graph Horizon web startup
 * Loads the local chat engine and assembles the static Web wrapper. The loaded
 * engine's resolved context becomes both the server request cap and the values
 * reported under `/props.default_generation_settings`; headless server flag
 * parsing remains separate.
 */

use color_eyre::eyre::{Result, eyre};

use crate::app::engine;
use crate::graph_horizon_server::{ServerConfig, ServerState};

use super::assets::Assets;
use super::config::WebConfig;
use super::server;

pub(crate) async fn run(model_path: Option<String>) -> Result<()> {
    let model = model_path.ok_or_else(|| eyre!("--mode web requires --model"))?;
    let config = WebConfig::from_args();
    let assets = Assets::new(config.asset_root.clone());
    assets.ensure_index()?;
    let context_limit = engine::config::context_tokens_from_args();
    let chat = engine::load_chat_engine(&model, context_limit)?;
    // The engine has already resolved and validated the immutable context limit.
    let chat_config = ServerConfig::chat(chat.context_limit() as usize);
    let state = ServerState::new(chat, chat_config);
    server::serve(config, assets, state).await
}
