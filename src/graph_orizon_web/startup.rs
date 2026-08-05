/*
 * Graph Orizon web startup
 * Single responsibility: load the local chat engine and start the static web
 * wrapper around the headless server. It depends on app engine loading, web
 * assets/config, and graph_orizon_server state; it creates no tools or reasoning.
 */

use color_eyre::eyre::{Result, eyre};

use crate::app::engine;
use crate::graph_orizon_server::{ServerConfig, ServerState};

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
    let chat_config = ServerConfig::chat(config.max_tokens, false);
    let state = ServerState::new(chat, chat_config);
    server::serve(config, assets, state).await
}
