/*
 * Graph Horizon web startup
 * Loads the local chat engine and assembles the bundled Web UI. The loaded
 * engine's immutable limits and runtime facts feed only the private browser
 * transport.
 */

use color_eyre::eyre::{Result, eyre};

use crate::app::engine;

use super::assets::Assets;
use super::chat::State;
use super::config::WebConfig;
use super::server;

pub(crate) async fn run(model_path: Option<String>) -> Result<()> {
    let model = model_path.ok_or_else(|| eyre!("--mode web requires --model"))?;
    let config = WebConfig::from_args();
    let assets = Assets::new(config.asset_root.clone());
    assets.ensure_index()?;
    let context_limit = engine::config::context_tokens_from_args();
    let chat = engine::load_chat_engine(&model, context_limit)?;
    let state = State::new(chat);
    server::serve(config, assets, state).await
}
