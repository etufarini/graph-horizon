/*
 * Graph Horizon web startup
 * Loads the local chat engine, initializes optional Web-search transport, and
 * assembles the bundled UI behind its private browser protocol.
 */

use color_eyre::eyre::{Result, eyre};

use crate::app::engine;

use super::assets::Assets;
use super::chat::State;
use super::config::WebConfig;
use super::search;
use super::server;

pub(crate) async fn run(model_path: Option<String>) -> Result<()> {
    let model = model_path.ok_or_else(|| eyre!("--mode web requires --model"))?;
    let config = WebConfig::from_args();
    let assets = Assets::new(config.asset_root.clone());
    assets.ensure_index()?;
    let search = search::Config::from_args()?;
    let context_limit = engine::config::context_tokens_from_args();
    let chat = engine::load_chat_engine(&model, context_limit)?;
    let max_tokens = engine::config::max_tokens_from_args(chat.context_limit() as usize);
    let state = State::new(chat, search, max_tokens);
    server::serve(config, assets, state).await
}
