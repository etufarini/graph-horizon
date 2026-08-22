/*
 * Graph Horizon CLI startup
 * Single responsibility: load and validate the CLI's local model before
 * initializing the terminal, then start the interactive chat session.
 */

use color_eyre::eyre::{Result, eyre};

use crate::app::engine;

use super::console::terminal_user_interface;
use super::plugins::attachments::FileAuthority;
use super::runtime::RuntimeInfo;
use super::runtime::{ContextBudget, RuntimeConfig, local};

const NO_PROMPT_SPACE: &str = "max_tokens leaves no room for the prompt";

fn context_budget(context_limit: usize, max_tokens: usize) -> Result<ContextBudget> {
    ContextBudget::new(context_limit, max_tokens).ok_or_else(|| eyre!(NO_PROMPT_SPACE))
}

pub(crate) async fn run(model_path: Option<String>) -> Result<()> {
    let files = FileAuthority::capture()
        .map_err(|_| color_eyre::eyre::eyre!("startup directory is unavailable"))?;
    let mut config = RuntimeConfig::from_args();
    // A prior conversation is restored later, in-session, via /import, so startup
    // always begins from a fresh history; the system prompt is read out here.
    let system = config.system.clone();
    let max_tokens = config.max_tokens;
    let model = model_path.ok_or_else(|| eyre!("--model must point to a mistral3 GGUF"))?;
    let chat = engine::load_chat_engine(&model, config.context_limit)?;
    // An explicit context override remains authoritative over model metadata.
    config.apply_model_context_limit(chat.context_limit());
    let budget = context_budget(
        config.context_limit.expect("model context is resolved"),
        max_tokens,
    )?;
    let runtime = RuntimeInfo {
        model_name: chat.model_name().unwrap_or("Local model").to_string(),
        backend: chat.backend_name(),
        memory: chat.memory(),
        placement: chat.placement(),
    };
    let generate = local::generation(chat, max_tokens);
    let mut terminal = ratatui::init();
    let result = terminal_user_interface(
        &mut terminal,
        system,
        budget,
        Some(runtime),
        &files,
        generate,
    )
    .await;
    ratatui::restore();
    result
}
