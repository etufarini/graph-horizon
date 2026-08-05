/*
 * Graph Orizon CLI startup
 * Single responsibility: start the interactive text-chat console with the
 * validated app/runtime configuration. It depends on app engine loading and CLI
 * runtime modules, and does not initialize tools, workspaces, or reasoning.
 */

use color_eyre::eyre::{Result, eyre};

use crate::app::{args, engine};

use super::console::terminal_user_interface;
use super::plugins::attachments::FileAuthority;
use super::runtime::{ClientConfig, generation_stream, local, pruning_threshold};

pub(crate) async fn run(model_path: Option<String>, files: FileAuthority) -> Result<()> {
    let mut client_config = ClientConfig::from_env();
    // A prior conversation is restored later, in-session, via /import, so startup
    // always begins from a fresh history; the system prompt is read out here.
    let system = client_config.system.clone();
    let max_tokens = client_config.max_tokens;

    // --provider local selects the in-process engine; anything else keeps the HTTP
    // path. The local Engine is built BEFORE the terminal is initialized so a
    // missing/invalid model fails cleanly at startup, and BEFORE the pruning threshold.
    if args::value("--provider").as_deref() == Some("local") {
        let model = model_path.clone().ok_or_else(|| {
            eyre!("--provider local requires --model to point to a mistral3 GGUF")
        })?;
        let chat = engine::load_chat_engine(&model, client_config.context_limit)?;
        // Fill the context limit from the model (override still wins), without any
        // network call, so pruning is sized correctly.
        client_config.apply_local_context_limit(chat.context_limit());
        let threshold = pruning_threshold(client_config.context_limit, max_tokens);
        let context_limit = client_config.context_limit;
        let generate = local::provider(chat, max_tokens);
        let mut terminal = ratatui::init();
        let result = terminal_user_interface(
            &mut terminal,
            system,
            threshold,
            context_limit,
            &files,
            generate,
        )
        .await;
        ratatui::restore();
        result
    } else {
        // HTTP path (unchanged): learn the context window from the provider when
        // there is no --context-tokens override.
        client_config.resolve_context_limit().await;
        let threshold = pruning_threshold(client_config.context_limit, max_tokens);
        let context_limit = client_config.context_limit;
        let mut terminal = ratatui::init();
        let result = terminal_user_interface(
            &mut terminal,
            system,
            threshold,
            context_limit,
            &files,
            move |msgs| generation_stream(msgs, client_config.clone()),
        )
        .await;
        ratatui::restore();
        result
    }
}
