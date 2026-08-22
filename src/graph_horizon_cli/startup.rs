/*
 * Graph Horizon CLI startup
 * Single responsibility: resolve and validate CLI provider capacity before
 * initializing the terminal, then start the interactive chat session.
 */

use color_eyre::eyre::{Result, eyre};

use crate::app::{args, engine};

use super::console::terminal_user_interface;
use super::plugins::attachments::FileAuthority;
use super::runtime::RuntimeInfo;
use super::runtime::{ClientConfig, ContextBudget, generation_stream, local};

const CONTEXT_UNAVAILABLE: &str = "context limit unavailable; specify --context-tokens";
const NO_PROMPT_SPACE: &str = "max_tokens leaves no room for the prompt";

fn context_budget(context_limit: Option<usize>, max_tokens: usize) -> Result<ContextBudget> {
    let context_limit = context_limit.ok_or_else(|| eyre!(CONTEXT_UNAVAILABLE))?;
    ContextBudget::new(context_limit, max_tokens).ok_or_else(|| eyre!(NO_PROMPT_SPACE))
}

async fn http_context_budget(config: &mut ClientConfig) -> Result<ContextBudget> {
    config
        .resolve_context_limit()
        .await
        .ok_or_else(|| eyre!(CONTEXT_UNAVAILABLE))?;
    context_budget(config.context_limit, config.max_tokens)
}

pub(crate) async fn run(model_path: Option<String>, files: FileAuthority) -> Result<()> {
    let mut client_config = ClientConfig::from_args();
    // A prior conversation is restored later, in-session, via /import, so startup
    // always begins from a fresh history; the system prompt is read out here.
    let system = client_config.system.clone();
    let max_tokens = client_config.max_tokens;

    // --provider local selects the in-process engine; anything else keeps the HTTP
    // path. Engine and capacity resolution both precede terminal initialization.
    if args::value("--provider").as_deref() == Some("local") {
        let model = model_path.clone().ok_or_else(|| {
            eyre!("--provider local requires --model to point to a mistral3 GGUF")
        })?;
        let chat = engine::load_chat_engine(&model, client_config.context_limit)?;
        // The explicit override still wins over the engine-resolved limit.
        client_config.apply_local_context_limit(chat.context_limit());
        let budget = context_budget(client_config.context_limit, max_tokens)?;
        let runtime = RuntimeInfo {
            model_name: chat.model_name().unwrap_or("Local model").to_string(),
            backend: chat.backend_name(),
            memory: chat.memory(),
            placement: chat.placement(),
        };
        let generate = local::provider(chat, max_tokens);
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
    } else {
        // HTTP discovery is mandatory unless an explicit override is present.
        let budget = http_context_budget(&mut client_config).await?;
        let mut terminal = ratatui::init();
        let result =
            terminal_user_interface(&mut terminal, system, budget, None, &files, move |msgs| {
                generation_stream(msgs, client_config.clone())
            })
            .await;
        ratatui::restore();
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn remote_context_failure_is_startup_error() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        drop(listener);
        let mut config = ClientConfig {
            base_url: format!("http://{address}/v1"),
            system: None,
            context_limit: None,
            max_tokens: 128,
        };

        assert_eq!(
            http_context_budget(&mut config)
                .await
                .unwrap_err()
                .to_string(),
            CONTEXT_UNAVAILABLE
        );
    }
}
