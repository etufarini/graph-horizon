/*
 * Graph Horizon CLI Modules - Runtime - Config
 * Single responsibility: hold chat-only provider configuration shared by CLI
 * local and HTTP generation. It depends on app args and the HTTP client helper,
 * and carries no tool mode, workspace, profile, or reasoning state.
 */

use super::client;
use crate::app::args;

#[derive(Clone)]
pub(crate) struct ClientConfig {
    pub base_url: String,
    // Optional fixed system prompt, prepended to every request by the session.
    pub system: Option<String>,
    // Optional context-window size in tokens. Sourced from the `--context-tokens`
    // flag when set, otherwise queried from the HTTP provider via
    // `resolve_context_limit`. Combined with `max_tokens` to derive pruning.
    pub context_limit: Option<usize>,
    pub max_tokens: usize,
}

impl ClientConfig {
    pub(crate) fn from_env() -> Self {
        Self {
            base_url: args::value("--base-url")
                .unwrap_or_else(|| client::DEFAULT_BASE_URL.to_string()),
            system: args::value("--system-prompt"),
            // A non-numeric or empty value is treated as "no limit" rather than a
            // fatal error: the program must keep running with pruning disabled.
            context_limit: args::value("--context-tokens")
                .and_then(|v| v.trim().parse::<usize>().ok()),
            max_tokens: crate::app::engine::config::max_tokens_from_args(),
        }
    }

    // Fills `context_limit` from the HTTP provider when the `--context-tokens` override
    // left it unset, so the pruning threshold is sized without manual configuration.
    // The override always wins: a valid `--context-tokens` short-circuits this
    // and the server is never queried. On any fetch failure `context_limit`
    // stays None — pruning disabled — and startup proceeds.
    pub(crate) async fn resolve_context_limit(&mut self) {
        if self.context_limit.is_none() {
            self.context_limit = client::fetch_context_limit(&self.base_url).await;
        }
    }

    // Local-provider counterpart of `resolve_context_limit`: fills
    // `context_limit` from the engine-resolved limit, with no network call.
    // The override always wins — a set `--context-tokens` leaves
    // `context_limit` untouched — so local pruning follows the loaded engine.
    pub(crate) fn apply_local_context_limit(&mut self, model_context: u32) {
        if self.context_limit.is_none() {
            self.context_limit = Some(model_context as usize);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(context_limit: Option<usize>, base_url: String) -> ClientConfig {
        ClientConfig {
            base_url,
            system: None,
            context_limit,
            max_tokens: 128,
        }
    }

    #[test]
    fn context_limit_policy_uses_engine_result_without_overriding_explicit_value() {
        let mut implicit = config(None, client::DEFAULT_BASE_URL.into());
        implicit.apply_local_context_limit(32_768);
        assert_eq!(implicit.context_limit, Some(32_768));

        let mut explicit = config(Some(65_536), client::DEFAULT_BASE_URL.into());
        explicit.apply_local_context_limit(32_768);
        assert_eq!(explicit.context_limit, Some(65_536));
    }

    #[tokio::test]
    async fn context_limit_policy_leaves_remote_failure_unresolved() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        drop(listener);

        let mut config = config(None, format!("http://{address}/v1"));
        config.resolve_context_limit().await;
        assert_eq!(config.context_limit, None);
    }
}
