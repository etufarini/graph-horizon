/*
 * Graph Horizon CLI Modules - Runtime - Config
 * Single responsibility: resolve provider configuration and a mandatory
 * positive context limit while preserving explicit-over-discovered precedence.
 */

use super::client;
use crate::app::args;

#[derive(Clone)]
pub(crate) struct ClientConfig {
    pub base_url: String,
    // Optional fixed system prompt, prepended to every request by the session.
    pub system: Option<String>,
    // Explicit context-window size when supplied; otherwise provider startup
    // resolves it before the terminal is initialized.
    pub context_limit: Option<usize>,
    pub max_tokens: usize,
}

impl ClientConfig {
    pub(crate) fn from_args() -> Self {
        Self {
            base_url: args::value("--base-url")
                .unwrap_or_else(|| client::DEFAULT_BASE_URL.to_string()),
            system: args::value("--system-prompt"),
            context_limit: crate::app::engine::config::context_tokens_from_args(),
            max_tokens: crate::app::engine::config::max_tokens_from_args(2048),
        }
    }

    // The explicit override always wins and performs no network request. Remote
    // discovery fails closed so callers cannot enter the terminal unbounded.
    pub(crate) async fn resolve_context_limit(&mut self) -> Option<usize> {
        if let Some(limit) = self.context_limit {
            return Some(limit);
        }
        let limit = client::fetch_context_limit(&self.base_url).await?;
        self.context_limit = Some(limit);
        Some(limit)
    }

    // Local discovery is immutable after engine loading; an explicit override
    // remains authoritative and is never replaced by the model result.
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
    fn local_context_uses_engine_result_without_overriding_explicit_value() {
        let mut implicit = config(None, client::DEFAULT_BASE_URL.into());
        implicit.apply_local_context_limit(32_768);
        assert_eq!(implicit.context_limit, Some(32_768));

        let mut explicit = config(Some(65_536), client::DEFAULT_BASE_URL.into());
        explicit.apply_local_context_limit(32_768);
        assert_eq!(explicit.context_limit, Some(65_536));
    }

    #[tokio::test]
    async fn explicit_context_skips_remote_discovery() {
        let mut config = config(Some(65_536), "http://127.0.0.1:1/v1".into());
        assert_eq!(config.resolve_context_limit().await, Some(65_536));
    }

    #[tokio::test]
    async fn context_limit_policy_reports_remote_failure() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        drop(listener);

        let mut config = config(None, format!("http://{address}/v1"));
        assert_eq!(config.resolve_context_limit().await, None);
        assert_eq!(config.context_limit, None);
    }
}
