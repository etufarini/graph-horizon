/*
 * Graph Horizon CLI Modules - Runtime - Config
 * Single responsibility: hold CLI generation settings and preserve an explicit
 * context override when model metadata supplies the default.
 */

use crate::app::args;

pub(crate) struct RuntimeConfig {
    // Optional fixed system prompt, prepended to every request by the session.
    pub system: Option<String>,
    // Explicit context-window size when supplied; otherwise the model resolves it.
    pub context_limit: Option<usize>,
    pub max_tokens: usize,
}

impl RuntimeConfig {
    pub(crate) fn from_args() -> Self {
        Self {
            system: args::value("--system-prompt"),
            context_limit: crate::app::engine::config::context_tokens_from_args(),
            max_tokens: crate::app::engine::config::max_tokens_from_args(2048),
        }
    }

    // Model discovery is immutable after engine loading; an explicit override
    // remains authoritative and is never replaced by the model result.
    pub(crate) fn apply_model_context_limit(&mut self, model_context: u32) {
        if self.context_limit.is_none() {
            self.context_limit = Some(model_context as usize);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(context_limit: Option<usize>) -> RuntimeConfig {
        RuntimeConfig {
            system: None,
            context_limit,
            max_tokens: 128,
        }
    }

    #[test]
    fn local_context_uses_engine_result_without_overriding_explicit_value() {
        let mut implicit = config(None);
        implicit.apply_model_context_limit(32_768);
        assert_eq!(implicit.context_limit, Some(32_768));

        let mut explicit = config(Some(65_536));
        explicit.apply_model_context_limit(32_768);
        assert_eq!(explicit.context_limit, Some(65_536));
    }
}
