/*
 * Graph Horizon headless server config
 * Single responsibility: hold bind address and chat generation cap for server
 * mode. It depends on parsed app args and does not expose tools, workspace, or
 * reasoning configuration.
 */

use crate::app::args;

#[derive(Clone)]
pub(crate) struct ServerConfig {
    pub host: String,
    pub port: u16,
    pub max_tokens: usize,
}

impl ServerConfig {
    pub(crate) fn chat(max_tokens: usize, _think: bool) -> ServerConfig {
        ServerConfig {
            host: "127.0.0.1".to_string(),
            port: 0,
            max_tokens,
        }
    }

    pub(crate) fn from_args() -> ServerConfig {
        ServerConfig {
            host: args::value("--host").unwrap_or_else(|| "127.0.0.1".to_string()),
            // Existing tolerant behavior: invalid numbers fall back to defaults.
            port: args::value("--port")
                .and_then(|v| v.trim().parse().ok())
                .unwrap_or(8080),
            max_tokens: args::value("--max-tokens")
                .and_then(|v| v.trim().parse().ok())
                .unwrap_or(1024),
        }
    }
}
