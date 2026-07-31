/*
 * GH Zero web config
 * Single responsibility: hold bind/static settings and chat generation cap for
 * web mode. It depends on parsed app args and does not expose tools, workspace,
 * or reasoning configuration.
 */

use std::path::PathBuf;

use crate::app::args;

#[derive(Clone)]
pub(super) struct WebConfig {
    pub host: String,
    pub port: String,
    pub asset_root: PathBuf,
    pub max_tokens: usize,
}

impl WebConfig {
    pub(super) fn from_args() -> WebConfig {
        WebConfig {
            host: args::value("--host").unwrap_or_else(|| "127.0.0.1".to_string()),
            // Keep the raw port string so invalid values fail at bind time.
            port: args::value("--port").unwrap_or_else(|| "8080".to_string()),
            asset_root: PathBuf::from("web/frontend/dist"),
            max_tokens: args::value("--max-tokens")
                .and_then(|v| v.trim().parse().ok())
                .unwrap_or(1024),
        }
    }
}
