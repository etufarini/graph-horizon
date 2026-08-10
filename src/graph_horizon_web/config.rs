/*
 * Graph Horizon web config
 * Holds only Web bind and static-asset settings. Generation capacity is
 * resolved from the loaded engine during startup, so Web argument parsing does
 * not own or validate a generation cap.
 */

use std::path::PathBuf;

use crate::app::args;

#[derive(Clone)]
pub(super) struct WebConfig {
    pub host: String,
    pub port: String,
    pub asset_root: PathBuf,
}

impl WebConfig {
    pub(super) fn from_args() -> WebConfig {
        WebConfig {
            host: args::value("--host").unwrap_or_else(|| "127.0.0.1".to_string()),
            // Keep the raw port string so invalid values fail at bind time.
            port: args::value("--port").unwrap_or_else(|| "8080".to_string()),
            asset_root: PathBuf::from("web/frontend/dist"),
        }
    }
}
