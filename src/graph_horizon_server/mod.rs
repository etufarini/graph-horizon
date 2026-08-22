/*
 * Graph Horizon legacy chat transport
 * Temporarily exposes the chat pipeline still used by the Web UI while the
 * public server contract is removed. It has no standalone startup or listener.
 */

mod api;
pub(crate) mod config;
pub(crate) mod handler;
pub(crate) mod state;
mod stream;

pub(crate) use config::ServerConfig;
pub(crate) use handler::ResponseBody;
pub(crate) use state::ServerState;
