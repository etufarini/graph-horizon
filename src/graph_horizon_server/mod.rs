/*
 * Graph Horizon headless server domain
 * Exposes the OpenAI-compatible HTTP server and the reusable chat pipeline. It
 * has no dependency on the CLI domain and contains no web asset handling.
 */

mod api;
pub(crate) mod config;
pub(crate) mod handler;
pub(crate) mod server;
pub(crate) mod startup;
pub(crate) mod state;
mod stream;

pub(crate) use config::ServerConfig;
pub(crate) use handler::ResponseBody;
pub(crate) use state::ServerState;
