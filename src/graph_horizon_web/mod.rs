/*
 * Graph Horizon local Web UI
 * Owns the bundled browser assets, private browser transport, local listener,
 * and engine startup. It exposes no standalone or compatibility API surface.
 */

mod assets;
mod chat;
mod config;
mod router;
mod runtime;
mod server;
pub(crate) mod startup;
