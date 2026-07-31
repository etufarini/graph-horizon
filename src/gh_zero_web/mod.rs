/*
 * GH Zero local web backend
 * Single responsibility: expose static browser assets wrapped around the
 * headless chat server. It depends on web config/assets/routing and does not
 * provide tools, confirmations, workspaces, or reasoning endpoints.
 */

mod assets;
mod config;
mod router;
mod server;
pub(crate) mod startup;
