/*
 * Graph Horizon headless server state
 * Owns the chat engine and admission/serialization gates used by server-mode
 * request handlers. It contains no routing, I/O loop, or wire-format logic.
 */

use std::sync::Arc;

use graph_horizon_engine::Engine;
use tokio::sync::{Mutex, Semaphore};

use super::ServerConfig;

const MAX_INFLIGHT: usize = 8;

#[derive(Clone)]
pub(crate) struct ServerState {
    pub(crate) chat: Arc<Engine>,
    pub(crate) serialize: Arc<Mutex<()>>,
    pub(crate) admission: Arc<Semaphore>,
    pub(crate) config: ServerConfig,
}

impl ServerState {
    pub(crate) fn new(chat: Arc<Engine>, config: ServerConfig) -> ServerState {
        ServerState {
            chat,
            // One generation at a time protects the in-process engine state.
            serialize: Arc::new(Mutex::new(())),
            // Admission is fixed so excess load fails immediately instead of
            // accumulating unbounded tasks behind the serialization lock.
            admission: Arc::new(Semaphore::new(MAX_INFLIGHT)),
            config,
        }
    }
}
