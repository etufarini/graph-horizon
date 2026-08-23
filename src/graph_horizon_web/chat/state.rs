/*
 * Graph Horizon Web chat state
 * Owns the local engine, optional search client, and bounded concurrency gates
 * used by bundled Web chat. Routing, wire parsing, and streaming stay elsewhere.
 */

use std::sync::Arc;

use graph_horizon_engine::Engine;
use tokio::sync::{Mutex, Semaphore};

use super::super::search;

const MAX_INFLIGHT: usize = 8;

#[derive(Clone)]
pub(in crate::graph_horizon_web) struct State {
    pub(in crate::graph_horizon_web) engine: Arc<Engine>,
    pub(super) serialize: Arc<Mutex<()>>,
    pub(super) admission: Arc<Semaphore>,
    pub(super) search: search::State,
}

impl State {
    pub(in crate::graph_horizon_web) fn new(engine: Arc<Engine>, search: search::State) -> State {
        State {
            engine,
            // One generation at a time protects the in-process engine state.
            serialize: Arc::new(Mutex::new(())),
            // Excess browser work fails immediately instead of growing a queue.
            admission: Arc::new(Semaphore::new(MAX_INFLIGHT)),
            search,
        }
    }
}
