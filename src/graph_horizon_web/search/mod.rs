/*
 * Graph Horizon Web search pipeline
 * Turns one admitted browser query into a bounded, explicitly untrusted model
 * context. Provider transport, HTML parsing, and prompt framing stay isolated
 * so the chat pipeline never handles remote markup directly.
 */

use std::sync::Arc;

use tokio::sync::Semaphore;

mod client;
mod context;
mod parser;

pub(super) const MAX_CONTEXT_CHARACTERS: usize = 6_144;
const MAX_CONCURRENT: usize = 1;

#[derive(Clone)]
pub(in crate::graph_horizon_web) struct State {
    admission: Arc<Semaphore>,
}

#[derive(Debug, PartialEq, Eq)]
pub(super) enum Error {
    Busy,
    Unavailable,
}

impl State {
    pub(in crate::graph_horizon_web) fn new() -> State {
        State {
            admission: Arc::new(Semaphore::new(MAX_CONCURRENT)),
        }
    }

    pub(super) async fn context(&self, query: &str, date: &str) -> Result<context::Framed, Error> {
        // One best-effort request at a time avoids amplifying upstream rate limits.
        let _permit = Arc::clone(&self.admission)
            .try_acquire_owned()
            .map_err(|_| Error::Busy)?;
        let html = client::fetch(query).await.map_err(|_| Error::Unavailable)?;
        let results = parser::parse(&html);
        context::frame(&results, date).ok_or(Error::Unavailable)
    }
}
