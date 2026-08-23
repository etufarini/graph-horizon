/*
 * Graph Horizon Web search pipeline
 * Turns one admitted browser query into a bounded, explicitly untrusted model
 * context. Provider transport, HTML parsing, and prompt framing stay isolated
 * so the chat pipeline never handles remote markup directly.
 */

use std::sync::Arc;

use color_eyre::eyre::{Result, eyre};
use tokio::sync::Semaphore;

mod client;
mod context;
mod parser;

pub(super) const MAX_CONTEXT_CHARACTERS: usize = 6_144;
const MAX_CONCURRENT: usize = 1;

#[derive(Clone)]
pub(in crate::graph_horizon_web) struct State {
    client: client::Client,
    admission: Arc<Semaphore>,
}

#[derive(Debug, PartialEq, Eq)]
pub(super) enum Error {
    Busy,
    Unavailable,
}

impl State {
    pub(in crate::graph_horizon_web) fn new() -> Result<State> {
        Ok(State {
            client: client::Client::new().map_err(|_| eyre!("failed to initialize Web search"))?,
            admission: Arc::new(Semaphore::new(MAX_CONCURRENT)),
        })
    }

    pub(super) async fn context(&self, query: &str) -> Result<String, Error> {
        // One best-effort request at a time avoids amplifying upstream rate limits.
        let _permit = Arc::clone(&self.admission)
            .try_acquire_owned()
            .map_err(|_| Error::Busy)?;
        let html = self
            .client
            .fetch(query)
            .await
            .map_err(|_| Error::Unavailable)?;
        let results = parser::parse(&html);
        context::frame(&results).ok_or(Error::Unavailable)
    }
}
