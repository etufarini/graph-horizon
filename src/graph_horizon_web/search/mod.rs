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

pub(super) const MAX_CONTEXT_CHARACTERS: usize = 12_288;
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
        let _permit = self.admission.try_acquire().map_err(|_| Error::Busy)?;
        let mut results = match client::fetch(query, date).await {
            Ok(html) => parser::parse(&html),
            Err(()) => Vec::new(),
        };
        if results.is_empty()
            && let Ok(Some(fallback)) = client::fetch_fallback(query, date).await
        {
            results = match fallback {
                client::Fallback::News(xml) => parser::parse_news(&xml),
                client::Fallback::Code(html) => parser::parse_code(&html),
            };
        }
        context::frame(&results, date).ok_or(Error::Unavailable)
    }
}
