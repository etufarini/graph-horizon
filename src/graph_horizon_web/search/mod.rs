/*
 * Graph Horizon Web search pipeline
 * Turns one admitted browser query into a bounded, explicitly untrusted model
 * context. Provider transport, HTML parsing, and prompt framing stay isolated
 * so the chat pipeline never handles remote markup directly.
 */

use std::sync::Arc;

use tokio::sync::Semaphore;

mod context;
mod provider;
mod request;
mod transport;

pub(in crate::graph_horizon_web) use request::Request;

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

    pub(super) async fn context(&self, request: &Request) -> Result<context::Framed, Error> {
        // One best-effort request at a time avoids amplifying upstream rate limits.
        let _permit = self.admission.try_acquire().map_err(|_| Error::Busy)?;
        let results = match request.category() {
            request::Category::Web => web_results(request).await,
            request::Category::News => news_results(request).await,
        };
        context::frame(&results, request).ok_or(Error::Unavailable)
    }
}

async fn web_results(request: &Request) -> Vec<provider::Result> {
    let results = provider::duckduckgo::search(request)
        .await
        .unwrap_or_default();
    if !results.is_empty() || request.published().is_some() {
        return results;
    }
    provider::brave::search(request).await.unwrap_or_default()
}

async fn news_results(request: &Request) -> Vec<provider::Result> {
    let results = provider::google_news::search(request)
        .await
        .unwrap_or_default();
    if !results.is_empty() || request.published().is_some() {
        return results;
    }
    let results = provider::duckduckgo::search(request)
        .await
        .unwrap_or_default();
    if !results.is_empty() {
        return results;
    }
    provider::brave::search(request).await.unwrap_or_default()
}
