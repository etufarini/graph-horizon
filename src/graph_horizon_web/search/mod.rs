/*
 * Configured Web search pipeline
 * Owns capability reporting, single-request admission, typed provider outcomes,
 * and conversion of one validated query into compact evidence plus provenance.
 * It never changes category, scrapes public pages, or fetches result URLs.
 */

use std::sync::Arc;

use serde::Serialize;
use tokio::sync::Semaphore;

mod config;
mod context;
mod provider;
mod request;
mod transport;

pub(in crate::graph_horizon_web) use config::Config;
pub(in crate::graph_horizon_web) use context::Report;
pub(in crate::graph_horizon_web) use request::Request;

pub(super) const MAX_CONTEXT_CHARACTERS: usize = 2_800;
const MAX_CONCURRENT: usize = 1;

#[derive(Clone)]
pub(in crate::graph_horizon_web) struct State {
    config: Option<Arc<Config>>,
    admission: Arc<Semaphore>,
}

#[derive(Debug, PartialEq, Eq)]
pub(super) enum Error {
    Busy,
    NotConfigured,
    NoResults,
    RateLimited,
    Timeout,
    Invalid,
    Unavailable,
}

#[derive(Serialize)]
pub(in crate::graph_horizon_web) struct Capability {
    provider: Option<String>,
    max_query_characters: usize,
    max_context_characters: usize,
}

impl State {
    pub(in crate::graph_horizon_web) fn new(config: Option<Config>) -> State {
        State {
            config: config.map(Arc::new),
            admission: Arc::new(Semaphore::new(MAX_CONCURRENT)),
        }
    }

    pub(in crate::graph_horizon_web) fn capability(&self) -> Capability {
        Capability {
            provider: self
                .config
                .as_ref()
                .map(|config| config.provider().to_string()),
            max_query_characters: request::MAX_QUERY_CHARACTERS,
            max_context_characters: MAX_CONTEXT_CHARACTERS,
        }
    }

    pub(super) async fn context(&self, request: &Request) -> Result<context::Framed, Error> {
        let config = self.config.as_ref().ok_or(Error::NotConfigured)?;
        let _permit = self.admission.try_acquire().map_err(|_| Error::Busy)?;
        let results = provider::search(config, request)
            .await
            .map_err(|error| match error {
                provider::Error::RateLimited => Error::RateLimited,
                provider::Error::Timeout => Error::Timeout,
                provider::Error::Invalid => Error::Invalid,
                provider::Error::Unavailable => Error::Unavailable,
            })?;
        if results.is_empty() {
            return Err(Error::NoResults);
        }
        context::frame(&results, request, config.provider()).ok_or(Error::NoResults)
    }
}
