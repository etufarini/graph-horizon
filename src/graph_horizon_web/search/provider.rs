/*
 * Configured JSON search adapter
 * Serializes the narrow Graph Horizon provider contract and converts its
 * strictly validated response into bounded, deduplicated evidence records.
 * It performs no fallback, page scraping, or result-URL fetching.
 */

use std::collections::HashSet;

use serde::{Deserialize, Serialize};
use url::Url;

use super::config::Config;
use super::request::{Category, MAX_TIMESTAMP_BOUND_MS, Published, Request};
use super::transport;

pub(super) const MAX_RESULTS: usize = 5;
const MAX_TITLE_CHARACTERS: usize = 160;
const MAX_EXCERPT_CHARACTERS: usize = 320;
const MAX_PUBLISHER_CHARACTERS: usize = 100;
const MAX_URL_CHARACTERS: usize = 2_048;

#[derive(Debug, PartialEq, Eq)]
pub(super) enum Error {
    RateLimited,
    Timeout,
    Invalid,
    Unavailable,
}

#[derive(Debug, PartialEq, Eq)]
pub(super) struct SearchResult {
    pub(super) title: String,
    pub(super) url: String,
    pub(super) excerpt: String,
    pub(super) publisher: Option<String>,
    pub(super) published_at_ms: Option<u64>,
}

#[derive(Serialize)]
struct ProviderRequest<'a> {
    query: &'a str,
    category: Category,
    language: &'a str,
    reference_date: &'a str,
    published: Option<Published>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ProviderResponse {
    results: Vec<ProviderResult>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ProviderResult {
    title: String,
    url: String,
    excerpt: String,
    publisher: Option<String>,
    published_at_ms: Option<u64>,
}

pub(super) async fn search(config: &Config, request: &Request) -> Result<Vec<SearchResult>, Error> {
    let body = serde_json::to_string(&ProviderRequest {
        query: request.terms(),
        category: request.category(),
        language: request.language(),
        reference_date: request.reference_date(),
        published: request.published(),
    })
    .map_err(|_| Error::Invalid)?;
    let body = transport::fetch(config, body)
        .await
        .map_err(|error| match error {
            transport::Error::Http(429) => Error::RateLimited,
            transport::Error::Timeout => Error::Timeout,
            transport::Error::Invalid => Error::Invalid,
            transport::Error::Unavailable
            | transport::Error::TooLarge
            | transport::Error::Http(_) => Error::Unavailable,
        })?;
    parse(&body, request.published())
}

fn parse(body: &str, published: Option<Published>) -> Result<Vec<SearchResult>, Error> {
    let response: ProviderResponse = serde_json::from_str(body).map_err(|_| Error::Invalid)?;
    let mut results = Vec::new();
    let mut urls = HashSet::new();
    for result in response.results {
        let title = normalized(&result.title, MAX_TITLE_CHARACTERS);
        let excerpt = normalized(&result.excerpt, MAX_EXCERPT_CHARACTERS);
        let Some(url) = result_url(&result.url) else {
            continue;
        };
        if title.is_empty()
            || excerpt.is_empty()
            || result
                .published_at_ms
                .is_some_and(|time| time >= MAX_TIMESTAMP_BOUND_MS)
            || published.is_some_and(|range| {
                result
                    .published_at_ms
                    .is_none_or(|time| !range.contains(time))
            })
            || !urls.insert(url.clone())
        {
            continue;
        }
        results.push(SearchResult {
            title,
            url,
            excerpt,
            publisher: result
                .publisher
                .map(|value| normalized(&value, MAX_PUBLISHER_CHARACTERS))
                .filter(|value| !value.is_empty()),
            published_at_ms: result.published_at_ms,
        });
        if results.len() == MAX_RESULTS {
            break;
        }
    }
    Ok(results)
}

fn result_url(value: &str) -> Option<String> {
    let mut url = Url::parse(value).ok()?;
    if !matches!(url.scheme(), "http" | "https")
        || url.host_str().is_none()
        || !url.username().is_empty()
        || url.password().is_some()
    {
        return None;
    }
    url.set_fragment(None);
    let value = url.to_string();
    (value.chars().count() <= MAX_URL_CHARACTERS).then_some(value)
}

fn normalized(value: &str, limit: usize) -> String {
    let text = value.split_whitespace().collect::<Vec<_>>().join(" ");
    if text.chars().count() <= limit {
        text
    } else {
        text.chars().take(limit - 1).chain(['…']).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(published: bool) -> Request {
        let range = if published {
            r#"{"from_ms":100,"to_ms":200}"#
        } else {
            "null"
        };
        serde_json::from_str::<Request>(&format!(r#"{{"terms":"Rust","category":"web","language":"it-IT","reference_date":"2026-08-24","published":{range}}}"#)).unwrap().validated().unwrap()
    }

    #[test]
    fn response_is_bounded_deduplicated_and_date_proven() {
        let body = r#"{"results":[
          {"title":" Current   Rust ","url":"https://example.com/a#part","excerpt":"Release 1.97","publisher":" Rust ","published_at_ms":150},
          {"title":"Duplicate","url":"https://example.com/a","excerpt":"duplicate","publisher":null,"published_at_ms":150},
          {"title":"Undated","url":"https://example.com/b","excerpt":"unknown","publisher":null,"published_at_ms":null},
          {"title":"Unsafe","url":"file:///tmp/x","excerpt":"bad","publisher":null,"published_at_ms":150}
          ,{"title":"Impossible date","url":"https://example.com/c","excerpt":"bad","publisher":null,"published_at_ms":253402300800000}
        ]}"#;
        let results = parse(body, request(true).published()).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].title, "Current Rust");
        assert_eq!(results[0].url, "https://example.com/a");
    }

    #[test]
    fn unknown_response_fields_are_rejected() {
        assert_eq!(
            parse(r#"{"results":[],"extra":true}"#, request(false).published()),
            Err(Error::Invalid)
        );
    }
}
