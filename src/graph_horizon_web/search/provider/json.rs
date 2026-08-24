/*
 * Advanced JSON provider adapter
 * Preserves the strict configured provider contract and converts one response
 * into the same bounded, date-proven result representation as public search.
 */

use std::collections::HashSet;

use serde::{Deserialize, Serialize};

use super::{
    Error, MAX_EXCERPT_CHARACTERS, MAX_PUBLISHER_CHARACTERS, MAX_RESULTS, MAX_TITLE_CHARACTERS,
    SearchResult, normalized, push, result_url, transport_error,
};
use crate::graph_horizon_web::search::config::Config;
use crate::graph_horizon_web::search::request::{Category, Published, Request};
use crate::graph_horizon_web::search::transport::{self, Accept, Body};

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
    let Config::Json { endpoint, bearer } = config else {
        return Err(Error::Invalid);
    };
    let body = serde_json::to_string(&ProviderRequest {
        query: request.terms(),
        category: request.category(),
        language: request.language(),
        reference_date: request.reference_date(),
        published: request.published(),
    })
    .map_err(|_| Error::Invalid)?;
    let body = transport::fetch(transport::Request {
        url: endpoint.to_string(),
        accept: Accept::Json,
        body: Body::Json(body),
        bearer: bearer.clone(),
    })
    .await
    .map_err(transport_error)?;
    parse(&body, request)
}

fn parse(body: &str, request: &Request) -> Result<Vec<SearchResult>, Error> {
    let response: ProviderResponse = serde_json::from_str(body).map_err(|_| Error::Invalid)?;
    let mut results = Vec::new();
    let mut urls = HashSet::new();
    for result in response.results {
        let title = normalized(std::iter::once(result.title.as_str()), MAX_TITLE_CHARACTERS);
        let excerpt = normalized(
            std::iter::once(result.excerpt.as_str()),
            MAX_EXCERPT_CHARACTERS,
        );
        let url = result_url(&result.url, "https://invalid.example/", false);
        let publisher = result
            .publisher
            .map(|value| normalized(std::iter::once(value.as_str()), MAX_PUBLISHER_CHARACTERS))
            .filter(|value| !value.is_empty());
        push(
            &mut results,
            &mut urls,
            url.map(|url| SearchResult {
                title,
                url,
                excerpt,
                publisher,
                published_at_ms: result.published_at_ms,
            }),
            request.published(),
        );
        if results.len() == MAX_RESULTS {
            break;
        }
    }
    Ok(results)
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
        serde_json::from_str::<Request>(&format!(r#"{{"terms":"Rust","category":"web","language":"it-IT","reference_date":"2026-08-25","published":{range}}}"#)).unwrap().validated().unwrap()
    }

    #[test]
    fn response_is_bounded_deduplicated_and_date_proven() {
        let body = r#"{"results":[
          {"title":" Current   Rust ","url":"https://example.com/a#part","excerpt":"Release 1.97","publisher":" Rust ","published_at_ms":150},
          {"title":"Duplicate","url":"https://example.com/a","excerpt":"duplicate","publisher":null,"published_at_ms":150},
          {"title":"Undated","url":"https://example.com/b","excerpt":"unknown","publisher":null,"published_at_ms":null},
          {"title":"Unsafe","url":"file:///tmp/x","excerpt":"bad","publisher":null,"published_at_ms":150}
        ]}"#;
        let results = parse(body, &request(true)).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].title, "Current Rust");
        assert_eq!(results[0].url, "https://example.com/a");
    }

    #[test]
    fn unknown_response_fields_are_rejected() {
        assert_eq!(
            parse(r#"{"results":[],"extra":true}"#, &request(false)),
            Err(Error::Invalid)
        );
    }
}
