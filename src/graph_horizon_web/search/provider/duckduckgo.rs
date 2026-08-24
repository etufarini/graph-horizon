/*
 * DuckDuckGo Lite Web adapter
 * Builds one fixed keyless form request and extracts bounded organic results,
 * including provider timestamps used to prove explicit publication intervals.
 */

use std::collections::HashSet;

use scraper::{ElementRef, Html, Selector};
use url::Url;

use super::{
    Error, MAX_EXCERPT_CHARACTERS, MAX_PUBLISHER_CHARACTERS, MAX_RESULTS, MAX_TITLE_CHARACTERS,
    SearchResult, iso_milliseconds, language_region, normalized, push, result_url, transport_error,
};
use crate::graph_horizon_web::search::request::Request;
use crate::graph_horizon_web::search::transport::{self, Accept, Body};

const ENDPOINT: &str = "https://lite.duckduckgo.com/lite/";

pub(super) async fn search(request: &Request) -> Result<Vec<SearchResult>, Error> {
    let html = transport::fetch(provider_request(request))
        .await
        .map_err(transport_error)?;
    if html.contains("challenge-form") || html.contains("anomaly-modal__title") {
        return Err(Error::RateLimited);
    }
    Ok(parse(&html, request))
}

fn provider_request(request: &Request) -> transport::Request {
    let (language, region) = language_region(request.language());
    let mut fields = vec![
        ("q".into(), request.terms().into()),
        (
            "kl".into(),
            format!("{}-{language}", region.to_ascii_lowercase()),
        ),
        ("kp".into(), "1".into()),
    ];
    if let Some(range) = request.published() {
        fields.push(("df".into(), range.duckduckgo_filter()));
    }
    transport::Request {
        url: ENDPOINT.into(),
        accept: Accept::Html,
        body: Body::Form(fields),
        bearer: None,
    }
}

fn parse(html: &str, request: &Request) -> Vec<SearchResult> {
    let document = Html::parse_document(html);
    let links = Selector::parse("a.result-link").expect("fixed result selector is valid");
    let snippets = Selector::parse(".result-snippet").expect("fixed snippet selector is valid");
    let timestamps = Selector::parse(".timestamp").expect("fixed timestamp selector is valid");
    let mut results = Vec::new();
    let mut urls = HashSet::new();

    for link in document.select(&links) {
        let Some(row) = link
            .ancestors()
            .filter_map(ElementRef::wrap)
            .find(|element| element.value().name() == "tr")
        else {
            continue;
        };
        if has_class(row.value().attr("class"), "result-sponsored") {
            continue;
        }
        let Some(url) = link
            .value()
            .attr("href")
            .and_then(|href| result_url(href, ENDPOINT, true))
        else {
            continue;
        };
        let title = normalized(link.text(), MAX_TITLE_CHARACTERS);
        let mut excerpt = String::new();
        let mut published_at_ms = None;
        // Only element rows count: provider whitespace and comments must not
        // hide the snippet or timestamp before the next organic result.
        for element in row.next_siblings().filter_map(ElementRef::wrap).take(3) {
            if element.select(&links).next().is_some() {
                break;
            }
            if excerpt.is_empty()
                && let Some(snippet) = element.select(&snippets).next()
            {
                excerpt = normalized(snippet.text(), MAX_EXCERPT_CHARACTERS);
            }
            if published_at_ms.is_none()
                && let Some(timestamp) = element.select(&timestamps).next()
            {
                let value = normalized(timestamp.text(), 64);
                published_at_ms = iso_milliseconds(&value);
            }
        }
        let publisher = Url::parse(&url)
            .ok()
            .and_then(|url| url.host_str().map(str::to_string))
            .map(|value| normalized(std::iter::once(value.as_str()), MAX_PUBLISHER_CHARACTERS));
        push(
            &mut results,
            &mut urls,
            Some(SearchResult {
                title,
                url,
                excerpt,
                publisher,
                published_at_ms,
            }),
            request.published(),
        );
        if results.len() == MAX_RESULTS {
            break;
        }
    }
    results
}

fn has_class(classes: Option<&str>, expected: &str) -> bool {
    classes.is_some_and(|classes| {
        classes
            .split_ascii_whitespace()
            .any(|name| name == expected)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(published: bool) -> Request {
        let range = if published {
            r#"{"from_ms":1754524800000,"to_ms":1754697600000}"#
        } else {
            "null"
        };
        serde_json::from_str::<Request>(&format!(r#"{{"terms":"Rust 1.89","category":"web","language":"it-IT","reference_date":"2026-08-25","published":{range}}}"#)).unwrap().validated().unwrap()
    }

    #[test]
    fn organic_result_is_normalized_and_date_proven() {
        let html = r#"<table>
          <tr><td><a class='result-link' href='https://example.com/a#part'> Rust <b>1.89</b> </a></td></tr>
          <!-- provider decoration does not consume the row budget -->
          <!-- provider decoration does not consume the row budget -->
          <!-- provider decoration does not consume the row budget -->
          <tr><td class='result-snippet'> Stable &amp; current </td></tr>
          <tr><td><span class='timestamp'>2025-08-07T12:30:00.0000000</span></td></tr>
          <tr class='result-sponsored'><td><a class='result-link' href='https://ad.example/'>Ad</a></td></tr>
        </table>"#;
        let results = parse(html, &request(true));
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].title, "Rust 1.89");
        assert_eq!(results[0].url, "https://example.com/a");
        assert_eq!(results[0].published_at_ms, Some(1_754_569_800_000));
    }

    #[test]
    fn dated_search_rejects_missing_or_outside_timestamps() {
        let html = r#"<table>
          <tr><td><a class='result-link' href='https://example.com/a'>Undated</a></td></tr>
          <tr><td class='result-snippet'>Missing date</td></tr>
          <tr><td><a class='result-link' href='https://example.com/b'>Outside</a></td></tr>
          <tr><td class='result-snippet'>Wrong date</td></tr>
          <tr><td><span class='timestamp'>2025-08-10T00:00:00</span></td></tr>
        </table>"#;
        assert!(parse(html, &request(true)).is_empty());
    }

    #[tokio::test]
    #[ignore = "requires live Internet"]
    async fn live_web_search_returns_real_date_proven_results() {
        let request = request(true);
        let results = search(&request).await.unwrap();
        assert!(!results.is_empty());
        assert!(results.len() <= MAX_RESULTS);
        assert!(results.iter().all(|result| {
            result.url.starts_with("http")
                && result
                    .publisher
                    .as_deref()
                    .is_some_and(|value| !value.is_empty())
                && result
                    .published_at_ms
                    .is_some_and(|time| request.published().unwrap().contains(time))
        }));
    }
}
