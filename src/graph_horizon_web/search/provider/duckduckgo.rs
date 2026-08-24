/*
 * DuckDuckGo Lite search adapter
 * Builds one fixed provider request and extracts bounded organic results while
 * reporting explicit block pages instead of treating them as empty searches.
 */

use std::collections::HashSet;

use scraper::{Html, Selector};

use super::{
    Error, MAX_RESULTS, MAX_SNIPPET_CHARACTERS, MAX_TITLE_CHARACTERS, Result, language_region,
    normalized, push, result_url, transport_error,
};
use crate::graph_horizon_web::search::{request::Request, transport};

const ENDPOINT: &str = "https://lite.duckduckgo.com/lite/";

pub(in crate::graph_horizon_web::search) async fn search(
    request: &Request,
) -> std::result::Result<Vec<Result>, Error> {
    let html = transport::fetch(provider_request(request))
        .await
        .map_err(transport_error)?;
    if html.contains("challenge-form") || html.contains("anomaly-modal__title") {
        return Err(Error::Blocked);
    }
    Ok(parse(&html))
}

fn provider_request(request: &Request) -> transport::Request {
    let (language, region) = language_region(request.language());
    let mut form = vec![
        ("q".into(), request.terms().into()),
        (
            "kl".into(),
            format!("{}-{language}", region.to_ascii_lowercase()),
        ),
        ("kp".into(), "1".into()),
    ];
    if let Some(range) = request.published() {
        form.push(("df".into(), range.duckduckgo_filter()));
    }
    transport::Request {
        url: ENDPOINT.into(),
        form,
    }
}

fn parse(html: &str) -> Vec<Result> {
    let document = Html::parse_document(html);
    let rows = Selector::parse("tr").expect("fixed row selector is valid");
    let links = Selector::parse("a.result-link").expect("fixed result selector is valid");
    let snippets = Selector::parse(".result-snippet").expect("fixed snippet selector is valid");
    let mut pending = None;
    let mut results = Vec::new();
    let mut urls = HashSet::new();

    for row in document.select(&rows) {
        if has_class(row.value().attr("class"), "result-sponsored") {
            continue;
        }
        if let Some(link) = row.select(&links).next() {
            push(&mut results, &mut urls, pending.take());
            if results.len() == MAX_RESULTS {
                break;
            }
            pending = link.value().attr("href").and_then(|href| {
                let url = result_url(href, ENDPOINT, true)?;
                let title = normalized(link.text(), MAX_TITLE_CHARACTERS);
                (!title.is_empty()).then_some(Result {
                    title,
                    url,
                    snippet: String::new(),
                    publisher: None,
                    published_at: None,
                })
            });
            continue;
        }
        if let Some(result) = &mut pending
            && let Some(snippet) = row.select(&snippets).next()
        {
            result.snippet = normalized(snippet.text(), MAX_SNIPPET_CHARACTERS);
            push(&mut results, &mut urls, pending.take());
            if results.len() == MAX_RESULTS {
                break;
            }
        }
    }
    if results.len() < MAX_RESULTS {
        push(&mut results, &mut urls, pending);
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

    const FIXTURE: &str = r#"
        <table>
          <tr class="result-sponsored"><td><a class="result-link" href="https://ad.example/">Ad</a></td></tr>
          <tr><td><a class="result-link" href="https://example.com/a#part"> First <b>result</b> </a></td></tr>
          <tr><td class="result-snippet"> Useful &amp; current <b>snippet</b> </td></tr>
          <tr><td><a class="result-link" href="/l/?uddg=https%3A%2F%2Frust-lang.org%2Flearn%3Fx%3D1&amp;rut=ignored">Rust</a></td></tr>
          <tr><td class="result-snippet">Language site</td></tr>
          <tr><td><a class="result-link" href="javascript:alert(1)">Unsafe</a></td></tr>
          <tr><td class="result-snippet">Ignored</td></tr>
          <tr><td><a class="result-link" href="https://example.com/a">Duplicate</a></td></tr>
        </table>
    "#;

    fn request() -> Request {
        serde_json::from_str::<Request>(
            r#"{"terms":"a b&@é","category":"web","language":"fr-FR","reference_date":"2026-08-24","published":{"from":"2026-08-14","to":"2026-08-15"}}"#,
        )
        .unwrap()
        .validated()
        .unwrap()
    }

    #[test]
    fn provider_form_preserves_terms_and_applies_explicit_dates() {
        let request = provider_request(&request());
        assert_eq!(request.url, ENDPOINT);
        assert!(request.form.contains(&("q".into(), "a b&@é".into())));
        assert!(request.form.contains(&("kl".into(), "fr-fr".into())));
        assert!(
            request
                .form
                .contains(&("df".into(), "2026-08-14..2026-08-14".into()))
        );
    }

    #[test]
    fn organic_results_are_normalized_resolved_and_deduplicated() {
        let results = parse(FIXTURE);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].title, "First result");
        assert_eq!(results[0].url, "https://example.com/a");
        assert_eq!(results[0].snippet, "Useful & current snippet");
        assert_eq!(results[1].url, "https://rust-lang.org/learn?x=1");
    }

    #[test]
    fn fields_and_result_count_are_bounded() {
        let title = "t".repeat(MAX_TITLE_CHARACTERS + 10);
        let snippet = "s".repeat(MAX_SNIPPET_CHARACTERS + 10);
        let rows = (0..12)
            .map(|index| format!("<tr><td><a class='result-link' href='https://example.com/{index}'>{title}</a></td></tr><tr><td class='result-snippet'>{snippet}</td></tr>"))
            .collect::<String>();
        let results = parse(&format!("<table>{rows}</table>"));
        assert_eq!(results.len(), MAX_RESULTS);
        assert_eq!(results[0].title.chars().count(), MAX_TITLE_CHARACTERS);
        assert_eq!(results[0].snippet.chars().count(), MAX_SNIPPET_CHARACTERS);
    }
}
