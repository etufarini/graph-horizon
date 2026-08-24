/*
 * Brave Web search adapter
 * Builds one fixed general-search request and extracts bounded Web cards while
 * distinguishing provider challenges and rate limits from an empty result set.
 */

use std::collections::HashSet;

use scraper::{Html, Selector};
use url::Url;

use super::{
    Error, MAX_RESULTS, MAX_SNIPPET_CHARACTERS, MAX_TITLE_CHARACTERS, Result, language_region,
    normalized, push, result_url, transport_error,
};
use crate::graph_horizon_web::search::{request::Request, transport};

const ENDPOINT: &str = "https://search.brave.com/search";

pub(in crate::graph_horizon_web::search) async fn search(
    request: &Request,
) -> std::result::Result<Vec<Result>, Error> {
    let html = transport::fetch(provider_request(request))
        .await
        .map_err(transport_error)?;
    if html.contains("challengeSet") || html.contains("Quick check before you continue searching") {
        return Err(Error::Blocked);
    }
    Ok(parse(&html))
}

fn provider_request(request: &Request) -> transport::Request {
    let mut url = Url::parse(ENDPOINT).expect("fixed search endpoint is valid");
    let (language, region) = language_region(request.language());
    url.query_pairs_mut()
        .append_pair("q", request.terms())
        .append_pair("source", "web")
        .append_pair("search_lang", &language)
        .append_pair("country", &region.to_ascii_lowercase());
    transport::Request {
        url: url.into(),
        form: Vec::new(),
    }
}

fn parse(html: &str) -> Vec<Result> {
    let document = Html::parse_document(html);
    let cards = Selector::parse(".snippet[data-type='web']").expect("fixed card selector is valid");
    let links = Selector::parse("a.l1[href]").expect("fixed Web link selector is valid");
    let titles = Selector::parse(".search-snippet-title").expect("fixed title selector is valid");
    let snippets =
        Selector::parse(".generic-snippet .content").expect("fixed snippet selector is valid");
    let mut results = Vec::new();
    let mut urls = HashSet::new();

    for card in document.select(&cards) {
        let Some(link) = card.select(&links).next() else {
            continue;
        };
        let Some(url) = link
            .value()
            .attr("href")
            .and_then(|href| result_url(href, ENDPOINT, false))
        else {
            continue;
        };
        let title = card
            .select(&titles)
            .next()
            .map(|title| normalized(title.text(), MAX_TITLE_CHARACTERS))
            .unwrap_or_default();
        if title.is_empty() {
            continue;
        }
        let snippet = card
            .select(&snippets)
            .next()
            .map(|snippet| normalized(snippet.text(), MAX_SNIPPET_CHARACTERS))
            .unwrap_or_default();
        push(
            &mut results,
            &mut urls,
            Some(Result {
                title,
                url,
                snippet,
                publisher: None,
                published_at: None,
            }),
        );
        if results.len() == MAX_RESULTS {
            break;
        }
    }
    results
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request() -> Request {
        serde_json::from_str::<Request>(
            r#"{"terms":"Come risolvo error class?","category":"web","language":"it-IT","reference_date":"2026-08-24","published":null}"#,
        )
        .unwrap()
        .validated()
        .unwrap()
    }

    #[test]
    fn provider_url_is_general_and_preserves_terms() {
        let request = provider_request(&request());
        let url = Url::parse(&request.url).unwrap();
        assert!(
            url.query_pairs()
                .any(|(key, value)| key == "q" && value == "Come risolvo error class?")
        );
        assert!(
            url.query_pairs()
                .any(|(key, value)| key == "search_lang" && value == "it")
        );
        assert!(!url.as_str().contains("official+documentation"));
    }

    #[test]
    fn web_cards_are_bounded_and_unsafe_cards_are_ignored() {
        let cards = (0..12)
            .map(|index| format!("<div class='snippet' data-type='web'><a class='l1' href='https://docs.example/{index}'><div class='search-snippet-title'>Docs {index}</div></a><div class='generic-snippet'><div class='content'>Example {index}</div></div></div>"))
            .collect::<String>();
        let html = format!(
            "<div class='snippet' data-type='ad'><a class='l1' href='https://ad.example/'><div class='search-snippet-title'>Ad</div></a></div><div class='snippet' data-type='web'><a class='l1' href='javascript:alert(1)'><div class='search-snippet-title'>Unsafe</div></a></div>{cards}"
        );
        let results = parse(&html);
        assert_eq!(results.len(), MAX_RESULTS);
        assert_eq!(results[0].title, "Docs 0");
        assert_eq!(results[0].url, "https://docs.example/0");
        assert_eq!(results[0].snippet, "Example 0");
    }
}
