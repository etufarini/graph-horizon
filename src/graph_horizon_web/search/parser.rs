/*
 * DuckDuckGo Lite result parser
 * Converts remote HTML rows into a small set of normalized text results. It
 * discards sponsored, duplicate, malformed, credential-bearing, and non-Web
 * URLs before any value can enter model context.
 */

use std::collections::HashSet;

use scraper::{Html, Selector};
use url::Url;

const MAX_RESULTS: usize = 5;
const MAX_TITLE_CHARACTERS: usize = 160;
const MAX_URL_CHARACTERS: usize = 2_048;
const MAX_SNIPPET_CHARACTERS: usize = 600;

#[derive(Debug, PartialEq, Eq)]
pub(super) struct Result {
    pub(super) title: String,
    pub(super) url: String,
    pub(super) snippet: String,
}

pub(super) fn parse(html: &str) -> Vec<Result> {
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
            pending = link
                .value()
                .attr("href")
                .and_then(result_url)
                .and_then(|url| {
                    let title = normalized(link.text(), MAX_TITLE_CHARACTERS);
                    (!title.is_empty()).then_some(Result {
                        title,
                        url,
                        snippet: String::new(),
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

fn push(results: &mut Vec<Result>, urls: &mut HashSet<String>, result: Option<Result>) {
    if let Some(result) = result
        && urls.insert(result.url.clone())
    {
        results.push(result);
    }
}

fn result_url(href: &str) -> Option<String> {
    let absolute = if href.starts_with("//") {
        format!("https:{href}")
    } else if href.starts_with('/') {
        format!("https://duckduckgo.com{href}")
    } else {
        href.to_string()
    };
    let mut url = Url::parse(&absolute).ok()?;
    if matches!(
        url.host_str(),
        Some("duckduckgo.com" | "www.duckduckgo.com")
    ) && url.path().starts_with("/l/")
    {
        let target = url
            .query_pairs()
            .find_map(|(key, value)| (key == "uddg").then(|| value.into_owned()))?;
        url = Url::parse(&target).ok()?;
    }
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

fn normalized<'a>(parts: impl Iterator<Item = &'a str>, limit: usize) -> String {
    let text = parts
        .flat_map(str::split_whitespace)
        .collect::<Vec<_>>()
        .join(" ");
    if text.chars().count() <= limit {
        text
    } else {
        text.chars().take(limit - 1).chain(['…']).collect()
    }
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
          <tr class="result-sponsored"><td class="result-snippet">Sponsored</td></tr>
          <tr><td><a class="result-link" href="https://example.com/a#part"> First <b>result</b> </a></td></tr>
          <tr><td class="result-snippet"> Useful &amp; current <b>snippet</b> </td></tr>
          <tr><td><a class="result-link" href="/l/?uddg=https%3A%2F%2Frust-lang.org%2Flearn%3Fx%3D1&amp;rut=ignored">Rust</a></td></tr>
          <tr><td class="result-snippet">Language site</td></tr>
          <tr><td><a class="result-link" href="javascript:alert(1)">Unsafe</a></td></tr>
          <tr><td class="result-snippet">Ignored</td></tr>
          <tr><td><a class="result-link" href="https://example.com/a">Duplicate</a></td></tr>
          <tr><td class="result-snippet">Ignored duplicate</td></tr>
        </table>
    "#;

    #[test]
    fn organic_results_are_normalized_resolved_and_deduplicated() {
        assert_eq!(
            parse(FIXTURE),
            vec![
                Result {
                    title: "First result".into(),
                    url: "https://example.com/a".into(),
                    snippet: "Useful & current snippet".into(),
                },
                Result {
                    title: "Rust".into(),
                    url: "https://rust-lang.org/learn?x=1".into(),
                    snippet: "Language site".into(),
                },
            ]
        );
    }

    #[test]
    fn fields_and_result_count_are_bounded() {
        let title = "t".repeat(MAX_TITLE_CHARACTERS + 10);
        let snippet = "s".repeat(MAX_SNIPPET_CHARACTERS + 10);
        let rows = (0..10)
            .map(|index| {
                format!(
                    "<tr><td><a class='result-link' href='https://example.com/{index}'>{title}</a></td></tr><tr><td class='result-snippet'>{snippet}</td></tr>"
                )
            })
            .collect::<String>();
        let html = format!("<table>{rows}</table>");
        let results = parse(&html);
        assert_eq!(results.len(), MAX_RESULTS);
        assert_eq!(results[0].title.chars().count(), MAX_TITLE_CHARACTERS);
        assert_eq!(results[0].snippet.chars().count(), MAX_SNIPPET_CHARACTERS);
    }

    #[test]
    fn result_urls_reject_credentials_and_non_web_schemes() {
        for href in [
            "javascript:alert(1)",
            "file:///etc/passwd",
            "https://user@example.com/",
            "https://user:secret@example.com/",
            "/l/?uddg=javascript%3Aalert%281%29",
        ] {
            assert_eq!(result_url(href), None, "accepted {href}");
        }
        assert_eq!(
            result_url("//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fa%23part"),
            Some("https://example.com/a".into())
        );
    }
}
