/*
 * DuckDuckGo Lite HTML parser
 * Extracts organic result rows, resolves provider redirect URLs, and preserves
 * only bounded normalized fields before shared URL validation and deduplication.
 */

use std::collections::HashSet;

use scraper::{Html, Selector};

use super::{
    MAX_RESULTS, MAX_SNIPPET_CHARACTERS, MAX_TITLE_CHARACTERS, Result, normalized, push, result_url,
};

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
        let results = parse(&format!("<table>{rows}</table>"));
        assert_eq!(results.len(), MAX_RESULTS);
        assert_eq!(results[0].title.chars().count(), MAX_TITLE_CHARACTERS);
        assert_eq!(results[0].snippet.chars().count(), MAX_SNIPPET_CHARACTERS);
    }
}
