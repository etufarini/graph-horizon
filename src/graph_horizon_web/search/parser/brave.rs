/*
 * Brave technical-search HTML parser
 * Extracts bounded Web result cards from the code fallback and validates every
 * target URL. Page controls, infoboxes, scripts, and non-Web cards are ignored.
 */

use std::collections::HashSet;

use scraper::{Html, Selector};

use super::{
    MAX_RESULTS, MAX_SNIPPET_CHARACTERS, MAX_TITLE_CHARACTERS, Result, normalized, push, result_url,
};

pub(super) fn parse(html: &str) -> Vec<Result> {
    let document = Html::parse_document(html);
    let cards = Selector::parse(".snippet[data-type='web']").expect("fixed card selector is valid");
    let links = Selector::parse("a.l1[href]").expect("fixed code link selector is valid");
    let titles =
        Selector::parse(".search-snippet-title").expect("fixed code title selector is valid");
    let snippets =
        Selector::parse(".generic-snippet .content").expect("fixed code snippet selector is valid");
    let mut results = Vec::new();
    let mut urls = HashSet::new();

    for card in document.select(&cards) {
        let Some(link) = card.select(&links).next() else {
            continue;
        };
        let Some(url) = link.value().attr("href").and_then(result_url) else {
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

    #[test]
    fn web_cards_are_bounded_and_unsafe_cards_are_ignored() {
        let cards = (0..12)
            .map(|index| {
                format!(
                    "<div class='snippet' data-type='web'><a class='l1' href='https://docs.example/{index}'><div class='search-snippet-title'>Docs {index}</div></a><div class='generic-snippet'><div class='content'>Example {index}</div></div></div>"
                )
            })
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
