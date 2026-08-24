/*
 * Google News RSS parser
 * Extracts bounded headline, publication, publisher, and validated link data
 * from fallback feed items without following the linked articles.
 */

use std::collections::HashSet;

use scraper::{Html, Selector};

use super::{
    MAX_RESULTS, MAX_SNIPPET_CHARACTERS, MAX_TITLE_CHARACTERS, Result, normalized, push, result_url,
};

pub(super) fn parse(xml: &str) -> Vec<Result> {
    let document = Html::parse_document(xml);
    let items = Selector::parse("item").expect("fixed news item selector is valid");
    let titles = Selector::parse("title").expect("fixed news title selector is valid");
    let descriptions =
        Selector::parse("description").expect("fixed news description selector is valid");
    let dates = Selector::parse("pubdate").expect("fixed news date selector is valid");
    let links = Selector::parse("a[href]").expect("fixed news link selector is valid");
    let mut results = Vec::new();
    let mut urls = HashSet::new();

    for item in document.select(&items) {
        let title = item
            .select(&titles)
            .next()
            .map(|title| normalized(title.text(), MAX_TITLE_CHARACTERS))
            .unwrap_or_default();
        let description = item
            .select(&descriptions)
            .next()
            .map(|description| description.text().collect::<String>())
            .unwrap_or_default();
        let fragment = Html::parse_fragment(&description);
        let url = fragment
            .select(&links)
            .next()
            .and_then(|link| link.value().attr("href"))
            .and_then(result_url);
        let Some(url) = url else {
            continue;
        };
        if title.is_empty() {
            continue;
        }
        let date = item
            .select(&dates)
            .next()
            .map(|date| normalized(date.text(), 80))
            .unwrap_or_default();
        let summary = normalized(fragment.root_element().text(), MAX_SNIPPET_CHARACTERS / 2);
        let snippet = normalized(
            format!("Published: {date}. {summary}").split_whitespace(),
            MAX_SNIPPET_CHARACTERS,
        );
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
    fn feed_provides_bounded_validated_results() {
        let items = (0..12)
            .map(|index| {
                format!(
                    "<item><title>News {index}</title><pubDate>Mon, 24 Aug 2026 08:09:00 GMT</pubDate><description>&lt;a href=\"https://news.google.com/rss/articles/{index}?oc=5\"&gt;News {index}&lt;/a&gt;&amp;nbsp;&amp;nbsp;&lt;font&gt;Publisher {index}&lt;/font&gt;</description></item>"
                )
            })
            .collect::<String>();
        let results = parse(&format!("<rss><channel>{items}</channel></rss>"));
        assert_eq!(results.len(), MAX_RESULTS);
        assert_eq!(results[0].title, "News 0");
        assert_eq!(
            results[0].url,
            "https://news.google.com/rss/articles/0?oc=5"
        );
        assert_eq!(
            results[0].snippet,
            "Published: Mon, 24 Aug 2026 08:09:00 GMT. News 0 Publisher 0"
        );

        let unsafe_xml = "<rss><channel><item><title>Unsafe</title><description>&lt;a href=\"javascript:alert(1)\"&gt;Unsafe&lt;/a&gt;</description></item></channel></rss>";
        assert!(parse(unsafe_xml).is_empty());
    }
}
