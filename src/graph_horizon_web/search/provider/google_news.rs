/*
 * Google News RSS search adapter
 * Builds one fixed feed request, parses XML as XML, and enforces any requested
 * half-open publication interval against each item's parsed UTC timestamp.
 */

use std::collections::HashSet;

use roxmltree::Document;
use scraper::{Html, Selector};
use url::Url;

use super::{
    Error, MAX_RESULTS, MAX_SNIPPET_CHARACTERS, MAX_TITLE_CHARACTERS, Result, language_region,
    normalized, push, result_url, transport_error,
};
use crate::graph_horizon_web::search::{request::Request, transport};

const ENDPOINT: &str = "https://news.google.com/rss/search";

pub(in crate::graph_horizon_web::search) async fn search(
    request: &Request,
) -> std::result::Result<Vec<Result>, Error> {
    let xml = transport::fetch(provider_request(request))
        .await
        .map_err(transport_error)?;
    parse(&xml, request)
}

fn provider_request(request: &Request) -> transport::Request {
    let mut url = Url::parse(ENDPOINT).expect("fixed news endpoint is valid");
    let query = request.published().map_or_else(
        || request.terms().to_string(),
        |range| {
            format!(
                "{} after:{} before:{}",
                request.terms(),
                range.from(),
                range.to()
            )
        },
    );
    let (language, region) = language_region(request.language());
    url.query_pairs_mut()
        .append_pair("q", &query)
        .append_pair("hl", request.language())
        .append_pair("gl", &region)
        .append_pair("ceid", &format!("{region}:{language}"));
    transport::Request {
        url: url.into(),
        form: Vec::new(),
    }
}

fn parse(xml: &str, request: &Request) -> std::result::Result<Vec<Result>, Error> {
    let document = Document::parse(xml).map_err(|_| Error::Invalid)?;
    let links = Selector::parse("a[href]").expect("fixed news link selector is valid");
    let publishers = Selector::parse("font").expect("fixed publisher selector is valid");
    let mut results = Vec::new();
    let mut urls = HashSet::new();

    for item in document
        .descendants()
        .filter(|node| node.has_tag_name("item"))
    {
        let child_text = |name| {
            item.children()
                .find(|child| child.has_tag_name(name))
                .and_then(|child| child.text())
        };
        let title = normalized(child_text("title").into_iter(), MAX_TITLE_CHARACTERS);
        let description = child_text("description").unwrap_or_default();
        let fragment = Html::parse_fragment(description);
        let Some(url) = fragment
            .select(&links)
            .next()
            .and_then(|link| link.value().attr("href"))
            .and_then(|href| result_url(href, ENDPOINT, false))
        else {
            continue;
        };
        if title.is_empty() {
            continue;
        }
        let published = child_text("pubDate").and_then(|date| httpdate::parse_http_date(date).ok());
        if request
            .published()
            .is_some_and(|range| published.is_none_or(|time| !range.contains(time)))
        {
            continue;
        }
        let publisher = child_text("source")
            .map(|value| normalized(std::iter::once(value), MAX_TITLE_CHARACTERS))
            .filter(|value| !value.is_empty())
            .or_else(|| {
                fragment
                    .select(&publishers)
                    .next_back()
                    .map(|node| normalized(node.text(), MAX_TITLE_CHARACTERS))
                    .filter(|value| !value.is_empty())
            });
        let snippet = normalized(fragment.root_element().text(), MAX_SNIPPET_CHARACTERS);
        push(
            &mut results,
            &mut urls,
            Some(Result {
                title,
                url,
                snippet,
                publisher,
                published_at: published.map(httpdate::fmt_http_date),
            }),
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
            r#"{"from":"2026-08-14","to":"2026-08-15"}"#
        } else {
            "null"
        };
        serde_json::from_str::<Request>(&format!(
            r#"{{"terms":"notizie Viterbo","category":"news","language":"it-IT","reference_date":"2026-08-24","published":{range}}}"#
        ))
        .unwrap()
        .validated()
        .unwrap()
    }

    #[test]
    fn provider_url_uses_language_and_half_open_dates() {
        let request = provider_request(&request(true));
        let url = Url::parse(&request.url).unwrap();
        assert!(url.query_pairs().any(|(key, value)| {
            key == "q" && value == "notizie Viterbo after:2026-08-14 before:2026-08-15"
        }));
        assert!(
            url.query_pairs()
                .any(|(key, value)| key == "ceid" && value == "IT:it")
        );
    }

    #[test]
    fn feed_is_structured_bounded_and_filtered_by_parsed_timestamp() {
        let items = (0..12)
            .map(|index| format!("<item><title>News {index}</title><pubDate>Fri, 14 Aug 2026 08:09:00 GMT</pubDate><description>&lt;a href=\"https://news.google.com/rss/articles/{index}?oc=5\"&gt;News {index}&lt;/a&gt;&amp;nbsp;&amp;nbsp;&lt;font&gt;Publisher {index}&lt;/font&gt;</description></item>"))
            .collect::<String>();
        let outside = "<item><title>Outside</title><pubDate>Sat, 15 Aug 2026 00:00:00 GMT</pubDate><description>&lt;a href=\"https://news.google.com/rss/articles/outside\"&gt;Outside&lt;/a&gt;</description></item>";
        let results = parse(
            &format!("<rss><channel>{outside}{items}</channel></rss>"),
            &request(true),
        )
        .unwrap();
        assert_eq!(results.len(), MAX_RESULTS);
        assert_eq!(results[0].title, "News 0");
        assert_eq!(results[0].publisher.as_deref(), Some("Publisher 0"));
        assert_eq!(
            results[0].published_at.as_deref(),
            Some("Fri, 14 Aug 2026 08:09:00 GMT")
        );
    }

    #[test]
    fn malformed_or_unsafe_feed_data_is_rejected() {
        assert_eq!(parse("<rss>", &request(false)), Err(Error::Invalid));
        let unsafe_xml = "<rss><channel><item><title>Unsafe</title><description>&lt;a href=\"javascript:alert(1)\"&gt;Unsafe&lt;/a&gt;</description></item></channel></rss>";
        assert!(parse(unsafe_xml, &request(false)).unwrap().is_empty());
    }
}
