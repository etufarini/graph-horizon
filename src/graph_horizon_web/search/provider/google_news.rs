/*
 * Google News RSS adapter
 * Builds one fixed keyless feed request, parses RSS as XML, and retains only
 * bounded news items whose UTC publication time proves any requested interval.
 */

use std::collections::HashSet;
use std::time::UNIX_EPOCH;

use roxmltree::Document;
use scraper::Html;
use url::Url;

use super::{
    Error, MAX_EXCERPT_CHARACTERS, MAX_PUBLISHER_CHARACTERS, MAX_RESULTS, MAX_TITLE_CHARACTERS,
    SearchResult, language_region, normalized, push, result_url, transport_error,
};
use crate::graph_horizon_web::search::request::Request;
use crate::graph_horizon_web::search::transport::{self, Accept, Body};

const ENDPOINT: &str = "https://news.google.com/rss/search";

pub(super) async fn search(request: &Request) -> Result<Vec<SearchResult>, Error> {
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
            let (after, before) = range.google_dates();
            format!("{} after:{after} before:{before}", request.terms())
        },
    );
    let (language, region) = language_region(request.language());
    url.query_pairs_mut()
        .append_pair("q", &query)
        // A primary-language `hl` avoids the feed's regional redirect.
        .append_pair("hl", &language)
        .append_pair("gl", &region)
        .append_pair("ceid", &format!("{region}:{language}"));
    transport::Request {
        url: url.into(),
        accept: Accept::Xml,
        body: Body::Empty,
        bearer: None,
    }
}

fn parse(xml: &str, request: &Request) -> Result<Vec<SearchResult>, Error> {
    let document = Document::parse(xml).map_err(|_| Error::Invalid)?;
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
        let Some(url) = child_text("link").and_then(|value| result_url(value, ENDPOINT, false))
        else {
            continue;
        };
        let publisher = child_text("source")
            .map(|value| normalized(std::iter::once(value), MAX_PUBLISHER_CHARACTERS))
            .filter(|value| !value.is_empty());
        let excerpt = child_text("description")
            .map(Html::parse_fragment)
            .map(|fragment| normalized(fragment.root_element().text(), MAX_EXCERPT_CHARACTERS))
            .filter(|value| !value.is_empty())
            .unwrap_or_else(|| title.clone());
        let published_at_ms = child_text("pubDate")
            .and_then(|value| httpdate::parse_http_date(value).ok())
            .and_then(|time| time.duration_since(UNIX_EPOCH).ok())
            .and_then(|duration| u64::try_from(duration.as_millis()).ok());
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
    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(published: bool) -> Request {
        let range = if published {
            r#"{"from_ms":1787522400000,"to_ms":1787695200000}"#
        } else {
            "null"
        };
        serde_json::from_str::<Request>(&format!(r#"{{"terms":"OpenAI","category":"news","language":"it-IT","reference_date":"2026-08-25","published":{range}}}"#)).unwrap().validated().unwrap()
    }

    #[test]
    fn rss_item_keeps_source_and_exact_in_range_date() {
        let xml = r#"<rss><channel><item>
          <title>OpenAI presenta un prodotto</title>
          <link>https://news.google.com/rss/articles/example?oc=5</link>
          <pubDate>Mon, 24 Aug 2026 11:40:26 GMT</pubDate>
          <description>&lt;a href="https://news.google.com/rss/articles/example?oc=5"&gt;OpenAI presenta un prodotto&lt;/a&gt;&amp;nbsp;&amp;nbsp;&lt;font&gt;Example&lt;/font&gt;</description>
          <source url="https://example.com">Example</source>
        </item></channel></rss>"#;
        let results = parse(xml, &request(true)).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].publisher.as_deref(), Some("Example"));
        assert_eq!(results[0].published_at_ms, Some(1_787_571_626_000));
    }

    #[test]
    fn malformed_feed_and_unproven_dates_fail_closed() {
        assert_eq!(parse("<rss>", &request(false)), Err(Error::Invalid));
        let undated = r#"<rss><channel><item><title>x</title><link>https://example.com/</link><description>x</description></item></channel></rss>"#;
        assert!(parse(undated, &request(true)).unwrap().is_empty());
    }

    #[tokio::test]
    #[ignore = "requires live Internet"]
    async fn live_news_search_returns_real_dated_results() {
        let results = search(&request(false)).await.unwrap();
        assert!(!results.is_empty());
        assert!(results.len() <= MAX_RESULTS);
        assert!(
            results
                .iter()
                .all(|result| result.published_at_ms.is_some())
        );
        assert!(results.iter().all(|result| {
            result.url.starts_with("https://")
                && result
                    .publisher
                    .as_deref()
                    .is_some_and(|value| !value.is_empty())
        }));
    }

    #[tokio::test]
    #[ignore = "requires live Internet"]
    async fn live_news_search_proves_requested_dates() {
        let request = request(true);
        let results = search(&request).await.unwrap();
        assert!(!results.is_empty());
        assert!(results.iter().all(|result| {
            result
                .published_at_ms
                .is_some_and(|time| request.published().unwrap().contains(time))
        }));
    }
}
