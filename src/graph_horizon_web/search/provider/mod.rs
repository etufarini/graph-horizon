/*
 * Shared search-provider result contract
 * Defines bounded plain-text fields, provider errors, URL resolution, and
 * exact-URL deduplication used by the fixed provider adapters.
 */

use std::collections::HashSet;

use url::Url;

use super::transport;

pub(super) mod brave;
pub(super) mod duckduckgo;
pub(super) mod google_news;

pub(super) const MAX_RESULTS: usize = 10;
pub(super) const MAX_TITLE_CHARACTERS: usize = 160;
const MAX_URL_CHARACTERS: usize = 2_048;
pub(super) const MAX_SNIPPET_CHARACTERS: usize = 600;

#[derive(Debug, PartialEq, Eq)]
pub(super) struct Result {
    pub(super) title: String,
    pub(super) url: String,
    pub(super) snippet: String,
    pub(super) publisher: Option<String>,
    pub(super) published_at: Option<String>,
}

#[derive(Debug, PartialEq, Eq)]
pub(super) enum Error {
    Blocked,
    RateLimited,
    Invalid,
    Unavailable,
}

pub(super) fn transport_error(error: transport::Error) -> Error {
    match error {
        transport::Error::Http(429) => Error::RateLimited,
        transport::Error::Invalid => Error::Invalid,
        transport::Error::Unavailable
        | transport::Error::Timeout
        | transport::Error::TooLarge
        | transport::Error::Http(_) => Error::Unavailable,
    }
}

pub(super) fn language_region(language: &str) -> (String, String) {
    let mut parts = language.split('-');
    let primary = parts.next().unwrap_or("en").to_ascii_lowercase();
    let region = parts
        .find(|part| part.len() == 2 && part.bytes().all(|byte| byte.is_ascii_alphabetic()))
        .map(str::to_ascii_uppercase)
        .unwrap_or_else(|| match primary.as_str() {
            "it" => "IT".into(),
            _ => "US".into(),
        });
    (primary, region)
}

pub(super) fn push(results: &mut Vec<Result>, urls: &mut HashSet<String>, result: Option<Result>) {
    if let Some(result) = result
        && urls.insert(result.url.clone())
    {
        results.push(result);
    }
}

pub(super) fn result_url(href: &str, base: &str, unwrap_duckduckgo: bool) -> Option<String> {
    let mut url = Url::parse(href)
        .or_else(|_| Url::parse(base).and_then(|base| base.join(href)))
        .ok()?;
    if unwrap_duckduckgo
        && matches!(
            url.host_str(),
            Some("duckduckgo.com" | "www.duckduckgo.com" | "lite.duckduckgo.com")
        )
        && url.path().starts_with("/l/")
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

pub(super) fn normalized<'a>(parts: impl Iterator<Item = &'a str>, limit: usize) -> String {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn result_urls_use_the_provider_base_and_reject_unsafe_targets() {
        for href in [
            "javascript:alert(1)",
            "file:///etc/passwd",
            "https://user@example.com/",
            "https://user:secret@example.com/",
            "/l/?uddg=javascript%3Aalert%281%29",
        ] {
            assert_eq!(
                result_url(href, "https://duckduckgo.com/", true),
                None,
                "accepted {href}"
            );
        }
        assert_eq!(
            result_url("/relative", "https://search.brave.com/search", false),
            Some("https://search.brave.com/relative".into())
        );
        assert_eq!(
            result_url(
                "//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fa%23part",
                "https://duckduckgo.com/",
                true,
            ),
            Some("https://example.com/a".into())
        );
    }

    #[test]
    fn language_tags_produce_provider_language_and_region() {
        assert_eq!(language_region("it-IT"), ("it".into(), "IT".into()));
        assert_eq!(language_region("en-GB"), ("en".into(), "GB".into()));
        assert_eq!(language_region("es-419"), ("es".into(), "US".into()));
    }
}
