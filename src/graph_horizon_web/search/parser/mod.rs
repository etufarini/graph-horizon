/*
 * Shared Web result contract
 * Defines bounded result fields, URL validation, normalization, and
 * deduplication shared by the three provider-specific document parsers.
 */

use std::collections::HashSet;

use url::Url;

mod brave;
mod duckduckgo;
mod google_news;

pub(super) const MAX_RESULTS: usize = 10;
pub(super) const MAX_TITLE_CHARACTERS: usize = 160;
const MAX_URL_CHARACTERS: usize = 2_048;
pub(super) const MAX_SNIPPET_CHARACTERS: usize = 600;

#[derive(Debug, PartialEq, Eq)]
pub(super) struct Result {
    pub(super) title: String,
    pub(super) url: String,
    pub(super) snippet: String,
}

pub(super) fn parse(html: &str) -> Vec<Result> {
    duckduckgo::parse(html)
}

pub(super) fn parse_news(xml: &str) -> Vec<Result> {
    google_news::parse(xml)
}

pub(super) fn parse_code(html: &str) -> Vec<Result> {
    brave::parse(html)
}

pub(super) fn push(results: &mut Vec<Result>, urls: &mut HashSet<String>, result: Option<Result>) {
    if let Some(result) = result
        && urls.insert(result.url.clone())
    {
        results.push(result);
    }
}

pub(super) fn result_url(href: &str) -> Option<String> {
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
