/*
 * Search provider boundary
 * Dispatches one validated request to the selected adapter and owns the common
 * bounded result, URL, date-proof, normalization, and transport-error contract.
 */

use std::collections::HashSet;

use url::Url;

use super::config::Config;
use super::request::{Category, MAX_TIMESTAMP_BOUND_MS, Published, Request};
use super::transport;

mod duckduckgo;
mod google_news;
mod json;

pub(super) const MAX_RESULTS: usize = 5;
pub(super) const MAX_TITLE_CHARACTERS: usize = 160;
pub(super) const MAX_EXCERPT_CHARACTERS: usize = 320;
pub(super) const MAX_PUBLISHER_CHARACTERS: usize = 100;
const MAX_URL_CHARACTERS: usize = 2_048;

#[derive(Debug, PartialEq, Eq)]
pub(super) enum Error {
    RateLimited,
    Timeout,
    Invalid,
    Unavailable,
}

#[derive(Debug, PartialEq, Eq)]
pub(super) struct SearchResult {
    pub(super) title: String,
    pub(super) url: String,
    pub(super) excerpt: String,
    pub(super) publisher: Option<String>,
    pub(super) published_at_ms: Option<u64>,
}

pub(super) async fn search(config: &Config, request: &Request) -> Result<Vec<SearchResult>, Error> {
    match config {
        Config::Public => match request.category() {
            Category::Web => duckduckgo::search(request).await,
            Category::News => google_news::search(request).await,
        },
        Config::Json { .. } => json::search(config, request).await,
    }
}

pub(super) fn label(config: &Config, category: Category) -> &str {
    match config {
        Config::Public => match category {
            Category::Web => "duckduckgo.com",
            Category::News => "news.google.com",
        },
        Config::Json { .. } => config.capability_label(),
    }
}

pub(super) fn transport_error(error: transport::Error) -> Error {
    match error {
        transport::Error::Http(429) => Error::RateLimited,
        transport::Error::Timeout => Error::Timeout,
        transport::Error::Invalid => Error::Invalid,
        transport::Error::Unavailable | transport::Error::TooLarge | transport::Error::Http(_) => {
            Error::Unavailable
        }
    }
}

pub(super) fn push(
    results: &mut Vec<SearchResult>,
    urls: &mut HashSet<String>,
    result: SearchResult,
    published: Option<Published>,
) {
    if result.title.is_empty()
        || result.excerpt.is_empty()
        || result
            .published_at_ms
            .is_some_and(|time| time >= MAX_TIMESTAMP_BOUND_MS)
        || published.is_some_and(|range| {
            result
                .published_at_ms
                .is_none_or(|time| !range.contains(time))
        })
        || !urls.insert(result.url.clone())
    {
        return;
    }
    results.push(result);
}

pub(super) fn result_url(value: &str, base: &str, unwrap_duckduckgo: bool) -> Option<String> {
    let mut url = Url::parse(value)
        .or_else(|_| Url::parse(base).and_then(|base| base.join(value)))
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

pub(super) fn iso_milliseconds(value: &str) -> Option<u64> {
    let bytes = value.as_bytes();
    if bytes.len() < 19
        || bytes[4] != b'-'
        || bytes[7] != b'-'
        || bytes[10] != b'T'
        || bytes[13] != b':'
        || bytes[16] != b':'
    {
        return None;
    }
    let number = |start: usize, end: usize| -> Option<u64> {
        bytes.get(start..end)?.iter().try_fold(0, |value, byte| {
            byte.is_ascii_digit()
                .then_some(value * 10 + u64::from(byte - b'0'))
        })
    };
    let year = number(0, 4)? as i64;
    let month = number(5, 7)? as i64;
    let day = number(8, 10)? as i64;
    let hour = number(11, 13)?;
    let minute = number(14, 16)?;
    let second = number(17, 19)?;
    if year < 1970 || !(1..=12).contains(&month) || hour > 23 || minute > 59 || second > 59 {
        return None;
    }
    let leap = year % 4 == 0 && (year % 100 != 0 || year % 400 == 0);
    let month_days = match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if leap => 29,
        2 => 28,
        _ => return None,
    };
    if day < 1 || day > month_days {
        return None;
    }
    let adjusted_year = year - i64::from(month <= 2);
    let era = adjusted_year / 400;
    let year_of_era = adjusted_year - era * 400;
    let shifted_month = month + if month > 2 { -3 } else { 9 };
    let day_of_year = (153 * shifted_month + 2) / 5 + day - 1;
    let days = era * 146_097 + year_of_era * 365 + year_of_era / 4 - year_of_era / 100
        + day_of_year
        - 719_468;
    let milliseconds = match bytes.get(19) {
        None => 0,
        Some(b'.') if bytes.len() > 20 && bytes[20..].iter().all(u8::is_ascii_digit) => {
            let mut value = number(20, bytes.len().min(23))?;
            for _ in bytes.len().min(23)..23 {
                value *= 10;
            }
            value
        }
        _ => return None,
    };
    Some((days as u64 * 86_400 + hour * 3_600 + minute * 60 + second) * 1_000 + milliseconds)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn provider_urls_reject_unsafe_targets_and_unwrap_duckduckgo() {
        for value in [
            "javascript:alert(1)",
            "file:///tmp/x",
            "https://user@example.com/",
        ] {
            assert!(result_url(value, "https://duckduckgo.com/", true).is_none());
        }
        assert_eq!(
            result_url(
                "/l/?uddg=https%3A%2F%2Fexample.com%2Fa%23x",
                "https://duckduckgo.com/",
                true
            ),
            Some("https://example.com/a".into())
        );
    }

    #[test]
    fn provider_timestamp_is_strict_utc_milliseconds() {
        assert_eq!(iso_milliseconds("1970-01-01T00:00:00.1230000"), Some(123));
        assert_eq!(
            iso_milliseconds("2024-02-29T12:30:01"),
            Some(1_709_209_801_000)
        );
        assert_eq!(iso_milliseconds("2026-02-29T00:00:00"), None);
        assert_eq!(iso_milliseconds("1970-01-01T00:00:00.bad"), None);
        assert_eq!(iso_milliseconds("1970-01-01T00:00:00Z"), None);
    }
}
