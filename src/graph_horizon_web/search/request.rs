/*
 * Provider-neutral Web search request
 * Validates the bounded terms, browser language hint, reference date, and
 * optional half-open publication interval before any provider sees them.
 */

use serde::Deserialize;
use std::time::{SystemTime, UNIX_EPOCH};

const MAX_QUERY_CHARACTERS: usize = 512;
const MAX_LANGUAGE_CHARACTERS: usize = 35;

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub(super) enum Category {
    Web,
    News,
}

#[derive(Debug, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(super) struct Published {
    from: String,
    to: String,
}

#[derive(Debug, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(in crate::graph_horizon_web) struct Request {
    terms: String,
    category: Category,
    language: String,
    reference_date: String,
    published: Option<Published>,
}

impl Request {
    pub(in crate::graph_horizon_web) fn validated(mut self) -> Result<Self, ()> {
        self.terms = self.terms.trim().to_string();
        if self.terms.is_empty()
            || self.terms.chars().count() > MAX_QUERY_CHARACTERS
            || !valid_language(&self.language)
            || !valid_date(&self.reference_date)
            || self.published.as_ref().is_some_and(|range| {
                !valid_date(&range.from) || !valid_date(&range.to) || range.from >= range.to
            })
        {
            return Err(());
        }
        Ok(self)
    }

    pub(super) fn terms(&self) -> &str {
        &self.terms
    }

    pub(super) fn category(&self) -> Category {
        self.category
    }

    pub(super) fn language(&self) -> &str {
        &self.language
    }

    pub(super) fn reference_date(&self) -> &str {
        &self.reference_date
    }

    pub(super) fn published(&self) -> Option<&Published> {
        self.published.as_ref()
    }
}

impl Published {
    pub(super) fn from(&self) -> &str {
        &self.from
    }

    pub(super) fn to(&self) -> &str {
        &self.to
    }

    pub(super) fn contains(&self, time: SystemTime) -> bool {
        let Ok(seconds) = time.duration_since(UNIX_EPOCH) else {
            return false;
        };
        let day = (seconds.as_secs() / 86_400) as i64;
        date_day(&self.from).is_some_and(|from| from <= day)
            && date_day(&self.to).is_some_and(|to| day < to)
    }

    pub(super) fn duckduckgo_filter(&self) -> String {
        format!("{}..{}", self.from, previous_date(&self.to))
    }
}

fn valid_language(value: &str) -> bool {
    if value.is_empty() || value.len() > MAX_LANGUAGE_CHARACTERS {
        return false;
    }
    let mut parts = value.split('-');
    let Some(primary) = parts.next() else {
        return false;
    };
    (2..=8).contains(&primary.len())
        && primary.bytes().all(|byte| byte.is_ascii_alphabetic())
        && parts.all(|part| {
            !part.is_empty()
                && part.len() <= 8
                && part.bytes().all(|byte| byte.is_ascii_alphanumeric())
        })
}

pub(super) fn valid_date(value: &str) -> bool {
    let bytes = value.as_bytes();
    if bytes.len() != 10
        || bytes[4] != b'-'
        || bytes[7] != b'-'
        || !bytes
            .iter()
            .enumerate()
            .all(|(index, byte)| matches!(index, 4 | 7) || byte.is_ascii_digit())
    {
        return false;
    }
    let year = value[0..4].parse::<u16>().unwrap_or(0);
    let month = value[5..7].parse::<u8>().unwrap_or(0);
    let day = value[8..10].parse::<u8>().unwrap_or(0);
    let leap = year % 4 == 0 && (year % 100 != 0 || year % 400 == 0);
    let days = match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if leap => 29,
        2 => 28,
        _ => return false,
    };
    year != 0 && day != 0 && day <= days
}

fn date_day(value: &str) -> Option<i64> {
    valid_date(value).then_some(())?;
    let mut year = value[0..4].parse::<i64>().ok()?;
    let month = value[5..7].parse::<i64>().ok()?;
    let day = value[8..10].parse::<i64>().ok()?;
    year -= i64::from(month <= 2);
    let era = if year >= 0 { year } else { year - 399 } / 400;
    let year_of_era = year - era * 400;
    let shifted_month = month + if month > 2 { -3 } else { 9 };
    let day_of_year = (153 * shifted_month + 2) / 5 + day - 1;
    Some(
        era * 146_097 + year_of_era * 365 + year_of_era / 4 - year_of_era / 100 + day_of_year
            - 719_468,
    )
}

fn previous_date(value: &str) -> String {
    let year = value[0..4].parse::<u16>().expect("validated year");
    let month = value[5..7].parse::<u8>().expect("validated month");
    let day = value[8..10].parse::<u8>().expect("validated day");
    if day > 1 {
        return format!("{year:04}-{month:02}-{:02}", day - 1);
    }
    let (year, month) = if month == 1 {
        (year - 1, 12)
    } else {
        (year, month - 1)
    };
    let leap = year % 4 == 0 && (year % 100 != 0 || year % 400 == 0);
    let day = match month {
        4 | 6 | 9 | 11 => 30,
        2 if leap => 29,
        2 => 28,
        _ => 31,
    };
    format!("{year:04}-{month:02}-{day:02}")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(value: &str) -> Result<Request, ()> {
        serde_json::from_str::<Request>(value)
            .map_err(|_| ())?
            .validated()
    }

    #[test]
    fn request_preserves_terms_and_separates_search_properties() {
        let request = parse(
            r#"{"terms":" notizie Rust 10 giorni fa ","category":"news","language":"it-IT","reference_date":"2026-08-24","published":{"from":"2026-08-14","to":"2026-08-15"}}"#,
        )
        .unwrap();
        assert_eq!(request.terms(), "notizie Rust 10 giorni fa");
        assert_eq!(request.category(), Category::News);
        assert_eq!(request.language(), "it-IT");
        assert_eq!(request.reference_date(), "2026-08-24");
        assert_eq!(request.published().unwrap().from(), "2026-08-14");
        assert_eq!(request.published().unwrap().to(), "2026-08-15");
    }

    #[test]
    fn request_accepts_languages_without_classifying_terms() {
        for language in ["it", "en-GB", "es-419", "de-DE", "ja-JP"] {
            let body = format!(
                r#"{{"terms":"error class latest","category":"web","language":"{language}","reference_date":"2026-08-24","published":null}}"#
            );
            assert!(parse(&body).is_ok(), "rejected {language}");
        }
    }

    #[test]
    fn invalid_language_dates_and_intervals_are_rejected() {
        for body in [
            r#"{"terms":"x","category":"web","language":"","reference_date":"2026-08-24","published":null}"#,
            r#"{"terms":"x","category":"web","language":"it--IT","reference_date":"2026-08-24","published":null}"#,
            r#"{"terms":"x","category":"web","language":"it-IT","reference_date":"2026-02-29","published":null}"#,
            r#"{"terms":"x","category":"news","language":"it-IT","reference_date":"2026-08-24","published":{"from":"2026-08-15","to":"2026-08-15"}}"#,
            r#"{"terms":"x","category":"news","language":"it-IT","reference_date":"2026-08-24","published":{"from":"2026-08-16","to":"2026-08-15"}}"#,
        ] {
            assert!(parse(body).is_err(), "accepted {body}");
        }
    }

    #[test]
    fn publication_intervals_are_half_open_utc_days() {
        let range = Published {
            from: "2026-08-14".into(),
            to: "2026-08-15".into(),
        };
        assert!(
            range.contains(httpdate::parse_http_date("Fri, 14 Aug 2026 23:59:59 GMT").unwrap())
        );
        assert!(
            !range.contains(httpdate::parse_http_date("Sat, 15 Aug 2026 00:00:00 GMT").unwrap())
        );
        assert_eq!(range.duckduckgo_filter(), "2026-08-14..2026-08-14");

        let leap = Published {
            from: "2024-02-28".into(),
            to: "2024-03-01".into(),
        };
        assert_eq!(leap.duckduckgo_filter(), "2024-02-28..2024-02-29");
    }
}
