/*
 * Provider-neutral Web search request
 * Validates bounded terms, browser language, local reference date, and an
 * optional half-open UTC millisecond interval before any provider sees them.
 */

use serde::{Deserialize, Serialize};

pub(super) const MAX_QUERY_CHARACTERS: usize = 512;
pub(super) const MAX_TIMESTAMP_BOUND_MS: u64 = 253_402_300_800_000;
const MAX_LANGUAGE_CHARACTERS: usize = 35;
const MILLISECONDS_PER_DAY: u64 = 86_400_000;
const LAST_SUPPORTED_DAY: u64 = MAX_TIMESTAMP_BOUND_MS / MILLISECONDS_PER_DAY - 1;

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub(super) enum Category {
    Web,
    News,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(super) struct Published {
    from_ms: u64,
    to_ms: u64,
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
            || self.published.is_some_and(|range| {
                range.from_ms >= range.to_ms || range.to_ms > MAX_TIMESTAMP_BOUND_MS
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

    pub(super) fn published(&self) -> Option<Published> {
        self.published
    }
}

impl Published {
    pub(super) fn start_ms(self) -> u64 {
        self.from_ms
    }

    pub(super) fn end_ms(self) -> u64 {
        self.to_ms
    }

    pub(super) fn contains(self, time_ms: u64) -> bool {
        self.from_ms <= time_ms && time_ms < self.to_ms
    }

    pub(super) fn duckduckgo_filter(self) -> String {
        let first = self.from_ms / MILLISECONDS_PER_DAY;
        let last = self.to_ms.saturating_sub(1) / MILLISECONDS_PER_DAY;
        format!("{}..{}", utc_date(first), utc_date(last))
    }

    pub(super) fn google_dates(self) -> (String, String) {
        let first = (self.from_ms / MILLISECONDS_PER_DAY).saturating_sub(1);
        let after_last =
            (self.to_ms.saturating_sub(1) / MILLISECONDS_PER_DAY + 2).min(LAST_SUPPORTED_DAY);
        (utc_date(first), utc_date(after_last))
    }
}

fn utc_date(days: u64) -> String {
    let z = days as i64 + 719_468;
    let era = z / 146_097;
    let day_of_era = z - era * 146_097;
    let year_of_era =
        (day_of_era - day_of_era / 1_460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let mut year = year_of_era + era * 400;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let shifted_month = (5 * day_of_year + 2) / 153;
    let day = day_of_year - (153 * shifted_month + 2) / 5 + 1;
    let month = shifted_month + if shifted_month < 10 { 3 } else { -9 };
    year += i64::from(month <= 2);
    format!("{year:04}-{month:02}-{day:02}")
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

fn valid_date(value: &str) -> bool {
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

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(value: &str) -> Result<Request, ()> {
        serde_json::from_str::<Request>(value)
            .map_err(|_| ())?
            .validated()
    }

    #[test]
    fn request_preserves_explicit_properties() {
        let request = parse(r#"{"terms":" current Rust ","category":"news","language":"it-IT","reference_date":"2026-08-24","published":{"from_ms":1787522400000,"to_ms":1787608800000}}"#).unwrap();
        assert_eq!(request.terms(), "current Rust");
        assert_eq!(request.category(), Category::News);
        assert_eq!(request.published().unwrap().start_ms(), 1_787_522_400_000);
    }

    #[test]
    fn invalid_properties_are_rejected() {
        for body in [
            r#"{"terms":"x","category":"web","language":"","reference_date":"2026-08-24","published":null}"#,
            r#"{"terms":"x","category":"web","language":"it--IT","reference_date":"2026-08-24","published":null}"#,
            r#"{"terms":"x","category":"web","language":"it-IT","reference_date":"2026-02-29","published":null}"#,
            r#"{"terms":"x","category":"news","language":"it-IT","reference_date":"2026-08-24","published":{"from_ms":2,"to_ms":2}}"#,
            r#"{"terms":"x","category":"news","language":"it-IT","reference_date":"2026-08-24","published":{"from_ms":2,"to_ms":253402300800001}}"#,
        ] {
            assert!(parse(body).is_err(), "accepted {body}");
        }
    }

    #[test]
    fn interval_is_half_open() {
        let range = Published {
            from_ms: 100,
            to_ms: 200,
        };
        assert!(range.contains(100));
        assert!(range.contains(199));
        assert!(!range.contains(200));
    }

    #[test]
    fn provider_dates_cover_the_exact_millisecond_interval() {
        let range = Published {
            from_ms: 1_787_522_400_000,
            to_ms: 1_787_608_800_000,
        };
        assert_eq!(range.duckduckgo_filter(), "2026-08-23..2026-08-24");
        assert_eq!(
            range.google_dates(),
            ("2026-08-22".into(), "2026-08-26".into())
        );

        let upper = Published {
            from_ms: MAX_TIMESTAMP_BOUND_MS - MILLISECONDS_PER_DAY,
            to_ms: MAX_TIMESTAMP_BOUND_MS,
        };
        assert_eq!(upper.google_dates().1, "9999-12-31");
    }
}
