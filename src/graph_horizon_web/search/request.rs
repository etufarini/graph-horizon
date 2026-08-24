/*
 * Provider-neutral Web search request
 * Validates the bounded terms, browser language hint, reference date, and
 * optional half-open publication interval before any provider sees them.
 */

use serde::Deserialize;

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
}
