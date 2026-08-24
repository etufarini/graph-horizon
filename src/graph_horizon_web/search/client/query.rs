/*
 * Search query classification and enrichment
 * Classifies bounded user text as general, recent-news, or programming intent
 * and adds only deterministic provider hints. It never removes or translates
 * the original query, so user-supplied subjects, versions, and errors survive.
 */

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Language {
    Italian,
    English,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Intent {
    General,
    News(Language),
    Code(Language),
}

pub(super) fn intent(query: &str) -> Intent {
    if let Some(language) = news_language(query) {
        return Intent::News(language);
    }
    if is_code_query(query) {
        return Intent::Code(query_language(query));
    }
    Intent::General
}

pub(super) fn primary(query: &str, date: &str) -> String {
    match intent(query) {
        Intent::News(language) => dated_news(query, date, language).unwrap_or_else(|| query.into()),
        Intent::Code(_) => format!("{query} official documentation"),
        Intent::General => query.into(),
    }
}

pub(super) fn news(query: &str, date: &str) -> Option<(String, Language)> {
    let Intent::News(language) = intent(query) else {
        return None;
    };
    dated_news(query, date, language).map(|query| (query, language))
}

pub(super) fn code(query: &str) -> Option<(String, Language)> {
    let Intent::Code(language) = intent(query) else {
        return None;
    };
    Some((format!("{query} official documentation"), language))
}

fn news_language(query: &str) -> Option<Language> {
    let mut english = false;
    for word in words(query) {
        if ["oggi", "notizie", "ultime", "recenti"]
            .iter()
            .any(|expected| word.eq_ignore_ascii_case(expected))
        {
            return Some(Language::Italian);
        }
        if ["today", "news", "latest", "recent"]
            .iter()
            .any(|expected| word.eq_ignore_ascii_case(expected))
        {
            english = true;
        }
    }
    english.then_some(Language::English)
}

fn is_code_query(query: &str) -> bool {
    words(query).any(|word| {
        [
            "api",
            "cargo",
            "class",
            "code",
            "codice",
            "coding",
            "compile",
            "compilare",
            "compilazione",
            "compiler",
            "css",
            "error",
            "errore",
            "exception",
            "funzione",
            "function",
            "github",
            "html",
            "java",
            "javascript",
            "kotlin",
            "node",
            "npm",
            "php",
            "programmare",
            "programmazione",
            "programming",
            "python",
            "react",
            "ruby",
            "rust",
            "sdk",
            "sql",
            "svelte",
            "swift",
            "tokio",
            "typescript",
            "vue",
        ]
        .iter()
        .any(|expected| word.eq_ignore_ascii_case(expected))
    })
}

fn query_language(query: &str) -> Language {
    if words(query).any(|word| {
        [
            "codice",
            "come",
            "compilare",
            "compilazione",
            "documentazione",
            "eccezione",
            "errore",
            "funzione",
            "libreria",
            "programmare",
            "programmazione",
        ]
        .iter()
        .any(|expected| word.eq_ignore_ascii_case(expected))
    }) {
        Language::Italian
    } else {
        Language::English
    }
}

fn dated_news(query: &str, date: &str, language: Language) -> Option<String> {
    const ITALIAN_MONTHS: [&str; 12] = [
        "gennaio",
        "febbraio",
        "marzo",
        "aprile",
        "maggio",
        "giugno",
        "luglio",
        "agosto",
        "settembre",
        "ottobre",
        "novembre",
        "dicembre",
    ];
    const ENGLISH_MONTHS: [&str; 12] = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ];

    let year = date.get(0..4)?;
    let month = date.get(5..7)?.parse::<usize>().ok()?.checked_sub(1)?;
    let day = date.get(8..10)?.parse::<u8>().ok()?;
    let month = match language {
        Language::Italian => ITALIAN_MONTHS.get(month)?,
        Language::English => ENGLISH_MONTHS.get(month)?,
    };
    Some(match language {
        Language::Italian => format!("notizie {day} {month} {year} {query}"),
        Language::English => format!("news {month} {day} {year} {query}"),
    })
}

fn words(query: &str) -> impl Iterator<Item = &str> {
    query.split(|character: char| !character.is_alphanumeric())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn intents_cover_bilingual_news_code_and_general_queries() {
        assert_eq!(intent("notizie Rust oggi"), Intent::News(Language::Italian));
        assert_eq!(intent("latest Rust news"), Intent::News(Language::English));
        assert_eq!(
            intent("Come uso Tokio in Rust?"),
            Intent::Code(Language::Italian)
        );
        assert_eq!(
            intent("How do I use Tokio in Rust?"),
            Intent::Code(Language::English)
        );
        assert_eq!(intent("History of Rome"), Intent::General);
    }

    #[test]
    fn enrichment_preserves_original_queries() {
        assert_eq!(
            primary("Cosa è successo oggi?", "2026-08-24"),
            "notizie 24 agosto 2026 Cosa è successo oggi?"
        );
        assert_eq!(
            primary("How do I use Tokio?", "2026-08-24"),
            "How do I use Tokio? official documentation"
        );
        assert_eq!(primary("History of Rome", "2026-08-24"), "History of Rome");
    }
}
