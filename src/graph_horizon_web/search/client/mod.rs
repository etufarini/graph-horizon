/*
 * Bounded Web search transport
 * Runs curl without a shell or user configuration against fixed HTTPS search
 * origins. It owns provider URLs and one intent-specific fallback request while
 * query classification remains isolated in the adjacent module.
 */

use std::process::Stdio;
use std::time::Duration;

use tokio::io::AsyncReadExt;
use tokio::process::Command;
use tokio::time::timeout;
use url::Url;

mod query;

const ENDPOINT: &str = "https://lite.duckduckgo.com/lite/";
const NEWS_ENDPOINT: &str = "https://news.google.com/rss/search";
const CODE_ENDPOINT: &str = "https://search.brave.com/search";
const MAX_RESPONSE_BYTES: usize = 512 * 1024;
const PROCESS_TIMEOUT: Duration = Duration::from_secs(9);

pub(super) async fn fetch(query: &str, date: &str) -> Result<String, ()> {
    fetch_url(&request_url(query, date)).await
}

pub(super) enum Fallback {
    News(String),
    Code(String),
}

pub(super) async fn fetch_fallback(query: &str, date: &str) -> Result<Option<Fallback>, ()> {
    match query::intent(query) {
        query::Intent::News(_) => match news_url(query, date) {
            Some(url) => fetch_url(&url).await.map(|body| Some(Fallback::News(body))),
            None => Ok(None),
        },
        query::Intent::Code(_) => match code_url(query) {
            Some(url) => fetch_url(&url).await.map(|body| Some(Fallback::Code(body))),
            None => Ok(None),
        },
        query::Intent::General => Ok(None),
    }
}

async fn fetch_url(url: &str) -> Result<String, ()> {
    let mut command = Command::new("curl");
    command
        // `-q` must be first so ~/.curlrc cannot widen this fixed request.
        .arg("-q")
        .args([
            "--silent",
            "--fail",
            "--compressed",
            "--connect-timeout",
            "4",
            "--max-time",
            "8",
            "--max-filesize",
        ])
        .arg(MAX_RESPONSE_BYTES.to_string())
        .args([
            "--noproxy",
            "*",
            "--proto",
            "=https",
            "--user-agent",
            concat!("graph-horizon/", env!("CARGO_PKG_VERSION")),
            &url,
        ])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .kill_on_drop(true);

    let mut child = command.spawn().map_err(|_| ())?;
    let stdout = child.stdout.take().ok_or(())?;
    timeout(PROCESS_TIMEOUT, async move {
        let mut body = Vec::new();
        stdout
            .take((MAX_RESPONSE_BYTES + 1) as u64)
            .read_to_end(&mut body)
            .await
            .map_err(|_| ())?;
        if body.len() > MAX_RESPONSE_BYTES {
            let _ = child.kill().await;
            let _ = child.wait().await;
            return Err(());
        }
        if !child.wait().await.map_err(|_| ())?.success() {
            return Err(());
        }
        String::from_utf8(body).map_err(|_| ())
    })
    .await
    .map_err(|_| ())?
}

fn request_url(query: &str, date: &str) -> String {
    let mut url = Url::parse(ENDPOINT).expect("fixed search endpoint is valid");
    let intent = query::intent(query);
    let provider_query = query::primary(query, date);
    let mut pairs = url.query_pairs_mut();
    pairs
        .append_pair("q", &provider_query)
        .append_pair("kl", "wt-wt")
        .append_pair("kp", "1");
    if matches!(intent, query::Intent::News(_)) {
        // The day filter keeps explicit news and recency queries out of archives.
        pairs.append_pair("df", "d");
    }
    drop(pairs);
    url.into()
}

fn news_url(query: &str, date: &str) -> Option<String> {
    let (provider_query, language) = query::news(query, date)?;
    let mut url = Url::parse(NEWS_ENDPOINT).expect("fixed news endpoint is valid");
    let mut pairs = url.query_pairs_mut();
    pairs.append_pair("q", &format!("{provider_query} when:1d"));
    match language {
        query::Language::Italian => {
            pairs
                .append_pair("hl", "it")
                .append_pair("gl", "IT")
                .append_pair("ceid", "IT:it");
        }
        query::Language::English => {
            pairs
                .append_pair("hl", "en-US")
                .append_pair("gl", "US")
                .append_pair("ceid", "US:en");
        }
    }
    drop(pairs);
    Some(url.into())
}

fn code_url(query: &str) -> Option<String> {
    let (provider_query, language) = query::code(query)?;
    let mut url = Url::parse(CODE_ENDPOINT).expect("fixed code endpoint is valid");
    let mut pairs = url.query_pairs_mut();
    pairs
        .append_pair("q", &provider_query)
        .append_pair("source", "web")
        // Technical documentation is predominantly indexed in English.
        .append_pair("search_lang", "en")
        .append_pair(
            "country",
            match language {
                query::Language::Italian => "it",
                query::Language::English => "us",
            },
        );
    drop(pairs);
    Some(url.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn query_is_encoded_into_the_fixed_get_url() {
        assert_eq!(
            request_url("a b&@\0é", "2026-08-24"),
            "https://lite.duckduckgo.com/lite/?q=a+b%26%40%00%C3%A9&kl=wt-wt&kp=1"
        );
    }

    #[test]
    fn recent_news_queries_receive_a_localized_date_and_day_filter() {
        let italian = Url::parse(&request_url(
            "Cosa è successo OGGI a Viterbo?",
            "2026-08-24",
        ))
        .unwrap();
        assert_eq!(
            italian.query_pairs().find(|(key, _)| key == "q").unwrap().1,
            "notizie 24 agosto 2026 Cosa è successo OGGI a Viterbo?"
        );
        assert!(
            italian
                .query_pairs()
                .any(|(key, value)| key == "df" && value == "d")
        );

        let english = Url::parse(&request_url(
            "What happened today in Viterbo?",
            "2026-08-24",
        ))
        .unwrap();
        assert_eq!(
            english.query_pairs().find(|(key, _)| key == "q").unwrap().1,
            "news August 24 2026 What happened today in Viterbo?"
        );
        assert!(
            english
                .query_pairs()
                .any(|(key, value)| key == "df" && value == "d")
        );

        assert!(!request_url("History magazine", "2026-08-24").contains("&df=d"));
    }

    #[test]
    fn news_fallback_is_bilingual_and_news_only() {
        let italian = Url::parse(&news_url("notizie Viterbo", "2026-08-24").unwrap()).unwrap();
        assert_eq!(italian.host_str(), Some("news.google.com"));
        assert!(italian.query_pairs().any(|(key, value)| {
            key == "q" && value == "notizie 24 agosto 2026 notizie Viterbo when:1d"
        }));
        assert!(
            italian
                .query_pairs()
                .any(|(key, value)| key == "ceid" && value == "IT:it")
        );

        let english = Url::parse(&news_url("Viterbo news", "2026-08-24").unwrap()).unwrap();
        assert!(english.query_pairs().any(|(key, value)| {
            key == "q" && value == "news August 24 2026 Viterbo news when:1d"
        }));
        assert!(
            english
                .query_pairs()
                .any(|(key, value)| key == "ceid" && value == "US:en")
        );

        assert!(news_url("Viterbo history", "2026-08-24").is_none());
    }

    #[test]
    fn code_fallback_uses_fixed_brave_search_with_language_country() {
        let italian = Url::parse(&code_url("Come uso Tokio in Rust?").unwrap()).unwrap();
        assert_eq!(italian.host_str(), Some("search.brave.com"));
        assert!(italian.query_pairs().any(|(key, value)| {
            key == "q" && value == "Come uso Tokio in Rust? official documentation"
        }));
        assert!(
            italian
                .query_pairs()
                .any(|(key, value)| key == "country" && value == "it")
        );

        let english = Url::parse(&code_url("How do I use Tokio in Rust?").unwrap()).unwrap();
        assert!(
            english
                .query_pairs()
                .any(|(key, value)| key == "country" && value == "us")
        );
        assert!(code_url("History of Rome").is_none());
    }
}
