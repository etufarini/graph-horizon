/*
 * Bounded Web search transport
 * Runs curl without a shell or user configuration against fixed HTTPS search
 * origins. It owns provider URLs and one category-specific fallback request;
 * provider preferences never rewrite or classify the user's search terms.
 */

use std::process::Stdio;
use std::time::Duration;

use tokio::io::AsyncReadExt;
use tokio::process::Command;
use tokio::time::timeout;
use url::Url;

use super::request::{Category, Request};

const ENDPOINT: &str = "https://lite.duckduckgo.com/lite/";
const NEWS_ENDPOINT: &str = "https://news.google.com/rss/search";
const CODE_ENDPOINT: &str = "https://search.brave.com/search";
const MAX_RESPONSE_BYTES: usize = 512 * 1024;
const PROCESS_TIMEOUT: Duration = Duration::from_secs(9);

pub(super) async fn fetch(request: &Request) -> Result<String, ()> {
    fetch_url(&request_url(request)).await
}

pub(super) enum Fallback {
    News(String),
    Code(String),
}

pub(super) async fn fetch_fallback(request: &Request) -> Result<Option<Fallback>, ()> {
    match request.category() {
        Category::News => fetch_url(&news_url(request))
            .await
            .map(|body| Some(Fallback::News(body))),
        Category::Web => fetch_url(&code_url(request))
            .await
            .map(|body| Some(Fallback::Code(body))),
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

fn request_url(request: &Request) -> String {
    let mut url = Url::parse(ENDPOINT).expect("fixed search endpoint is valid");
    let mut pairs = url.query_pairs_mut();
    pairs
        .append_pair("q", request.terms())
        .append_pair("kl", &duckduckgo_region(request.language()))
        .append_pair("kp", "1");
    drop(pairs);
    url.into()
}

fn news_url(request: &Request) -> String {
    let mut url = Url::parse(NEWS_ENDPOINT).expect("fixed news endpoint is valid");
    let mut pairs = url.query_pairs_mut();
    let query = match request.published() {
        Some(range) => format!(
            "{} after:{} before:{}",
            request.terms(),
            range.from(),
            range.to()
        ),
        None => request.terms().to_string(),
    };
    let (language, region) = language_region(request.language());
    pairs
        .append_pair("q", &query)
        .append_pair("hl", request.language())
        .append_pair("gl", &region)
        .append_pair("ceid", &format!("{region}:{language}"));
    drop(pairs);
    url.into()
}

fn code_url(request: &Request) -> String {
    let mut url = Url::parse(CODE_ENDPOINT).expect("fixed code endpoint is valid");
    let mut pairs = url.query_pairs_mut();
    let (language, region) = language_region(request.language());
    pairs
        .append_pair("q", request.terms())
        .append_pair("source", "web")
        .append_pair("search_lang", &language)
        .append_pair("country", &region.to_ascii_lowercase());
    drop(pairs);
    url.into()
}

fn language_region(language: &str) -> (String, String) {
    let mut parts = language.split('-');
    let primary = parts.next().unwrap_or("en").to_ascii_lowercase();
    let region = parts
        .find(|part| part.len() == 2 && part.bytes().all(|byte| byte.is_ascii_alphabetic()))
        .map(str::to_ascii_uppercase)
        .unwrap_or_else(|| match primary.as_str() {
            "it" => "IT".into(),
            "en" => "US".into(),
            _ => "US".into(),
        });
    (primary, region)
}

fn duckduckgo_region(language: &str) -> String {
    let (language, region) = language_region(language);
    format!("{}-{language}", region.to_ascii_lowercase())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(json: &str) -> Request {
        serde_json::from_str::<Request>(json)
            .unwrap()
            .validated()
            .unwrap()
    }

    #[test]
    fn query_is_encoded_into_the_fixed_get_url() {
        assert_eq!(
            request_url(&request(
                r#"{"terms":"a b&@\u0000é","category":"web","language":"fr-FR","reference_date":"2026-08-24","published":null}"#
            )),
            "https://lite.duckduckgo.com/lite/?q=a+b%26%40%00%C3%A9&kl=fr-fr&kp=1"
        );
    }

    #[test]
    fn primary_search_preserves_terms_without_keyword_classification() {
        for terms in [
            "latest Rust API changes",
            "error class swift react",
            "noticias de hace diez días",
        ] {
            let json = format!(
                r#"{{"terms":"{terms}","category":"web","language":"es-ES","reference_date":"2026-08-24","published":null}}"#
            );
            let url = Url::parse(&request_url(&request(&json))).unwrap();
            assert_eq!(
                url.query_pairs().find(|(key, _)| key == "q").unwrap().1,
                terms
            );
            assert!(!url.as_str().contains("official+documentation"));
            assert!(!url.query_pairs().any(|(key, _)| key == "df"));
        }
    }

    #[test]
    fn news_fallback_uses_explicit_language_and_half_open_dates() {
        let request = request(
            r#"{"terms":"notizie Viterbo","category":"news","language":"it-IT","reference_date":"2026-08-24","published":{"from":"2026-08-14","to":"2026-08-15"}}"#,
        );
        let url = Url::parse(&news_url(&request)).unwrap();
        assert_eq!(url.host_str(), Some("news.google.com"));
        assert!(url.query_pairs().any(|(key, value)| {
            key == "q" && value == "notizie Viterbo after:2026-08-14 before:2026-08-15"
        }));
        assert!(
            url.query_pairs()
                .any(|(key, value)| key == "ceid" && value == "IT:it")
        );
    }

    #[test]
    fn web_fallback_is_general_and_preserves_non_english_terms() {
        let request = request(
            r#"{"terms":"Come risolvo questo errore Rust?","category":"web","language":"it-IT","reference_date":"2026-08-24","published":null}"#,
        );
        let url = Url::parse(&code_url(&request)).unwrap();
        assert_eq!(url.host_str(), Some("search.brave.com"));
        assert!(
            url.query_pairs()
                .any(|(key, value)| { key == "q" && value == "Come risolvo questo errore Rust?" })
        );
        assert!(
            url.query_pairs()
                .any(|(key, value)| key == "search_lang" && value == "it")
        );
        assert!(
            url.query_pairs()
                .any(|(key, value)| key == "country" && value == "it")
        );
    }
}
