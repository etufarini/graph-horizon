/*
 * DuckDuckGo Lite transport
 * Owns the only outbound Web-search request and accepts one size-bounded UTF-8
 * HTML response from a fixed HTTPS origin. It follows no redirects and exposes
 * neither upstream status nor response content to callers.
 */

use std::time::Duration;

use reqwest::header::CONTENT_TYPE;
use reqwest::redirect::Policy;

const ENDPOINT: &str = "https://lite.duckduckgo.com/lite/";
const MAX_RESPONSE_BYTES: usize = 512 * 1024;
const TIMEOUT: Duration = Duration::from_secs(8);

#[derive(Clone)]
pub(super) struct Client {
    inner: reqwest::Client,
}

impl Client {
    pub(super) fn new() -> Result<Client, reqwest::Error> {
        Ok(Client {
            inner: reqwest::Client::builder()
                .https_only(true)
                .redirect(Policy::none())
                .timeout(TIMEOUT)
                .user_agent(concat!("graph-horizon/", env!("CARGO_PKG_VERSION")))
                .build()?,
        })
    }

    pub(super) async fn fetch(&self, query: &str) -> Result<String, ()> {
        let mut response = self
            .inner
            .post(ENDPOINT)
            .form(&[("q", query), ("kl", "wt-wt"), ("kp", "1")])
            .send()
            .await
            .map_err(|_| ())?;
        if response.status() != reqwest::StatusCode::OK
            || !is_html(response.headers().get(CONTENT_TYPE))
            || response
                .content_length()
                .is_some_and(|length| length > MAX_RESPONSE_BYTES as u64)
        {
            return Err(());
        }

        let mut body = Vec::new();
        while let Some(chunk) = response.chunk().await.map_err(|_| ())? {
            extend(&mut body, &chunk)?;
        }
        String::from_utf8(body).map_err(|_| ())
    }
}

fn is_html(value: Option<&reqwest::header::HeaderValue>) -> bool {
    value
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.split(';').next())
        .is_some_and(|mime| mime.trim().eq_ignore_ascii_case("text/html"))
}

fn extend(body: &mut Vec<u8>, chunk: &[u8]) -> Result<(), ()> {
    let length = body.len().checked_add(chunk.len()).ok_or(())?;
    if length > MAX_RESPONSE_BYTES {
        return Err(());
    }
    body.extend_from_slice(chunk);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn content_type_is_exact_html_with_optional_parameters() {
        assert!(is_html(Some(&"text/html; charset=UTF-8".parse().unwrap())));
        assert!(is_html(Some(&"TEXT/HTML".parse().unwrap())));
        assert!(!is_html(Some(&"application/json".parse().unwrap())));
        assert!(!is_html(None));
    }

    #[test]
    fn decoded_body_limit_accepts_equality_and_rejects_one_byte_over() {
        let mut body = vec![0; MAX_RESPONSE_BYTES - 1];
        assert_eq!(extend(&mut body, &[0]), Ok(()));
        assert_eq!(extend(&mut body, &[0]), Err(()));
    }
}
