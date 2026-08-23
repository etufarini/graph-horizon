/*
 * DuckDuckGo Lite transport
 * Runs the project's existing curl prerequisite without a shell or user
 * configuration, bounds its output and lifetime, and returns only UTF-8 HTML
 * from one fixed HTTPS origin. Remote details never escape this module.
 */

use std::process::Stdio;
use std::time::Duration;

use tokio::io::AsyncReadExt;
use tokio::process::Command;
use tokio::time::timeout;
use url::Url;

const ENDPOINT: &str = "https://lite.duckduckgo.com/lite/";
const MAX_RESPONSE_BYTES: usize = 512 * 1024;
const PROCESS_TIMEOUT: Duration = Duration::from_secs(9);

pub(super) async fn fetch(query: &str) -> Result<String, ()> {
    let url = request_url(query);
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
            "524288",
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

fn request_url(query: &str) -> String {
    let mut url = Url::parse(ENDPOINT).expect("fixed search endpoint is valid");
    let today = requests_today(query);
    let mut pairs = url.query_pairs_mut();
    pairs
        .append_pair("q", query)
        .append_pair("kl", "wt-wt")
        .append_pair("kp", "1");
    if today {
        // DuckDuckGo's day filter keeps explicit "today" queries out of almanacs.
        pairs.append_pair("df", "d");
    }
    drop(pairs);
    url.into()
}

fn requests_today(query: &str) -> bool {
    query
        .split(|character: char| !character.is_alphanumeric())
        .any(|word| matches!(word.to_lowercase().as_str(), "oggi" | "today"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn query_is_encoded_into_the_fixed_get_url() {
        assert_eq!(
            request_url("a b&@\0é"),
            "https://lite.duckduckgo.com/lite/?q=a+b%26%40%00%C3%A9&kl=wt-wt&kp=1"
        );
    }

    #[test]
    fn explicit_current_day_queries_receive_the_day_filter() {
        assert!(request_url("Cosa è successo OGGI?").ends_with("&df=d"));
        assert!(request_url("News today").ends_with("&df=d"));
        assert!(!request_url("History magazine").contains("&df=d"));
    }
}
