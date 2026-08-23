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

const ENDPOINT: &str = "https://lite.duckduckgo.com/lite/";
const MAX_RESPONSE_BYTES: usize = 512 * 1024;
const PROCESS_TIMEOUT: Duration = Duration::from_secs(9);

#[derive(Clone, Copy)]
pub(super) struct Client;

impl Client {
    pub(super) fn new() -> Client {
        Client
    }

    pub(super) async fn fetch(&self, query: &str) -> Result<String, ()> {
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
                "--header",
                "Content-Type: application/x-www-form-urlencoded",
                "--data-raw",
                &form(query),
                ENDPOINT,
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
}

fn form(query: &str) -> String {
    const HEX: &[u8; 16] = b"0123456789ABCDEF";
    let mut value = String::with_capacity(query.len() * 3 + 24);
    value.push_str("q=");
    for byte in query.bytes() {
        if byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'.' | b'_' | b'~') {
            value.push(char::from(byte));
        } else {
            value.push('%');
            value.push(char::from(HEX[(byte >> 4) as usize]));
            value.push(char::from(HEX[(byte & 0x0f) as usize]));
        }
    }
    value.push_str("&kl=wt-wt&kp=1");
    value
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn query_is_form_encoded_without_curl_file_syntax() {
        assert_eq!(form("a b&@\0é"), "q=a%20b%26%40%00%C3%A9&kl=wt-wt&kp=1");
    }
}
