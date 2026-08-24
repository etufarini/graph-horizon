/*
 * Bounded JSON-provider transport
 * Executes one fixed configured HTTPS POST without shell, redirects, user curl
 * settings, proxies, or cookies. An optional bearer token reaches curl only
 * through its stdin configuration and never appears in process arguments.
 */

use std::process::Stdio;
use std::time::Duration;

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::process::Command;
use tokio::time::timeout;

use super::config::Config;

const MAX_RESPONSE_BYTES: usize = 512 * 1024;
const PROCESS_TIMEOUT: Duration = Duration::from_secs(9);

#[derive(Debug, PartialEq, Eq)]
pub(super) enum Error {
    Unavailable,
    Timeout,
    TooLarge,
    Http(u16),
    Invalid,
}

pub(super) async fn fetch(config: &Config, body: String) -> Result<String, Error> {
    let mut command = Command::new("curl");
    command
        // `-q` must be first so ~/.curlrc cannot widen this fixed request.
        .arg("-q")
        .args([
            "--config",
            "-",
            "--silent",
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
            "=http,https",
            "--request",
            "POST",
            "--header",
            "content-type: application/json",
            "--header",
            "accept: application/json",
            "--user-agent",
            concat!("graph-horizon/", env!("CARGO_PKG_VERSION")),
            "--write-out",
            "%{http_code}",
            "--data-binary",
        ])
        .arg(body)
        .arg(config.endpoint.as_str())
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .kill_on_drop(true);

    let mut child = command.spawn().map_err(|_| Error::Unavailable)?;
    let mut stdin = child.stdin.take().ok_or(Error::Unavailable)?;
    if let Some(token) = &config.bearer {
        stdin
            .write_all(format!("header = \"Authorization: Bearer {token}\"\n").as_bytes())
            .await
            .map_err(|_| Error::Unavailable)?;
    }
    stdin.shutdown().await.map_err(|_| Error::Unavailable)?;
    // curl parses `--config -` only after EOF; releasing the pipe is the EOF
    // invariant even on runtimes where AsyncWriteExt::shutdown only flushes.
    drop(stdin);
    let stdout = child.stdout.take().ok_or(Error::Unavailable)?;

    timeout(PROCESS_TIMEOUT, async move {
        let mut output = Vec::new();
        stdout
            .take((MAX_RESPONSE_BYTES + 4) as u64)
            .read_to_end(&mut output)
            .await
            .map_err(|_| Error::Unavailable)?;
        if output.len() > MAX_RESPONSE_BYTES + 3 {
            let _ = child.kill().await;
            let _ = child.wait().await;
            return Err(Error::TooLarge);
        }
        let status = child.wait().await.map_err(|_| Error::Unavailable)?;
        if !status.success() {
            return Err(curl_error(status.code()));
        }
        response(output)
    })
    .await
    .map_err(|_| Error::Timeout)?
}

fn curl_error(code: Option<i32>) -> Error {
    match code {
        Some(28) => Error::Timeout,
        Some(63) => Error::TooLarge,
        _ => Error::Unavailable,
    }
}

fn response(mut output: Vec<u8>) -> Result<String, Error> {
    if output.len() < 3 {
        return Err(Error::Invalid);
    }
    let status = output.split_off(output.len() - 3);
    let status = std::str::from_utf8(&status)
        .ok()
        .and_then(|value| value.parse::<u16>().ok())
        .ok_or(Error::Invalid)?;
    if !(200..300).contains(&status) {
        return Err(Error::Http(status));
    }
    String::from_utf8(output).map_err(|_| Error::Invalid)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::net::TcpListener;
    use url::Url;

    #[test]
    fn response_separates_status_from_arbitrary_json_bytes() {
        assert_eq!(
            response(br#"{"results":[]}200"#.to_vec()),
            Ok(r#"{"results":[]}"#.into())
        );
        assert_eq!(response(b"{}429".to_vec()), Err(Error::Http(429)));
        assert_eq!(response(b"20".to_vec()), Err(Error::Invalid));
    }

    #[test]
    fn curl_timeout_and_size_codes_remain_distinct() {
        assert_eq!(curl_error(Some(28)), Error::Timeout);
        assert_eq!(curl_error(Some(63)), Error::TooLarge);
        assert_eq!(curl_error(Some(7)), Error::Unavailable);
        assert_eq!(curl_error(None), Error::Unavailable);
    }

    #[tokio::test]
    async fn fetch_closes_config_input_and_posts_json() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let endpoint = format!("http://{}/search", listener.local_addr().unwrap());
        let server = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut request = Vec::new();
            loop {
                let mut chunk = [0; 4096];
                let count = socket.read(&mut chunk).await.unwrap();
                request.extend_from_slice(&chunk[..count]);
                let Some(headers_end) = request.windows(4).position(|part| part == b"\r\n\r\n")
                else {
                    continue;
                };
                let headers = String::from_utf8_lossy(&request[..headers_end]);
                let length = headers
                    .lines()
                    .find_map(|line| {
                        line.to_ascii_lowercase()
                            .strip_prefix("content-length: ")
                            .map(str::to_string)
                    })
                    .and_then(|value| value.parse::<usize>().ok())
                    .unwrap();
                if request.len() >= headers_end + 4 + length {
                    break;
                }
            }
            socket
                .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 14\r\n\r\n{\"results\":[]}")
                .await
                .unwrap();
            String::from_utf8(request).unwrap()
        });
        let config = Config {
            endpoint: Url::parse(&endpoint).unwrap(),
            bearer: None,
        };

        assert_eq!(
            fetch(&config, r#"{"query":"exact"}"#.into()).await.unwrap(),
            r#"{"results":[]}"#
        );
        assert!(server.await.unwrap().ends_with(r#"{"query":"exact"}"#));
    }
}
