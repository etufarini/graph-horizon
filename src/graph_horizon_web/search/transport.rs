/*
 * Bounded search-provider transport
 * Runs curl without a shell or user configuration against provider-built
 * fixed HTTPS requests, enforcing process, time, status, and body limits.
 */

use std::process::Stdio;
use std::time::Duration;

use tokio::io::AsyncReadExt;
use tokio::process::Command;
use tokio::time::timeout;

const MAX_RESPONSE_BYTES: usize = 512 * 1024;
const PROCESS_TIMEOUT: Duration = Duration::from_secs(9);

pub(super) struct Request {
    pub(super) url: String,
    pub(super) form: Vec<(String, String)>,
}

#[derive(Debug, PartialEq, Eq)]
pub(super) enum Error {
    Unavailable,
    Timeout,
    TooLarge,
    Http(u16),
    Invalid,
}

pub(super) async fn fetch(request: Request) -> Result<String, Error> {
    let mut command = Command::new("curl");
    command
        // `-q` must be first so ~/.curlrc cannot widen this fixed request.
        .arg("-q")
        .args([
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
            "=https",
            "--user-agent",
            concat!("graph-horizon/", env!("CARGO_PKG_VERSION")),
            "--write-out",
            "%{http_code}",
        ]);
    for (name, value) in request.form {
        command
            .arg("--data-urlencode")
            .arg(format!("{name}={value}"));
    }
    command
        .arg(request.url)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .kill_on_drop(true);

    let mut child = command.spawn().map_err(|_| Error::Unavailable)?;
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
        if !child
            .wait()
            .await
            .map_err(|_| Error::Unavailable)?
            .success()
        {
            return Err(Error::Unavailable);
        }
        response(output)
    })
    .await
    .map_err(|_| Error::Timeout)?
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

    #[test]
    fn response_separates_the_fixed_status_suffix_from_arbitrary_body_bytes() {
        assert_eq!(response(b"body200".to_vec()), Ok("body".into()));
        assert_eq!(response(b"body429".to_vec()), Err(Error::Http(429)));
        assert_eq!(response(b"20".to_vec()), Err(Error::Invalid));
        assert_eq!(response(vec![0xff, b'2', b'0', b'0']), Err(Error::Invalid));
    }
}
