/*
 * Configured JSON search provider
 * Validates one fixed endpoint and optionally loads one bearer token from a
 * permission-restricted file before the listener starts. Secrets never enter
 * process arguments, logs, errors, or browser-visible runtime metadata.
 */

use std::fs;
use std::path::Path;

use color_eyre::eyre::{Result, eyre};
use url::{Host, Url};

use crate::app::args;

#[derive(Clone)]
pub(in crate::graph_horizon_web) struct Config {
    pub(super) endpoint: Url,
    pub(super) bearer: Option<String>,
}

impl Config {
    pub(in crate::graph_horizon_web) fn from_args() -> Result<Option<Config>> {
        let endpoint = args::value("--search-url");
        let key_file = args::value("--search-key-file");
        match (endpoint, key_file) {
            (None, None) => Ok(None),
            (None, Some(_)) => Err(eyre!("--search-key-file requires --search-url")),
            (Some(endpoint), key_file) => {
                let endpoint = valid_endpoint(&endpoint)
                    .ok_or_else(|| eyre!("invalid Web search provider URL"))?;
                let bearer = key_file.as_deref().map(load_bearer).transpose()?;
                Ok(Some(Config { endpoint, bearer }))
            }
        }
    }

    pub(super) fn provider(&self) -> &str {
        self.endpoint
            .host_str()
            .expect("validated endpoint has a host")
    }
}

fn valid_endpoint(value: &str) -> Option<Url> {
    let url = Url::parse(value).ok()?;
    let loopback = matches!(url.host(), Some(Host::Domain("localhost")))
        || matches!(url.host(), Some(Host::Ipv4(ip)) if ip.is_loopback())
        || matches!(url.host(), Some(Host::Ipv6(ip)) if ip.is_loopback());
    if (url.scheme() != "https" && !(url.scheme() == "http" && loopback))
        || url.host().is_none()
        || !url.username().is_empty()
        || url.password().is_some()
        || url.query().is_some()
        || url.fragment().is_some()
    {
        return None;
    }
    Some(url)
}

fn load_bearer(path: &str) -> Result<String> {
    let path = Path::new(path);
    let metadata = fs::metadata(path).map_err(|_| eyre!("Web search key unavailable"))?;
    if !metadata.is_file() {
        return Err(eyre!("Web search key unavailable"));
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        if metadata.permissions().mode() & 0o077 != 0 {
            return Err(eyre!("Web search key file permissions are too broad"));
        }
    }
    let raw = fs::read_to_string(path).map_err(|_| eyre!("Web search key unavailable"))?;
    let token = raw.strip_suffix('\n').unwrap_or(&raw);
    let token = token.strip_suffix('\r').unwrap_or(token);
    if token.is_empty()
        || token.len() > 1_024
        || token.bytes().any(|byte| {
            !byte.is_ascii_alphanumeric()
                && !matches!(byte, b'-' | b'.' | b'_' | b'~' | b'+' | b'/' | b'=')
        })
    {
        return Err(eyre!("invalid Web search key"));
    }
    Ok(token.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn endpoint_requires_https_except_for_loopback() {
        assert!(valid_endpoint("https://search.example/v1/search").is_some());
        assert!(valid_endpoint("http://127.0.0.1:9000/search").is_some());
        assert!(valid_endpoint("http://[::1]:9000/search").is_some());
        for invalid in [
            "http://search.example/search",
            "https://user@search.example/search",
            "https://search.example/search?q=x",
            "file:///tmp/search",
        ] {
            assert!(valid_endpoint(invalid).is_none(), "accepted {invalid}");
        }
    }
}
