/*
 * Graph Orizon web assets
 * The only filesystem boundary for compiled frontend assets. It normalizes
 * untrusted URL paths, rejects traversal, reads files under `dist`, and assigns
 * the minimal MIME types needed by the v1 frontend.
 */

use std::path::{Component, Path, PathBuf};

use color_eyre::eyre::{Result, eyre};

pub(super) struct Assets {
    root: PathBuf,
}

pub(super) struct AssetFile {
    pub bytes: Vec<u8>,
    pub mime: &'static str,
}

impl Assets {
    pub(super) fn new(root: PathBuf) -> Assets {
        Assets { root }
    }

    pub(super) fn ensure_index(&self) -> Result<()> {
        if self.root.join("index.html").is_file() {
            Ok(())
        } else {
            Err(eyre!("web frontend build missing"))
        }
    }

    pub(super) fn index(&self) -> Option<AssetFile> {
        self.load_relative(Path::new("index.html"))
    }

    pub(super) fn asset(&self, uri_path: &str) -> Option<AssetFile> {
        let raw = uri_path.strip_prefix("/assets/")?;
        let relative = safe_relative(raw)?;
        // The URL prefix mirrors the `assets/` directory inside the Vite build,
        // so the on-disk path keeps that directory under the root.
        self.load_relative(&Path::new("assets").join(relative))
    }

    fn load_relative(&self, relative: &Path) -> Option<AssetFile> {
        let path = self.root.join(relative);
        let bytes = std::fs::read(&path).ok()?;
        if !path.is_file() {
            return None;
        }
        Some(AssetFile {
            bytes,
            mime: mime_for(relative),
        })
    }
}

fn safe_relative(raw: &str) -> Option<PathBuf> {
    let decoded = percent_decode(raw)?;
    let path = Path::new(&decoded);
    let mut out = PathBuf::new();
    for component in path.components() {
        match component {
            Component::Normal(part) => out.push(part),
            // Only normal components are allowed; this rejects absolute paths,
            // `..`, `.`, Windows prefixes and encoded traversal after decoding.
            _ => return None,
        }
    }
    if out.as_os_str().is_empty() {
        None
    } else {
        Some(out)
    }
}

fn percent_decode(raw: &str) -> Option<String> {
    let bytes = raw.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' {
            let hi = *bytes.get(i + 1)?;
            let lo = *bytes.get(i + 2)?;
            out.push(from_hex(hi)? * 16 + from_hex(lo)?);
            i += 3;
        } else {
            out.push(bytes[i]);
            i += 1;
        }
    }
    String::from_utf8(out).ok()
}

fn from_hex(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}

fn mime_for(path: &Path) -> &'static str {
    match path.extension().and_then(|ext| ext.to_str()) {
        Some("html") => "text/html; charset=utf-8",
        Some("js") => "text/javascript; charset=utf-8",
        Some("css") => "text/css; charset=utf-8",
        Some("svg") => "image/svg+xml",
        Some("ico") => "image/x-icon",
        _ => "application/octet-stream",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn safe_relative_rejects_traversal() {
        assert!(safe_relative("../Cargo.toml").is_none());
        assert!(safe_relative("%2e%2e/Cargo.toml").is_none());
        assert!(safe_relative("/tmp/file").is_none());
    }

    #[test]
    fn safe_relative_accepts_asset_paths() {
        assert_eq!(safe_relative("app.js").unwrap(), PathBuf::from("app.js"));
        assert_eq!(
            safe_relative("nested/app.css").unwrap(),
            PathBuf::from("nested/app.css")
        );
    }

    #[test]
    fn mime_types_cover_frontend_build_assets() {
        assert_eq!(
            mime_for(Path::new("index.html")),
            "text/html; charset=utf-8"
        );
        assert_eq!(
            mime_for(Path::new("app.js")),
            "text/javascript; charset=utf-8"
        );
        assert_eq!(mime_for(Path::new("style.css")), "text/css; charset=utf-8");
    }
}
