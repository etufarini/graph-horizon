/*
 * Graph Horizon web config
 * Holds Web bind and static-asset location settings. Installed commands resolve
 * assets from their prefix while source runs retain the documented local dist
 * fallback. Generation capacity remains owned by the loaded engine.
 */

use std::path::PathBuf;

use crate::app::args;

#[derive(Clone)]
pub(super) struct WebConfig {
    pub host: String,
    pub port: String,
    pub asset_root: PathBuf,
}

impl WebConfig {
    pub(super) fn from_args() -> WebConfig {
        WebConfig {
            host: args::value("--host").unwrap_or_else(|| "127.0.0.1".to_string()),
            // Keep the raw port string so invalid values fail at bind time.
            port: args::value("--port").unwrap_or_else(|| "8080".to_string()),
            asset_root: asset_root(std::env::current_exe().ok()),
        }
    }
}

fn asset_root(executable: Option<PathBuf>) -> PathBuf {
    // A complete installed prefix wins; absence retains the source-run contract.
    executable
        .as_deref()
        .and_then(|path| path.parent())
        .and_then(|bindir| bindir.parent())
        .map(|prefix| prefix.join("share/graph-horizon/web"))
        .filter(|root| root.join("index.html").is_file())
        .unwrap_or_else(|| PathBuf::from("web/frontend/dist"))
}

#[cfg(test)]
mod tests {
    use std::fs;

    use super::*;

    #[test]
    fn installed_assets_win_over_the_source_fallback() {
        let fixture =
            std::env::temp_dir().join(format!("graph-horizon-web-config-{}", std::process::id()));
        let assets = fixture.join("share/graph-horizon/web");
        fs::create_dir_all(&assets).unwrap();
        fs::write(assets.join("index.html"), b"fixture").unwrap();

        assert_eq!(asset_root(Some(fixture.join("bin/graph-horizon"))), assets);
        assert_eq!(
            asset_root(Some(fixture.join("other/bin/graph-horizon"))),
            PathBuf::from("web/frontend/dist")
        );
        assert_eq!(asset_root(None), PathBuf::from("web/frontend/dist"));

        fs::remove_dir_all(fixture).unwrap();
    }
}
