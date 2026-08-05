/*
 * graph_horizon_engine — sole startup family loader
 * Opens an untrusted model without exposing its path, parses GGUF once, and
 * constructs the only supported family. Architecture and profile validation
 * remain in mistral's capability contract.
 */

pub(crate) mod mistral;

use std::path::Path;

use color_eyre::eyre::{Result, eyre};

use crate::api::engine::EngineConfig;
use crate::gguf::loader::GgufFile;

pub(crate) fn load(model_path: &Path, config: &EngineConfig) -> Result<mistral::RuntimeModel> {
    if std::fs::File::open(model_path).is_err() {
        return Err(eyre!("E01 model file is missing or unreadable"));
    }
    let file = GgufFile::open(model_path).map_err(|_| eyre!("E02 invalid GGUF file"))?;
    mistral::RuntimeModel::load(&file, config)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_matrix_e01_e02_normalizes_file_failures() {
        let missing = std::env::temp_dir().join(format!(
            "graph_horizon_missing_{}_{}.gguf",
            std::process::id(),
            unique()
        ));
        let e01 = load(&missing, &EngineConfig::default())
            .err()
            .expect("missing file must fail")
            .to_string();
        assert_eq!(e01, "E01 model file is missing or unreadable");
        assert!(!e01.contains(&missing.to_string_lossy().to_string()));

        let malformed = std::env::temp_dir().join(format!(
            "graph_horizon_invalid_{}_{}.gguf",
            std::process::id(),
            unique()
        ));
        std::fs::write(&malformed, b"not a GGUF").unwrap();
        let e02 = load(&malformed, &EngineConfig::default())
            .err()
            .expect("malformed file must fail")
            .to_string();
        let _ = std::fs::remove_file(&malformed);
        assert_eq!(e02, "E02 invalid GGUF file");
        assert!(!e02.contains(&malformed.to_string_lossy().to_string()));
    }

    fn unique() -> u128 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    }
}
