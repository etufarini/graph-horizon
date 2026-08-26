/*
 * graph_horizon_engine — closed model-family dispatcher
 * Opens an untrusted model without exposing its path, parses GGUF once, and
 * selects one family from validated architecture metadata. Family-specific
 * validation, policy, resources, and request lifecycles remain in that family.
 */

pub(crate) mod mistral;

use std::path::Path;

use color_eyre::eyre::{Result, bail, eyre};

use crate::api::engine::{EngineConfig, ModelMemory, PlacementReport};
use crate::api::request::{EventSink, Request, SamplingParams};
use crate::gguf::loader::GgufFile;

pub(crate) enum Model {
    Mistral(mistral::RuntimeModel),
}

pub(crate) fn load(model_path: &Path, config: &EngineConfig) -> Result<Model> {
    if std::fs::File::open(model_path).is_err() {
        return Err(eyre!("E01 model file is missing or unreadable"));
    }
    let file = GgufFile::open(model_path).map_err(|_| eyre!("E02 invalid GGUF file"))?;
    let architecture = file
        .metadata()
        .get("general.architecture")
        .and_then(|value| value.as_str())
        .ok_or_else(|| eyre!("E06 missing or invalid GGUF metadata 'general.architecture'"))?;
    match architecture {
        "mistral3" => Ok(Model::Mistral(mistral::RuntimeModel::load(&file, config)?)),
        other => bail!("E03 unsupported architecture '{other}'; supported architecture: mistral3"),
    }
}

impl Model {
    pub(crate) fn context_limit(&self) -> u32 {
        match self {
            Self::Mistral(model) => model.context_limit(),
        }
    }

    pub(crate) fn name(&self) -> Option<&str> {
        match self {
            Self::Mistral(model) => model.name.as_deref(),
        }
    }

    pub(crate) fn memory(&self) -> ModelMemory {
        match self {
            Self::Mistral(model) => model.memory,
        }
    }

    pub(crate) fn default_sampling(&self) -> SamplingParams {
        match self {
            Self::Mistral(model) => model.default_sampling(),
        }
    }

    pub(crate) fn placement(&self) -> Option<PlacementReport> {
        match self {
            Self::Mistral(model) => model.placement(),
        }
    }

    pub(crate) fn generate(&self, request: Request, sink: &mut dyn EventSink) {
        match self {
            Self::Mistral(model) => mistral::generation::generate(model, request, sink),
        }
    }

    pub(crate) fn generate_cached(
        &self,
        cache_key: [u8; 16],
        request: Request,
        sink: &mut dyn EventSink,
    ) {
        match self {
            Self::Mistral(model) => model.generate_cached(cache_key, request, sink),
        }
    }

    pub(crate) fn validate_parity(
        &self,
        prompt_ids: &str,
        completion_ids: &str,
    ) -> Result<crate::harness::ParityReport> {
        match self {
            Self::Mistral(model) => mistral::parity::validate(model, prompt_ids, completion_ids),
        }
    }
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
