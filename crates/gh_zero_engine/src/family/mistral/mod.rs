/*
 * gh_zero_engine — Ministral 3 family boundary
 * Validates one mistral3 GGUF contract, resolves the versioned implicit
 * context, and constructs the compile-time-selected CPU, Vulkan, or immutable
 * hybrid backend. It owns no request lifecycle or graph operations.
 */

pub(crate) mod config;
pub(crate) mod decode;
pub(crate) mod detect;
pub(crate) mod graph;
#[cfg(feature = "hybrid")]
pub(crate) mod hybrid;
#[cfg(test)]
mod parity;
pub(crate) mod run;
pub(crate) mod template;
pub(crate) mod tensors;
pub(crate) mod tokenizer;
mod version;
#[cfg(all(test, feature = "vulkan"))]
mod vulkan;

use color_eyre::eyre::Result;

use crate::api::engine::EngineConfig;
#[cfg(any(test, not(feature = "hybrid")))]
use crate::backend::Backend;
use crate::gguf::loader::GgufFile;
#[cfg(any(test, not(feature = "hybrid")))]
use crate::gguf::metadata::ModelMetadata;
use crate::gguf::tensor_index::TensorIndex;

pub use config::MistralConfig;
pub use detect::WeightProfile;
pub(crate) use tensors::MistralTensors;
pub use tokenizer::TekkenTokenizer;

// Validated contract pieces borrowed from the GGUF file. The struct has no
// backend fields: it is the last pure-data gate before future allocation code.
pub(crate) struct MistralContract<'a> {
    pub(crate) config: MistralConfig,
    #[cfg(test)]
    pub(crate) profile: WeightProfile,
    pub(crate) tensors: MistralTensors<'a>,
    pub(crate) tokenizer: TekkenTokenizer,
}

impl<'a> MistralContract<'a> {
    pub(crate) fn from_gguf(file: &'a GgufFile) -> Result<Self> {
        let profile = detect::detect(file.metadata(), file.tensors())?;
        let tokenizer = TekkenTokenizer::from_metadata(file.metadata())?;
        let config = MistralConfig::from_metadata(
            file.metadata(),
            tokenizer.vocab_size(),
            tokenizer.bos_id(),
            tokenizer.eos_id(),
        )?;
        let index = TensorIndex::new(file.tensors());
        let tensors = MistralTensors::build(&config, profile, &index)?;
        Ok(Self {
            config,
            #[cfg(test)]
            profile,
            tensors,
            tokenizer,
        })
    }
}

#[cfg(test)]
pub(crate) struct MistralModel<B: Backend> {
    pub(crate) config: MistralConfig,
    pub(crate) backend: B,
}

#[cfg(test)]
impl<B: Backend> MistralModel<B> {
    pub(crate) fn load(file: &GgufFile, context: usize) -> Result<Self> {
        let contract = MistralContract::from_gguf(file)?;
        let metadata = ModelMetadata::from_gguf(file)?;
        // WeightSource is consumed synchronously: after load, only backend-owned
        // buffers remain, including one allocation for tied embedding/output.
        let backend = B::load(&metadata, &contract.tensors, file, context)?;
        Ok(Self {
            config: contract.config,
            backend,
        })
    }
}

pub(crate) struct RuntimeModel {
    pub(crate) config: MistralConfig,
    pub(crate) tokenizer: TekkenTokenizer,
    pub(crate) context: usize,
    pub(crate) scheme: crate::kv_cache::scheme::KvQuant,
    #[cfg(all(feature = "cpu", not(feature = "hybrid")))]
    pub(crate) backend: crate::backend::cpu::CpuBackend,
    #[cfg(all(feature = "vulkan", not(feature = "cpu")))]
    pub(crate) backend: crate::backend::vulkan::VulkanBackend,
    #[cfg(feature = "hybrid")]
    pub(crate) backend: hybrid::LoadedHybrid,
}

impl RuntimeModel {
    pub(crate) fn load(file: &GgufFile, settings: &EngineConfig) -> Result<Self> {
        let contract = MistralContract::from_gguf(file)?;
        let context = resolve_context(settings.context_tokens, contract.config.context_length)?;
        #[cfg(all(feature = "cpu", not(feature = "hybrid")))]
        let backend = {
            let metadata = ModelMetadata::from_gguf(file)?;
            crate::backend::cpu::CpuBackend::load(&metadata, &contract.tensors, file, context)?
        };
        #[cfg(all(feature = "vulkan", not(feature = "cpu")))]
        let backend = {
            let metadata = ModelMetadata::from_gguf(file)?;
            crate::backend::vulkan::VulkanBackend::load(
                &metadata,
                &contract.tensors,
                file,
                context,
            )?
        };
        #[cfg(feature = "hybrid")]
        let backend = hybrid::loader::load(
            file,
            &contract,
            context,
            settings.kv_quant,
            settings.vram_weights_percent,
            settings.vram_reserve_mib,
        )?;
        Ok(Self {
            config: contract.config,
            tokenizer: contract.tokenizer,
            context,
            scheme: settings.kv_quant,
            backend,
        })
    }

    pub(crate) fn context_limit(&self) -> u32 {
        self.context as u32
    }
}

fn resolve_context(requested: Option<usize>, model_maximum: usize) -> Result<usize> {
    let context = requested.unwrap_or_else(|| version::DEFAULT_CONTEXT.min(model_maximum));
    if context == 0 || context > model_maximum {
        color_eyre::eyre::bail!(
            "E17 context {context} does not fit the selected backend; context was not reduced"
        );
    }
    Ok(context)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn context_resolution_caps_only_the_implicit_context() {
        assert_eq!(resolve_context(None, 262_144).unwrap(), 32_768);
        assert_eq!(resolve_context(None, 8_192).unwrap(), 8_192);
        assert_eq!(resolve_context(None, 32_768).unwrap(), 32_768);
        assert_eq!(resolve_context(Some(32_769), 262_144).unwrap(), 32_769);
    }

    #[test]
    fn context_resolution_rejects_invalid_explicit_context_without_reduction() {
        for requested in [0, 262_145] {
            let error = resolve_context(Some(requested), 262_144)
                .unwrap_err()
                .to_string();
            assert_eq!(
                error,
                format!(
                    "E17 context {requested} does not fit the selected backend; context was not reduced"
                )
            );
        }
    }
}
