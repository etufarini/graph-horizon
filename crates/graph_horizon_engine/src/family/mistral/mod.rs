/*
 * graph_horizon_engine — Ministral 3 family boundary
 * Applies the pure Q4-only detection gate, validates one mistral3 GGUF
 * contract, resolves context, and constructs the statically selected resource
 * owner. It owns no concrete backend choice or request lifecycle.
 */

pub(crate) mod config;
pub(crate) mod decode;
pub(crate) mod detect;
pub(crate) mod generation;
pub(crate) mod graph;
mod memory;
pub(crate) mod parity;
pub(crate) mod template;
pub(crate) mod tensors;
pub(crate) mod tokenizer;
mod version;

use color_eyre::eyre::Result;

#[cfg(hybrid_backend)]
use crate::api::engine::BackendMemory;
use crate::api::engine::{EngineConfig, ModelMemory, PlacementReport};
use crate::api::request::{EventSink, Request, SamplingParams};
#[cfg(test)]
use crate::backend::Backend;
use crate::backend::selection;
use crate::gguf::loader::GgufFile;
use crate::gguf::loader::GgufValue;
use crate::gguf::metadata::ModelMetadata;
use crate::gguf::tensor_index::TensorIndex;
use crate::runtime::contract::LayeredGraph;

pub use config::MistralConfig;
pub(crate) use tensors::MistralTensors;
pub use tokenizer::TekkenTokenizer;

// Validated contract pieces borrowed from the GGUF file. The struct has no
// backend fields: it is the last pure-data gate before backend allocation.
pub(crate) struct MistralContract<'a> {
    pub(crate) config: MistralConfig,
    pub(crate) tensors: MistralTensors<'a>,
    pub(crate) tokenizer: TekkenTokenizer,
}

impl<'a> MistralContract<'a> {
    pub(crate) fn from_gguf(file: &'a GgufFile) -> Result<Self> {
        detect::detect(file.metadata(), file.tensors())?;
        let tokenizer = TekkenTokenizer::from_metadata(file.metadata())?;
        let config = MistralConfig::from_metadata(
            file.metadata(),
            tokenizer.vocab_size(),
            tokenizer.bos_id(),
            tokenizer.eos_id(),
        )?;
        let index = TensorIndex::new(file.tensors());
        let tensors = MistralTensors::build(&config, &index)?;
        Ok(Self {
            config,
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

pub(crate) struct RuntimeModel {
    pub(crate) name: Option<String>,
    pub(crate) config: MistralConfig,
    pub(crate) tokenizer: TekkenTokenizer,
    pub(crate) context: usize,
    pub(crate) scheme: crate::kv_cache::scheme::KvQuant,
    pub(crate) memory: ModelMemory,
    pub(crate) backend: selection::SelectedBackend,
    #[cfg(any(feature = "vulkan", feature = "metal"))]
    pub(in crate::family::mistral) session_cache:
        std::sync::Mutex<Option<generation::SessionCache>>,
}

impl RuntimeModel {
    pub(crate) fn load(file: &GgufFile, settings: &EngineConfig) -> Result<Self> {
        let name = display_name(file.metadata());
        let contract = MistralContract::from_gguf(file)?;
        let context = resolve_context(settings.context_tokens, contract.config.context_length)?;
        let metadata = ModelMetadata::from_gguf(file)?;
        let shape = graph::MistralGraph::shape(&contract.config);
        let backend = selection::load(
            file,
            &contract.tensors,
            &metadata,
            shape,
            context,
            settings.kv_quant,
            settings.vram_weights_percent,
            settings.vram_reserve_mib,
        )?;
        #[cfg(not(hybrid_backend))]
        let memory = memory::homogeneous(
            &contract.tensors,
            &contract.config,
            context,
            settings.kv_quant,
            &backend,
        )?;
        #[cfg(hybrid_backend)]
        let memory = memory::hybrid(selection::placement(&backend))?;
        Ok(Self {
            name,
            config: contract.config,
            tokenizer: contract.tokenizer,
            context,
            scheme: settings.kv_quant,
            memory,
            backend,
            #[cfg(any(feature = "vulkan", feature = "metal"))]
            session_cache: std::sync::Mutex::new(None),
        })
    }

    pub(crate) fn context_limit(&self) -> u32 {
        self.context as u32
    }

    pub(crate) fn default_sampling(&self) -> SamplingParams {
        if self.tokenizer.uses_reasoning_profile() {
            // Keep the production Reasoning path identical to the qualified policy.
            SamplingParams {
                temperature: 0.7,
                ..SamplingParams::greedy()
            }
        } else {
            SamplingParams::greedy()
        }
    }

    pub(crate) fn placement(&self) -> Option<PlacementReport> {
        #[cfg(hybrid_backend)]
        {
            selection::placement(&self.backend).map(|plan| {
                let memory = |bytes: crate::backend::hybrid::BackendBytes| BackendMemory {
                    weights: bytes.weights,
                    kv: bytes.kv,
                    scratch: bytes.scratch,
                    fixed: bytes.fixed,
                    staging: bytes.staging,
                    crossing: bytes.crossing,
                    reserve: bytes.reserve,
                    total: bytes.total,
                };
                PlacementReport {
                    mode: selection::placement_mode(plan.mode),
                    cpu_layers: plan.cpu_layers,
                    gpu_layers: plan.gpu_layers,
                    cpu: memory(plan.cpu),
                    gpu: memory(plan.gpu),
                }
            })
        }
        #[cfg(not(hybrid_backend))]
        {
            None
        }
    }

    pub(crate) fn generate_cached(
        &self,
        cache_key: [u8; 16],
        request: Request,
        sink: &mut dyn EventSink,
    ) {
        #[cfg(any(feature = "vulkan", feature = "metal"))]
        generation::generate_cached(self, cache_key, request, sink);
        #[cfg(not(any(feature = "vulkan", feature = "metal")))]
        {
            let _ = cache_key;
            generation::generate(self, request, sink);
        }
    }

    pub(crate) fn shape(&self) -> crate::backend::hybrid::weights::runtime::RuntimeShape {
        graph::MistralGraph::shape(&self.config)
    }
}

fn display_name(md: &std::collections::HashMap<String, GgufValue>) -> Option<String> {
    let raw = md.get("general.name").and_then(GgufValue::as_str)?;
    if raw.chars().any(char::is_control) {
        return None;
    }
    let name = raw.trim();
    let length = name.chars().count();
    (length > 0 && length <= 128).then(|| name.to_owned())
}

#[cfg(any(feature = "vulkan", feature = "metal"))]
impl Drop for RuntimeModel {
    fn drop(&mut self) {
        let slot = self
            .session_cache
            .get_mut()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(cache) = slot.take() {
            generation::free_cache(&self.backend, cache);
        }
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

    #[test]
    fn display_name_accepts_bounded_plain_metadata_only() {
        let md = |name: &str| {
            std::collections::HashMap::from([(
                "general.name".into(),
                GgufValue::String(name.into()),
            )])
        };
        assert_eq!(
            display_name(&md("  Ministral 3B  ")).as_deref(),
            Some("Ministral 3B")
        );
        assert_eq!(display_name(&md("\nmodel")), None);
        assert_eq!(display_name(&md(&"x".repeat(129))), None);
        assert_eq!(display_name(&std::collections::HashMap::new()), None);
    }

    #[test]
    fn sampling_defaults_follow_the_validated_chat_profile() {
        let instruct = SamplingParams::greedy();
        assert_eq!(instruct.temperature, 0.0);
        assert_eq!(instruct.repeat_penalty, 1.0);

        let reasoning = SamplingParams {
            temperature: 0.7,
            ..SamplingParams::greedy()
        };
        assert_eq!(reasoning.temperature, 0.7);
        assert_eq!(reasoning.top_p, 1.0);
        assert_eq!(reasoning.top_k, 0);
        assert_eq!(reasoning.min_p, 0.0);
        assert_eq!(reasoning.repeat_penalty, 1.0);
        assert_eq!(reasoning.seed, 0);
    }
}
