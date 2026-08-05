/*
 * gh_zero_engine — persistent public engine
 * Applies statically selected backend settings, owns the single Ministral model,
 * reports immutable placement, and submits cancellation-safe text requests.
 */

use std::path::Path;

use color_eyre::eyre::Result;

use super::request::{EventSink, Request};
use crate::family::{self, mistral};
use crate::kv_cache::scheme::KvQuant;

pub struct EngineConfig {
    pub context_tokens: Option<usize>,
    pub vram_weights_percent: Option<u8>,
    pub vram_reserve_mib: Option<u64>,
    pub cpu_threads: Option<usize>,
    pub no_attn_simd: bool,
    pub kv_quant: KvQuant,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BackendMemory {
    pub weights: u64,
    pub kv: u64,
    pub scratch: u64,
    pub fixed: u64,
    pub staging: u64,
    pub crossing: u64,
    pub reserve: u64,
    pub total: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PlacementReport {
    pub mode: &'static str,
    pub cpu_layers: usize,
    pub gpu_layers: usize,
    pub cpu: BackendMemory,
    pub gpu: BackendMemory,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            context_tokens: None,
            vram_weights_percent: None,
            vram_reserve_mib: None,
            cpu_threads: None,
            no_attn_simd: false,
            kv_quant: KvQuant::F16,
        }
    }
}

pub struct Engine {
    model: mistral::RuntimeModel,
}

impl Engine {
    pub fn new(model_path: &Path, config: EngineConfig) -> Result<Self> {
        crate::backend::selection::configure(
            config.cpu_threads,
            config.no_attn_simd,
            config.vram_weights_percent,
            config.vram_reserve_mib,
        );
        Ok(Self {
            model: family::load(model_path, &config)?,
        })
    }

    pub fn context_limit(&self) -> u32 {
        self.model.context_limit()
    }

    pub fn placement(&self) -> Option<PlacementReport> {
        #[cfg(any(feature = "vulkan-hybrid", feature = "metal-hybrid"))]
        {
            crate::backend::selection::placement(&self.model.backend).map(|plan| {
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
                    mode: crate::backend::selection::placement_mode(plan.mode),
                    cpu_layers: plan.cpu_layers,
                    gpu_layers: plan.gpu_layers,
                    cpu: memory(plan.cpu),
                    gpu: memory(plan.gpu),
                }
            })
        }
        #[cfg(not(any(feature = "vulkan-hybrid", feature = "metal-hybrid")))]
        {
            None
        }
    }

    pub fn generate(&self, request: Request, sink: &mut dyn EventSink) {
        mistral::generation::generate(&self.model, request, sink);
    }

    pub(crate) fn validate_parity(
        &self,
        prompt_ids: &str,
        completion_ids: &str,
    ) -> Result<mistral::parity::ParityReport> {
        mistral::parity::validate(&self.model, prompt_ids, completion_ids)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_configuration_is_portable() {
        let config = EngineConfig::default();
        assert_eq!(config.context_tokens, None);
        assert_eq!(config.vram_weights_percent, None);
        assert_eq!(config.kv_quant, KvQuant::F16);
    }
}
