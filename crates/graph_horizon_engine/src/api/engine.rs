/*
 * graph_horizon_engine — persistent public engine
 * Applies statically selected backend settings, owns one selected family model,
 * reports immutable memory/placement, and submits cancellation-safe requests.
 */

use std::path::Path;

use color_eyre::eyre::Result;

use super::request::{EventSink, Request, SamplingParams};
use crate::family;
use crate::kv_cache::scheme::KvQuant;

pub struct EngineConfig {
    pub context_tokens: Option<usize>,
    pub vram_weights_percent: Option<u8>,
    pub vram_reserve_mib: Option<u64>,
    pub cpu_threads: Option<usize>,
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

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ModelMemory {
    pub weights: u64,
    pub kv: u64,
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
            kv_quant: KvQuant::F16,
        }
    }
}

pub struct Engine {
    model: family::Model,
}

impl Engine {
    pub fn new(model_path: &Path, config: EngineConfig) -> Result<Self> {
        crate::backend::selection::configure(config.cpu_threads);
        Ok(Self {
            model: family::load(model_path, &config)?,
        })
    }

    pub fn context_limit(&self) -> u32 {
        self.model.context_limit()
    }

    pub fn model_name(&self) -> Option<&str> {
        self.model.name()
    }

    pub fn backend_name(&self) -> &'static str {
        #[cfg(feature = "cpu")]
        return "cpu";
        #[cfg(feature = "vulkan")]
        return "vulkan";
        #[cfg(feature = "vulkan-hybrid")]
        return "vulkan-hybrid";
        #[cfg(feature = "metal")]
        return "metal";
        #[cfg(feature = "metal-hybrid")]
        return "metal-hybrid";
        #[cfg(feature = "cuda")]
        return "cuda";
    }

    // Planned retained weights and full-context KV capacity. This is immutable
    // load-time accounting, not process RSS or live allocator telemetry.
    pub fn memory(&self) -> ModelMemory {
        self.model.memory()
    }

    pub fn default_sampling(&self) -> SamplingParams {
        self.model.default_sampling()
    }

    pub fn placement(&self) -> Option<PlacementReport> {
        self.model.placement()
    }

    pub fn generate(&self, request: Request, sink: &mut dyn EventSink) {
        self.model.generate(request, sink);
    }

    pub fn generate_cached(&self, cache_key: [u8; 16], request: Request, sink: &mut dyn EventSink) {
        self.model.generate_cached(cache_key, request, sink);
    }

    pub(crate) fn validate_parity(
        &self,
        prompt_ids: &str,
        completion_ids: &str,
    ) -> Result<crate::harness::ParityReport> {
        self.model.validate_parity(prompt_ids, completion_ids)
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
