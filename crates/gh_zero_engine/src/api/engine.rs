/*
 * gh_zero_engine — persistent public engine
 * Applies process-wide CPU settings before loading the single Ministral family,
 * owns its immutable placement for the session, and submits cancellation-safe
 * text requests to the family lifecycle.
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
        #[cfg(feature = "cpu")]
        {
            crate::backend::cpu::parallel::set_threads(config.cpu_threads);
            crate::backend::cpu::set_no_simd(config.no_attn_simd);
        }
        #[cfg(all(feature = "vulkan", not(feature = "hybrid")))]
        {
            crate::backend::vulkan::set_weights_percent(config.vram_weights_percent);
            crate::backend::vulkan::set_reserve_mib(config.vram_reserve_mib);
        }
        Ok(Self {
            model: family::load(model_path, &config)?,
        })
    }

    pub fn context_limit(&self) -> u32 {
        self.model.context_limit()
    }

    pub fn placement(&self) -> Option<PlacementReport> {
        #[cfg(feature = "hybrid")]
        {
            let plan = &self.model.backend.plan;
            let memory = |bytes: crate::family::mistral::hybrid::BackendBytes| BackendMemory {
                weights: bytes.weights,
                kv: bytes.kv,
                scratch: bytes.scratch,
                fixed: bytes.fixed,
                staging: bytes.staging,
                crossing: bytes.crossing,
                reserve: bytes.reserve,
                total: bytes.total,
            };
            Some(PlacementReport {
                mode: plan.mode.name(),
                cpu_layers: plan.cpu_layers,
                gpu_layers: plan.gpu_layers,
                cpu: memory(plan.cpu),
                gpu: memory(plan.gpu),
            })
        }
        #[cfg(not(feature = "hybrid"))]
        {
            None
        }
    }

    pub fn generate(&self, request: Request, sink: &mut dyn EventSink) {
        mistral::run::generate(&self.model, request, sink);
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
