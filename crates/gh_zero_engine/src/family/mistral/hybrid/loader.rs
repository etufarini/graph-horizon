/*
 * gh_zero_engine — Ministral hybrid load planning and ownership
 * Owns host/GPU budgets, immutable placement, and backend construction; excludes graph execution.
 */

use color_eyre::eyre::{Result, bail, eyre};

use super::placement::{self, PlacementInput};
use super::weights::{RuntimeBytes, WeightBytes};
use super::{HybridBackends, HybridMode, HybridPlan, LoadedHybrid};
use crate::backend::cpu::CpuBackend;
use crate::backend::source::WeightSelection;
use crate::backend::vulkan::{VulkanBackend, hybrid_device, vram_for_auto};
use crate::family::mistral::{MistralConfig, MistralTensors};
use crate::gguf::loader::GgufFile;
use crate::gguf::metadata::ModelMetadata;
use crate::kv_cache::scheme::KvQuant;

const MIB: u64 = 1024 * 1024;

#[derive(Clone, Copy, Debug)]
pub(crate) struct AvailableMemory {
    pub(crate) cpu: u64,
    pub(crate) gpu: Option<u64>,
}

pub(crate) fn load(
    file: &GgufFile,
    contract: &crate::family::mistral::MistralContract<'_>,
    context: usize,
    scheme: KvQuant,
    weights_percent: Option<u8>,
    reserve_mib: Option<u64>,
) -> Result<LoadedHybrid> {
    let weights_percent = weights_percent.unwrap_or(100);
    // Explicit zero forbids even a transient Vulkan probe.
    let mut device = acquire_device(weights_percent)?;
    let plan = select_plan(
        &contract.config,
        &contract.tensors,
        context,
        scheme,
        weights_percent,
        reserve_mib,
        AvailableMemory {
            cpu: std::fs::read_to_string("/proc/meminfo")
                .ok()
                .and_then(|text| parse_mem_available(&text))
                .map(usable_ram_budget)
                .unwrap_or(0),
            gpu: device.as_ref().map(vram_for_auto),
        },
    )?;
    let metadata = ModelMetadata::from_gguf(file)?;
    let backends = match plan.mode {
        HybridMode::AllGpu => {
            let gpu = VulkanBackend::load_selected(
                device.take().expect("all-GPU plan requires Vulkan"),
                &metadata,
                &contract.tensors,
                file,
                &WeightSelection::full(plan.block_count),
            )?;
            HybridBackends::AllGpu(gpu)
        }
        HybridMode::Mixed => {
            let cpu = CpuBackend::load_selected(
                &metadata,
                &contract.tensors,
                file,
                &WeightSelection {
                    layers: 0..plan.split,
                    embedding: true,
                    tail: false,
                },
            )?;
            // The prefix stays owned until suffix construction completes; errors drop it once.
            let gpu = VulkanBackend::load_selected(
                device.take().expect("mixed plan requires Vulkan"),
                &metadata,
                &contract.tensors,
                file,
                &WeightSelection {
                    layers: plan.split..plan.block_count,
                    embedding: false,
                    tail: true,
                },
            )?;
            HybridBackends::Mixed { cpu, gpu }
        }
        HybridMode::CpuOnly => {
            drop(device);
            HybridBackends::CpuOnly(CpuBackend::load_selected(
                &metadata,
                &contract.tensors,
                file,
                &WeightSelection::full(plan.block_count),
            )?)
        }
    };
    eprintln!(
        "hybrid: mode={} cpu_layers={} gpu_layers={} cpu_bytes={} gpu_bytes={}",
        plan.mode.name(),
        plan.cpu_layers,
        plan.gpu_layers,
        plan.cpu.total,
        plan.gpu.total
    );
    Ok(LoadedHybrid { plan, backends })
}

fn acquire_device(weights_percent: u8) -> Result<Option<crate::backend::vulkan::Device>> {
    if weights_percent == 0 {
        return Ok(None);
    }
    hybrid_device()
}

pub(crate) fn select_plan(
    config: &MistralConfig,
    tensors: &MistralTensors<'_>,
    context: usize,
    scheme: KvQuant,
    weights_percent: u8,
    reserve_mib: Option<u64>,
    available: AvailableMemory,
) -> Result<HybridPlan> {
    if weights_percent > 100 {
        bail!("invalid Vulkan weight percentage");
    }
    let weights = WeightBytes::from_tensors(tensors)?;
    let gpu_enabled = weights_percent > 0 && available.gpu.is_some();
    let gpu_available = available.gpu.unwrap_or(0);
    let reserve = if gpu_enabled {
        reserve_bytes(gpu_available, reserve_mib)?
    } else {
        0
    };
    let weight_total = weights
        .globals
        .all()
        .and_then(|globals| {
            weights
                .layer_range(0..weights.layers.len())?
                .checked_add(globals)
        })
        .ok_or_else(|| eyre!("hybrid placement arithmetic overflow"))?;
    let gpu_weight_available = if gpu_enabled {
        ((weight_total as u128 * weights_percent as u128) / 100)
            .min(gpu_available.saturating_sub(reserve) as u128) as u64
    } else {
        0
    };
    let runtime = RuntimeBytes::new(config, context, scheme)?;

    placement::select(
        &weights,
        PlacementInput {
            cpu_available: available.cpu,
            gpu_available,
            gpu_weight_available,
            gpu_enabled,
            context,
            cpu_kv_per_layer: runtime.kv_per_layer,
            gpu_kv_per_layer: runtime.kv_per_layer,
            cpu_scratch: runtime.scratch,
            cpu_fixed: runtime.logits,
            gpu_host_fixed: runtime.logits,
            gpu_scratch: runtime.scratch,
            gpu_fixed: runtime.gpu_fixed,
            gpu_reserve: reserve,
            crossing: runtime.crossing,
        },
    )
}

fn parse_mem_available(text: &str) -> Option<u64> {
    let mut found = None;
    for line in text.lines() {
        let Some((key, raw)) = line.split_once(':') else {
            continue;
        };
        if key != "MemAvailable" {
            continue;
        }
        if found.is_some() {
            return None;
        }
        let mut fields = raw.split_whitespace();
        let kib = fields.next()?.parse::<u64>().ok()?;
        if fields.next()? != "kB" || fields.next().is_some() {
            return None;
        }
        found = kib.checked_mul(1024);
    }
    found
}

fn usable_ram_budget(available: u64) -> u64 {
    ((available as u128 * 90) / 100) as u64
}

fn reserve_bytes(gpu_available: u64, override_mib: Option<u64>) -> Result<u64> {
    match override_mib {
        Some(mib) => mib
            .checked_mul(MIB)
            .ok_or_else(|| eyre!("hybrid placement arithmetic overflow")),
        None => Ok((256 * MIB).max(gpu_available / 20)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::family::mistral::hybrid::HybridMode;
    use crate::family::mistral::tensors::{MistralLayer, OutputTensor};
    use crate::gguf::tensor_index::{GgmlType, TensorInfo};

    #[test]
    fn explicit_zero_skips_the_vulkan_probe() {
        crate::backend::vulkan::reset_probe_count();
        assert!(acquire_device(0).unwrap().is_none());
        assert_eq!(crate::backend::vulkan::probe_count(), 0);
    }

    fn config() -> MistralConfig {
        MistralConfig {
            block_count: 2,
            context_length: 128,
            embedding_length: 32,
            feed_forward_length: 64,
            head_count: 4,
            kv_head_count: 2,
            key_length: 8,
            value_length: 8,
            q_width: 32,
            k_width: 16,
            v_width: 16,
            attention_width: 32,
            rope_dimension: 8,
            rope_freq_base: 10_000.0,
            rms_epsilon: 0.00001,
            yarn_factor: 1.0,
            yarn_beta_fast: 32.0,
            yarn_beta_slow: 1.0,
            yarn_log_multiplier: 1.0,
            yarn_original_context: 128,
            attention_temperature_scale: 1.0,
            vocab_size: 32,
            bos_id: 1,
            eos_id: 2,
        }
    }

    fn tensor(name: &str, dims: &[u64], ggml_type: GgmlType) -> TensorInfo {
        TensorInfo {
            name: name.into(),
            dims: dims.into(),
            ggml_type,
            offset: 0,
        }
    }

    #[test]
    fn mem_available_is_strict_and_never_counts_swap() {
        assert_eq!(
            parse_mem_available("MemAvailable: 1234 kB\nSwapFree: 999999 kB\n"),
            Some(1_263_616)
        );
        assert_eq!(parse_mem_available("MemTotal: 1234 kB\n"), None);
        assert_eq!(parse_mem_available("MemAvailable: nope kB\n"), None);
        assert_eq!(parse_mem_available("MemAvailable: 1 MB\n"), None);
        assert_eq!(
            parse_mem_available("MemAvailable: 1 kB\nMemAvailable: 2 kB\n"),
            None
        );
        assert_eq!(
            parse_mem_available("MemAvailable: 18014398509481984 kB\n"),
            None
        );
    }

    #[test]
    fn usable_ram_budget_is_exact_and_overflow_safe() {
        for (available, expected) in [
            (0, 0),
            (1, 0),
            (10, 9),
            (11, 9),
            (1_000, 900),
            (u64::MAX, ((u64::MAX as u128 * 90) / 100) as u64),
        ] {
            let budget = usable_ram_budget(available);
            assert_eq!(budget, expected);
            assert!(budget <= available);
        }
    }

    #[test]
    fn reserve_override_and_default_are_checked() {
        assert_eq!(reserve_bytes(16 << 30, None).unwrap(), (16 << 30) / 20);
        assert_eq!(reserve_bytes(1 << 30, None).unwrap(), 256 * MIB);
        assert_eq!(reserve_bytes(8 << 30, Some(512)).unwrap(), 512 * MIB);
        assert!(reserve_bytes(u64::MAX, Some(u64::MAX)).is_err());
    }

    #[test]
    fn loader_plan_honors_zero_one_and_full_weight_percent() {
        let embedding = tensor("token_embd.weight", &[32, 32], GgmlType::Q8_0);
        let norm = tensor("norm.weight", &[32], GgmlType::F32);
        let matrix = tensor("matrix.weight", &[32, 32], GgmlType::Q8_0);
        let layer = || MistralLayer {
            attn_norm: &norm,
            attn_q: &matrix,
            attn_k: &matrix,
            attn_v: &matrix,
            attn_output: &matrix,
            ffn_norm: &norm,
            ffn_gate: &matrix,
            ffn_up: &matrix,
            ffn_down: &matrix,
        };
        let tensors = MistralTensors {
            token_embd: &embedding,
            output_norm: &norm,
            output: OutputTensor::Tied,
            layers: vec![layer(), layer()],
        };
        let roomy = AvailableMemory {
            cpu: 1 << 30,
            gpu: Some(1 << 30),
        };

        let cpu = select_plan(&config(), &tensors, 16, KvQuant::F16, 0, Some(0), roomy)
            .expect("explicit zero has a CPU plan");
        assert_eq!(cpu.mode, HybridMode::CpuOnly);
        assert_eq!(cpu.gpu.total, 0);
        select_plan(
            &config(),
            &tensors,
            16,
            KvQuant::F16,
            0,
            Some(0),
            AvailableMemory {
                // Direct inputs are already usable budgets and must fit exactly.
                cpu: cpu.cpu.total,
                gpu: None,
            },
        )
        .expect("direct placement input is not reduced again");

        let tiny_cap = select_plan(&config(), &tensors, 16, KvQuant::Int8, 1, Some(0), roomy)
            .expect("one percent falls back deterministically");
        assert_eq!(tiny_cap.mode, HybridMode::CpuOnly);

        let gpu = select_plan(&config(), &tensors, 16, KvQuant::F16, 100, Some(0), roomy)
            .expect("full GPU plan");
        assert_eq!(gpu.mode, HybridMode::AllGpu);
        assert_eq!(gpu.cpu_layers, 0);

        let unavailable = select_plan(
            &config(),
            &tensors,
            16,
            KvQuant::F16,
            100,
            Some(0),
            AvailableMemory {
                cpu: 1 << 30,
                gpu: None,
            },
        )
        .expect("unavailable Vulkan falls back before allocation");
        assert_eq!(unavailable.mode, HybridMode::CpuOnly);
    }
}
