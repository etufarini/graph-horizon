/*
 * Graph Horizon Web runtime properties
 * Serializes the loaded model's bounded display name, compile-time backend,
 * effective hybrid placement, and planned allocation bytes for the local Web
 * surface. Paths, device identifiers, and physical memory never enter the DTO.
 */

use graph_horizon_engine::{BackendMemory, Engine, ModelMemory, PlacementReport};
use serde::Serialize;

#[derive(Serialize)]
pub(super) struct RuntimeProperties<'a> {
    model_name: Option<&'a str>,
    backend: &'static str,
    memory: MemorySummary,
    placement: Option<Placement>,
}

#[derive(Serialize)]
struct MemorySummary {
    weights_bytes: String,
    kv_bytes: String,
}

#[derive(Serialize)]
struct Placement {
    mode: &'static str,
    cpu_layers: usize,
    accelerator_layers: usize,
    cpu: Memory,
    accelerator: Memory,
}

#[derive(Serialize)]
struct Memory {
    weights_bytes: String,
    kv_bytes: String,
    scratch_bytes: String,
    fixed_bytes: String,
    staging_bytes: String,
    crossing_bytes: String,
    reserve_bytes: String,
    total_bytes: String,
}

pub(super) fn payload(engine: &Engine) -> RuntimeProperties<'_> {
    fields(
        engine.model_name(),
        engine.backend_name(),
        engine.memory(),
        engine.placement(),
    )
}

fn fields<'a>(
    model_name: Option<&'a str>,
    backend: &'static str,
    model_memory: ModelMemory,
    placement: Option<PlacementReport>,
) -> RuntimeProperties<'a> {
    RuntimeProperties {
        model_name,
        backend,
        memory: MemorySummary {
            weights_bytes: model_memory.weights.to_string(),
            kv_bytes: model_memory.kv.to_string(),
        },
        placement: placement.map(|report| Placement {
            mode: report.mode,
            cpu_layers: report.cpu_layers,
            accelerator_layers: report.gpu_layers,
            cpu: memory(report.cpu),
            accelerator: memory(report.gpu),
        }),
    }
}

fn memory(value: BackendMemory) -> Memory {
    Memory {
        weights_bytes: value.weights.to_string(),
        kv_bytes: value.kv.to_string(),
        scratch_bytes: value.scratch.to_string(),
        fixed_bytes: value.fixed.to_string(),
        staging_bytes: value.staging.to_string(),
        crossing_bytes: value.crossing.to_string(),
        reserve_bytes: value.reserve.to_string(),
        total_bytes: value.total.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn payload_contains_only_public_runtime_fields_and_exact_bytes() {
        let report = PlacementReport {
            mode: "mixed",
            cpu_layers: 12,
            gpu_layers: 20,
            cpu: BackendMemory {
                weights: u64::MAX,
                kv: 42,
                total: u64::MAX,
                ..BackendMemory::default()
            },
            gpu: BackendMemory::default(),
        };
        let value = serde_json::to_value(fields(
            Some("Ministral 3B"),
            "vulkan-hybrid",
            ModelMemory {
                weights: u64::MAX,
                kv: 42,
            },
            Some(report),
        ))
        .unwrap();

        assert_eq!(value["model_name"], "Ministral 3B");
        assert_eq!(value["memory"]["weights_bytes"], u64::MAX.to_string());
        assert_eq!(value["memory"]["kv_bytes"], "42");
        assert_eq!(value["placement"]["cpu_layers"], 12);
        assert_eq!(
            value["placement"]["cpu"]["total_bytes"],
            u64::MAX.to_string()
        );
        let raw = value.to_string();
        for prohibited in ["path", "device", "physical", "available"] {
            assert!(!raw.contains(prohibited));
        }
    }

    #[test]
    fn homogeneous_payload_keeps_memory_without_fabricating_placement() {
        let value = serde_json::to_value(fields(
            None,
            "cpu",
            ModelMemory {
                weights: 10,
                kv: 20,
            },
            None,
        ))
        .unwrap();

        assert_eq!(value["memory"]["weights_bytes"], "10");
        assert_eq!(value["memory"]["kv_bytes"], "20");
        assert!(value["placement"].is_null());
    }
}
