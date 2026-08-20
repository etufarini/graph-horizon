/*
 * Graph Horizon CLI runtime banner
 * Formats immutable local identity, universal weight/KV planning, effective
 * placement, and owner totals into scrollable secondary lines. It performs no
 * runtime lookup and never receives paths or physical-memory information.
 */

use ratatui::prelude::*;

use super::SectionStyle;
use super::wrap::push_wrapped_lines;
use crate::graph_horizon_cli::runtime::RuntimeInfo;

pub(super) fn push_banner(lines: &mut Vec<Line<'static>>, width: u16, info: &RuntimeInfo) {
    let accelerator = if info.backend.starts_with("metal") {
        "Metal"
    } else {
        "GPU"
    };
    push_wrapped_lines(
        lines,
        width,
        SectionStyle::Secondary,
        &format!("Graph Horizon · {}", info.model_name),
    );
    let placement = info.placement.map(|value| {
        format!(
            " · {} · {} CPU / {} {accelerator}",
            value.mode, value.cpu_layers, value.gpu_layers
        )
    });
    push_wrapped_lines(
        lines,
        width,
        SectionStyle::Secondary,
        &format!("{}{}", info.backend, placement.as_deref().unwrap_or("")),
    );
    push_wrapped_lines(
        lines,
        width,
        SectionStyle::Secondary,
        &format!(
            "pesi {} · KV max {}",
            bytes(info.memory.weights),
            bytes(info.memory.kv)
        ),
    );
    if let Some(value) = info.placement {
        push_wrapped_lines(
            lines,
            width,
            SectionStyle::Secondary,
            &format!(
                "budget CPU {} · {accelerator} {}",
                bytes(value.cpu.total),
                bytes(value.gpu.total)
            ),
        );
    }
}

fn bytes(value: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
    let mut unit = 0;
    let mut divisor = 1_u64;
    while unit < UNITS.len() - 1
        && let Some(next) = divisor.checked_mul(1024)
        && value >= next
    {
        divisor = next;
        unit += 1;
    }
    if unit == 0 {
        format!("{value} B")
    } else {
        format!("{:.1} {}", value as f64 / divisor as f64, UNITS[unit])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph_horizon_cli::runtime::ModelMemory;

    #[test]
    fn homogeneous_banner_includes_weight_and_kv_summary() {
        let info = RuntimeInfo {
            model_name: "local".into(),
            backend: "cpu",
            memory: ModelMemory {
                weights: 1024,
                kv: 1536,
            },
            placement: None,
        };
        let mut lines = Vec::new();

        push_banner(&mut lines, 80, &info);

        let text = lines
            .iter()
            .flat_map(|line| line.spans.iter())
            .map(|span| span.content.as_ref())
            .collect::<Vec<_>>()
            .join("");
        assert!(text.contains("pesi 1.0 KiB · KV max 1.5 KiB"));
    }

    #[test]
    fn metal_placement_uses_metal_and_budget_labels() {
        let info = RuntimeInfo {
            model_name: "local".into(),
            backend: "metal-hybrid",
            memory: ModelMemory::default(),
            placement: Some(crate::graph_horizon_cli::runtime::PlacementReport {
                mode: "all-metal",
                cpu_layers: 0,
                gpu_layers: 32,
                cpu: Default::default(),
                gpu: Default::default(),
            }),
        };
        let mut lines = Vec::new();

        push_banner(&mut lines, 80, &info);

        let text = lines
            .iter()
            .flat_map(|line| line.spans.iter())
            .map(|span| span.content.as_ref())
            .collect::<Vec<_>>()
            .join("");
        assert!(text.contains("0 CPU / 32 Metal"));
        assert!(text.contains("budget CPU 0 B · Metal 0 B"));
    }
}
