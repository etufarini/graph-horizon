/*
 * Graph Horizon CLI runtime banner
 * Formats the immutable local model, backend, effective placement, and planned
 * owner totals into scrollable secondary lines. It performs no runtime lookup
 * and never receives model paths or physical-memory information.
 */

use ratatui::prelude::*;

use super::SectionStyle;
use super::wrap::push_wrapped_lines;
use crate::graph_horizon_cli::runtime::RuntimeInfo;

pub(super) fn push_banner(lines: &mut Vec<Line<'static>>, width: u16, info: &RuntimeInfo) {
    push_wrapped_lines(
        lines,
        width,
        SectionStyle::Secondary,
        &format!("Graph Horizon · {}", info.model_name),
    );
    let placement = info.placement.map(|value| {
        format!(
            " · {} · {} CPU / {} GPU",
            value.mode, value.cpu_layers, value.gpu_layers
        )
    });
    push_wrapped_lines(
        lines,
        width,
        SectionStyle::Secondary,
        &format!("{}{}", info.backend, placement.as_deref().unwrap_or("")),
    );
    if let Some(value) = info.placement {
        let accelerator = if info.backend.starts_with("metal") {
            "Metal"
        } else {
            "GPU"
        };
        push_wrapped_lines(
            lines,
            width,
            SectionStyle::Secondary,
            &format!(
                "memoria CPU {} · {accelerator} {}",
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
