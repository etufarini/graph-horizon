/*
 * Graph Horizon CLI Modules - Console - Render - Status
 * Single responsibility: format compact context occupancy, total generation
 * duration, capacity errors, and semantic conversation spans.
 */

use std::time::Duration;

use ratatui::prelude::*;

use crate::graph_horizon_cli::runtime::{CapacityError, ContextUsage};
use crate::graph_horizon_cli::style;

pub(super) fn context_status(usage: ContextUsage, duration: Option<Duration>) -> String {
    let mut label = format!(
        "ctx ~{}/{} tok",
        count(usage.estimated_messages),
        count(usage.context_limit)
    );
    if let Some(duration) = duration {
        label.push_str(&format!(" gen {:.1}s", duration.as_secs_f64()));
    }
    label
}

pub(super) fn capacity_error(error: CapacityError) -> String {
    format!(
        "Contesto insufficiente: ~{} token + {} riservati superano il budget sicuro di {} token",
        error.estimated_messages, error.max_tokens, error.safe_total_budget
    )
}

fn count(value: usize) -> String {
    if value < 1000 {
        value.to_string()
    } else {
        format!("{:.1}k", value as f64 / 1000.0)
    }
}

// Identifies which visual section a span of text belongs to.
#[derive(Clone, Copy)]
pub(crate) enum SectionStyle {
    Input,
    Secondary,
    Response,
    Hint,
}

// Applies the color and style for the given section to a text string.
pub(crate) fn format_span(section: SectionStyle, text: String) -> Span<'static> {
    let palette = match section {
        SectionStyle::Input => style::Palette::Input,
        SectionStyle::Secondary => style::Palette::Secondary,
        SectionStyle::Response => style::Palette::Response,
        SectionStyle::Hint => style::Palette::Hint,
    };
    style::format(palette, text)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn context_status_formats_exact_counts() {
        let usage = ContextUsage {
            estimated_messages: 5200,
            context_limit: 32_800,
        };
        assert_eq!(
            context_status(usage, Some(Duration::from_millis(12_440))),
            "ctx ~5.2k/32.8k tok gen 12.4s"
        );
        assert_eq!(
            context_status(
                ContextUsage {
                    estimated_messages: 0,
                    context_limit: 999,
                },
                None,
            ),
            "ctx ~0/999 tok"
        );
    }

    #[test]
    fn context_status_has_no_percentage_or_rate() {
        let label = context_status(
            ContextUsage {
                estimated_messages: 1000,
                context_limit: 2000,
            },
            Some(Duration::ZERO),
        );
        assert!(!label.contains('%'));
        assert!(!label.contains("tok/s"));
    }
}
