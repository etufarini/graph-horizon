/*
 * Graph Horizon CLI Modules - Console - Render - Status
 * Single responsibility: format compact context occupancy, total generation
 * duration, capacity errors, and semantic conversation spans.
 */

use std::time::Duration;

use ratatui::prelude::*;

use crate::graph_horizon_cli::runtime::{
    CapacityError, ContextUsage, GenerationPhase, GenerationStats,
};
use crate::graph_horizon_cli::style;

pub(super) fn generation_status(
    usage: ContextUsage,
    duration: Option<Duration>,
    stats: Option<GenerationStats>,
    phase: Option<GenerationPhase>,
    phase_duration: Option<Duration>,
    streaming: bool,
    width: u16,
) -> String {
    let context = format!(
        "ctx ~{}/{} tok",
        count(usage.estimated_messages),
        count(usage.context_limit)
    );
    if streaming {
        let phase = match phase {
            None => "attesa",
            Some(GenerationPhase::Prefill) => "prefill",
            Some(GenerationPhase::Decode) => "decode",
        };
        let elapsed = phase_duration.unwrap_or_default().as_secs_f64();
        return if width >= 60 {
            format!("{context} | {phase} {elapsed:.1}s")
        } else {
            format!("{phase} {elapsed:.1}s")
        };
    }
    if let Some(stats) = stats {
        let prefill_rate = rate(stats.prefill_tokens, stats.prefill_ms);
        let decode_rate = rate(stats.completion_tokens, stats.decode_ms);
        if width >= 100 {
            return format!(
                "{context} | {} in · {} pf · {} out | pf {:.2}s {prefill_rate} · dec {:.2}s {decode_rate}",
                count(stats.prompt_tokens),
                count(stats.prefill_tokens),
                count(stats.completion_tokens),
                stats.prefill_ms as f64 / 1000.0,
                stats.decode_ms as f64 / 1000.0,
            );
        }
        if width >= 60 {
            return format!(
                "{context} | {} out | pf {prefill_rate} · dec {decode_rate}",
                count(stats.completion_tokens),
            );
        }
        return format!("{} tok · {decode_rate}", count(stats.completion_tokens));
    }
    duration.map_or(context.clone(), |elapsed| {
        format!("{context} gen {:.1}s", elapsed.as_secs_f64())
    })
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

fn rate(tokens: usize, milliseconds: u64) -> String {
    if milliseconds == 0 {
        "—".to_string()
    } else {
        format!("{:.1}/s", tokens as f64 * 1000.0 / milliseconds as f64)
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
            generation_status(
                usage,
                Some(Duration::from_millis(12_440)),
                None,
                None,
                None,
                false,
                80,
            ),
            "ctx ~5.2k/32.8k tok gen 12.4s"
        );
        assert_eq!(
            generation_status(
                ContextUsage {
                    estimated_messages: 0,
                    context_limit: 999,
                },
                None,
                None,
                None,
                None,
                false,
                80,
            ),
            "ctx ~0/999 tok"
        );
    }

    #[test]
    fn context_status_has_no_percentage_or_rate() {
        let label = generation_status(
            ContextUsage {
                estimated_messages: 1000,
                context_limit: 2000,
            },
            Some(Duration::ZERO),
            None,
            None,
            None,
            false,
            80,
        );
        assert!(!label.contains('%'));
        assert!(!label.contains("tok/s"));
    }

    #[test]
    fn status_prioritizes_phase_and_adapts_final_metrics_to_width() {
        let usage = ContextUsage {
            estimated_messages: 1200,
            context_limit: 32_768,
        };
        assert_eq!(
            generation_status(
                usage,
                None,
                None,
                Some(GenerationPhase::Prefill),
                Some(Duration::from_millis(800)),
                true,
                80,
            ),
            "ctx ~1.2k/32.8k tok | prefill 0.8s"
        );
        let stats = GenerationStats {
            prompt_tokens: 1200,
            prefill_tokens: 96,
            completion_tokens: 42,
            prefill_ms: 800,
            decode_ms: 1400,
        };
        let wide = generation_status(usage, None, Some(stats), None, None, false, 120);
        assert!(wide.contains("1.2k in · 96 pf · 42 out"));
        assert!(wide.contains("pf 0.80s 120.0/s"));
        assert!(wide.contains("dec 1.40s 30.0/s"));
        assert_eq!(
            generation_status(usage, None, Some(stats), None, None, false, 40),
            "42 tok · 30.0/s"
        );
    }
}
