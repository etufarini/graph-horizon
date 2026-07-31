/*
 * GH Zero CLI Modules - Console - Render - Status
 * Status-bar indicators and section styling: token status, speed/count labels, span colors.
 * Pure formatting from runtime metrics; the draw module composes these into the bottom row.
 */

use ratatui::prelude::*;

use crate::gh_zero_cli::runtime::Throughput;
use crate::gh_zero_cli::style;

// State of the status-bar token indicator. Derived solely from whether a pruning
// threshold exists, so it is fixed for the whole run: pruning is either off or
// active, never both. The status only switches the label shown before the count;
// the count itself is always plain grey.
#[derive(Clone, Copy)]
pub(crate) enum TokenStatus {
    // Pruning disabled (no context window set, or too small for the reserve):
    // shows "pruning off".
    Off,
    // Pruning active: the chat is effectively unbounded since old turns are
    // dropped, shown as the "∞" infinite-chat marker.
    Active,
}

// Formats throughput as "↑ 320 ↓ 48 tok/s" — every token single-spaced, glyph
// included, so the field lines up with the rest of the uniformly spaced bar.
// Each arrow is dropped below 1 tok/s, so the connecting phase and sub-unit noise
// yield an empty string and render nothing.
pub(super) fn speed_label(throughput: Throughput) -> String {
    let mut parts: Vec<String> = Vec::new();
    if throughput.input >= 1.0 {
        parts.push(format!("↑ {:.0}", throughput.input));
    }
    if throughput.output >= 1.0 {
        parts.push(format!("↓ {:.0}", throughput.output));
    }
    if parts.is_empty() {
        return String::new();
    }
    parts.push("tok/s".to_string());
    parts.join(" ")
}

// Formats the current token estimates as "a 240 Σ 240 tok" —
// every token single-spaced like speed_label, so the whole bar shares one gap
// width. A zero total means no reply has completed
// yet, so it yields "" and renders nothing — mirroring speed_label before the
// first token. During streaming the session passes the live estimate; during
// input it passes the previous completed turn's final estimate.
pub(super) fn count_label(output_tokens: usize) -> String {
    if output_tokens == 0 {
        return String::new();
    }
    format!("a {output_tokens} Σ {output_tokens} tok")
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
    fn speed_label_hides_rates_below_one() {
        // Connecting phase: no rates yet -> nothing shown.
        assert_eq!(speed_label(Throughput::default()), "");
        // Each component appears independently once its rate reaches 1 tok/s.
        let t = Throughput {
            input: 320.4,
            output: 0.6,
        };
        assert_eq!(speed_label(t), "↑ 320 tok/s");
        let t = Throughput {
            input: 320.4,
            output: 48.3,
        };
        assert_eq!(speed_label(t), "↑ 320 ↓ 48 tok/s");
    }

    #[test]
    fn count_label_hidden_until_first_reply() {
        // No reply completed yet (zero total): blank, like the speed before the
        // first token, so the bar starts empty.
        assert_eq!(count_label(0), "");
        assert_eq!(count_label(240), "a 240 Σ 240 tok");
    }
}
