/*
 * GH Zero CLI Modules - Console - Render - Conversation - Suggestions
 * Renders completion suggestions under the input, with a scrollable window past a threshold.
 * Pure line builder called by the conversation orchestrator; owns the suggestion threshold.
 */

use ratatui::prelude::*;

use super::super::SectionStyle;
use super::super::wrap::push_prefixed_lines;

// Most matches shown at once under the input; beyond this the suggestion list
// becomes a scrollable window of this many items (see push_suggestions).
pub(crate) const SUGGESTION_THRESHOLD: usize = 10;

// Adds completion suggestions under the current input when there is more than one match.
// When matches exceed SUGGESTION_THRESHOLD a scrollable window of N items is shown;
// the Left/Right keys shift the window, and ← / → markers flag hidden items
// before and after it.
pub(super) fn push_suggestions(
    lines: &mut Vec<Line<'static>>,
    width: u16,
    suggestions: &[String],
    scroll: usize,
) {
    // A single match is already visible in the inline hint.
    if suggestions.len() <= 1 {
        return;
    }

    if suggestions.len() <= SUGGESTION_THRESHOLD {
        for suggestion in suggestions {
            push_prefixed_lines(
                lines,
                width,
                SectionStyle::Hint,
                "  ",
                "  ",
                suggestion,
                None,
            );
        }
        return;
    }

    let start = scroll.min(suggestions.len().saturating_sub(SUGGESTION_THRESHOLD));
    let end = (start + SUGGESTION_THRESHOLD).min(suggestions.len());

    if start > 0 {
        push_prefixed_lines(lines, width, SectionStyle::Hint, "  ", "  ", "←", None);
    }
    for suggestion in &suggestions[start..end] {
        push_prefixed_lines(
            lines,
            width,
            SectionStyle::Hint,
            "  ",
            "  ",
            suggestion,
            None,
        );
    }
    if suggestions.len() > end {
        push_prefixed_lines(lines, width, SectionStyle::Hint, "  ", "  ", "→", None);
    }
}
