/*
 * Graph Horizon CLI Modules - Console - Render - Conversation - Caret
 * Draws the live input line with a reverse-video caret at the cursor character.
 * Used only for the editable prompt; history and streaming reuse the plain prompt path.
 */

use ratatui::prelude::*;

use super::super::wrap::{display_width, take_fitting_segment};
use super::super::{SectionStyle, format_span};

// Draws the live input line with a visible caret at character index `cursor`.
// The caret cell is rendered reverse-video: the character under the cursor when it
// sits inside the text, or — at end of text — the hint's first character if present,
// else a trailing space. Wrapping mirrors push_prefixed_lines (prefixes "> " and
// "  ", both two columns wide), so the caret lands on the wrapped segment and column
// that actually contain the cursor. Used only for the editable prompt, never history.
pub(super) fn push_input_caret(
    lines: &mut Vec<Line<'static>>,
    width: u16,
    prompt: &str,
    hint: Option<&str>,
    cursor: usize,
) {
    // Both prefixes are two ASCII columns, so first and continuation share one width.
    let available = usize::from(width).saturating_sub(2).max(1);
    let len = prompt.chars().count();

    let mut remaining = prompt;
    let mut consumed = 0; // characters already emitted on previous wrapped lines
    let mut is_first = true;

    loop {
        let prefix = if is_first { "> " } else { "  " };
        let (segment, rest) = take_fitting_segment(remaining, available);
        let seg_chars = segment.chars().count();

        let mut spans = vec![format_span(SectionStyle::Input, prefix.to_string())];

        // The caret falls in this segment when cursor is within [consumed, consumed+seg_chars).
        // Splitting the segment's span into before/caret/after keeps the caret cell on the
        // exact column the character would occupy, leaving the wrapping untouched.
        if cursor >= consumed && cursor < consumed + seg_chars {
            let split = segment
                .char_indices()
                .nth(cursor - consumed)
                .map_or(segment.len(), |(i, _)| i);
            let caret_len = segment[split..].chars().next().map_or(0, char::len_utf8);
            let before = &segment[..split];
            let caret = &segment[split..split + caret_len];
            let after = &segment[split + caret_len..];
            if !before.is_empty() {
                spans.push(format_span(SectionStyle::Input, before.to_string()));
            }
            spans.push(format_span(SectionStyle::Input, caret.to_string()).reversed());
            if !after.is_empty() {
                spans.push(format_span(SectionStyle::Input, after.to_string()));
            }
        } else {
            spans.push(format_span(SectionStyle::Input, segment.to_string()));
        }

        if rest.is_empty() {
            // End-of-text caret: highlight the hint's first character, or a space when
            // there is no hint, so the cursor is always visible past the last character.
            if cursor >= len {
                // The caret occupies one cell past the last character; when the segment
                // already fills the line that cell would be clipped (the chat Paragraph
                // truncates overflow), so it wraps onto a fresh continuation line.
                if display_width(segment) >= available {
                    lines.push(Line::from(spans));
                    spans = vec![format_span(SectionStyle::Input, "  ".to_string())];
                }
                match hint {
                    Some(h) if !h.is_empty() => {
                        let first_len = h.chars().next().map_or(0, char::len_utf8);
                        spans.push(
                            format_span(SectionStyle::Hint, h[..first_len].to_string()).reversed(),
                        );
                        if h.len() > first_len {
                            spans.push(format_span(SectionStyle::Hint, h[first_len..].to_string()));
                        }
                    }
                    _ => spans.push(format_span(SectionStyle::Input, " ".to_string()).reversed()),
                }
            } else if let Some(h) = hint {
                spans.push(format_span(SectionStyle::Hint, h.to_string()));
            }
            lines.push(Line::from(spans));
            break;
        }

        lines.push(Line::from(spans));
        remaining = rest;
        consumed += seg_chars;
        is_first = false;
    }
}
