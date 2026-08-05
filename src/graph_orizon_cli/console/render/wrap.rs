/*
 * Graph Orizon CLI Modules - Console - Render - Wrap
 * Text wrapping and prefix formatting utilities used to turn conversation text into terminal lines.
*/

use super::{SectionStyle, format_span};
use ratatui::prelude::*;
use unicode_width::UnicodeWidthChar;

// Wraps `text` to fit within `width` display columns and pushes each wrapped line into `lines`.
// `first_prefix` is prepended to the first line; `continuation_prefix` to subsequent lines.
// The inline `hint` (dimmed ghost text) is appended after the final character of the last line.
pub(crate) fn push_prefixed_lines(
    lines: &mut Vec<Line<'static>>,
    width: u16,
    section: SectionStyle,
    first_prefix: &str,
    continuation_prefix: &str,
    text: &str,
    hint: Option<&str>,
) {
    let first_available = usize::from(width)
        .saturating_sub(display_width(first_prefix))
        .max(1);
    let cont_available = usize::from(width)
        .saturating_sub(display_width(continuation_prefix))
        .max(1);

    let mut remaining = text;
    let mut is_first_line = true;

    loop {
        let (prefix, available_width) = if is_first_line {
            (first_prefix, first_available)
        } else {
            (continuation_prefix, cont_available)
        };
        let (segment, rest) = take_fitting_segment(remaining, available_width);

        let mut spans = vec![
            format_span(section, prefix.to_string()),
            format_span(section, segment.to_string()),
        ];

        if rest.is_empty() {
            if let Some(h) = hint {
                spans.push(format_span(SectionStyle::Hint, h.to_string()));
            }
            lines.push(Line::from(spans));
            break;
        }

        lines.push(Line::from(spans));
        remaining = rest;
        is_first_line = false;
    }
}

// Wraps `text` to fit within `width` display columns and pushes each wrapped line into `lines`,
// with no prefix. Empty text produces a single empty line to preserve blank-line spacing.
pub(crate) fn push_wrapped_lines(
    lines: &mut Vec<Line<'static>>,
    width: u16,
    section: SectionStyle,
    text: &str,
) {
    let available_width = usize::from(width).max(1);
    let mut remaining = text;

    loop {
        let (segment, rest) = take_fitting_segment(remaining, available_width);
        lines.push(Line::from(format_span(section, segment.to_string())));

        if rest.is_empty() {
            break;
        }

        remaining = rest;
    }
}

// Splits `text` at the first character boundary that would exceed `max_width` terminal display
// columns, returning (head, tail). Uses Unicode East Asian Width so wide characters (CJK,
// emoji) count as 2 columns each. Characters are never split mid-byte. When max_width is 0
// the first character is still emitted to avoid an infinite loop in callers that
// unconditionally push each segment.
pub(crate) fn take_fitting_segment(text: &str, max_width: usize) -> (&str, &str) {
    if text.is_empty() {
        return ("", "");
    }

    let mut columns_used = 0;
    let mut end = 0;

    for (index, ch) in text.char_indices() {
        let ch_width = UnicodeWidthChar::width(ch).unwrap_or(1);

        if columns_used + ch_width > max_width {
            if end == 0 {
                // Emit at least one character to avoid infinite loops when max_width is 0
                // or smaller than the first character's display width.
                end = index + ch.len_utf8();
            }
            break;
        }

        columns_used += ch_width;
        end = index + ch.len_utf8();
    }

    (&text[..end], &text[end..])
}

// Returns the number of terminal display columns occupied by a string.
pub(crate) fn display_width(s: &str) -> usize {
    s.chars()
        .map(|c| UnicodeWidthChar::width(c).unwrap_or(1))
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_text_returns_empty_pair() {
        assert_eq!(take_fitting_segment("", 10), ("", ""));
    }

    #[test]
    fn text_shorter_than_width_returns_whole_text() {
        assert_eq!(take_fitting_segment("hi", 10), ("hi", ""));
    }

    #[test]
    fn text_exactly_at_width_returns_whole_text() {
        assert_eq!(take_fitting_segment("abcd", 4), ("abcd", ""));
    }

    #[test]
    fn text_longer_than_width_splits_correctly() {
        assert_eq!(take_fitting_segment("abcdef", 4), ("abcd", "ef"));
    }

    #[test]
    fn multibyte_single_width_chars_are_not_split() {
        // "é" is 2 bytes but 1 display column; with max_width=1 we get exactly one character.
        let (head, tail) = take_fitting_segment("éàü", 1);
        assert_eq!(head, "é");
        assert_eq!(tail, "àü");
    }

    #[test]
    fn multibyte_single_width_chars_counted_by_display_column() {
        let (head, tail) = take_fitting_segment("éàü", 2);
        assert_eq!(head, "éà");
        assert_eq!(tail, "ü");
    }

    #[test]
    fn wide_chars_count_as_two_columns() {
        // "中" is a CJK character that occupies 2 terminal columns.
        let (head, tail) = take_fitting_segment("中文ab", 4);
        assert_eq!(head, "中文");
        assert_eq!(tail, "ab");
    }

    #[test]
    fn wide_char_that_does_not_fit_is_deferred_to_tail() {
        // Width=3 cannot fit two CJK chars (4 cols); only one fits (2 cols), leaving 1 unused.
        let (head, tail) = take_fitting_segment("中文", 3);
        assert_eq!(head, "中");
        assert_eq!(tail, "文");
    }

    #[test]
    fn zero_width_still_emits_first_char_to_prevent_infinite_loop() {
        let (head, tail) = take_fitting_segment("abc", 0);
        assert_eq!(head, "a");
        assert_eq!(tail, "bc");
    }

    #[test]
    fn zero_width_with_wide_first_char_still_emits_it() {
        // Even though "中" is 2 columns wide, it must be emitted when max_width=0
        // to prevent callers from looping forever.
        let (head, tail) = take_fitting_segment("中bc", 0);
        assert_eq!(head, "中");
        assert_eq!(tail, "bc");
    }
}
