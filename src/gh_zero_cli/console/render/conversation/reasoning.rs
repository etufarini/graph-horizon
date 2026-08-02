/*
 * This module recognizes current raw Ministral Reasoning markers for rendering
 * only. Runtime chunks, history, transcripts, protocol fields, and model policy
 * remain owned by their existing modules.
 */

const THINK_OPEN: &str = "[THINK]";
const THINK_CLOSE: &str = "[/THINK]";

pub(super) struct ReasoningView<'a> {
    pub(super) thinking: Option<&'a str>,
    pub(super) answer: &'a str,
    pub(super) pending: bool,
}

pub(super) fn split_reasoning(raw: &str, streaming: bool) -> ReasoningView<'_> {
    let candidate = raw.trim_start();

    if streaming && candidate != THINK_OPEN && THINK_OPEN.starts_with(candidate) {
        return ReasoningView {
            thinking: None,
            answer: "",
            pending: true,
        };
    }

    let Some(content) = candidate.strip_prefix(THINK_OPEN) else {
        return ReasoningView {
            thinking: None,
            answer: raw,
            pending: false,
        };
    };

    let (thinking, answer) = content.split_once(THINK_CLOSE).unwrap_or((content, ""));
    ReasoningView {
        thinking: Some(thinking),
        answer,
        pending: false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_view(
        raw: &str,
        streaming: bool,
        thinking: Option<&str>,
        answer: &str,
        pending: bool,
    ) {
        let view = split_reasoning(raw, streaming);
        assert_eq!(view.thinking, thinking);
        assert_eq!(view.answer, answer);
        assert_eq!(view.pending, pending);
    }

    #[test]
    fn empty_and_whitespace_streams_are_pending() {
        assert_view("", true, None, "", true);
        assert_view(" \n\t", true, None, "", true);
    }

    #[test]
    fn empty_and_whitespace_completed_content_is_ordinary() {
        assert_view("", false, None, "", false);
        assert_view(" \n\t", false, None, " \n\t", false);
    }

    #[test]
    fn every_strict_opening_prefix_is_pending_while_streaming() {
        for end in 0..THINK_OPEN.len() {
            assert_view(&THINK_OPEN[..end], true, None, "", true);
        }
    }

    #[test]
    fn leading_whitespace_before_a_prefix_is_pending() {
        assert_view("\n  [TH", true, None, "", true);
    }

    #[test]
    fn contradicted_streaming_prefix_becomes_ordinary() {
        assert_view("  [THX", true, None, "  [THX", false);
    }

    #[test]
    fn incomplete_completed_prefix_is_ordinary() {
        assert_view("[THIN", false, None, "[THIN", false);
    }

    #[test]
    fn ordinary_response_is_unchanged() {
        assert_view(
            "Risposta ordinaria",
            true,
            None,
            "Risposta ordinaria",
            false,
        );
    }

    #[test]
    fn complete_markers_preserve_present_but_empty_thinking() {
        assert_view("[THINK][/THINK]", false, Some(""), "", false);
    }

    #[test]
    fn leading_whitespace_is_omitted_only_from_reasoning_view() {
        let raw = " \n[THINK]  ragiona \n[/THINK]  rispondi ";
        assert_view(raw, false, Some("  ragiona \n"), "  rispondi ", false);
        assert_eq!(raw, " \n[THINK]  ragiona \n[/THINK]  rispondi ");
    }

    #[test]
    fn opening_without_close_exposes_thinking_while_streaming() {
        assert_view("[THINK]passo", true, Some("passo"), "", false);
    }

    #[test]
    fn opening_without_close_exposes_thinking_after_completion() {
        assert_view("[THINK]passo", false, Some("passo"), "", false);
    }

    #[test]
    fn lowercase_and_mixed_case_markers_are_ordinary() {
        for raw in ["[think]x[/think]y", "[Think]x[/THINK]y"] {
            assert_view(raw, true, None, raw, false);
        }
    }

    #[test]
    fn opening_after_non_whitespace_text_is_literal() {
        let raw = "intro [THINK]x[/THINK]y";
        assert_view(raw, false, None, raw, false);
    }

    #[test]
    fn closing_without_opening_is_literal() {
        let raw = "[/THINK]risposta";
        assert_view(raw, false, None, raw, false);
    }

    #[test]
    fn extra_opening_remains_inside_thinking() {
        assert_view(
            "[THINK]a[THINK]b[/THINK]c",
            false,
            Some("a[THINK]b"),
            "c",
            false,
        );
    }

    #[test]
    fn markers_after_first_close_remain_literal_answer_content() {
        assert_view(
            "[THINK]a[/THINK]b[/THINK][THINK]c[/THINK]",
            false,
            Some("a"),
            "b[/THINK][THINK]c[/THINK]",
            false,
        );
    }

    #[test]
    fn newlines_and_unicode_are_preserved_exactly() {
        assert_view(
            "[THINK]\nπensa 🧠\n[/THINK]\nrisposta ✓\n",
            false,
            Some("\nπensa 🧠\n"),
            "\nrisposta ✓\n",
            false,
        );
    }
}
