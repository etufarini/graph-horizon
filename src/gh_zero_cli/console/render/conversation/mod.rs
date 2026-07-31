/*
 * GH Zero CLI Modules - Console - Render - Conversation
 * Builds scrollable terminal lines from completed chat turns plus the active input or stream.
 * This folder owns only conversation-to-lines conversion; frame drawing and caching stay in render.
 */

use ratatui::prelude::*;

use crate::gh_zero_cli::runtime::{ChatMessage, estimate_tokens};

use super::RenderContent;
use answer::{push_answer, push_prompt, trim_trailing_blank};
use caret::push_input_caret;
use suggestions::push_suggestions;

mod answer;
mod caret;
mod suggestions;

pub(crate) use suggestions::SUGGESTION_THRESHOLD;

// One completed exchange shown in the terminal history.
#[derive(Clone)]
pub(crate) struct ChatTurn {
    pub(crate) prompt: String,
    pub(crate) response: String,
    // LLM messages produced by this turn; empty for failed turns.
    pub(crate) messages: Vec<ChatMessage>,
}

impl ChatTurn {
    pub(crate) fn new(prompt: String, response: String, messages: Vec<ChatMessage>) -> Self {
        Self {
            prompt,
            response,
            messages,
        }
    }
}

// Estimates the tokens of what occupies the model's context: the fixed system
// prompt plus the user/assistant content committed to history. The system
// prompt is included even though it is invisible in history, so the shown number
// stays consistent with the pruning that the same prompt drives. The
// in-progress prompt and
// failed turns (no messages) are excluded, so the count stays still while
// typing and streaming and only moves once a completed turn enters history.
// Characters are summed first and divided once (see estimate_tokens) to avoid
// the per-message truncation that would undercount many short messages.
pub(crate) fn conversation_tokens(system: Option<&str>, history: &[ChatTurn]) -> usize {
    let system_chars = system.map_or(0, |s| s.chars().count());
    let history_chars: usize = history
        .iter()
        .flat_map(|turn| turn.messages.iter())
        .map(ChatMessage::chars)
        .sum();
    estimate_tokens(system_chars + history_chars)
}

// Converts RenderContent into a flat list of terminal lines at the given width.
pub(crate) fn build_lines(
    width: u16,
    revision: u64,
    content: &RenderContent<'_>,
) -> Vec<Line<'static>> {
    if width == 0 {
        return Vec::new();
    }

    let mut lines = Vec::new();

    for turn in content.history {
        push_prompt(&mut lines, width, &turn.prompt, None);
        push_answer(&mut lines, width, &turn.response, false, revision);
        lines.push(Line::default());
    }

    // The live input line draws a caret at the cursor; the streaming phase (caret
    // None) reuses the plain prompt path, so output frames stay unchanged.
    match content.caret {
        Some(cursor) => push_input_caret(&mut lines, width, content.prompt, content.hint, cursor),
        None => push_prompt(&mut lines, width, content.prompt, content.hint),
    }
    push_suggestions(
        &mut lines,
        width,
        content.suggestions,
        content.suggestion_scroll,
    );
    push_answer(&mut lines, width, content.resp, content.loading, revision);

    trim_trailing_blank(&mut lines);
    lines
}

#[cfg(test)]
mod tests {
    use super::super::TokenStatus;
    use super::*;
    use crate::gh_zero_cli::runtime::Throughput;

    #[test]
    fn builds_history_before_current_prompt() {
        let history = vec![ChatTurn::new(
            "hello".to_string(),
            "world".to_string(),
            Vec::new(),
        )];
        let content = RenderContent::input(
            &history,
            "next",
            "",
            0,
            &[],
            0,
            TokenStatus::Active,
            0,
            Throughput::default(),
            0,
            None,
        );

        let visible: Vec<String> = build_lines(20, 1, &content).iter().map(line_text).collect();

        assert_eq!(visible, vec!["> hello", "", "world", "", "> next"]);
    }

    #[test]
    fn wraps_long_prompt_into_scrollable_lines() {
        let content = RenderContent::input(
            &[],
            "abcdefghi",
            "",
            0,
            &[],
            0,
            TokenStatus::Active,
            0,
            Throughput::default(),
            0,
            None,
        );

        let visible: Vec<String> = build_lines(6, 1, &content).iter().map(line_text).collect();

        assert_eq!(visible, vec!["> abcd", "  efgh", "  i"]);
    }

    #[test]
    fn shows_generating_before_first_chunk() {
        let content = RenderContent::output(
            &[],
            "prompt",
            "",
            TokenStatus::Active,
            0,
            Throughput::default(),
            0,
            None,
        );

        let visible: Vec<String> = build_lines(20, 1, &content).iter().map(line_text).collect();

        assert_eq!(visible[0], "> prompt");
        assert_eq!(visible[1], "");
        assert_eq!(visible[2], "[⠙ generating]");
    }

    #[test]
    fn no_generating_once_content_arrives() {
        let content = RenderContent::output(
            &[],
            "prompt",
            "content",
            TokenStatus::Active,
            0,
            Throughput::default(),
            0,
            None,
        );

        let visible: Vec<String> = build_lines(20, 1, &content).iter().map(line_text).collect();

        assert!(!visible.iter().any(|l| l.contains("[generating")));
    }

    #[test]
    fn no_generating_in_input_mode() {
        let content = RenderContent::input(
            &[],
            "hello",
            "",
            0,
            &[],
            0,
            TokenStatus::Active,
            0,
            Throughput::default(),
            0,
            None,
        );

        let visible: Vec<String> = build_lines(20, 1, &content).iter().map(line_text).collect();

        assert!(!visible.iter().any(|l| l.contains("[generating")));
    }

    #[test]
    fn preserves_empty_lines_in_response() {
        let content = RenderContent::output(
            &[],
            "prompt",
            "one\n\ntwo",
            TokenStatus::Active,
            0,
            Throughput::default(),
            0,
            None,
        );

        let visible: Vec<String> = build_lines(10, 1, &content).iter().map(line_text).collect();

        assert_eq!(visible, vec!["> prompt", "", "one", "", "two"]);
    }

    #[test]
    fn counts_only_committed_messages_not_display_or_input() {
        // Only stored messages count: display text and the in-progress prompt do
        // not affect the number until a completed turn enters history.
        let turn = ChatTurn::new(
            "ignored prompt text".to_string(),
            "ignored response text".to_string(),
            vec![
                ChatMessage::user("hello world".to_string()),
                ChatMessage::assistant("hi".to_string()),
            ],
        );
        let history = vec![turn];

        // 11 + 2 = 13 chars / 4 = 3. Summing characters first matters: counting
        // per message ("hi" -> 0) would lose the short message and yield 2.
        assert_eq!(conversation_tokens(None, &history), 3);
    }

    #[test]
    fn includes_system_prompt_in_the_count() {
        // The system prompt is invisible in history but drives pruning, so it
        // must contribute to the shown count. 8 system chars / 4 = 2.
        assert_eq!(conversation_tokens(Some("sys text"), &[]), 2);
    }

    // Flattens a styled line into plain text for assertions.
    fn line_text(line: &Line<'_>) -> String {
        line.spans
            .iter()
            .map(|span| span.content.as_ref())
            .collect::<String>()
    }

    // Returns the text of the single reverse-styled span — the caret cell — across all
    // rendered lines, so caret position can be asserted independently of the line text.
    fn caret_cell(lines: &[Line<'_>]) -> String {
        lines
            .iter()
            .flat_map(|line| line.spans.iter())
            .filter(|span| span.style.add_modifier.contains(Modifier::REVERSED))
            .map(|span| span.content.as_ref())
            .collect::<String>()
    }

    fn input_lines(prompt: &str, hint: &str, cursor: usize, width: u16) -> Vec<Line<'static>> {
        let content = RenderContent::input(
            &[],
            prompt,
            hint,
            cursor,
            &[],
            0,
            TokenStatus::Active,
            0,
            Throughput::default(),
            0,
            None,
        );
        build_lines(width, 1, &content)
    }

    #[test]
    fn caret_highlights_character_under_cursor() {
        let lines = input_lines("abc", "", 1, 20);
        assert_eq!(caret_cell(&lines), "b");
        assert_eq!(line_text(&lines[0]), "> abc");
    }

    #[test]
    fn caret_at_end_without_hint_is_a_trailing_space() {
        let lines = input_lines("ab", "", 2, 20);
        assert_eq!(caret_cell(&lines), " ");
        assert_eq!(line_text(&lines[0]), "> ab ");
    }

    #[test]
    fn caret_at_end_with_hint_highlights_first_hint_char() {
        let lines = input_lines("ab", "cd", 2, 20);
        assert_eq!(caret_cell(&lines), "c");
        assert_eq!(line_text(&lines[0]), "> abcd");
    }

    #[test]
    fn end_caret_wraps_to_a_new_line_when_the_last_line_is_full() {
        // width 6 wraps "abcdefgh" into "> abcd" / "  efgh": the second line is full,
        // so the end-of-text caret cell must wrap to a third line instead of being
        // clipped by the (non-wrapping) chat Paragraph.
        let lines = input_lines("abcdefgh", "", 8, 6);
        assert_eq!(caret_cell(&lines), " ");
        assert_eq!(line_text(&lines[2]), "   ");
    }

    #[test]
    fn caret_lands_on_the_wrapped_segment_holding_the_cursor() {
        // width 6 wraps "abcdefghi" into "> abcd" / "  efgh" / "  i"; cursor 5 is 'f',
        // which lives on the second wrapped line.
        let lines = input_lines("abcdefghi", "", 5, 6);
        assert_eq!(caret_cell(&lines), "f");
        assert_eq!(line_text(&lines[1]), "  efgh");
    }
}
