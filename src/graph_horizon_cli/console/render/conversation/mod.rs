/*
 * Graph Horizon CLI Modules - Console - Render - Conversation
 * Builds scrollable terminal lines from completed chat turns plus the active input or stream.
 * This folder includes display-only Reasoning separation; raw history remains
 * in turn/message fields, while frame drawing and caching stay in render.
 */

use ratatui::prelude::*;

use crate::graph_horizon_cli::runtime::ChatMessage;

use super::RenderContent;
use answer::{push_answer, push_prompt, trim_trailing_blank};
use caret::push_input_caret;
use suggestions::push_suggestions;

mod answer;
mod caret;
mod reasoning;
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

// Counts only raw messages that would be sent again. None records arithmetic
// overflow so a later admission cannot mistake an unrepresentable count for a
// small context.
pub(crate) fn conversation_characters(system: Option<&str>, history: &[ChatTurn]) -> Option<usize> {
    history
        .iter()
        .flat_map(|turn| turn.messages.iter())
        .map(ChatMessage::chars)
        .chain(system.map(|text| text.chars().count()))
        .try_fold(0_usize, usize::checked_add)
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
        push_answer(&mut lines, width, &turn.response, false, false, revision);
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
    push_answer(
        &mut lines,
        width,
        content.resp,
        content.loading,
        content.caret.is_none(),
        revision,
    );

    trim_trailing_blank(&mut lines);
    lines
}

#[cfg(test)]
mod tests {
    use super::super::{SectionStyle, TokenStatus, format_span};
    use super::*;
    use crate::graph_horizon_cli::runtime::Throughput;

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
    fn separates_complete_thinking_and_answer() {
        let content = output("[THINK]passo[/THINK]risposta");
        let lines = build_lines(20, 1, &content);

        assert_eq!(
            texts(&lines),
            vec!["> prompt", "", "[THINK]", "passo", "", "risposta"]
        );
        assert_line_style(&lines[2], SectionStyle::Secondary);
        assert_line_style(&lines[3], SectionStyle::Secondary);
        assert_line_style(&lines[5], SectionStyle::Response);
    }

    #[test]
    fn streams_unclosed_thinking_without_placeholder() {
        let content = output("[THINK]passo");

        assert_eq!(
            texts(&build_lines(20, 1, &content)),
            vec!["> prompt", "", "[THINK]", "passo"]
        );
    }

    #[test]
    fn completed_unclosed_thinking_has_missing_response_label() {
        let history = vec![ChatTurn::new(
            "prompt".to_string(),
            "[THINK]passo".to_string(),
            Vec::new(),
        )];
        let content = input(&history, "next");

        assert_eq!(
            texts(&build_lines(20, 1, &content)),
            vec![
                "> prompt",
                "",
                "[THINK]",
                "passo",
                "",
                "[no response]",
                "",
                "> next"
            ]
        );
    }

    #[test]
    fn empty_thinking_keeps_label_before_answer() {
        let content = output("[THINK][/THINK]risposta");

        assert_eq!(
            texts(&build_lines(20, 1, &content)),
            vec!["> prompt", "", "[THINK]", "", "risposta"]
        );
    }

    #[test]
    fn pending_prefix_uses_spinner_and_hides_raw_text() {
        let content = output("[TH");
        let visible = texts(&build_lines(20, 1, &content));

        assert_eq!(visible, vec!["> prompt", "", "[⠙ generating]"]);
        assert!(!visible.iter().any(|line| line.contains("[TH")));
    }

    #[test]
    fn contradicted_prefix_is_an_ordinary_response() {
        let content = output("[THX");

        assert_eq!(
            texts(&build_lines(20, 1, &content)),
            vec!["> prompt", "", "[THX"]
        );
    }

    #[test]
    fn additional_markers_remain_literal() {
        let content = output("[THINK]a[THINK]b[/THINK]c[/THINK]");

        assert_eq!(
            texts(&build_lines(40, 1, &content)),
            vec!["> prompt", "", "[THINK]", "a[THINK]b", "", "c[/THINK]"]
        );
    }

    #[test]
    fn completed_and_streaming_views_have_equal_sections() {
        let raw = "[THINK]passo[/THINK]risposta";
        let mut completed = Vec::new();
        let mut streaming = Vec::new();
        push_answer(&mut completed, 20, raw, false, false, 1);
        push_answer(&mut streaming, 20, raw, false, true, 1);

        assert_eq!(texts(&completed), texts(&streaming));
    }

    #[test]
    fn narrow_unicode_reasoning_wraps_without_loss() {
        let mut lines = Vec::new();
        push_answer(&mut lines, 4, "[THINK]中文[/THINK]答案", false, false, 1);

        assert_eq!(texts(&lines), vec!["", "[THI", "NK]", "中文", "", "答案"]);
    }

    #[test]
    fn section_and_content_owned_blank_lines_remain_distinct() {
        let content = output("[THINK]\none\n\n[/THINK]\nanswer\n");
        let lines = build_lines(20, 1, &content);

        assert_eq!(
            texts(&lines),
            vec![
                "> prompt", "", "[THINK]", "", "one", "", "", "", "", "answer", ""
            ]
        );
        assert_eq!(lines[6].spans.len(), 1);
        assert!(lines[7].spans.is_empty());
        assert_eq!(lines[8].spans.len(), 1);
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
        assert_eq!(conversation_characters(None, &history), Some(13));
    }

    #[test]
    fn includes_system_prompt_in_the_count() {
        // The system prompt is invisible in history but drives pruning, so it
        // must contribute to the shown count. 8 system chars / 4 = 2.
        assert_eq!(conversation_characters(Some("sys text"), &[]), Some(8));
    }

    // Flattens a styled line into plain text for assertions.
    fn line_text(line: &Line<'_>) -> String {
        line.spans
            .iter()
            .map(|span| span.content.as_ref())
            .collect::<String>()
    }

    fn texts(lines: &[Line<'_>]) -> Vec<String> {
        lines.iter().map(line_text).collect()
    }

    fn assert_line_style(line: &Line<'_>, section: SectionStyle) {
        let expected = format_span(section, String::new()).style;
        assert!(line.spans.iter().all(|span| span.style == expected));
    }

    fn output(response: &str) -> RenderContent<'_> {
        RenderContent::output(
            &[],
            "prompt",
            response,
            TokenStatus::Active,
            0,
            Throughput::default(),
            0,
            None,
        )
    }

    fn input<'a>(history: &'a [ChatTurn], prompt: &'a str) -> RenderContent<'a> {
        RenderContent::input(
            history,
            prompt,
            "",
            0,
            &[],
            0,
            TokenStatus::Active,
            0,
            Throughput::default(),
            0,
            None,
        )
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
