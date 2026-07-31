/*
 * GH Zero CLI Modules - Console - Render - Conversation - Answer
 * Single responsibility: push prompt, assistant text, and loading spinner lines
 * for one turn. It depends on wrapping/style helpers and does not render tools,
 * confirmations, or a separately labelled internal channel.
 */

use ratatui::prelude::*;

use super::super::SectionStyle;
use super::super::wrap::{push_prefixed_lines, push_wrapped_lines};

// Adds one user prompt, preserving multiline input and inline completion on the last line.
pub(super) fn push_prompt(
    lines: &mut Vec<Line<'static>>,
    width: u16,
    prompt: &str,
    hint: Option<&str>,
) {
    let prompt_lines: Vec<&str> = prompt.split('\n').collect();

    for (index, line) in prompt_lines.iter().enumerate() {
        let is_last = index == prompt_lines.len() - 1;
        push_prefixed_lines(
            lines,
            width,
            SectionStyle::Input,
            if index == 0 { "> " } else { "  " },
            "  ",
            line,
            if is_last { hint } else { None },
        );
    }
}

// Adds assistant response text or the spinner while waiting for the first token.
pub(super) fn push_answer(
    lines: &mut Vec<Line<'static>>,
    width: u16,
    response: &str,
    loading: bool,
    revision: u64,
) {
    if loading {
        // Spinner frames keep the loading state visible while the HTTP stream opens.
        const SPINNER: &[char] = &['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];
        let frame = SPINNER[revision as usize % SPINNER.len()];
        lines.push(Line::default());
        push_wrapped_lines(
            lines,
            width,
            SectionStyle::Secondary,
            &format!("[{frame} generating]"),
        );
    }

    if !response.is_empty() {
        lines.push(Line::default());
        for line in response.split('\n') {
            push_wrapped_lines(lines, width, SectionStyle::Response, line);
        }
    }
}

// Removes only the separator added after the final completed turn.
pub(super) fn trim_trailing_blank(lines: &mut Vec<Line<'static>>) {
    if lines.last().is_some_and(|line| line.spans.is_empty()) {
        lines.pop();
    }
}
