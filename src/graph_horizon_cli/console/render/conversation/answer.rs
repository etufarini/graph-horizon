/*
 * Graph Horizon CLI Modules - Console - Render - Conversation - Answer
 * Pushes prompt, separately labelled Reasoning, final answer, and spinner lines
 * for one turn. Parsing is delegated to the sibling module; runtime and history
 * ownership remain elsewhere.
 */

use ratatui::prelude::*;

use super::super::SectionStyle;
use super::super::wrap::{push_prefixed_lines, push_wrapped_lines};
use super::reasoning::split_reasoning;

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
    streaming: bool,
    revision: u64,
) {
    let view = split_reasoning(response, streaming);
    // Undecided marker bytes stay hidden behind the established loading signal.
    if loading || view.pending {
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
        return;
    }

    if let Some(thinking) = view.thinking {
        lines.push(Line::default());
        push_wrapped_lines(lines, width, SectionStyle::Secondary, "[THINK]");
        if !thinking.is_empty() {
            for line in thinking.split('\n') {
                push_wrapped_lines(lines, width, SectionStyle::Secondary, line);
            }
        }
        lines.push(Line::default());
        if view.answer.is_empty() {
            if !streaming {
                push_wrapped_lines(lines, width, SectionStyle::Secondary, "[no response]");
            }
        } else {
            for line in view.answer.split('\n') {
                push_wrapped_lines(lines, width, SectionStyle::Response, line);
            }
        }
    } else if !view.answer.is_empty() {
        lines.push(Line::default());
        for line in view.answer.split('\n') {
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
