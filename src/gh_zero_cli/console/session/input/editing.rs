/*
 * GH Zero CLI Modules - Console - Session - Input - Editing
 * Text-editing key handlers for the input phase: Enter, Char, Backspace, Tab.
 * Pure caret/prompt mutators called from the input loop's key match; no terminal I/O.
 */

use super::{bump, byte_offset};

// Result of an Enter keypress: whether the loop should send the prompt now.
pub(super) enum Edit {
    Continue,
    Send,
}

// Enter sends the prompt only when no @-completion is in progress. If there is
// something to complete, it completes; if the @-token still matches several
// files, it waits for disambiguation instead of sending; an empty prompt is
// ignored so Esc stays the single exit path.
pub(super) fn on_enter(
    prompt: &mut String,
    cursor: &mut usize,
    hint: &str,
    suggestions_len: usize,
    content_revision: &mut u64,
) -> Edit {
    if !hint.is_empty() {
        prompt.push_str(hint);
        *cursor = prompt.chars().count();
        bump(content_revision);
        Edit::Continue
    } else if suggestions_len > 1 {
        // Ambiguous open @-token: hold the send until the user narrows it.
        Edit::Continue
    } else if !prompt.trim().is_empty() {
        // Send only a non-empty prompt; trim guards whitespace-only input, which
        // would otherwise reach the model as an effectively empty turn.
        Edit::Send
    } else {
        Edit::Continue
    }
}

// Inserts a character at the caret (char index -> byte offset) and steps over it.
pub(super) fn on_char(
    c: char,
    prompt: &mut String,
    cursor: &mut usize,
    history_index: &mut Option<usize>,
    suggestion_scroll: &mut usize,
    content_revision: &mut u64,
) {
    let offset = byte_offset(prompt, *cursor);
    prompt.insert(offset, c);
    *cursor += 1;
    // Editing detaches from history: this is now a free draft again.
    *history_index = None;
    *suggestion_scroll = 0;
    bump(content_revision);
}

// Deletes the character left of the caret; a no-op on an empty prompt or with the
// caret at the start (no detach/redraw).
pub(super) fn on_backspace(
    prompt: &mut String,
    cursor: &mut usize,
    history_index: &mut Option<usize>,
    suggestion_scroll: &mut usize,
    content_revision: &mut u64,
) {
    if *cursor > 0 {
        let offset = byte_offset(prompt, *cursor - 1);
        prompt.remove(offset);
        *cursor -= 1;
        *history_index = None;
        *suggestion_scroll = 0;
        bump(content_revision);
    }
}

// Tab applies the same completion already shown as the hint, be it a command name
// or a path, so both share one source.
pub(super) fn on_tab(
    prompt: &mut String,
    cursor: &mut usize,
    hint: &str,
    content_revision: &mut u64,
) {
    if !hint.is_empty() {
        prompt.push_str(hint);
        *cursor = prompt.chars().count();
        bump(content_revision);
    }
}
