/*
 * GH Zero CLI Modules - Console - Session - Input - History
 * Prompt-recall key handlers: Up/Down walk previously sent prompts, saving the live draft.
 * Falls through to viewport scrolling past the recall window; called from the input loop.
 */

use super::{ChatTurn, HISTORY_RECALL_LIMIT, KeyCode, ViewportState, bump, handle_scroll_events};

// Up recalls an older prompt while editing. Recall is capped at the last
// HISTORY_RECALL_LIMIT prompts: pressing Up past that window (or with no history)
// falls through to scrolling the viewport up. The first recall saves the live
// draft so Down can restore it.
#[allow(clippy::too_many_arguments)]
pub(super) fn on_up(
    prompt: &mut String,
    cursor: &mut usize,
    history_index: &mut Option<usize>,
    draft: &mut String,
    suggestion_scroll: &mut usize,
    history: &[ChatTurn],
    viewport: &mut ViewportState,
    content_revision: &mut u64,
) {
    let oldest = history.len().saturating_sub(HISTORY_RECALL_LIMIT);
    let next = match *history_index {
        None if !history.is_empty() => Some(history.len() - 1), // newest prompt
        Some(i) if i > oldest => Some(i - 1),
        _ => None, // no older prompt within the window
    };
    if let Some(i) = next {
        if history_index.is_none() {
            *draft = prompt.clone(); // save the live draft to restore later
        }
        *prompt = history[i].prompt.clone();
        *cursor = prompt.chars().count();
        *history_index = Some(i);
        *suggestion_scroll = 0;
        bump(content_revision);
    } else {
        handle_scroll_events(KeyCode::Up, viewport);
    }
}

// Down recalls a newer prompt, returns to the saved draft past the newest, or —
// while already editing the live draft — scrolls the viewport down.
#[allow(clippy::too_many_arguments)]
pub(super) fn on_down(
    prompt: &mut String,
    cursor: &mut usize,
    history_index: &mut Option<usize>,
    draft: &mut String,
    suggestion_scroll: &mut usize,
    history: &[ChatTurn],
    viewport: &mut ViewportState,
    content_revision: &mut u64,
) {
    match *history_index {
        Some(i) if i + 1 < history.len() => {
            *prompt = history[i + 1].prompt.clone();
            *cursor = prompt.chars().count();
            *history_index = Some(i + 1);
            *suggestion_scroll = 0;
            bump(content_revision);
        }
        Some(_) => {
            // Past the newest prompt: return to the saved live draft.
            *prompt = std::mem::take(draft);
            *cursor = prompt.chars().count();
            *history_index = None;
            *suggestion_scroll = 0;
            bump(content_revision);
        }
        // Already editing the live draft: nothing newer to recall, so Down scrolls
        // the viewport down (mirroring Up past the window).
        None => handle_scroll_events(KeyCode::Down, viewport),
    }
}
