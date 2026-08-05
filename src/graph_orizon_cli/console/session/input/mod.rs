/*
 * Graph Orizon CLI Modules - Console - Session - Input
 * Input phase: reads one prompt from the user, handling editing, @-token completion, and scroll.
 * Returns None only on the explicit exit path (Esc); orchestration stays in session/mod.rs.
*/

use super::super::render::{
    ChatTurn, RenderCache, RenderContent, SUGGESTION_THRESHOLD, TokenStatus, draw_viewport,
};
use super::super::scroll::{ViewportState, handle_scroll_events};
use super::super::{INPUT_POLL_INTERVAL, bump};
use crate::graph_orizon_cli::plugins::attachments::FileAuthority;
use crate::graph_orizon_cli::plugins::{attachments, command};
use crate::graph_orizon_cli::runtime::Throughput;
use color_eyre::eyre::Result;
use crossterm::event::{self, Event, KeyCode, KeyEventKind};
use ratatui::DefaultTerminal;

mod editing;
mod history;

// Most recent prompts reachable by Up/Down recall; older history stays out of reach
// so that pressing Up past the window resumes scrolling the viewport upward instead.
const HISTORY_RECALL_LIMIT: usize = 10;

// Byte offset of the character at index `index` in `text`; the text length when the
// index is at or past the end. Keeps caret edits on UTF-8 boundaries so multibyte
// characters are never split mid-byte.
fn byte_offset(text: &str, index: usize) -> usize {
    text.char_indices()
        .nth(index)
        .map_or(text.len(), |(i, _)| i)
}

// Reads one prompt from the user, returning None only on the explicit exit path (Esc).
// The prompt can start prefilled with `initial` (re-entry after an interrupted turn);
// an empty `initial` gives the usual blank field. The remaining arguments are the
// render/scroll state this phase drives plus the status-bar data it draws; bundling
// them into a struct would add indirection without clarity.
#[allow(clippy::too_many_arguments)]
pub(super) fn read_prompt(
    terminal: &mut DefaultTerminal,
    document: &mut RenderCache,
    viewport: &mut ViewportState,
    content_revision: &mut u64,
    history: &[ChatTurn],
    initial: String,
    status: TokenStatus,
    token_count: usize,
    throughput: Throughput,
    output_tokens: usize,
    context_limit: Option<usize>,
    files: &FileAuthority,
) -> Result<Option<String>> {
    let mut prompt = initial;
    // Caret position as a character index into `prompt`, in 0..=prompt.chars().count().
    // Left/Right move it; edits insert/delete at it; any prompt replacement (history
    // recall, draft restore, completion) snaps it back to the end of the new text.
    // A prefilled prompt starts with the caret at the end, as if just typed.
    let mut cursor: usize = prompt.chars().count();
    let mut suggestion_scroll: usize = 0;
    // Prompt-history recall: while editing, Up/Down walk previously sent prompts
    // instead of scrolling the viewport (which keeps PageUp/PageDown/Home/End).
    // `history_index` is None while editing the live draft, Some(i) while showing
    // history[i].prompt; `draft` preserves the live text so Down can restore it.
    let mut history_index: Option<usize> = None;
    let mut draft = String::new();

    loop {
        // Completion source depends on context: a prompt opening with '/' is a
        // plugin command (name, then optional path argument); anything else uses
        // the '@' attachment completion. Both feed the same hint/suggestions UI.
        let (tail, suggestions) = if prompt.trim_start().starts_with('/') {
            command::complete(files, &prompt)
        } else {
            attachments::complete_at_token(files, &prompt)
        };
        let hint = tail.unwrap_or_default();
        let content = RenderContent::input(
            history,
            &prompt,
            &hint,
            cursor,
            &suggestions,
            suggestion_scroll,
            status,
            token_count,
            throughput,
            output_tokens,
            context_limit,
        );
        draw_viewport(terminal, document, *content_revision, &content, viewport)?;

        if event::poll(INPUT_POLL_INTERVAL)?
            && let Event::Key(k) = event::read()?
            && k.kind == KeyEventKind::Press
        {
            match k.code {
                KeyCode::Enter => {
                    if let editing::Edit::Send = editing::on_enter(
                        &mut prompt,
                        &mut cursor,
                        &hint,
                        suggestions.len(),
                        content_revision,
                    ) {
                        break;
                    }
                }
                KeyCode::Right if suggestions.len() > SUGGESTION_THRESHOLD => {
                    let max_scroll = suggestions.len().saturating_sub(SUGGESTION_THRESHOLD);
                    suggestion_scroll = (suggestion_scroll + 1).min(max_scroll);
                    bump(content_revision);
                }
                KeyCode::Left if suggestions.len() > SUGGESTION_THRESHOLD => {
                    suggestion_scroll = suggestion_scroll.saturating_sub(1);
                    bump(content_revision);
                }
                // With no scrollable suggestion window open, Left/Right walk the caret
                // through the prompt one character at a time. At the text edges they are
                // a no-op (they must not fall through to scrolling the viewport).
                KeyCode::Left => {
                    if cursor > 0 {
                        cursor -= 1;
                        bump(content_revision);
                    }
                }
                KeyCode::Right => {
                    if cursor < prompt.chars().count() {
                        cursor += 1;
                        bump(content_revision);
                    }
                }
                // Up/Down recall older/newer prompts while editing; past the recall
                // window they fall through to scrolling the viewport (see history.rs).
                KeyCode::Up => history::on_up(
                    &mut prompt,
                    &mut cursor,
                    &mut history_index,
                    &mut draft,
                    &mut suggestion_scroll,
                    history,
                    viewport,
                    content_revision,
                ),
                KeyCode::Down => history::on_down(
                    &mut prompt,
                    &mut cursor,
                    &mut history_index,
                    &mut draft,
                    &mut suggestion_scroll,
                    history,
                    viewport,
                    content_revision,
                ),
                KeyCode::Char(c) => editing::on_char(
                    c,
                    &mut prompt,
                    &mut cursor,
                    &mut history_index,
                    &mut suggestion_scroll,
                    content_revision,
                ),
                KeyCode::Backspace => editing::on_backspace(
                    &mut prompt,
                    &mut cursor,
                    &mut history_index,
                    &mut suggestion_scroll,
                    content_revision,
                ),
                KeyCode::Tab => editing::on_tab(&mut prompt, &mut cursor, &hint, content_revision),
                KeyCode::Esc => return Ok(None),
                code => handle_scroll_events(code, viewport),
            }
        }
    }

    // Loop only breaks on a non-empty prompt (see the Enter branch), so by here
    // the prompt is guaranteed valid: no empty-prompt exit path remains.
    bump(content_revision);
    viewport.manual_scroll = None;
    Ok(Some(prompt))
}
