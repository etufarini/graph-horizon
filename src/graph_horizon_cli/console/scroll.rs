/*
 * Graph Horizon CLI Modules - Console - Scroll
 * Viewport scroll state and event handling.
*/

use color_eyre::eyre::Result;
use crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use std::time::Duration;

// Tracks the scroll position of the output viewport.
// `manual_scroll` is Some when the user has scrolled manually; None means auto-scroll to
// the bottom (following new content). `scroll` holds the last known auto-scroll position
// and is used as the baseline when the user begins a manual scroll from the current bottom.
// `last_max_scroll` records the document bottom from the last render so a user scroll that
// reaches it can resume auto-follow without confusing it with a resize-induced clamp.
#[derive(Default)]
pub(crate) struct ViewportState {
    pub(crate) manual_scroll: Option<u16>,
    scroll: u16,
    last_max_scroll: u16,
}

// Applies a single key event to the viewport scroll state. Any directional key switches
// the viewport into manual mode; End resets it back to auto-scroll. A downward scroll that
// reaches the document bottom also resumes auto-follow, since that is an explicit user action
// (unlike a resize/reflow that merely clamps the position against a shrunken document).
pub(crate) fn handle_scroll_events(code: KeyCode, viewport: &mut ViewportState) {
    let base = viewport.manual_scroll.unwrap_or(viewport.scroll);
    let target = match code {
        KeyCode::Up => Some(base.saturating_sub(1)),
        KeyCode::Down => Some(base.saturating_add(1)),
        KeyCode::PageUp => Some(base.saturating_sub(super::PAGE_SCROLL_LINES)),
        KeyCode::PageDown => Some(base.saturating_add(super::PAGE_SCROLL_LINES)),
        KeyCode::Home => Some(0),
        KeyCode::End => None,
        _ => return,
    };
    // Resume auto-follow when the user scrolls down to (or past) the document bottom.
    viewport.manual_scroll = match target {
        Some(value) if value >= viewport.last_max_scroll => None,
        other => other,
    };
}

// Outcome of draining the terminal event queue for one streaming frame.
// `redraw` requests a viewport repaint; `quit` signals the user asked to exit;
// `interrupt` signals the user asked to stop the generation, not the app.
#[derive(Default)]
pub(crate) struct DrainOutcome {
    pub(crate) redraw: bool,
    pub(crate) quit: bool,
    pub(crate) interrupt: bool,
}

// Drains all pending terminal events during streaming, applying scroll keypresses and
// reporting whether the user asked to quit. Quit detection lives here because the event
// queue can only be consumed once: scroll and exit must be read in the same pass.
pub(crate) fn drain_stream_events(viewport: &mut ViewportState) -> Result<DrainOutcome> {
    let mut outcome = DrainOutcome::default();

    while event::poll(Duration::from_millis(0))? {
        match event::read()? {
            Event::Key(k) if k.kind == KeyEventKind::Press => {
                // Ctrl+C means "stop the generation, not the app". The modifier check is
                // explicit: a plain 'c' must never trigger it. Raw mode delivers Ctrl+C
                // as Char('c') + CONTROL, so no uppercase variant is handled.
                if k.modifiers.contains(KeyModifiers::CONTROL) && k.code == KeyCode::Char('c') {
                    outcome.interrupt = true;
                    continue;
                }
                // Esc means "exit the app" during streaming.
                if k.code == KeyCode::Esc {
                    outcome.quit = true;
                    continue;
                }
                let previous_scroll = viewport.manual_scroll;
                handle_scroll_events(k.code, viewport);
                outcome.redraw |= viewport.manual_scroll != previous_scroll;
            }
            Event::Resize(..) => outcome.redraw = true,
            _ => {}
        }
    }

    Ok(outcome)
}

// Called after each render with the new max scroll. Clamps manual scroll so it can't exceed
// the document bottom, keeping the viewport in manual mode: a resize/reflow that shrinks the
// document must not be mistaken for the user scrolling to the bottom. Resuming auto-follow is
// handled in handle_scroll_events, where it can be tied to an explicit user action.
// In auto-scroll mode records the position as the baseline for when the user next scrolls manually.
pub(crate) fn sync_scroll_state(viewport: &mut ViewportState, max_scroll: u16) {
    viewport.last_max_scroll = max_scroll;
    if let Some(value) = viewport.manual_scroll {
        viewport.manual_scroll = Some(value.min(max_scroll));
    } else {
        viewport.scroll = max_scroll;
    }
}
