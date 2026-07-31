/*
 * GH Zero CLI Modules - Console
 * This module provides the console user interface.
*/

use std::time::Duration;

pub(crate) mod render;
pub(crate) mod scroll;
pub(crate) mod session;

pub(crate) use session::terminal_user_interface;

pub(crate) const INPUT_POLL_INTERVAL: Duration = Duration::from_millis(50);
pub(crate) const PAGE_SCROLL_LINES: u16 = 10;

// Advances the content revision so RenderCache knows to rewrap on the next draw.
fn bump(revision: &mut u64) {
    *revision = revision.wrapping_add(1);
}
