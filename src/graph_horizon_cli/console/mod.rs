/*
 * Graph Horizon CLI Modules - Console
 * Wires the console renderer, scrolling state, and interactive session. It owns
 * the shared input/render polling constants and the content-revision bump used
 * to invalidate wrapping, but no terminal loop or drawing implementation.
*/

use std::time::Duration;

pub(crate) mod render;
pub(crate) mod scroll;
pub(crate) mod session;

pub(crate) use session::terminal_user_interface;

pub(crate) const INPUT_POLL_INTERVAL: Duration = Duration::from_millis(50);
pub(crate) const PAGE_SCROLL_LINES: u16 = 10;

// Wrapping is cached by revision; wrapping addition deliberately makes cache
// invalidation total even after the counter reaches `u64::MAX`.
fn bump(revision: &mut u64) {
    *revision = revision.wrapping_add(1);
}
