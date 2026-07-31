/*
 * GH Zero CLI Modules - Console - Render
 * Public rendering API for the terminal chat view: content snapshot, line cache, frame drawing.
 * Entry point that wires the render submodules and re-exports their public surface.
 */

mod cache;
mod content;
mod conversation;
mod draw;
mod status;
mod wrap;

pub(crate) use cache::RenderCache;
pub(crate) use content::RenderContent;
pub(crate) use conversation::{ChatTurn, SUGGESTION_THRESHOLD, conversation_tokens};
pub(crate) use draw::draw_viewport;
pub(crate) use status::{SectionStyle, TokenStatus, format_span};
