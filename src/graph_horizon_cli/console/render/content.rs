/*
 * Graph Horizon CLI Modules - Console - Render - Content
 * Single responsibility: describe one text-chat frame for line building and
 * drawing. It carries input/output state only and does not model shell
 * confirmations, tools, workspace, or separate reasoning views.
 */

use super::{ChatTurn, TokenStatus};
use crate::graph_horizon_cli::runtime::Throughput;

// Snapshot of content to be rendered in a single frame.
pub(crate) struct RenderContent<'a> {
    pub(crate) history: &'a [ChatTurn],
    pub(crate) prompt: &'a str,
    pub(crate) hint: Option<&'a str>,
    // Cursor position (character index) in the live input prompt, used to draw the
    // caret. None during streaming, where there is no editable prompt.
    pub(crate) caret: Option<usize>,
    pub(crate) suggestions: &'a [String],
    pub(crate) suggestion_scroll: usize,
    pub(crate) resp: &'a str,
    pub(super) loading: bool,
    // Drives the token indicator: the "pruning off" label and the grey/orange
    // count, computed once per turn by the session (see TokenStatus).
    pub(super) status: TokenStatus,
    // Pre-computed by the session once per turn; the count is frozen between
    // turns, so passing it in avoids recomputing it on every rendered frame.
    pub(super) token_count: usize,
    // Token/s rates shown left of the count: updated live by the streaming phase
    // and frozen by the session between turns, plumbed exactly like token_count.
    pub(super) throughput: Throughput,
    // Per-turn output-token estimate shown right of the speed. It is recomputed
    // from accumulated assistant text while streaming and frozen between turns.
    pub(super) output_tokens: usize,
    // Raw context window, constant for the whole run. Drives the bottom-left
    // pruning reference; None (no limit) leaves the left side empty.
    pub(super) context_limit: Option<usize>,
}

impl<'a> RenderContent<'a> {
    // Constructs a content snapshot for the input phase.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn input(
        history: &'a [ChatTurn],
        prompt: &'a str,
        hint: &'a str,
        caret: usize,
        suggestions: &'a [String],
        suggestion_scroll: usize,
        status: TokenStatus,
        token_count: usize,
        throughput: Throughput,
        output_tokens: usize,
        context_limit: Option<usize>,
    ) -> Self {
        Self {
            history,
            prompt,
            hint: (!hint.is_empty()).then_some(hint),
            caret: Some(caret),
            suggestions,
            suggestion_scroll,
            resp: "",
            loading: false,
            status,
            token_count,
            throughput,
            output_tokens,
            context_limit,
        }
    }

    // Constructs a content snapshot for the streaming/output phase.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn output(
        history: &'a [ChatTurn],
        prompt: &'a str,
        resp: &'a str,
        status: TokenStatus,
        token_count: usize,
        throughput: Throughput,
        output_tokens: usize,
        context_limit: Option<usize>,
    ) -> Self {
        Self {
            history,
            prompt,
            hint: None,
            caret: None,
            suggestions: &[],
            suggestion_scroll: 0,
            resp,
            loading: resp.is_empty(),
            status,
            token_count,
            throughput,
            output_tokens,
            context_limit,
        }
    }

    // Whether the turn is still waiting on its first token (drives the spinner).
    // The predicate lives only here so the streaming loop need not recompute it.
    pub(crate) fn loading(&self) -> bool {
        self.loading
    }
}
