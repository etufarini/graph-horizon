/*
 * Graph Horizon CLI Modules - Console - Render - Content
 * Single responsibility: describe one text-chat frame plus its current
 * capacity and monotonic-duration presentation state.
 */

use std::time::Duration;

use super::ChatTurn;
use crate::graph_horizon_cli::runtime::{
    CapacityError, ContextUsage, GenerationPhase, GenerationStats, RuntimeInfo,
};

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
    pub(super) runtime: Option<&'a RuntimeInfo>,
    pub(super) loading: bool,
    pub(super) usage: ContextUsage,
    pub(super) duration: Option<Duration>,
    pub(super) phase: Option<GenerationPhase>,
    pub(super) phase_duration: Option<Duration>,
    pub(super) stats: Option<GenerationStats>,
    pub(super) capacity_error: Option<CapacityError>,
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
        usage: ContextUsage,
        runtime: Option<&'a RuntimeInfo>,
        duration: Option<Duration>,
        stats: Option<GenerationStats>,
        capacity_error: Option<CapacityError>,
    ) -> Self {
        Self {
            history,
            prompt,
            hint: (!hint.is_empty()).then_some(hint),
            caret: Some(caret),
            suggestions,
            suggestion_scroll,
            resp: "",
            runtime,
            loading: false,
            usage,
            duration,
            phase: None,
            phase_duration: None,
            stats,
            capacity_error,
        }
    }

    // Constructs a content snapshot for the streaming/output phase.
    pub(crate) fn output(
        history: &'a [ChatTurn],
        prompt: &'a str,
        resp: &'a str,
        usage: ContextUsage,
        runtime: Option<&'a RuntimeInfo>,
        phase: Option<GenerationPhase>,
        phase_duration: Duration,
    ) -> Self {
        Self {
            history,
            prompt,
            hint: None,
            caret: None,
            suggestions: &[],
            suggestion_scroll: 0,
            resp,
            runtime,
            loading: resp.is_empty(),
            usage,
            // Streaming status is phase-local; total time is retained only for
            // providers that finish without Graph Horizon metrics.
            duration: None,
            phase,
            phase_duration: Some(phase_duration),
            stats: None,
            capacity_error: None,
        }
    }

    // Whether the turn is still waiting on its first token (drives the spinner).
    // The predicate lives only here so the streaming loop need not recompute it.
    pub(crate) fn loading(&self) -> bool {
        self.loading
    }
}
