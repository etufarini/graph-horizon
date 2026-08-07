/*
 * Graph Horizon CLI Modules - Console - Session - Stream
 * Single responsibility: consume one admitted assistant stream while rendering
 * live occupancy and total monotonic duration until its terminal outcome.
*/

use super::super::render::{ChatTurn, RenderCache, RenderContent, draw_viewport};
use super::super::scroll::{ViewportState, drain_stream_events};
use crate::graph_horizon_cli::runtime::{self, ChunkStream, ContextBudget};
use color_eyre::eyre::Result;
use ratatui::DefaultTerminal;
use std::time::{Duration, Instant};
use tokio_stream::StreamExt;

// ~60 fps polling keeps streaming output smooth without busy-spinning. Shared
// with the busy-wait spinner (busy.rs) so both animate at the same cadence.
pub(super) const STREAM_POLL_INTERVAL: Duration = Duration::from_millis(16);

pub(crate) enum StreamOutcome {
    Quit,        // Esc: exit the app
    Interrupted, // Ctrl+C: discard partial output, back to input
    Completed(String, Duration),
}

// Streams the response for one prompt and returns how the phase ended: Quit on Esc,
// Interrupted on Ctrl+C mid-stream (partial output is discarded), or Completed
// with the full assistant response text.
// The arguments are the terminal/render/scroll state this phase drives plus the data it draws;
// bundling them into a struct would add an indirection without making the flow any clearer.
#[allow(clippy::too_many_arguments)]
pub(crate) async fn stream_response<Fut>(
    terminal: &mut DefaultTerminal,
    document: &mut RenderCache,
    content_revision: &mut u64,
    viewport: &mut ViewportState,
    history: &[ChatTurn],
    prompt: &str,
    budget: ContextBudget,
    input_characters: usize,
    started: Instant,
    stream_future: Fut,
) -> Result<StreamOutcome>
where
    Fut: std::future::Future<Output = Result<ChunkStream>>,
{
    // Animate [generating...] while waiting for the HTTP stream to open. Esc is honoured
    // here too: connecting can be the slowest phase (model loading), so the user must be
    // able to exit before the first token arrives.
    tokio::pin!(stream_future);
    let mut stream = loop {
        draw_viewport(
            terminal,
            document,
            *content_revision,
            &RenderContent::output(
                history,
                prompt,
                "",
                budget.usage(input_characters),
                started.elapsed(),
            ),
            viewport,
        )?;
        // Only quit is honoured while connecting: `interrupt` is deliberately ignored
        // here, keeping the connection window's behaviour unchanged (Ctrl+C acts only
        // once the stream is open and tokens are arriving).
        if drain_stream_events(viewport)?.quit {
            return Ok(StreamOutcome::Quit);
        }
        match tokio::time::timeout(STREAM_POLL_INTERVAL, &mut stream_future).await {
            Ok(result) => break result?,
            Err(_) => super::super::bump(content_revision),
        }
    };

    let mut resp = String::new();
    loop {
        let content_changed = match tokio::time::timeout(STREAM_POLL_INTERVAL, stream.next()).await
        {
            Ok(Some(chunk)) => {
                let c = chunk?;
                runtime::response(&c).is_some_and(|r| {
                    resp.push_str(r);
                    true
                })
            }
            Ok(None) => break,
            Err(_) => false,
        };

        // Drain events every frame (not only when idle) so a quit press is acted on
        // immediately even while tokens are streaming in fast.
        let events = drain_stream_events(viewport)?;
        // Quit checked before interrupt: if both arrive in the same drain, quit wins.
        // Either return drops `stream` (and the in-flight network task) on the way out.
        if events.quit {
            return Ok(StreamOutcome::Quit);
        }
        if events.interrupt {
            return Ok(StreamOutcome::Interrupted);
        }

        let live_characters = input_characters.saturating_add(resp.chars().count());

        // `loading` (no token yet) lives in RenderContent; build the snapshot
        // once and reuse its flag instead of recomputing the predicate here.
        let content = RenderContent::output(
            history,
            prompt,
            &resp,
            budget.usage(live_characters),
            started.elapsed(),
        );
        if content_changed || content.loading() {
            super::super::bump(content_revision);
        }

        if content_changed || events.redraw || content.loading() {
            draw_viewport(terminal, document, *content_revision, &content, viewport)?;
        }
    }

    Ok(StreamOutcome::Completed(resp, started.elapsed()))
}
