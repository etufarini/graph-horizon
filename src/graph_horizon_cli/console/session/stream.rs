/*
 * Graph Horizon CLI Modules - Console - Session - Stream
 * Single responsibility: consume one assistant text stream and report terminal
 * state to the session. It depends on rendering, input events, and runtime
 * chunks, and does not render tools or a separate reasoning channel.
*/

use super::super::render::{ChatTurn, RenderCache, RenderContent, TokenStatus, draw_viewport};
use super::super::scroll::{ViewportState, drain_stream_events};
use crate::graph_horizon_cli::runtime::{self, ChunkStream, Throughput, rate};
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
    Completed(String, Throughput),
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
    status: TokenStatus,
    token_count: usize,
    input_tokens: usize,
    // Previous turn's output estimate. It is shown only while connecting; once
    // current text arrives, the status bar uses the live estimate.
    output_tokens: usize,
    context_limit: Option<usize>,
    stream_future: Fut,
) -> Result<StreamOutcome>
where
    Fut: std::future::Future<Output = Result<ChunkStream>>,
{
    // The prefill window starts when this phase begins and closes on the first
    // real token. That keeps input throughput tied to the request that is now in
    // flight, while generation throughput still starts at the first token.
    let started = Instant::now();
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
                status,
                token_count,
                Throughput::default(),
                output_tokens,
                context_limit,
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
    let mut first_token_at: Option<Instant> = None;
    let mut throughput = Throughput::default();
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

        // Mark the first real token: it closes prefill and starts generation.
        // Both rates stay hidden until this boundary exists.
        if content_changed && first_token_at.is_none() {
            first_token_at = Some(Instant::now());
        }
        if let Some(first) = first_token_at {
            let output_tokens = runtime::estimate_tokens(resp.chars().count());
            throughput = Throughput {
                input: rate(input_tokens, first.saturating_duration_since(started)),
                output: rate(output_tokens, first.elapsed()),
            };
        }
        let live_output_tokens = runtime::estimate_tokens(resp.chars().count());

        // `loading` (no token yet) lives in RenderContent; build the snapshot
        // once and reuse its flag instead of recomputing the predicate here.
        let content = RenderContent::output(
            history,
            prompt,
            &resp,
            status,
            token_count,
            throughput,
            live_output_tokens,
            context_limit,
        );
        if content_changed || content.loading() {
            super::super::bump(content_revision);
        }

        if content_changed || events.redraw || content.loading() {
            draw_viewport(terminal, document, *content_revision, &content, viewport)?;
        }
    }

    Ok(StreamOutcome::Completed(resp, throughput))
}
