/*
 * Graph Orizon CLI Modules - Console - Session
 * Single responsibility: drive one interactive text-chat lifecycle with input,
 * slash commands, streaming, and history commit. It depends on console render,
 * plugins, and runtime streams, and does not own tools or reasoning channels.
*/

mod dispatch;
mod history;
mod input;
mod request;
mod stream;

use super::render::{ChatTurn, RenderCache, TokenStatus, conversation_tokens};
use super::scroll::ViewportState;
use crate::graph_orizon_cli::plugins::attachments::FileAuthority;
use crate::graph_orizon_cli::runtime::{ChatMessage, ChunkStream, Throughput};
use color_eyre::eyre::Result;
use ratatui::DefaultTerminal;

// Runs the terminal chat lifecycle until the user presses Esc.
pub(crate) async fn terminal_user_interface<Fut>(
    terminal: &mut DefaultTerminal,
    // Mutable for the run: a /import replaces the active system prompt with the
    // one recorded in the restored transcript.
    mut system: Option<String>,
    // Pruning threshold for the whole run; None disables pruning and drives the
    // static "pruning off" indicator.
    threshold: Option<usize>,
    // Raw context window for the whole run; shown bottom-left as a fixed pruning
    // reference. None (no limit) leaves the left side empty.
    context_limit: Option<usize>,
    files: &FileAuthority,
    generate: impl Fn(Vec<ChatMessage>) -> Fut,
) -> Result<()>
where
    Fut: std::future::Future<Output = Result<ChunkStream>>,
{
    let mut viewport = ViewportState::default();
    let mut document = RenderCache::default();
    let mut content_revision = 0_u64;
    // The conversation starts empty; a /import restores a saved one in-session.
    let mut history: Vec<ChatTurn> = Vec::new();
    // Computed once per turn (not per frame) and frozen until a turn completes:
    // the status bar shows this without recounting the whole history each draw.
    let mut token_count = conversation_tokens(system.as_deref(), &history);
    // Last measured token/s rates, frozen between turns and shown while the user
    // types the next prompt; the streaming phase replaces it on each turn.
    let mut throughput = Throughput::default();
    // Last completed turn's output-token estimate, frozen while the user types
    // the next prompt. An errored or empty turn clears it back to zero.
    let mut output_tokens = 0;
    // Text that prefills the next input field; set only by a Ctrl+C interruption,
    // consumed (taken) by every read_prompt call so it never survives past one turn.
    let mut prefill = String::new();
    loop {
        // The indicator depends only on whether a threshold exists: no threshold
        // means pruning is off, otherwise pruning is active (the "∞" marker).
        let status = match threshold {
            None => TokenStatus::Off,
            Some(_) => TokenStatus::Active,
        };
        let Some(prompt) = input::read_prompt(
            terminal,
            &mut document,
            &mut viewport,
            &mut content_revision,
            &history,
            std::mem::take(&mut prefill),
            status,
            token_count,
            throughput,
            output_tokens,
            context_limit,
            files,
        )?
        else {
            return Ok(());
        };

        // Slash-commands and @attachment expansion are handled out of line; the
        // loop only sees whether to exit, restart, or proceed with the model.
        let expanded = match dispatch::dispatch(
            &mut content_revision,
            &mut viewport,
            &mut system,
            &mut history,
            &mut token_count,
            &mut output_tokens,
            &mut throughput,
            &prompt,
            files,
        )? {
            dispatch::Dispatch::Handled => continue,
            dispatch::Dispatch::Proceed(expanded) => expanded,
        };

        // Prepare this turn's outgoing request (assembly, pruning).
        let (request, input_tokens) =
            request::assemble(system.as_deref(), &history, &expanded, threshold);
        let stream_future = generate(request);
        let outcome = stream::stream_response(
            terminal,
            &mut document,
            &mut content_revision,
            &mut viewport,
            &history,
            &prompt,
            status,
            token_count,
            input_tokens,
            output_tokens,
            context_limit,
            stream_future,
        )
        .await;

        // Commit the finished turn (history + counters) and read back the loop
        // control signal; Quit/Interrupted keep every counter frozen.
        match history::commit_turn(
            outcome,
            &mut history,
            system.as_deref(),
            prompt,
            expanded,
            &mut token_count,
            &mut output_tokens,
            &mut throughput,
        ) {
            history::Commit::Quit => return Ok(()),
            history::Commit::Interrupted(prompt) => {
                prefill = prompt;
                viewport.manual_scroll = None;
                super::bump(&mut content_revision);
                continue;
            }
            history::Commit::Continued => {
                viewport.manual_scroll = None;
                super::bump(&mut content_revision);
            }
        }
    }
}
