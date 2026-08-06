/*
 * Graph Horizon CLI Modules - Console - Session
 * Single responsibility: drive one interactive text-chat lifecycle with input,
 * slash commands, streaming, and history commit. It depends on console render,
 * plugins, and runtime streams, and does not own tools or reasoning channels.
*/

mod dispatch;
mod history;
mod input;
mod request;
mod stream;

use super::render::{ChatTurn, RenderCache, TokenStatus, conversation_characters};
use super::scroll::ViewportState;
use crate::graph_horizon_cli::plugins::attachments::FileAuthority;
use crate::graph_horizon_cli::runtime::{
    CapacityError, ChatMessage, ChunkStream, ContextBudget, Throughput,
};
use color_eyre::eyre::Result;
use ratatui::DefaultTerminal;

// Runs the terminal chat lifecycle until the user presses Esc.
pub(crate) async fn terminal_user_interface<Fut>(
    terminal: &mut DefaultTerminal,
    // Mutable for the run: a /import replaces the active system prompt with the
    // one recorded in the restored transcript.
    mut system: Option<String>,
    budget: ContextBudget,
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
    let mut committed_characters = conversation_characters(system.as_deref(), &history);
    // Last measured token/s rates, frozen between turns and shown while the user
    // types the next prompt; the streaming phase replaces it on each turn.
    let mut throughput = Throughput::default();
    // Last completed turn's output-token estimate, frozen while the user types
    // the next prompt. An errored or empty turn clears it back to zero.
    let mut output_tokens = 0;
    // Text that prefills the next input field; set only by a Ctrl+C interruption,
    // consumed (taken) by every read_prompt call so it never survives past one turn.
    let mut prefill = String::new();
    let mut capacity_error = None;
    loop {
        let status = TokenStatus::Active;
        let Some(prompt) = input::read_prompt(
            terminal,
            &mut document,
            &mut viewport,
            &mut content_revision,
            &history,
            std::mem::take(&mut prefill),
            status,
            committed_characters,
            throughput,
            output_tokens,
            budget,
            &mut capacity_error,
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
            &mut committed_characters,
            &mut output_tokens,
            &mut throughput,
            &prompt,
            files,
        )? {
            dispatch::Dispatch::Handled => continue,
            dispatch::Dispatch::Proceed(expanded) => expanded,
        };

        let prepared = request::assemble(system.as_deref(), &history, &expanded, budget);
        let (stream_future, input_characters) = match provider_future(prepared, &generate) {
            Ok(started) => started,
            Err(error) => {
                restore_rejected_prompt(&mut prefill, &mut capacity_error, prompt, error);
                viewport.manual_scroll = None;
                super::bump(&mut content_revision);
                continue;
            }
        };
        let input_tokens = budget.usage(input_characters).estimated_messages;
        let context_limit = Some(budget.usage(0).context_limit);
        let outcome = stream::stream_response(
            terminal,
            &mut document,
            &mut content_revision,
            &mut viewport,
            &history,
            &prompt,
            status,
            budget
                .usage(committed_characters.unwrap_or(usize::MAX))
                .estimated_messages,
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
            &mut committed_characters,
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

fn provider_future<F, Fut>(
    prepared: Result<request::PreparedRequest, CapacityError>,
    generate: &F,
) -> Result<(Fut, usize), CapacityError>
where
    F: Fn(Vec<ChatMessage>) -> Fut,
{
    let prepared = prepared?;
    let characters = prepared.characters;
    Ok((generate(prepared.messages), characters))
}

fn restore_rejected_prompt(
    prefill: &mut String,
    capacity_error: &mut Option<CapacityError>,
    prompt: String,
    error: CapacityError,
) {
    // Rejection happens before transport and restores the raw spelling, never
    // the attachment-expanded request content used by admission.
    *prefill = prompt;
    *capacity_error = Some(error);
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    #[test]
    fn over_budget_request_preserves_prompt_without_invoking_provider() {
        let budget = ContextBudget::new(100, 10).unwrap();
        let prepared = request::assemble(None, &[], &"x".repeat(400), budget);
        let calls = Cell::new(0);

        let result = provider_future(prepared, &|_| {
            calls.set(calls.get() + 1);
            std::future::ready(())
        });

        assert!(result.is_err());
        assert_eq!(calls.get(), 0);
    }

    #[test]
    fn expanded_attachment_is_counted_but_raw_prompt_is_restored() {
        let budget = ContextBudget::new(100, 10).unwrap();
        let error = match request::assemble(None, &[], &"file".repeat(100), budget) {
            Ok(_) => panic!("expanded request should exceed capacity"),
            Err(error) => error,
        };
        let mut prefill = String::new();
        let mut capacity_error = None;

        restore_rejected_prompt(
            &mut prefill,
            &mut capacity_error,
            "read @file.txt".into(),
            error,
        );

        assert_eq!(prefill, "read @file.txt");
        assert_eq!(capacity_error.unwrap().estimated_messages, 100);
    }
}
