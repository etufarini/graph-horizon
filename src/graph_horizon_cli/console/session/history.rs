/*
 * Graph Horizon CLI Modules - Console - Session - History
 * Single responsibility: commit a completed text-chat turn into history and
 * refresh token counters. It depends on render/runtime message types and does
 * not persist tools or a separate reasoning channel.
 */

use super::super::render::{ChatTurn, conversation_tokens};
use super::stream::StreamOutcome;
use crate::graph_horizon_cli::runtime::{ChatMessage, Throughput, estimate_tokens};
use color_eyre::eyre::Result;

// What the loop should do after a turn was committed.
pub(super) enum Commit {
    // Esc mid-generation: quit the app, dropping the in-flight stream.
    Quit,
    // Ctrl+C: the turn vanishes without trace; carries the original prompt to
    // prefill the next input, with every counter left frozen at the prior turn.
    Interrupted(String),
    // The turn entered history (completed or errored); counters were refreshed.
    Continued,
}

// Applies a finished turn's outcome to the session state. On a completed or
// errored turn it pushes the turn into history, recomputes the per-turn counts,
// and refreshes the frozen token count so it updates in one frame. Quit and
// Interrupted touch no state here: they only signal the loop, which keeps every
// counter frozen at the previous turn's values.
#[allow(clippy::too_many_arguments)]
pub(super) fn commit_turn(
    outcome: Result<StreamOutcome>,
    history: &mut Vec<ChatTurn>,
    system: Option<&str>,
    prompt: String,
    expanded: String,
    token_count: &mut usize,
    output_tokens: &mut usize,
    throughput: &mut Throughput,
) -> Commit {
    match outcome {
        // The user pressed Esc mid-generation: quit the app.
        Ok(StreamOutcome::Quit) => return Commit::Quit,
        // Ctrl+C: the turn vanishes; the original prompt (not the expanded/outgoing
        // variants) prefills the next input.
        Ok(StreamOutcome::Interrupted) => return Commit::Interrupted(prompt),
        Ok(StreamOutcome::Completed(resp, tp)) => {
            *throughput = tp;
            *output_tokens = estimate_tokens(resp.chars().count());
            // An empty reply does not re-enter the context: no messages go into
            // history, so the turn is shown but never resent.
            let msgs = if resp.trim().is_empty() {
                Vec::new()
            } else {
                vec![
                    ChatMessage::user(expanded),
                    ChatMessage::assistant(resp.clone()),
                ]
            };
            history.push(ChatTurn::new(prompt, resp, msgs));
        }
        Err(e) => {
            // The turn produced no model output (only an error notice), so the
            // breakdown is cleared rather than left showing the prior turn.
            *output_tokens = 0;
            history.push(ChatTurn::new(prompt, format!("⚠ {e}"), Vec::new()));
        }
    }

    // Refresh the count only now, reflecting the turn that just entered history,
    // so it updates in one frame instead of mid-stream.
    *token_count = conversation_tokens(system, history);
    Commit::Continued
}
