/*
 * Graph Horizon CLI Modules - Console - Session - History
 * Single responsibility: commit successful raw messages and refresh the
 * checked character count while excluding failed or empty turns.
 */

use super::super::render::{ChatTurn, conversation_characters};
use super::stream::StreamOutcome;
use crate::graph_horizon_cli::runtime::ChatMessage;
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
    committed_characters: &mut Option<usize>,
    duration: &mut Option<std::time::Duration>,
) -> Commit {
    match outcome {
        // The user pressed Esc mid-generation: quit the app.
        Ok(StreamOutcome::Quit) => return Commit::Quit,
        // Ctrl+C: the turn vanishes; the original prompt (not the expanded/outgoing
        // variants) prefills the next input.
        Ok(StreamOutcome::Interrupted) => return Commit::Interrupted(prompt),
        Ok(StreamOutcome::Completed(resp, elapsed)) => {
            *duration = Some(elapsed);
            // An empty reply does not re-enter the context: no messages go into
            // history, so the turn is shown but never resent.
            let msgs = if resp.trim().is_empty() {
                Vec::new()
            } else {
                vec![
                    // Attachments affect only the submitted request. Successful
                    // committed history retains the user's raw prompt spelling.
                    ChatMessage::user(prompt.clone()),
                    ChatMessage::assistant(resp.clone()),
                ]
            };
            history.push(ChatTurn::new(prompt, resp, msgs));
        }
        Err(e) => {
            *duration = None;
            history.push(ChatTurn::new(prompt, format!("⚠ {e}"), Vec::new()));
        }
    }

    // Refresh the count only now, reflecting the turn that just entered history,
    // so it updates in one frame instead of mid-stream.
    *committed_characters = conversation_characters(system, history);
    Commit::Continued
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn completed_stream_freezes_total_duration() {
        let mut history = Vec::new();
        let mut characters = Some(0);
        let mut duration = None;

        let commit = commit_turn(
            Ok(StreamOutcome::Completed(
                "answer".into(),
                std::time::Duration::from_millis(1234),
            )),
            &mut history,
            None,
            "prompt".into(),
            &mut characters,
            &mut duration,
        );

        assert!(matches!(commit, Commit::Continued));
        assert_eq!(duration, Some(std::time::Duration::from_millis(1234)));
    }

    #[test]
    fn interrupted_stream_publishes_no_duration() {
        let mut history = Vec::new();
        let mut characters = Some(0);
        let mut duration = None;

        let commit = commit_turn(
            Ok(StreamOutcome::Interrupted),
            &mut history,
            None,
            "prompt".into(),
            &mut characters,
            &mut duration,
        );

        assert!(matches!(commit, Commit::Interrupted(_)));
        assert_eq!(duration, None);
        assert!(history.is_empty());
    }
}
