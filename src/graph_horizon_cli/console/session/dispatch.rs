/*
 * Graph Horizon CLI Modules - Console - Session - Dispatch
 * Single responsibility: handle retained slash commands, reset imported
 * context accounting, and expand attachments before generation.
 */

use super::super::bump;
use super::super::render::{ChatTurn, conversation_characters};
use super::super::scroll::ViewportState;
use crate::graph_horizon_cli::plugins::attachments::FileAuthority;
use crate::graph_horizon_cli::plugins::{attachments, command};
use crate::graph_horizon_cli::runtime::Throughput;
use color_eyre::eyre::Result;

// Outcome of dispatching one prompt back to the session loop.
pub(super) enum Dispatch {
    // A command consumed the prompt; the loop should start the next iteration.
    Handled,
    // Not a command: carries the attachment-expanded prompt for the model path.
    Proceed(String),
}

// Recognises retained slash commands, then expands @attachments and returns the
// text for the model path. Whole-word matching stays delegated to command::run.
#[allow(clippy::too_many_arguments)]
pub(super) fn dispatch(
    content_revision: &mut u64,
    viewport: &mut ViewportState,
    system: &mut Option<String>,
    history: &mut Vec<ChatTurn>,
    committed_characters: &mut Option<usize>,
    output_tokens: &mut usize,
    throughput: &mut Throughput,
    prompt: &str,
    files: &FileAuthority,
) -> Result<Dispatch> {
    // Intercept registry plugin commands (/export, /import) before they can reach
    // the model. A non-command prompt returns None and flows on unchanged.
    match command::run(files, prompt, system.as_deref(), history) {
        Some(command::CommandResult::Notice(text)) => {
            // Shown as a notice turn with no messages, so it stays out of the
            // model context and out of any later export.
            history.push(ChatTurn::new(prompt.to_string(), text, Vec::new()));
            viewport.manual_scroll = None;
            bump(content_revision);
            return Ok(Dispatch::Handled);
        }
        Some(command::CommandResult::Restore {
            system: restored_system,
            history: restored_history,
        }) => {
            // /import replaces the whole conversation: swap the system prompt and
            // history, then reset the per-turn token and throughput readouts that
            // no longer describe the restored state.
            *system = restored_system;
            *history = restored_history;
            *committed_characters = conversation_characters(system.as_deref(), history);
            *output_tokens = 0;
            *throughput = Throughput::default();
            viewport.manual_scroll = None;
            bump(content_revision);
            return Ok(Dispatch::Handled);
        }
        None => {}
    }

    Ok(Dispatch::Proceed(attachments::attach_local_files(
        files, prompt,
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph_horizon_cli::runtime::ChatMessage;

    #[test]
    fn import_recomputes_committed_characters() {
        let history = vec![ChatTurn::new(
            "shown".into(),
            "shown".into(),
            vec![
                ChatMessage::user("😀".into()),
                ChatMessage::assistant("reply".into()),
            ],
        )];

        assert_eq!(conversation_characters(Some("sys"), &history), Some(9));
    }
}
