/*
 * Graph Horizon CLI Modules - Console - Session - Dispatch
 * Single responsibility: apply session/transcript slash commands, refresh
 * context accounting, and expand attachments before generation.
 */

use super::super::bump;
use super::super::render::{ChatTurn, conversation_characters};
use super::super::scroll::ViewportState;
use crate::graph_horizon_cli::plugins::attachments::FileAuthority;
use crate::graph_horizon_cli::plugins::{attachments, command};
use crate::graph_horizon_cli::runtime::GenerationStats;
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
    duration: &mut Option<std::time::Duration>,
    stats: &mut Option<GenerationStats>,
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
        Some(command::CommandResult::Clear) => {
            history.clear();
            *committed_characters = conversation_characters(system.as_deref(), history);
            *duration = None;
            *stats = None;
            viewport.manual_scroll = None;
            bump(content_revision);
            return Ok(Dispatch::Handled);
        }
        Some(command::CommandResult::SetSystem(value)) => {
            *system = value;
            *committed_characters = conversation_characters(system.as_deref(), history);
            *duration = None;
            *stats = None;
            // Never retain the entered prompt: it may contain private
            // instructions. The generic notice remains outside model context.
            history.push(ChatTurn::new(
                "/system".into(),
                "✓ system prompt updated".into(),
                Vec::new(),
            ));
            viewport.manual_scroll = None;
            bump(content_revision);
            return Ok(Dispatch::Handled);
        }
        Some(command::CommandResult::Restore {
            system: restored_system,
            history: restored_history,
        }) => {
            // /import replaces the whole conversation: swap the system prompt and
            // history, then clear the prior generation duration because it no
            // longer describes the restored session.
            *system = restored_system;
            *history = restored_history;
            *committed_characters = conversation_characters(system.as_deref(), history);
            *duration = None;
            *stats = None;
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

    #[test]
    fn system_notice_does_not_retain_private_text() {
        let mut revision = 0;
        let mut viewport = ViewportState::default();
        let mut system = None;
        let mut history = Vec::new();
        let mut characters = Some(0);
        let mut duration = None;
        let mut stats = None;
        let files = FileAuthority::capture().unwrap();

        let result = dispatch(
            &mut revision,
            &mut viewport,
            &mut system,
            &mut history,
            &mut characters,
            &mut duration,
            &mut stats,
            "/system private",
            &files,
        )
        .unwrap();

        assert!(matches!(result, Dispatch::Handled));
        assert_eq!(system.as_deref(), Some("private"));
        assert_eq!(characters, Some(7));
        assert_eq!(history[0].prompt, "/system");
        assert!(!history[0].response.contains("private"));
    }

    #[test]
    fn clear_preserves_only_the_system_context() {
        let mut revision = 0;
        let mut viewport = ViewportState::default();
        let mut system = Some("system".to_string());
        let mut history = vec![ChatTurn::new(
            "user".into(),
            "answer".into(),
            vec![
                ChatMessage::user("user".into()),
                ChatMessage::assistant("answer".into()),
            ],
        )];
        let mut characters = Some(16);
        let mut duration = Some(std::time::Duration::from_secs(1));
        let mut stats = None;
        let files = FileAuthority::capture().unwrap();

        let result = dispatch(
            &mut revision,
            &mut viewport,
            &mut system,
            &mut history,
            &mut characters,
            &mut duration,
            &mut stats,
            "/clear",
            &files,
        )
        .unwrap();

        assert!(matches!(result, Dispatch::Handled));
        assert!(history.is_empty());
        assert_eq!(system.as_deref(), Some("system"));
        assert_eq!(characters, Some(6));
        assert_eq!(duration, None);
    }
}
