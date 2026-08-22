/*
 * Graph Horizon CLI Modules - Plugins - Command - Action
 * Registry actions for session mutation and workspace-confined transcript I/O.
 * Each returns a CommandResult; failures stay generic so no internal detail leaks.
 */

use super::CommandResult;
use crate::graph_horizon_cli::console::render::ChatTurn;
use crate::graph_horizon_cli::plugins::attachments::FileAuthority;
use crate::graph_horizon_cli::plugins::transcript;

pub(super) fn clear_command(
    _files: &FileAuthority,
    _arg: &str,
    _system: Option<&str>,
    _history: &[ChatTurn],
) -> CommandResult {
    CommandResult::Clear
}

pub(super) fn system_command(
    _files: &FileAuthority,
    arg: &str,
    _system: Option<&str>,
    _history: &[ChatTurn],
) -> CommandResult {
    match arg {
        "" => CommandResult::Notice("usage: /system <text> or /system --clear".into()),
        "--clear" => CommandResult::SetSystem(None),
        text => CommandResult::SetSystem(Some(text.to_string())),
    }
}

// Exports the conversation to `arg` (default graph-horizon-export.json). The outcome is
// a notice so it is shown but never re-enters the model context. The path is
// confined to the workspace exactly like /import reads are, and failures stay
// generic — no OS or path detail leaks into the notice (mirroring import).
pub(super) fn export_command(
    files: &FileAuthority,
    arg: &str,
    system: Option<&str>,
    history: &[ChatTurn],
) -> CommandResult {
    let path = if arg.is_empty() {
        "graph-horizon-export.json"
    } else {
        arg
    };
    let Ok(file) = files.open_create(path) else {
        return CommandResult::Notice(
            "⚠ export failed: path is invalid or outside the workspace".into(),
        );
    };
    let notice = match transcript::export(file, system, history) {
        Ok(()) => format!("✓ exported to {path}"),
        Err(_) => "⚠ export failed: could not write the file".into(),
    };
    CommandResult::Notice(notice)
}

// Imports a transcript and asks the session to replace its system prompt and
// history with it. The path is confined to the workspace (no absolute paths, no
// traversal) like the attachment plugin; a missing, outside, or malformed file
// surfaces as a notice and leaves the current conversation untouched.
pub(super) fn import_command(
    files: &FileAuthority,
    arg: &str,
    _system: Option<&str>,
    _history: &[ChatTurn],
) -> CommandResult {
    if arg.is_empty() {
        return CommandResult::Notice("⚠ import failed: missing file path".into());
    }
    let Ok(file) = files.open_read(arg) else {
        return CommandResult::Notice(
            "⚠ import failed: file is missing or outside the workspace".into(),
        );
    };
    match transcript::import(file) {
        Ok((system, history)) => CommandResult::Restore { system, history },
        Err(e) => CommandResult::Notice(format!("⚠ import failed: {e}")),
    }
}
