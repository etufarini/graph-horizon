/*
 * GH Zero CLI Modules - Plugins - Command - Completion
 * Single responsibility: complete retained slash command names and confined path
 * arguments. It depends on attachments/path completion and does not complete
 * tools, confirmations, profiles, workspace, or reasoning commands.
 */

use super::args::path_arg;
use super::{COMMANDS, parse};
use crate::gh_zero_cli::plugins::attachments::{self, FileAuthority};
use crate::gh_zero_cli::plugins::completion::completion_tail;

// Completion tail and candidate list shown while typing, both from one pass so
// the input phase never recomputes them. Cases: still typing the command name
// ("/imp" -> "ort") or the path argument of a path-completing command.
pub(crate) fn complete(files: &FileAuthority, prompt: &str) -> (Option<String>, Vec<String>) {
    match parse(prompt) {
        Some((command, rest)) if command.completes_path => match path_arg(rest) {
            // Path stage: complete the file name and list the matching .json entries.
            Some(partial) => attachments::complete_path(files, partial, Some("json")),
            // Command typed in full but not yet separated from its argument: offer
            // the space so the next completion targets the path.
            None if rest.is_empty() => (Some(" ".to_string()), Vec::new()),
            None => (None, Vec::new()),
        },
        // A fully recognised command with nothing left to complete.
        Some(_) => (None, Vec::new()),
        None => (complete_name(prompt), name_matches(prompt)),
    }
}

// Completion tail for a partially typed command name ("/imp" -> "ort"), using the
// longest common prefix of every matching command. None when the name is complete
// or matches nothing.
fn complete_name(prompt: &str) -> Option<String> {
    let typed = prompt.trim_start();
    let matches = name_matches(prompt);
    let tail = completion_tail(typed, &matches)?;
    // Once the common prefix is a whole path-taking command, append the separating
    // space so completing the name lands ready to complete the path in one step.
    let completed = format!("{typed}{tail}");
    let space = COMMANDS
        .iter()
        .any(|command| command.name == completed && command.completes_path);
    Some(if space { format!("{tail} ") } else { tail })
}

// Command names that start with what the user has typed so far. Empty unless the
// prompt opens with '/', so ordinary prompts are never read as command typing.
fn name_matches(prompt: &str) -> Vec<String> {
    let typed = prompt.trim_start();
    if !typed.starts_with('/') {
        return Vec::new();
    }
    COMMANDS
        .iter()
        .map(|command| command.name)
        .filter(|name| name.starts_with(typed))
        .map(str::to_string)
        .collect()
}
