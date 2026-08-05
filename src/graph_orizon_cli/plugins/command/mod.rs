/*
 * Graph Orizon CLI Modules - Plugins - Command
 * Single responsibility: map retained chat slash commands to transcript actions
 * and completion. It depends on plugin actions/render history and does not
 * expose tools, confirmations, profiles, workspace, or reasoning commands.
 */

use crate::graph_orizon_cli::console::render::ChatTurn;
use crate::graph_orizon_cli::plugins::attachments::FileAuthority;
use action::{export_command, import_command};

mod action;
mod args;
mod completion;

pub(crate) use completion::complete;

// What running a command produces. A command never reaches the model: it either
// shows a local notice (kept out of the model context) or restores a whole
// conversation, replacing the current system prompt and history.
pub(crate) enum CommandResult {
    Notice(String),
    Restore {
        system: Option<String>,
        history: Vec<ChatTurn>,
    },
}

// One registered command. `completes_path` is the only per-command knob the
// autocomplete needs: true means the trailing argument is a workspace path that
// should be completed (currently only /import, restricted to JSON files).
struct Command {
    name: &'static str,
    completes_path: bool,
    action: fn(
        files: &FileAuthority,
        arg: &str,
        system: Option<&str>,
        history: &[ChatTurn],
    ) -> CommandResult,
}

const COMMANDS: &[Command] = &[
    Command {
        name: "/export",
        completes_path: false,
        action: export_command,
    },
    Command {
        name: "/import",
        completes_path: true,
        action: import_command,
    },
];

// Runs the command in `prompt`, or None when the prompt is not a command (it then
// flows on to the model unchanged). The match is on the whole command word only,
// so "/exporting ..." is an ordinary prompt, not a malformed /export.
pub(crate) fn run(
    files: &FileAuthority,
    prompt: &str,
    system: Option<&str>,
    history: &[ChatTurn],
) -> Option<CommandResult> {
    let (command, rest) = parse(prompt)?;
    Some((command.action)(files, rest.trim(), system, history))
}

// Splits the prompt into a known command and the raw text after its name. The
// name must be a whole word (end of input or followed by whitespace), so a longer
// word that merely starts with the name is not treated as the command.
fn parse(prompt: &str) -> Option<(&'static Command, &str)> {
    let typed = prompt.trim_start();
    COMMANDS.iter().find_map(|command| {
        let rest = typed.strip_prefix(command.name)?;
        (rest.is_empty() || rest.starts_with(char::is_whitespace)).then_some((command, rest))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn complete(prompt: &str) -> (Option<String>, Vec<String>) {
        super::complete(&FileAuthority::capture().unwrap(), prompt)
    }

    fn run(prompt: &str, system: Option<&str>, history: &[ChatTurn]) -> Option<CommandResult> {
        super::run(&FileAuthority::capture().unwrap(), prompt, system, history)
    }

    #[test]
    fn completes_partial_command_name() {
        // A partial name completes to the rest of the unique match. /import takes
        // a path, so its name completion also appends the separating space.
        assert_eq!(complete("/exp").0.as_deref(), Some("ort"));
        assert_eq!(complete("/imp").0.as_deref(), Some("ort "));
        // A lone '/' is ambiguous: every command listed, no auto-tail.
        let (tail, matches) = complete("/");
        assert_eq!(tail, None);
        assert_eq!(matches.len(), 2);
        // A non-command prompt yields nothing to complete.
        let (tail, matches) = complete("hello");
        assert!(tail.is_none());
        assert!(matches.is_empty());
    }

    #[test]
    fn parses_command_as_whole_word_only() {
        // "/exporting" is an ordinary prompt, not the /export command.
        assert!(run("/exporting now", None, &[]).is_none());
        // The bare command word is recognised even without an argument.
        assert!(run("/import", None, &[]).is_some());
    }

    #[test]
    fn import_without_path_is_a_notice() {
        let result = run("/import", None, &[]).expect("recognised command");
        assert!(matches!(result, CommandResult::Notice(_)));
    }

    #[test]
    fn import_rejects_paths_outside_the_workspace() {
        // Traversal must not import arbitrary files: it stays a notice.
        let result = run("/import ../secret.md", None, &[]).expect("recognised command");
        match result {
            CommandResult::Notice(msg) => assert!(msg.contains("import failed")),
            _ => panic!("traversal should not restore a conversation"),
        }
    }

    #[test]
    fn import_completes_a_space_then_lists_json() {
        // The fully typed command, still unseparated, completes to the space the
        // system supplies; no files are listed until that space exists.
        let (tail, matches) = complete("/import");
        assert_eq!(tail.as_deref(), Some(" "));
        assert!(matches.is_empty());

        // After the space, only .json files (and directories) are listed/completed.
        let dir = "cmd_import_probe";
        std::fs::create_dir_all(dir).unwrap();
        std::fs::File::create(format!("{dir}/only.json")).unwrap();
        let (tail, matches) = complete(&format!("/import {dir}/"));
        assert_eq!(matches, vec!["only.json"]);
        assert_eq!(tail.as_deref(), Some("only.json"));
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn removed_embedding_commands_do_not_complete() {
        // /embeddings, /search and /deep-search are no longer commands; they
        // must fall through like any other unknown slash-command spelling.
        for prompt in [
            "/emb",
            "/embeddings",
            "/embeddings import notes.bin",
            "/sea",
            "/search",
            "/search --source g",
            "/deep",
            "/deep-search",
        ] {
            let (tail, matches) = complete(prompt);
            assert!(tail.is_none());
            assert!(matches.is_empty());
        }
        let (tail, matches) = complete("/searchfoo");
        assert!(tail.is_none());
        assert!(matches.is_empty());
    }

    #[test]
    fn tools_command_is_not_exposed() {
        for prompt in ["/too", "/tools", "/tools read", "/toolsfoo"] {
            let (tail, matches) = complete(prompt);
            assert!(tail.is_none());
            assert!(matches.is_empty());
            assert!(run(prompt, None, &[]).is_none());
        }
    }

    #[test]
    fn export_rejects_paths_outside_the_workspace() {
        // Writes get the same confinement as reads: traversal and absolute
        // paths surface as a generic failure notice and write nothing.
        for prompt in ["/export ../evil.json", "/export /tmp/evil.json"] {
            let result = run(prompt, None, &[]).expect("recognised command");
            match result {
                CommandResult::Notice(msg) => assert!(msg.contains("export failed")),
                _ => panic!("confined export should not restore a conversation"),
            }
        }
        assert!(!std::path::Path::new("../evil.json").exists());
    }
}
