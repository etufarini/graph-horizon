/*
 * GH Zero CLI Modules - Transcript
 * Persists and restores the conversation as a JSON array of {role, content}
 * records (the common chat-transcript shape). Owns only this format: the full
 * conversion to/from `ChatTurn` lives here and consumes only already-open owned
 * files from the startup authority. JSON escaping preserves exact message text;
 * this module never accepts or reopens a filesystem path.
 */

use std::fs::File;
use std::io;

use color_eyre::eyre::{Result, bail};
use serde::{Deserialize, Serialize};

use crate::gh_zero_cli::console::render::ChatTurn;
use crate::gh_zero_cli::runtime::ChatMessage;

// One record of the transcript file. Local to this module on purpose: the file
// shape stays decoupled from the runtime's ChatMessage, whose fields remain
// private to the request path.
#[derive(Serialize, Deserialize)]
struct Record {
    role: Role,
    content: String,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Copy)]
#[serde(rename_all = "lowercase")]
enum Role {
    System,
    User,
    Assistant,
}

// Writes the system prompt (if any) followed by every completed turn as
// alternating user/assistant records. Only turns that entered the model context
// are exported: a turn with empty `messages` is a failed turn or a local notice
// (e.g. an /export result) and must never re-enter a future conversation. The
// human-typed `prompt` and the `response` are written verbatim; attachment
// expansion is an input-time convenience and is intentionally not preserved.
pub(crate) fn export(file: File, system: Option<&str>, history: &[ChatTurn]) -> io::Result<()> {
    let mut records = Vec::new();
    if let Some(system) = system {
        records.push(Record {
            role: Role::System,
            content: system.to_string(),
        });
    }
    for turn in history.iter().filter(|t| !t.messages.is_empty()) {
        records.push(Record {
            role: Role::User,
            content: turn.prompt.clone(),
        });
        records.push(Record {
            role: Role::Assistant,
            content: turn.response.clone(),
        });
    }
    // Pretty-printed so the file stays reviewable by eye even though the
    // canonical consumer is /import.
    serde_json::to_writer_pretty(file, &records).map_err(io::Error::other)
}

// Reads a transcript file and rebuilds the optional system prompt plus the
// conversation turns. Errors carry no path or OS detail so callers can surface
// them without leaking the filesystem. Invariant: the records must be an
// optional leading system record followed by strictly alternating
// user/assistant pairs; any other order is a malformed transcript and is
// rejected, leaving the current conversation untouched.
pub(crate) fn import(file: File) -> Result<(Option<String>, Vec<ChatTurn>)> {
    let Ok(records) = serde_json::from_reader::<_, Vec<Record>>(file) else {
        bail!("transcript is not a valid JSON message array");
    };

    let mut records = records.into_iter().peekable();
    let system = if records.peek().is_some_and(|r| r.role == Role::System) {
        records.next().map(|r| r.content)
    } else {
        None
    };

    let mut history = Vec::new();
    loop {
        let prompt = match records.next() {
            None => break,
            Some(Record {
                role: Role::User,
                content,
            }) => content,
            // A second system record, or an assistant without its user, breaks
            // the alternation the format guarantees.
            Some(_) => bail!("malformed transcript: expected a user message"),
        };
        let Some(Record {
            role: Role::Assistant,
            content: response,
        }) = records.next()
        else {
            bail!("malformed transcript: user message without a reply");
        };
        // Restore the context messages from the visible text so display and the
        // model context stay identical after import.
        let messages = vec![
            ChatMessage::user(prompt.clone()),
            ChatMessage::assistant(response.clone()),
        ];
        history.push(ChatTurn::new(prompt, response, messages));
    }

    Ok((system, history))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;
    use std::fs;

    // Builds a completed turn whose context messages are derived from prompt and
    // response, mirroring how the session stores a real turn.
    fn turn(prompt: &str, response: &str) -> ChatTurn {
        ChatTurn::new(
            prompt.to_string(),
            response.to_string(),
            vec![
                ChatMessage::user(prompt.to_string()),
                ChatMessage::assistant(response.to_string()),
            ],
        )
    }

    fn temp_path(name: &str) -> std::path::PathBuf {
        env::temp_dir().join(format!("gh-zero-transcript-{name}.json"))
    }

    #[test]
    fn round_trips_system_and_turns() {
        let path = temp_path("roundtrip");
        let history = vec![turn("ciao\nmulti", "risposta"), turn("seconda", "ok")];
        export(File::create(&path).unwrap(), Some("sei utile"), &history).unwrap();

        let (system, restored) = import(File::open(&path).unwrap()).unwrap();
        assert_eq!(system.as_deref(), Some("sei utile"));
        assert_eq!(restored.len(), 2);
        assert_eq!(restored[0].prompt, "ciao\nmulti");
        assert_eq!(restored[0].response, "risposta");
        assert_eq!(restored[1].prompt, "seconda");
        // Imported turns must carry context messages so the chat resumes.
        assert_eq!(restored[1].messages.len(), 2);
    }

    #[test]
    fn round_trip_is_exact_for_edge_whitespace_and_json_text() {
        let path = temp_path("exact");
        // JSON escaping must preserve content the old delimited format could
        // not: edge newlines and text shaped like the format itself.
        let tricky = "\n\nstarts and ends blank\n{\"role\":\"user\"}\n\n";
        export(File::create(&path).unwrap(), None, &[turn("q", tricky)]).unwrap();

        let (_, restored) = import(File::open(&path).unwrap()).unwrap();
        assert_eq!(restored[0].response, tricky);
    }

    #[test]
    fn skips_turns_without_context_messages() {
        let path = temp_path("skip");
        // A notice/failed turn (empty messages) must not be written, so it can
        // never re-enter the context on a later import.
        let notice = ChatTurn::new("/export".into(), "done".into(), Vec::new());
        export(
            File::create(&path).unwrap(),
            None,
            &[notice, turn("real", "reply")],
        )
        .unwrap();

        let (system, restored) = import(File::open(&path).unwrap()).unwrap();
        assert!(system.is_none());
        assert_eq!(restored.len(), 1);
        assert_eq!(restored[0].prompt, "real");
    }

    #[test]
    fn rejects_user_without_reply() {
        let path = temp_path("malformed");
        fs::write(&path, r#"[{"role":"user","content":"lonely"}]"#).unwrap();
        assert!(import(File::open(&path).unwrap()).is_err());
    }

    #[test]
    fn rejects_invalid_json_and_unknown_roles() {
        let path = temp_path("invalid");
        fs::write(&path, "not json at all").unwrap();
        assert!(import(File::open(&path).unwrap()).is_err());

        // An unrecognised role must fail parsing, not silently become a turn.
        fs::write(&path, r#"[{"role":"tool","content":"x"}]"#).unwrap();
        assert!(import(File::open(&path).unwrap()).is_err());
    }
}
