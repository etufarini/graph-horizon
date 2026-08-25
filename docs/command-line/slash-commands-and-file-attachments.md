<!--
This document owns the terminal slash-command, transcript, attachment,
completion, and file-authority behavior. It does not describe browser file
handling or developer build commands.
-->

# Slash Commands And File Attachments

The TUI intercepts four local commands and can expand text files in the prompt. A
recognized command never reaches the model: it produces a notice or applies one
explicit session/transcript mutation.

## Commands

| Command | Effect |
|---|---|
| `/clear` | Clears the conversation and metrics while preserving the system prompt |
| `/export [path]` | Saves the conversation as JSON; defaults to `graph-horizon-export.json` |
| `/import <path>` | Validates and restores a transcript, replacing the current one |
| `/system <text>` | Replaces the session system prompt without retaining the command text in history |
| `/system --clear` | Removes the session system prompt |

The name must be a complete word. `/exporting` is an ordinary prompt, as is any
unrecognized command.

Bare `/system` reports its usage and does not reveal the current prompt. System
prompt changes clear the prior generation metrics and recompute context
occupancy; `/clear` keeps the prompt but removes every visible and contextual
turn without reloading the model.

## TUI Transcript

The format is a JSON array of `{role, content}` records. It may begin with one
`system` record, followed by strictly alternating `user`/`assistant` pairs.

Export includes the active system prompt and completed turns only. Notices,
failed turns, and expanded attachment content are excluded: the original prompt
is preserved. Import rejects malformed JSON, unknown roles, and incomplete turns
without changing the existing session.

The TUI format differs from the versioned container used by the web UI; the two
importers validate different contracts.

## `@file` Attachments

A UTF-8-readable `@path/to/file` token is replaced in the request copy by a
Markdown block headed `### File: <path>`. The expanded copy affects only that
request and its admission estimate. If the turn finishes with text, later
requests, the view, and export use the typed prompt with its raw `@path`
spelling. The fence language comes from the extension. A missing, non-regular,
or non-UTF-8 token remains literal text.

The fence uses more backticks than the longest sequence found in the file, so
the content cannot close the block early.

## File Authority

A descriptor for the current directory is opened at startup. Every read,
creation, and completion operation is anchored to that descriptor:

- reads and creates reject absolute paths, `.` and `..` components, empty
  components, and trailing slashes;
- completion rejects the same unsafe components but accepts one trailing slash
  to list an already authorized directory;
- opened directories and files use descriptor-relative, no-follow operations;
- reads, creates, and directory descent reject symlinks even when they point
  inside the directory; a name surfaced by completion still fails when opened
  if it is a symlink;
- `/export` creates or truncates regular files with `0600` permissions;
- displayed errors omit absolute paths, OS details, and stack traces.

The captured directory remains the authority even if its name changes during
the session. There is no runtime flag for changing it.

## Completion

`Tab` completes the longest common prefix for:

- `@` tokens with accessible files and directories;
- `/export` and `/import` names;
- the `/import` argument, limited to `.json` files and traversable directories.

More than ten candidates are shown in a scrollable window. `Enter` resolves a
pending completion first and submits only when the token is unambiguous.
