<!--
This document owns the TUI slash-command, transcript, attachment, completion,
file-authority, and developer build-command behavior. It does not describe
server or browser file handling.
-->

# Slash Commands And Attachments

The TUI intercepts two local commands and can expand text files in the prompt. A
recognized command never reaches the model: it produces a notice or replaces the
current conversation.

## Commands

| Command | Effect |
|---|---|
| `/export [path]` | Saves the conversation as JSON; defaults to `graph-orizon-export.json` |
| `/import <path>` | Validates and restores a transcript, replacing the current one |

The name must be a complete word. `/exporting` is an ordinary prompt, as is any
unrecognized command.

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
Markdown block headed `### File: <path>`. If the turn finishes with text, that
copy enters the session's internal context; the view and export continue to show
the typed prompt. The fence language comes from the extension. A missing,
non-regular, or non-UTF-8 token remains literal text.

The fence uses more backticks than the longest sequence found in the file, so
the content cannot close the block early.

## File Authority

A descriptor for the current directory is opened at startup. Every read,
creation, and completion operation is relative to that descriptor:

- absolute paths, `.` and `..` components, empty components, and trailing slashes are forbidden;
- every directory and file is opened with `openat` and `O_NOFOLLOW`;
- symlinks are rejected even when they point inside the directory;
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

## Build And Toolchain Commands

Backend selection is static and explicit:

```sh
cargo build --locked --no-default-features --features cpu
cargo build --locked --no-default-features --features vulkan
cargo build --locked --no-default-features --features vulkan-hybrid
cargo build --locked --no-default-features --features metal
cargo build --locked --no-default-features --features metal-hybrid
xcrun -f metal
xcrun -f metallib
```

Exactly one feature is permitted. Metal commands require macOS arm64 and the
Xcode tools; no command installs them or falls back to CPU.
