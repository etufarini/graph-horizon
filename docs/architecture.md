<!--
This document owns the model-neutral application map and request flows. Detailed
surface contracts and family-specific engine behavior belong to linked documents.
-->

# Architecture

GH Zero Engine separates the inference engine from the three surfaces that use
it: the interactive console, HTTP server, and web UI. The `gh_zero_engine` crate
does not depend on the terminal, Hyper, or Svelte; surfaces only use its public API.

## Workspace Layout

```text
.
├── src/
│   ├── app/                     flags, engine configuration, and mode selection
│   ├── gh_zero_cli/             TUI and local/HTTP providers
│   ├── gh_zero_server/          chat endpoint and SSE streaming
│   ├── gh_zero_web/             static assets and chat server wrapper
│   └── main.rs                  initialization and asynchronous entry point
├── crates/gh_zero_engine/       GGUF loader, families, backends, and public API
├── examples/                    tools built on the public API
├── support/                     installation and validation procedures
└── web/frontend/                Svelte application built as static assets
```

## Application Domains

| Domain | Responsibility | Reference |
|---|---|---|
| `app` | Parses a closed flag table, validates common options, and selects `cli`, `server`, or `web` | [configuration.md](configuration.md) |
| `gh_zero_cli` | Manages input, history, transcripts, attachments, and the local/HTTP provider | [console.md](console.md), [commands.md](commands.md) |
| `gh_zero_server` | Converts text-only requests into engine requests and serializes events as SSE | [server.md](server.md) |
| `gh_zero_web` | Serves the built frontend and delegates chat to the same server state | [web.md](web.md) |
| `gh_zero_engine` | Validates a supported GGUF and runs prefill/decode on the compiled backend | [backend.md](backend.md) |

## Common Startup

`main` initializes the parser before the terminal, listener, or model.
`app::run` validates engine options, captures a descriptor for the initial
directory for TUI file operations, and selects exactly one mode.

Local model paths are passed to the engine without changing the current
directory. Server and web modes always require `--model`; the TUI requires it
only with `--provider local`.

## TUI Flow

```text
input
  |
  +-- /export or /import command --> local action, no model request
  |
  +-- ordinary prompt
        |
        +-- expand @file attachments in the outgoing copy
        +-- system prompt + completed turns + new prompt
        +-- prune the oldest pairs when configured
        +-- local provider or HTTP POST /chat/completions
        +-- text-only stream
        +-- completed turn added to history
```

The view and export preserve the typed prompt. For completed turns, the internal
history sent back to the model instead retains the expanded copy, so the model
can use an attachment in later turns of the same session. Notices and failed
turns do not enter the context. The TUI models only `system`, `user`, and
`assistant` messages; tool calls and separate Reasoning channels are not part of
its state. Raw assistant content, including Reasoning markers, remains the TUI
context, history, and transcript representation. Only terminal conversation
rendering derives THINK and final-answer sections from it.

## Server And Web Flow

The server loads one `Engine`, accepts validated text-only requests, and
serializes one generation at a time. A semaphore limits admitted requests,
including waiting requests, to eight.

Web mode adds only static assets and a browser UI to the same chat endpoint.
There are no workspace, tool, or command-confirmation routes.

The engine and server expose raw Reasoning markers only as ordinary
`delta.content` text; neither creates a structured Reasoning field. The browser
bubble and terminal conversation renderer independently derive their visual
THINK sections at the presentation boundary. Web context and transfer state also
retain raw assistant content. A separate visual section is therefore not a
separate Reasoning protocol channel, and the same boundary applies to both the
local engine and CLI HTTP provider paths.

## Engine Boundaries

`Engine` gives surfaces a stable entry point that does not expose the concrete
backend. In the current structure, `family/mod.rs` opens the GGUF and forwards
to one family implementation; there is not yet a multi-family registry or
dispatcher. The concrete gate uses metadata and capabilities, not the filename.
Backends, KV layouts, and graphs remain internal details; selected GGUF and
tokenizer types are exported for inspection.

The actual set of accepted families and profiles is a versioned library
contract, not part of the application architecture. See the
[crate README](../crates/gh_zero_engine/README.md) for current support.
