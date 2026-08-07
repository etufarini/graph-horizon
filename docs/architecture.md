<!--
This document owns the model-neutral application map, request flows, and Web
persistence boundary. Detailed surface contracts and family-specific engine
behavior belong to linked documents.
-->

# Architecture

Graph Horizon separates the inference engine from the three surfaces that use
it: the interactive console, HTTP server, and web UI. The `graph_horizon_engine` crate
does not depend on the terminal, Hyper, or Svelte; surfaces only use its public API.

## Workspace Layout

```text
.
├── src/
│   ├── app/                     flags, engine configuration, and mode selection
│   ├── graph_horizon_cli/             TUI and local/HTTP providers
│   ├── graph_horizon_server/          chat endpoint and SSE streaming
│   ├── graph_horizon_web/             static assets and chat server wrapper
│   └── main.rs                  initialization and asynchronous entry point
├── crates/graph_horizon_engine/       GGUF loader, families, backends, and public API
├── examples/                    tools built on the public API
├── support/                     installation and validation procedures
└── web/frontend/                Svelte application built as static assets
```

## Application Domains

| Domain | Responsibility | Reference |
|---|---|---|
| `app` | Parses a closed flag table, validates common options, and selects `cli`, `server`, or `web` | [configuration.md](configuration.md) |
| `graph_horizon_cli` | Manages input, history, transcripts, attachments, and the local/HTTP provider | [console.md](console.md), [commands.md](commands.md) |
| `graph_horizon_server` | Converts text-only requests into engine requests and serializes events as SSE | [server.md](server.md) |
| `graph_horizon_web` | Serves the built frontend and delegates chat to the same server state | [web.md](web.md) |
| `graph_horizon_engine` | Validates a supported GGUF and runs prefill/decode on the compiled backend | [backend.md](backend.md) |

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
        +-- assemble system prompt + every completed turn + new prompt
        +-- reject or admit the complete request without modifying it
        +-- local provider or HTTP POST /chat/completions after admission
        +-- text-only stream
        +-- completed turn added to history
```

The view, model history, and export preserve the typed prompt. Attachment
contents participate in the one request for which they are expanded, without
replacing that raw prompt in committed history. Admission uses the shared
[context-capacity contract](context.md) before provider invocation. Notices and
failed turns do not enter the context. The TUI models only `system`, `user`, and
`assistant` messages; tool calls and separate Reasoning channels are not part of
its state. Raw assistant content, including Reasoning markers, remains the TUI
context, history, and transcript representation. Only terminal conversation
rendering derives THINK and final-answer sections from it.

## Server And Web Flow

The server loads one `Engine`, serves its immutable resolved context through
`GET /props`, accepts validated text-only chat requests, and serializes one
generation at a time. A semaphore limits admitted chat requests, including
waiting requests, to eight; property reads do not acquire it.

Web mode adds static assets and a browser UI to the same chat and properties
boundaries. The browser loads context before enabling send, assembles the exact
wire messages, admits them locally, and only then starts chat transport and its
monotonic timer. Direct external API clients retain the server's existing chat
contract and receive no new server-side capacity gate. There are no workspace,
tool, or command-confirmation routes.

Browser conversation ownership remains entirely in the frontend:

```text
transfer -----> transcript <----- state -----> client -----> HTTP/SSE
                                  |
                                  +----------> persistence -----> localStorage
```

`transcript` owns the pure complete-pair invariant shared by file transfer and
persistence. `state` owns the stable checkpoints: successful completion,
settled stop, valid import, and new-chat clear. Incremental deltas and failed
turns do not reach persistence. Import/export and browser persistence remain
separate versioned formats despite sharing transcript validation.

`localStorage` is an origin-scoped browser boundary, not application storage.
No server route, engine component, or CLI path reads or writes the saved Web
conversation; there is no server-side or CLI persistence and no cross-tab
coordination.

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

The engine dependency direction is deliberately narrow:

```text
family -> runtime <- backend
```

The family supplies `WeightSource` and `LayeredGraph`; a statically selected
backend supplies buffers and encoders. The runtime owns either one homogeneous
session or one immutable partitioned session. A partitioned pass runs a CPU
prefix, crosses once, then runs the device suffix. Loading is transactional;
generation emits one terminal event, cancellation emits none, and normalized
public errors expose no device paths or driver internals.

The actual set of accepted families and profiles is a versioned library
contract, not part of the application architecture. See the
[crate README](../crates/graph_horizon_engine/README.md) for current support.
