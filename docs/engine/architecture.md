<!--
This document owns the model-neutral application map, CLI flow, and Web UI
boundaries. Family-specific engine behavior belongs to linked documents.
-->

# Architecture

Graph Horizon separates the inference engine from two supported product
surfaces: the interactive CLI and the integrated Web UI. There is no standalone
server mode or supported public HTTP API.

The `graph_horizon_engine` crate does not depend on terminal, Hyper, or Svelte
code. Both surfaces load it in-process and run one local generation at a time.

## Workspace Layout

```text
.
├── src/
│   ├── app/                         flags, engine configuration, and mode selection
│   ├── graph_horizon_cli/           terminal session and local generation
│   ├── graph_horizon_web/           bundled assets and private browser transport
│   └── main.rs                      initialization and asynchronous entry point
├── crates/graph_horizon_engine/     GGUF loader, families, backends, and library API
├── examples/                        tools built on the library API
├── support/                         installation and validation procedures
└── web/frontend/                    Svelte application built as static assets
```

## Application Domains

| Domain | Responsibility | Reference |
|---|---|---|
| `app` | Parses a closed flag table, validates common options, and selects `cli` or `web` | [runtime options](../command-line/runtime-options.md) |
| `graph_horizon_cli` | Manages input, history, transcripts, attachments, and local generation | [terminal interface](../command-line/terminal-interface.md), [slash commands and file attachments](../command-line/slash-commands-and-file-attachments.md) |
| `graph_horizon_web` | Serves bundled assets and owns private browser generation plus optional bounded search | [local Web interface](../web-interface/README.md) |
| `graph_horizon_engine` | Validates a supported GGUF and runs prefill/decode on the compiled backend | [backend support status](backend-support-status.md) |

`main` installs error handling and parses flags before a model or surface is
initialized. `app::run` validates common engine options and dispatches exactly
one mode. Both modes require `--model`. The CLI alone captures the initial
directory as an immutable authority for transcript and attachment operations.

## CLI Flow

```text
input
  |
  +-- /export or /import command --> local action, no model request
  |
  +-- ordinary prompt
        |
        +-- expand @file attachments in the outgoing copy
        +-- assemble system prompt + completed turns + new prompt
        +-- admit the complete request against local model capacity
        +-- run the in-process engine and consume its event stream
        +-- add the completed turn to history
```

The visible prompt and committed history retain the text the user typed.
Expanded attachment contents exist only in the submitted request. Notices and
failed turns do not enter model context. Only `system`, `user`, and `assistant`
text messages are representable.

## Web UI Flow

Web mode loads one engine and starts the local HTTP listener required to deliver
the bundled application. Its request paths and JSON/SSE frames are private
implementation details shared only by the checked-in Rust backend and Svelte
frontend. They are deliberately strict and carry no compatibility or versioning
promise for external clients.

```text
browser assets --> browser state --> private same-origin transport --> engine
                         |                    |
                         |                    +--> optional structured search
                         |                         +--> DuckDuckGo Lite (Web)
                         |                         +--> Google News RSS (News)
                         |                         +--> JSON override (advanced)
                         +--> localStorage chat archive
                         +--> IndexedDB Markdown files
```

The browser loads engine capacity before enabling Send, admits the outgoing
prompt locally, and then starts generation. When explicitly enabled, a separate
raw query obtains bounded snippets before the engine mutex is acquired; there is
no model-directed tool loop. One mutex protects engine state and semaphores bound
browser work and outbound search. Static asset paths reject traversal; chat
request bodies are size-limited and reject unknown fields.

Browser conversation and Markdown-file persistence remain origin-scoped. No
engine, CLI, or filesystem component reads browser archives. Files are framed as
untrusted reference data in one outgoing user-message copy; they are never
uploaded into product storage or indexed by a retrieval service.

Search is the only optional non-local Web path. The Rust host invokes `curl`
without a shell and dispatches an explicit Web request to DuckDuckGo Lite or an
explicit News request to Google News RSS. `--search-url` replaces that fixed
dispatch with one JSON endpoint. The browser chooses the category and exact
displayed query; no term classification or cross-provider fallback changes
that choice. Strict parsing keeps provenance
separate from snippets and requires timestamp proof for dated results. The
bounded transport follows no redirects and never fetches a result URL. Search
excerpts exist only in the engine-request copy. Structured provenance is stored
for display but projected out of every later model-history request.

Reasoning markers remain ordinary raw assistant text in both surfaces. The CLI
and browser derive visual THINK sections at presentation time without creating
a separate protocol or model-history field.

## Engine Boundary

`Engine` gives both surfaces a stable library entry point without exposing the
concrete backend. Family code supplies weights and graph operations; the
statically selected backend supplies buffers and encoders. Loading is
transactional, generation emits one terminal event, and cancellation emits
none.

`family/mod.rs` owns a closed `Model` dispatcher selected from validated GGUF
architecture metadata. Its only current variant is Ministral; adding a family
requires a new explicit variant rather than backend-pair modules or dynamic
registration.

```text
Engine -> family::Model -> selected family -> runtime <- backend
```

Static backend selection keeps each build to one dependency and resource
policy. Supported families, profiles, and the Rust library facade are defined
in the [engine crate contract](../../crates/graph_horizon_engine/README.md).

The shared partitioned runtime owns every hybrid plan, CPU prefix, accelerator
suffix, and crossing. Concrete Vulkan, Metal, and CUDA modules own only their
device resources, memory policy, and numeric operations. In particular,
`cuda-hybrid` reuses the existing CUDA operation dispatch and PTX module; no
family module or backend-pair module owns alternate CUDA numerics.
