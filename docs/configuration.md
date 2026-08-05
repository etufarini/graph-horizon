<!--
This document owns the current runtime flag contract and mode-specific defaults.
Build options, model support details and validation-only variables are delegated.
-->

# Configuration

The binary reads runtime configuration only from flags. Flags with values use
`--flag value`; `--flag=value` is not accepted. Unknown flags and missing values
terminate execution before loading a model or initializing a surface.

Profile selection is a build-time decision described in
[backend.md](backend.md) and [support/README.md](../support/README.md).

## Minimal Usage

```sh
# TUI with the local engine
graph-orizon --provider local --model /path/to/model.gguf

# TUI with an OpenAI-compatible HTTP provider
graph-orizon --base-url http://127.0.0.1:8080/v1

# Local server
graph-orizon --mode server --model /path/to/model.gguf

# Local web UI
graph-orizon --mode web --model /path/to/model.gguf
```

The file must satisfy the current library contract. Supported families and
profiles are listed in the
[crate README](../crates/graph_orizon_engine/README.md), not on this page.

## Accepted Flags

| Flag | Default | Effect |
|---|---|---|
| `--model <path>` | none | GGUF required by the local provider, server, and web modes |
| `--mode <cli\|server\|web>` | `cli` | Selects one surface; any other value is an error |
| `--provider <value>` | HTTP | TUI only: `local` uses the in-process engine; every other value uses HTTP |
| `--host <host>` | `127.0.0.1` | Server and web bind host |
| `--port <value>` | `8080` | Server and web port, with mode-specific parsing |
| `--context-tokens <n>` | engine policy or HTTP detection | Explicit context, integer `>= 1` |
| `--system-prompt <text>` | none | TUI-only system prompt |
| `--base-url <url>` | `http://127.0.0.1:8080/v1` | Base URL for the TUI HTTP provider |
| `--max-tokens <n>` | TUI `2048`; server/web `1024` | Default maximum generation, integer `>= 0` |
| `--vram-weights-percent <n>` | automatic | Explicit `0..=100` weight-placement limit |
| `--vram-reserve-mib <n>` | engine policy | Non-negative VRAM reserve; part of the hybrid automatic plan |
| `--cpu-threads <n>` | host parallelism | CPU workers, integer `>= 1` |
| `--kv-quant <f16\|int8>` | `f16` | Lowercase, case-sensitive KV scheme |
| `--no-attn-simd` | absent | Disables the CPU attention SIMD path |

`--context-tokens`, VRAM percentage, reserve, thread count, KV, and
`--max-tokens` are validated before dispatch. In particular, a percentage above
100 is an error and is not reduced automatically.

## Local And HTTP TUI

With `--provider local`, `--model` is required and the GGUF is loaded before the
terminal. Without `--context-tokens`, pruning uses the context resolved by the
engine.

With any other provider, the TUI sends a text-only body containing `messages`,
`stream: true`, and `max_tokens` to `<base-url>/chat/completions`. It sends no
model name or credentials. Without an explicit context, it tries `GET /props` at
the provider root for three seconds and reads
`default_generation_settings.n_ctx`; failure disables pruning without blocking
startup.

The current integrated server does not expose `/props`. Connecting the TUI to
that server therefore requires `--context-tokens` to enable client-side pruning.

## Local Context

`--context-tokens` requests exactly the specified positive value. When absent,
the engine applies its versioned policy within the maximum declared by the GGUF.
A value above the maximum or unsupported by the backend fails; it is not
silently truncated.

The reserve used by TUI pruning is `--max-tokens`. The mechanism is described in
[pruning.md](pruning.md).

## Server

`--mode server` ignores `--provider` and always loads the local model. Port
parsing is tolerant: a non-numeric value falls back to `8080`. The server value
of `--max-tokens` is used only when the HTTP request omits `max_tokens`; an
explicit positive request value is not clamped to that default.

```sh
graph-orizon --mode server --model /path/to/model.gguf \
  --host 127.0.0.1 --port 8080 --max-tokens 1024
```

## Web

`--mode web` always loads the local model and requires built assets in
`web/frontend/dist`. It keeps the port as a string until bind time, so an invalid
value causes a listen error instead of the server mode's fallback.

The included UI explicitly sends `max_tokens: 1024`. Consequently,
`--max-tokens` is only the default for HTTP clients that omit the field, not a
setting for the current web composer.

## Build And Environment

```sh
support/install.sh --backend cpu|vulkan|vulkan-hybrid|metal|metal-hybrid \
  --profile release|fast --prefix /path/to/prefix
```

The backend is required; there is no default or runtime backend setting. Build
profile defaults to `release` and prefix to `${HOME}/.local`. The only user
environment variable is `GRAPH_ORIZON_INSTALL_PREFIX`, an alternative to `--prefix`.
The `GRAPH_ORIZON_*` variables used by tests, profiling, and diagnostics are not
binary runtime configuration; their scripts and sources remain authoritative
for those development interfaces.

On separate-memory Vulkan, `--vram-weights-percent` limits only device weights
after reserve. On unified-memory Metal it selects the intended share while CPU
and Metal categories still compete for one capacity. `0` is CPU-only, `100` is
the device-only endpoint, and omission requests automatic planning. No value
causes a retry on another profile, percentage, context, or KV scheme.
