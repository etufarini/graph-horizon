<!--
This document owns the current runtime flag contract, mode-specific defaults,
and installer inputs as build-time configuration. Model support details and
validation-only variables are delegated.
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
graph-horizon --provider local --model /path/to/model.gguf

# TUI with an OpenAI-compatible HTTP provider
graph-horizon --base-url http://127.0.0.1:8080/v1

# Local server
graph-horizon --mode server --model /path/to/model.gguf

# Local web UI
graph-horizon --mode web --model /path/to/model.gguf
```

The file must satisfy the current library contract. Supported families and
profiles are listed in the
[crate README](../crates/graph_horizon_engine/README.md), not on this page.

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
| `--max-tokens <n>` | TUI `2048`; server `1024`; Web ignored | CLI response reserve or server fallback, integer `>= 0` |
| `--vram-weights-percent <n>` | automatic | Explicit `0..=100` weight-placement limit |
| `--vram-reserve-mib <n>` | engine policy | Non-negative VRAM reserve; part of the hybrid automatic plan |
| `--cpu-threads <n>` | host parallelism | CPU workers, integer `>= 1` |
| `--kv-quant <f16\|int8>` | `f16` | Lowercase, case-sensitive KV scheme |

`--context-tokens`, VRAM percentage, reserve, thread count, and KV are validated
before engine loading. CLI and server modes also validate `--max-tokens`. In
particular, a percentage above 100 is an error and is not reduced automatically.

## Local And HTTP TUI

With `--provider local`, `--model` is required and the GGUF is loaded before the
terminal. Without `--context-tokens`, capacity admission uses the context
resolved by the engine.

With any other provider, the TUI sends a text-only body containing `messages`,
`stream: true`, and `max_tokens` to `<base-url>/chat/completions`. It sends no
model name or credentials. Without an explicit context, it tries `GET /props` at
the provider root for three seconds and reads
`default_generation_settings.n_ctx`. A valid explicit override skips discovery.
Failure, timeout, malformed JSON, or a non-positive value terminates before the
terminal with `limite di contesto non disponibile; specificare
--context-tokens`; there is no unknown or unlimited fallback.

The integrated server and Web wrapper expose `/props`, so the HTTP TUI example
works without an override. External providers that do not implement the route
require `--context-tokens`.

## Local Context

`--context-tokens` requests exactly the specified positive value. When absent,
the engine applies its versioned policy within the maximum declared by the GGUF.
A value above the maximum or unsupported by the backend fails; it is not
silently truncated.

The TUI uses `--max-tokens` as its response reserve. If the reserve leaves no
safe prompt space, startup fails before the terminal. Complete estimation and
admission rules are defined in [context.md](context.md).

## Server

`--mode server` ignores `--provider` and always loads the local model. Port
parsing is tolerant: a non-numeric value falls back to `8080`. The server value
of `--max-tokens` is used only when the HTTP request omits `max_tokens`; an
explicit positive request value is not clamped to that default.

```sh
graph-horizon --mode server --model /path/to/model.gguf \
  --host 127.0.0.1 --port 8080 --max-tokens 1024
```

## Web

`--mode web` always loads the local model and requires built assets in
`web/frontend/dist`. It keeps the port as a string until bind time, so an invalid
value causes a listen error instead of the server mode's fallback.

`--max-tokens` is ignored in Web mode. This includes explicit and otherwise
invalid values. After loading the engine, the wrapper publishes its positive
`n_ctx` as both `/props` capacity fields and uses the same value as the
wrapped-server fallback. The included UI requires those fields to be equal,
applies its 90% margin only to estimated prompt admission, and sends the full
`n_ctx` with every chat request.

## Build And Environment

```sh
support/install.sh --backend cpu|vulkan|vulkan-hybrid|metal|metal-hybrid \
  [--profile release|fast] [--prefix /path/to/prefix]
```

The backend is required; there is no default or runtime backend setting. Build
profile defaults to `release`. The prefix must be absolute, non-root, and have
no `.` or `..` component. An explicit `--prefix` takes precedence over
`GRAPH_HORIZON_INSTALL_PREFIX`, which takes precedence over `$HOME/.local`.
The installer places both command names in `<prefix>/bin`; if that exact entry
is absent from `PATH`, it reports the directory without editing shell files.

| Platform | Accepted build backends |
|---|---|
| macOS arm64 | `cpu`, `vulkan`, `vulkan-hybrid`, `metal`, `metal-hybrid` |
| Linux x86_64 | `cpu`, `vulkan`, `vulkan-hybrid` |

Build acceptance does not assign production, qualified, or reference status.
Those labels and their hardware/software scopes belong to the
[backend contract](backend.md#support-status).

Every other platform/backend tuple is rejected before `npm` or Cargo starts;
there is no backend or profile fallback. The local form above builds the
checkout directly. After the repository is public, the public acquisition form
downloads immutable `v0.1.0` and delegates to that same local installer:

```sh
curl --fail --location --silent --show-error https://raw.githubusercontent.com/etufarini/graph-horizon/v0.1.0/install.sh | bash -s -- --backend cpu
```

See the [product installation guide](../README.md) and
[support contract](../support/README.md) for prerequisites, reinstall behavior,
and the anonymous-access limitation while the repository remains private.
The `GRAPH_HORIZON_*` variables used by tests, profiling, and diagnostics are not
binary runtime configuration; their scripts and sources remain authoritative
for those development interfaces.

On separate-memory Vulkan, `--vram-weights-percent` limits only device weights
after reserve. On unified-memory Metal it selects the intended share while CPU
and Metal categories still compete for one capacity. `0` is CPU-only, `100` is
the device-only endpoint, and omission requests automatic planning. No value
causes a retry on another profile, percentage, context, or KV scheme.
