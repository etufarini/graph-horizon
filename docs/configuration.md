<!--
This document owns the current CLI and Web runtime flags plus installer inputs.
Model support details and validation-only variables are delegated.
-->

# Configuration

The binary reads runtime configuration only from flags. Use `--flag value`;
`--flag=value` is not accepted. Unknown flags and missing values fail before a
model or surface starts. When a flag repeats, the first value wins.

Graph Horizon supports exactly two modes:

```sh
# Interactive CLI (the default mode)
graph-horizon --model /path/to/model.gguf

# Integrated Web UI
graph-horizon --mode web --model /path/to/model.gguf
```

The file must satisfy the current [engine crate
contract](../crates/graph_horizon_engine/README.md).

## Accepted Flags

| Flag | Default | Effect |
|---|---|---|
| `--model <path>` | none | GGUF required by both modes |
| `--mode <cli\|web>` | `cli` | Selects the supported surface |
| `--host <host>` | `127.0.0.1` | Web UI bind host; ignored by CLI |
| `--port <value>` | `8080` | Web UI bind port; ignored by CLI |
| `--search-url <url>` | built-in public providers | Advanced JSON search endpoint override; HTTPS except HTTP loopback |
| `--search-key-file <path>` | none | Optional override bearer token; requires `--search-url` |
| `--context-tokens <n>` | engine policy | Exact positive context request |
| `--system-prompt <text>` | none | CLI-only system prompt |
| `--max-tokens <n>` | CLI: `2048`; Web: context limit | Maximum generated tokens |
| `--vram-weights-percent <n>` | automatic | Explicit `0..=100` hybrid weight-placement limit |
| `--vram-reserve-mib <n>` | engine policy | Non-negative hybrid VRAM reserve |
| `--cpu-threads <n>` | host parallelism | CPU workers, integer `>= 1` |
| `--kv-quant <f16\|int8>` | `f16` | Lowercase, case-sensitive KV scheme |

Engine numeric and KV flags are validated before loading. An invalid explicit
value is an error; it is never silently reduced or replaced with a fallback.

## CLI

The CLI always loads the model in-process. Without `--context-tokens`, the
engine applies its versioned policy within the GGUF maximum. `--max-tokens`
reserves response capacity; startup fails before terminal initialization when
that reserve leaves no safe prompt space. Complete estimation and admission
rules are in [context.md](context.md).

## Web UI

Web mode requires built assets in `web/frontend/dist`. Host and port remain Web
settings because the browser application needs a local listener. An invalid
bind value fails startup. Binding outside loopback deliberately makes the Web UI
reachable from that network; Graph Horizon does not add TLS or authentication.

The Web UI uses the engine's resolved context and model-profile sampling.
`--system-prompt` remains CLI-only. `--max-tokens` bounds Web generation too;
when omitted, Web uses the resolved context limit as its existing default.

Without search flags, Web mode exposes the built-in keyless providers:
DuckDuckGo Lite for `web` and Google News RSS for `news`. Search remains an
explicit per-request choice in the browser; starting the UI does not send a
query.

`--search-url` replaces both built-in routes with one advanced JSON provider.
The URL may contain a path but no credentials, query, or fragment. It must use
HTTPS, except that HTTP is accepted for `localhost`, `127.0.0.1`, and `::1` to
support local adapters and tests. `--search-key-file` contains one bearer token;
on Unix it must be a regular file with no group or other permission bits.
Configuration errors stop startup before the listener opens. The provider
contract and privacy boundary are documented in [web.md](web.md#web-search).

## Build And Environment

```sh
support/install.sh --backend cpu|vulkan|vulkan-hybrid|metal|metal-hybrid \
  [--profile release|fast] [--prefix /path/to/prefix]
```

The backend is required and selected at build time. The prefix must be absolute,
non-root, and contain no `.` or `..` component. An explicit `--prefix` wins over
`GRAPH_HORIZON_INSTALL_PREFIX`, which wins over `$HOME/.local`.

| Platform | Accepted build backends |
|---|---|
| macOS arm64 | `cpu`, `vulkan`, `vulkan-hybrid`, `metal`, `metal-hybrid` |
| Linux x86_64 | `cpu`, `vulkan`, `vulkan-hybrid` |

Build acceptance does not assign production, qualified, or reference status;
see the [backend contract](backend.md#support-status). Variables used by profiling, testing, and
oracle scripts are development interfaces, not binary runtime configuration.
