<!--
This index owns navigation for the repository documentation. It separates stable,
model-neutral behavior from the current support contract and validation evidence.
-->

# Documentation

These pages describe Graph Horizon surfaces and processes without assuming a
specific model family, size, or release. Examples therefore use the
`<model.gguf>` placeholder.

Concrete support changes with the code: currently accepted architectures, GGUF
profiles, and limits are defined in the
[library contract](../crates/graph_horizon_engine/README.md), while reviewed results
belong in the [validation register](../VALIDATION.md).

## Usage And Structure

| Document | Contents |
|---|---|
| [configuration.md](configuration.md) | Runtime flags, defaults, and build-time backend selection |
| [architecture.md](architecture.md) | Workspace layout and application flows |
| [backend.md](backend.md) | CPU, Vulkan, Metal, hybrid placement, and KV cache |
| [server.md](server.md) | Text-only HTTP server and SSE protocol |
| [web.md](web.md) | Local web UI, assets, and available routes |
| [context.md](context.md) | Shared context estimate and request admission rules |

## Console

| Document | Contents |
|---|---|
| [console.md](console.md) | TUI session, keyboard controls, rendering, and status bar |
| [commands.md](commands.md) | Slash commands, transcripts, and local attachments |

## Development And Validation

| Document | Contents |
|---|---|
| [model-addition-process.md](model-addition-process.md) | Process for adding a family or profile |
| [backend-addition-process.md](backend-addition-process.md) | Process for adding a compute backend |
| [model-validation-process.md](model-validation-process.md) | Repeatable technical and semantic qualification |
| [kv-quant-mistral-validation.md](kv-quant-mistral-validation.md) | Ministral f16/int8 KV comparison contract |
| [oracle-validation-process.md](oracle-validation-process.md) | Numeric and external-oracle comparison process |
| [performance-investigation-process.md](performance-investigation-process.md) | Correctness-gated performance investigation |
| [metal-long-context-investigation.md](metal-long-context-investigation.md) | Vulkan-to-Metal long-context kernel investigation |
| [vulkan-decode-investigation.md](vulkan-decode-investigation.md) | Phase 1 Vulkan long-context decode and GQA KV reuse |
| [vulkan-decode-phase2-investigation.md](vulkan-decode-phase2-investigation.md) | Phase 2 Vulkan decode bottleneck attribution and retained optimizations |
| [prefill-short-context-investigation.md](prefill-short-context-investigation.md) | Phase 5--9 Vulkan prefill attribution, ceiling verification, and retained optimizations |
| [throughput-bench.md](throughput-bench.md) | End-to-end benchmark of the public API |

For installation, catalogs, and operational scripts, also see
[support/README.md](../support/README.md).
