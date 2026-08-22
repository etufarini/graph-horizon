<!--
This index owns navigation for the repository documentation. It separates stable,
model-neutral behavior from the current support contract and validation evidence.
-->

# Documentation

These pages describe Graph Horizon surfaces and processes without assuming a
specific model family, size, or release. Examples therefore use the
`<model.gguf>` placeholder.

Ministral 3 is the model family integrated by the current implementation. It is
not part of the Graph Horizon project name or a separate product edition.

Concrete support changes with the code: currently accepted architectures, GGUF
profiles, and limits are defined in the
[library contract](../crates/graph_horizon_engine/README.md), while reviewed results
belong in the [validation register](validation.md).

Performance and cleanup reports preserve measurements at their recorded SHAs.
Their mission-close Git state is historical; later push and merge status is
recorded in each affected report and summarized in
[current-optimization-state.md](current-optimization-state.md).

## Usage And Structure

| Document | Contents |
|---|---|
| [configuration.md](configuration.md) | Runtime flags, defaults, and build-time backend selection |
| [architecture.md](architecture.md) | Workspace layout and application flows |
| [backend.md](backend.md) | CPU, Vulkan, Metal, hybrid placement, and KV cache |
| [web.md](web.md) | Local Web UI, assets, and browser behavior |
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
| [vulkan-nvidia-long-context-prefill.md](vulkan-nvidia-long-context-prefill.md) | NVIDIA Vulkan long-context prefill attribution, routing fix, and quantitative stop checkpoint |
| [vulkan-nvidia-long-context-decode.md](vulkan-nvidia-long-context-decode.md) | NVIDIA Vulkan long-context decode attribution and final routing checkpoint |
| [metal-llamacpp-competitive-optimization.md](metal-llamacpp-competitive-optimization.md) | Metal performance comparison and optimization checkpoint against llama.cpp Metal |
| [metal-vulkan-optimization-parity.md](metal-vulkan-optimization-parity.md) | Vulkan-to-Metal capability mapping, retained candidates, and global-stop evidence |
| [metal-long-context-prefill-optimization.md](metal-long-context-prefill-optimization.md) | Apple Metal long-context prefill attribution, retained attention optimization, and final checkpoint |
| [metal-long-context-decode-optimization.md](metal-long-context-decode-optimization.md) | Completed Apple Metal long-context decode checkpoint and candidate ledger |
| [throughput-bench.md](throughput-bench.md) | End-to-end benchmark of the engine library |
| [vulkan-amd-long-context-decode.md](vulkan-amd-long-context-decode.md) | AMD Vulkan long-context decode investigation and checkpoint |
| [vulkan-amd-long-context-prefill.md](vulkan-amd-long-context-prefill.md) | AMD Vulkan long-context prefill attribution, split-GQA route, and qualification |
| [final-backend-model-family-cleanup.md](final-backend-model-family-cleanup.md) | Cleanup-candidate ownership inventory, decisions, qualification snapshot, and later repair link |
| [amd-deep-clean-regression-repair.md](amd-deep-clean-regression-repair.md) | AMD post-cleanup correctness diagnosis, repair, and qualification evidence |

## Reviewed Evidence

| Document | Contents |
|---|---|
| [Current optimization state](current-optimization-state.md) | Retained CPU, Vulkan, AMD, and Metal results and remaining bottlenecks |
| [Validation register](validation.md) | Artifact identities, reviewed qualification evidence, and release readiness |
| [Pre-release qualification](release-qualification.md) | Historical v0.1.0 candidate gates that must be refreshed before release |
| [Release notes](release-notes.md) | Draft user-facing v0.1.0 contract, installation, capabilities, and limitations |

For installation, catalogs, and operational scripts, also see
[support/README.md](../support/README.md).
