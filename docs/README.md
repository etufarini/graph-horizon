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
| [prefill-long-attention-phase10.md](prefill-long-attention-phase10.md) | Phase 10 Vulkan long-attention attribution, reuse/resource ceilings, and stop decision |
| [prefill-global-phase11.md](prefill-global-phase11.md) | Phase 11 global prefill attribution, quantized-matmul experiment, and exact-path stop decision |
| [vulkan-predecoded-weights-phase12.md](vulkan-predecoded-weights-phase12.md) | Phase 12 Vulkan persistent predecoded-weight economics, retained gate/up path, and qualification |
| [vulkan-native-weight-frontier-phase13.md](vulkan-native-weight-frontier-phase13.md) | Phase 13 marginal TTFT/VRAM frontier, rejected down expansion, compact-representation audit, and exact-path stop decision |
| [vulkan-native-lowprecision-weights-phase14.md](vulkan-native-lowprecision-weights-phase14.md) | Phase 14 native low-precision capability audit, rejected INT8 prototypes, numeric evidence, and quality-gated stop decision |
| [vulkan-attention-algorithm-phase15.md](vulkan-attention-algorithm-phase15.md) | Phase 15 long-context attention physical model, rejected exact/numeric architectures, and dense-attention stop decision |
| [vulkan-sparse-attention-phase16.md](vulkan-sparse-attention-phase16.md) | Phase 16 sparse/window/global/layer-hybrid quality-performance frontier, rejected prototype, and model-adaptation stop decision |
| [vulkan-sparse-execution-phase17.md](vulkan-sparse-execution-phase17.md) | Phase 17 fixed-mask realization efficiency, rejected traversal/runtime candidates, measured physical floor, and model-adaptation boundary |
| [vulkan-llamacpp-gap-phase18.md](vulkan-llamacpp-gap-phase18.md) | Phase 18 same-device llama.cpp benchmark, competitive gap attribution, architectural comparison, and Phase 19 candidate ranking |
| [vulkan-ampere-attention-phase19.md](vulkan-ampere-attention-phase19.md) | Phase 19 Ampere Matrix2 Q64 attention implementation, resource/ISA audit, causal attribution, and qualified 28K improvement |
| [vulkan-specialization-architecture-phase20.md](vulkan-specialization-architecture-phase20.md) | Phase 20 NVIDIA Matrix2 attention capability/shape routing, portability inventory, preserved performance, and future GPU/model policy |
| [vulkan-matmul-architecture-phase21.md](vulkan-matmul-architecture-phase21.md) | Phase 21 quantized-matmul attribution, down-projection architecture experiment, rejected direct Matrix2 feed, and evidence-based stop decision |
| [vulkan-second-model-bringup-phase22.md](vulkan-second-model-bringup-phase22.md) | Phase 22 8B weight bring-up, generic and optimized Vulkan qualification, reuse matrix, baseline, and next bottleneck |
| [vulkan-model2-largek-matmul-phase23.md](vulkan-model2-largek-matmul-phase23.md) | Phase 23 Model 2 large-K down-projection attribution, retained Matrix2 route, full qualification, and next bottleneck |
| [vulkan-model2-gate-up-phase24.md](vulkan-model2-gate-up-phase24.md) | Phase 24 Model 2 gate/up attribution, retained compressed Matrix2 route, full qualification, and attention handoff |
| [vulkan-model2-post-mlp-phase25.md](vulkan-model2-post-mlp-phase25.md) | Phase 25 post-MLP global reprofile, competitive removable-cost ranking, attention causal audit, and exact-path stop decision |
| [vulkan-whole-inference-phase26.md](vulkan-whole-inference-phase26.md) | Phase 26 whole-inference checkpoint, retained decode kernels, six-model quality matrix, competitive gaps, and next-cycle decision |
| [throughput-bench.md](throughput-bench.md) | End-to-end benchmark of the public API |

For installation, catalogs, and operational scripts, also see
[support/README.md](../support/README.md).
