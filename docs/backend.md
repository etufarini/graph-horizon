<!--
This document owns the model-neutral local inference and backend contract. Exact
supported model families and artifact profiles remain in the engine README.
-->

# Local Inference And Backends

The `graph_horizon_engine` crate loads a supported GGUF read-only and exposes a
synchronous facade for text generation. The local console, server, and web UI
share the same entry point; no surface imports a concrete backend directly.

## Model Boundary

The `Engine` facade hides the family from application surfaces, but the current
loader forwards to one concrete implementation. That contract validates
`general.architecture`, required metadata, the tokenizer, and tensor shapes and
types. The filename, directory, and total size do not authorize loading.

An unsupported format must fail before backend allocation. The current set of
architectures and GGUF profiles belongs to the
[library contract](../crates/graph_horizon_engine/README.md). Adding a family or
profile follows the [model addition process](model-addition-process.md); adding
a compute target follows the
[backend addition process](backend-addition-process.md).

## Public API

`Engine::new(path, EngineConfig)` validates the file and builds its resources. A
`Request` contains text messages, sampling parameters, and `max_tokens`.
`Engine::generate` sends only these events to the sink:

- `TextDelta(String)` for incremental text;
- `Finished(GenerationStats)` for completion and metrics;
- `Error("generation failed")` as the normalized terminal error.

Each generation has exactly one terminal event. If the sink cancels the request,
decoding stops and no further events are emitted.

## Build Backends

The build profile is selected at build time, not with a runtime flag. There are
exactly three numeric backend families: CPU, Vulkan, and Metal.
`vulkan-hybrid` and `metal-hybrid` are public composition profiles, not numeric
backend families:

| Feature | Execution | Main Policy |
|---|---|---|
| `cpu` | Multi-core CPU | Portable path and numeric reference |
| `vulkan` | Vulkan GPU | Entire model on the GPU or an error |
| `vulkan-hybrid` | CPU and Vulkan | All GPU, layer split, or all CPU |
| `metal` | Apple Metal | Entire model on Metal or an error |
| `metal-hybrid` | CPU and Metal | All Metal, layer split, or all CPU |

The library crate has no default feature, and neither does the root binary.
`support/install.sh` requires the choice explicitly:

```sh
support/install.sh --backend cpu
support/install.sh --backend vulkan
support/install.sh --backend vulkan-hybrid
support/install.sh --backend metal
support/install.sh --backend metal-hybrid
```

For reproducible Cargo builds:

```sh
cargo check --workspace --no-default-features --features cpu
cargo check --workspace --no-default-features --features vulkan
cargo check --workspace --no-default-features --features vulkan-hybrid
cargo check --workspace --no-default-features --features metal
cargo check --workspace --no-default-features --features metal-hybrid
```

## Hybrid Placement

The hybrid plan is decided during loading and remains immutable. A mixed plan
assigns a contiguous prefix of layers to the CPU and the suffix to the GPU; each
layer's KV remains on its owning backend. Activations cross the boundary once per
pass.

The effective placement, rather than the Cargo profile name, selects batching
and numeric dispatch:

| Effective placement | Prefill rows | Numeric execution |
|---|---:|---|
| CPU standalone or `CpuOnly` | 4 | CPU |
| Vulkan standalone or `AllGpu` | 32 | Vulkan |
| Metal standalone or `AllGpu` (`all-metal`) | 32 | Metal |
| Vulkan or Metal `Mixed` | 4 | CPU prefix, one crossing, GPU suffix |

Feature conditions control availability, dependencies, and loading. They do not
select operation variants. Metal stores the immutable `Mixed` fact and applies
only the two qualified operation-local exceptions: per-row matmul and serial
attention. Vulkan uses the same kernels for homogeneous and mixed placement.
Weights and KV stay with the backend that owns each layer.

`--vram-weights-percent` sets an explicit limit from `0` to `100`; when absent,
hybrid calculates placement from available resources. `--vram-reserve-mib`
withholds VRAM from the budget and participates in the hybrid automatic plan.
`Engine::placement()` exposes the final decision and planned CPU/GPU memory
breakdown.

Correct 32-row scratch accounting can change a capacity-bound `AllGpu` split.
The requested percentage, reserve, candidate order, immutable result, and
no-retry behavior remain unchanged; the runtime never reduces context or
replans after an allocation or execution error.

Vulkan owns device memory and explicit transfer/staging buffers. Metal uses
shared/private storage as required by CPU visibility, one command queue, and
offline-compiled kernels; its planner counts weights, KV, scratch, fixed,
staging, crossing, and reserve against unified memory. Both support Q4_K_M
weights and f16/int8 KV. Standalone backends perform no implicit fallback;
device, allocation, pipeline, command, and readback failures remain errors.

Metal support is currently qualified on Apple M4 with macOS 26.3. Other Apple
silicon generations and operating-system versions remain unqualified rather
than implicitly supported.

## KV Cache And Sampling

`EngineConfig` selects the KV cache once during loading:

- `f16`, the default;
- `int8`, a compact format with per-vector quantization metadata.

Sampling supports greedy/argmax and the public parameters defined by
`SamplingParams`. Application surfaces may expose only a subset. The server and
Web wrapper keep Instruct greedy and select the qualified temperature `0.7`
policy for a loaded Reasoning profile; request sampling fields remain ignored.

Internal kernel numeric formats do not automatically expand the accepted GGUF
contract. Technical support and numeric qualification must be recorded
separately in [VALIDATION.md](../VALIDATION.md).
