<!--
This document owns the model-neutral local inference and backend contract. Exact
supported model families and artifact profiles remain in the engine README.
-->

# Local Inference And Backends

The `graph_horizon_engine` crate loads a supported GGUF read-only and exposes a
synchronous facade for text generation. The local console, server, and web UI
share the same entry point; no surface imports a concrete backend directly.

It is designed for one active chat generation at a time. This narrow execution
model keeps KV state, cancellation, allocation lifetime, and error handling
local to one request. Multi-request batching and serving throughput are outside
the runtime contract; the HTTP surface exists to carry the same chat flow used
by the console and Web UI.

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
Messages contain at most one initial `System`, then strictly alternate `User`
and non-empty `Assistant`, and end with `User`; any other sequence is a
generation error.
`Engine::generate` sends only these events to the sink:

- `Phase(Prefill)` immediately before prompt evaluation;
- `Phase(Decode)` immediately before autoregressive generation;
- `TextDelta(String)` for incremental text;
- `Finished(GenerationStats)` for completion and metrics;
- `Error("generation failed")` as the normalized terminal error.

The two phase events are ordered and occur at most once. `GenerationStats`
contains prompt, actually-prefilled, and completion token counts plus separate
prefill and decode durations. Cached prompt tokens remain part of
`prompt_tokens` but not `prefill_tokens`, so prefill throughput is not inflated.
Each generation has exactly one terminal event. If the sink cancels the request,
decoding stops and no further events are emitted.

`Engine::generate_cached(cache_key, request, sink)` has the same event and
cancellation contract. Standalone Vulkan and Metal retain one caller-keyed KV
slot and reuse only an exact rendered-token prefix with the same 16-byte key;
another key or prefix may replace it. CPU and hybrid profiles execute the same
path as `generate` without prefix reuse. The cache key is caller-supplied
namespacing, not a credential.

After a successful load, `Engine::model_name()` exposes only the bounded,
control-free `general.name` display value when present; it never derives an
identity from the model path. `Engine::backend_name()` reports the statically
selected profile. `Engine::memory()` reports retained model weights and the
full-context KV capacity for every profile; these immutable planned bytes are
not process RSS or live allocator telemetry. `Engine::placement()` remains the
source of effective hybrid ownership and its complete allocation breakdown.
`Engine::context_limit()` returns the positive resolved context, and
`Engine::default_sampling()` returns greedy parameters for Instruct or the
qualified Reasoning policy with `temperature=0.7`.

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

Compile-time selection is deliberate: a binary contains only the dependencies,
initialization, and memory policy of its chosen backend. This keeps platform
builds small and predictable without preventing deployment on different
machines, which use separately built profiles. Within a profile, kernel routing
remains capability- and shape-based and preserves the generic fallback.

For reproducible Cargo builds:

```sh
cargo check --workspace --no-default-features --features cpu
cargo check --workspace --no-default-features --features vulkan
cargo check --workspace --no-default-features --features vulkan-hybrid
cargo check --workspace --no-default-features --features metal
cargo check --workspace --no-default-features --features metal-hybrid
```

## Support Status

Build availability and support status are separate claims. An accepted build
tuple means that the installer and feature guards admit it; it does not make
the resulting profile production or qualified.

- **production** is the recommended accelerated profile inside its declared
  scope. Its public model, format, lifecycle, fallback, correctness, and
  performance gates are maintained as release regressions.
- **qualified** is supported only inside an explicit hardware and software
  tuple. Its evidence is either hardware-specific or does not yet cover the
  complete production promotion matrix.
- **reference** identifies the canonical numeric path used as an oracle. It is
  a role rather than a lower maturity tier and carries no performance promise.

The current primary labels are:

| Profile | Status | Declared scope |
|---|---|---|
| `cpu` | **reference** | Public portable numeric path on accepted build platforms; Q4_K_M and f16/int8 KV |
| `vulkan` | **production** | Linux x86_64, compatible NVIDIA or AMD Vulkan device, Q4_K_M, f16/int8 KV, all-GPU |
| `vulkan-hybrid` | **qualified** | Linux x86_64; all-GPU, mixed, or CPU-only placement |
| `metal` | **qualified** | Apple M4 10-core GPU, macOS 26.3, Q4_K_M, f16/int8 KV, all-Metal |
| `metal-hybrid` | **qualified** | The same Apple M4/macOS tuple; all-Metal, mixed, or CPU-only placement |

The CPU profile remains a public supported path even though its primary role is
reference. Vulkan support outside Linux x86_64 is not implied by build
availability. `vulkan-hybrid` remains qualified because its placement and
implementation gates pass, but a fresh complete real-model `Mixed` campaign is
still required before production promotion. The homogeneous endpoints retain
their underlying numeric status: all-GPU uses production Vulkan, while CPU-only
uses the CPU reference path.

Metal and Metal-hybrid qualification is evidence for the stated M4/macOS tuple,
not a promise for every Apple GPU, Apple silicon generation, or operating-system
version. Current terminal evidence remains in
[VALIDATION.md](../VALIDATION.md); that register records results and does not
assign product status.

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

## KV Cache And Sampling

`EngineConfig` selects the KV cache once during loading:

- `f16`, the default;
- `int8`, a compact format with per-vector quantization metadata.

Sampling supports greedy/argmax and the public parameters defined by
`SamplingParams`. Application surfaces may expose only a subset. The server and
Web wrapper keep Instruct greedy and select the Reasoning sampling policy
defined by the qualification protocol, with `temperature=0.7`; request
sampling fields remain ignored.

Internal kernel numeric formats do not automatically expand the accepted GGUF
contract. Technical support and numeric qualification must be recorded
separately in [VALIDATION.md](../VALIDATION.md).

## Qualification And Performance References

The reviewed matrix contains only hardware available to the project's sole
maintainer. A qualified row is concrete evidence for that tuple; a missing row
is not silently promoted to support or failure. This deliberately narrow claim
keeps validation reproducible as the backend evolves.

`llama.cpp` measurements are external performance references used to identify
and rank bottlenecks under comparable model, quantization, context, and hardware
settings. They do not define Graph Horizon's feature scope: this runtime remains
focused on a simple Ministral chat experience through its console and Web UI.
