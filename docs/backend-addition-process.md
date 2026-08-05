<!--
This document owns the model-neutral process for adding a compute backend. It
defines implementation boundaries and evidence gates, not current support scope.
-->

# Backend Addition Process

This process applies when adding a hardware-facing implementation to
`graph_horizon_engine`. It does not claim that a proposed backend is supported. Current
build features and support limits remain in the
[engine documentation](../crates/graph_horizon_engine/README.md).

Backend selection is currently a build-time contract. Runtime backend switching
must not be introduced unless it is the explicitly approved feature.

## Current Boundary

The current build exposes five mutually exclusive features:

| Feature | Role |
|---|---|
| `cpu` | Compiles `CpuBackend`, the portable numeric reference |
| `vulkan` | Compiles `VulkanBackend`, with all-GPU-or-error placement |
| `vulkan-hybrid` | Compiles CPU and Vulkan for an immutable split plan |
| `metal` | Compiles `MetalBackend`, with all-Metal-or-error placement |
| `metal-hybrid` | Compiles CPU and Metal for a unified-memory split plan |

CPU, Vulkan, and Metal implement the model-neutral `Backend` trait. A
hybrid-capable device also implements `HybridDevice`; the neutral runtime owns
the immutable partition, backend resources, and crossing between them. Model
families do not contain backend-pair orchestration.

The stable boundaries are:

```text
crates/graph_horizon_engine/
├── Cargo.toml                    backend features and optional dependencies
├── build.rs                      feature-gated build-time assets
└── src/
    ├── backend/
    │   ├── contract.rs           model-neutral Backend trait
    │   ├── mod.rs                feature selection and shared modules
    │   └── <name>/               one concrete hardware backend
    ├── runtime/                  homogeneous and partitioned execution
    └── family/<family>/          graph and model loading
```

A concrete backend owns device or runtime state, buffers, weight upload, kernel
dispatch, readback, and backend-local memory policy. It must not own tokenizer,
model-family detection, tensor naming, chat templates, sampling, or public
application behavior.

## Phase 1: Freeze The Scope

Before implementation, define:

- the backend feature name and supported operating systems;
- required SDK, loader, driver, compiler, and device capabilities;
- supported weight formats, KV schemes, and context limits;
- whether the backend runs the full graph or participates in a combined mode;
- numeric oracles, thresholds, hardware rows, and external prerequisites;
- files to add or change, with their single responsibilities and line estimates.

Keep a new backend under `backend/<name>/`. Split genuine domains such as
initialization, memory, execution, pipelines, and kernels into subfolders only
when each domain needs more than one file. Do not create prefix-named sibling
files or a directory containing only `mod.rs`.

## Phase 2: Add Feature Selection

Add one Cargo feature and forward it through the root package when the binary
must expose it. Gate every dependency, build dependency, link directive, build
step, and platform requirement behind that feature.

Preserve these compile-time invariants:

- selecting no concrete backend is an error;
- incompatible concrete backend combinations are errors;
- an intentional combined mode has its own explicit feature;
- a CPU-only build does not require a GPU SDK, loader, or compiler;
- unaffected existing features keep building independently.

Update `support/install.sh` only when the new feature is ready to become an
installer-supported choice. The installer must validate prerequisites but must
not install SDKs, invoke `sudo`, or modify system configuration.

## Phase 3: Implement Resource Ownership

Initialize the runtime, select a device, create queues or streams, and validate
required capabilities before model allocation. Missing hardware or runtime
support returns a bounded error; it must not panic or silently select another
backend.

Implement allocation, views, upload, readback, alignment, and cleanup before
graph operations. Preserve these invariants:

- a view aliases storage and never owns the parent allocation;
- parent buffers are released exactly once;
- partial initialization releases every acquired resource;
- byte sizes, offsets, alignments, and copy ranges use checked arithmetic;
- readback validates destination size before copying;
- cancellation and execution errors release per-request KV resources.

Treat model bytes, device properties, driver responses, and compiler diagnostics
as untrusted input. Public errors must not disclose local paths, unbounded driver
output, or internal stack traces.

## Phase 4: Implement The Trait

Implement the complete `Backend` contract in the order exposed by
`backend/contract.rs`: lifecycle and buffers, KV writes, encoder submission,
embedding and projections, normalization and RoPE, elementwise operations,
attention, and readback.

The implementation file is one `impl Backend` block of thin delegators. Put
resource algorithms and numeric kernels in their owning modules. Keep opaque
buffer and encoder types inside the backend boundary; family graphs must remain
generic over `Backend`.

Start with the simplest correct path. Add batching, fusion, SIMD, subgroup, or
device-specific kernels only after a scalar, unfused, or host oracle exists.
Optimized paths must retain a deterministic way to test the reference result.

## Phase 5: Integrate Loading And Combined Modes

Load only explicitly supported weight formats. Unsupported formats fail during
model loading and never enter a fallback that changes the numeric contract.

Wire a standalone backend through the existing family loader without importing
its concrete API into the public engine, CLI, server, or web modules. When a
combined mode is approved, keep placement explicit and immutable. Record which
backend owns each layer and its KV storage, validate transfer boundaries, and do
not retry with a different split after observing an error.

Adding a backend does not add a model family. Validate every family and profile
claimed for the backend through the
[model addition process](model-addition-process.md).

## Phase 6: Run Synthetic Gates

Run model-free tests before using external artifacts. Cover:

- feature guards and dependency isolation;
- capability detection and unavailable-device errors;
- buffer allocation, aliasing, bounds, alignment, and cleanup failpoints;
- each trait operation against a host or existing-backend oracle;
- every claimed weight format and KV scheme;
- fused and optimized paths against their reference paths;
- malformed sizes, overflow, submission failure, cancellation, and readback;
- combined-mode placement arithmetic and ownership boundaries, when applicable.

Existing build rows must remain green:

```sh
cargo check --workspace --no-default-features --features cpu
cargo check --workspace --no-default-features --features vulkan
cargo check --workspace --no-default-features --features vulkan-hybrid
cargo check --workspace --no-default-features --features metal
cargo check --workspace --no-default-features --features metal-hybrid
```

Add the new standalone or combined feature to this matrix. Also verify that
unsupported feature combinations fail with the intended compile-time error.

## Phase 7: Run Real-Model Validation

A backend is qualified per artifact, family/profile, weight format, KV scheme,
context, and placement row. Pin the artifact digest and configuration before the
run. Compare tokenizer and rendered prompt IDs before comparing generation.

Use a predeclared numeric contract such as logits drift, KL divergence,
perplexity ratio, teacher-forced top-k, or greedy token identity. Missing
hardware, drivers, artifacts, or oracle programs are `external-verification`,
not passes. Out-of-memory is capacity evidence for the fixed row and must not
trigger an implicit smaller-context or different-backend retry.

Combined modes require all-device, all-host when supported, and true mixed rows
with positive work on both sides. Record requested and final placement.

## Phase 8: Measure Performance

Measure only after correctness gates pass. The current benchmark takes the model
as its first positional argument and requires explicit context and KV values:

```sh
cargo run --no-default-features --features cpu --example bench -- \
  /path/to/model.gguf --context 4096 --kv f16 \
  --prompt "Hello" --max-tokens 32 --warmup 1 --reps 3
```

Replace `cpu` with the backend feature under test while keeping the artifact,
context, KV, prompt, generation length, warmup, and repetitions fixed. Record
hardware and driver/runtime versions with TTFT, prompt throughput, decode
throughput, and capacity failures. Performance does not override correctness.

## Documentation And Evidence

Before claiming support, update:

- the engine README support table and limitations;
- root configuration and installation instructions;
- `support/install.sh` and profiling or validation scripts, when applicable;
- `VALIDATION.md` with one terminal state per approved matrix row;
- [backend.md](backend.md) and benchmark notes when their current contracts change.

Do not list speculative platforms, SDKs, devices, or formats as supported. A
future backend may be discussed only as an example until its implementation and
evidence are present.

## Definition Of Done

A standalone backend implements the neutral `Backend` contract. If it supports
partitioned execution it also implements `HybridDevice`; the shared hybrid
runtime owns placement and crossing. Model families and graphs remain unchanged
and no family/backend pair file is added.

A backend addition is complete when feature selection is explicit, unrelated
builds do not require its dependencies, the full trait is implemented without
hardware APIs escaping its domain, resource and kernel tests pass, every claimed
model/KV/format/placement row has a terminal validation result, performance is
measured or explicitly out of scope, and documentation states the exact support
boundary.
