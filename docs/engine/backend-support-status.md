<!--
This reference owns build-profile availability and the current reference,
qualified, and production support labels. Numeric implementation and placement
details remain in linked engine documents.
-->

# Backend Support Status

Graph Horizon loads its inference engine in-process and executes one active
generation at a time. Every build selects exactly one profile; the binary has
no runtime backend switch or fallback.

## Build Profiles

There are four numeric backend families: CPU, Vulkan, Metal, and CUDA.
`vulkan-hybrid`, `metal-hybrid`, and `cuda-hybrid` are composition profiles,
not additional numeric backend families.

| Feature | Execution | Main policy |
|---|---|---|
| `cpu` | Multi-core CPU | Portable path and numeric reference |
| `vulkan` | Vulkan GPU | Entire model on the GPU or an error |
| `vulkan-hybrid` | CPU and Vulkan | All GPU, layer split, or all CPU |
| `metal` | Apple Metal | Entire model on Metal or an error |
| `metal-hybrid` | CPU and Metal | All Metal, layer split, or all CPU |
| `cuda` | NVIDIA CUDA | Entire model on visible device ordinal 0 or an error |
| `cuda-hybrid` | CPU and CUDA | All GPU, layer split, or all CPU |

Neither the library crate nor the root binary has a default backend feature.
For reproducible builds:

```sh
cargo check --workspace --no-default-features --features cpu
cargo check --workspace --no-default-features --features vulkan
cargo check --workspace --no-default-features --features vulkan-hybrid
cargo check --workspace --no-default-features --features metal
cargo check --workspace --no-default-features --features metal-hybrid
cargo check --workspace --no-default-features --features cuda
cargo check --workspace --no-default-features --features cuda-hybrid
```

Build an executable with the same explicit selection:

```sh
cargo build --locked --no-default-features --features cpu
cargo build --locked --no-default-features --features vulkan
cargo build --locked --no-default-features --features vulkan-hybrid
cargo build --locked --no-default-features --features metal
cargo build --locked --no-default-features --features metal-hybrid
cargo build --locked --no-default-features --features cuda
cargo build --locked --no-default-features --features cuda-hybrid
```

Feature selection controls compiled dependencies, initialization, and resource
policy. Numeric operation variants remain capability- and shape-based.

## Status Definitions

Build availability and support status are separate claims:

- **production** is the recommended accelerated profile inside its declared
  scope and maintains public correctness and performance regression gates;
- **qualified** is supported only inside a recorded hardware and software
  tuple whose evidence does not yet cover the full production matrix;
- **reference** is the canonical numeric path and carries no performance
  promise.

## Support Status

| Profile | Status | Declared scope |
|---|---|---|
| `cpu` | **reference** | Public portable numeric path on accepted build platforms; Q4_K_M and f16/int8 KV |
| `vulkan` | **production** | Linux x86_64, required Vulkan capability set, Q4_K_M, f16/int8 KV, all-GPU; qualified tuples are recorded in validation evidence |
| `vulkan-hybrid` | **qualified** | Linux x86_64; all-GPU, mixed, or CPU-only placement |
| `metal` | **qualified** | Q4_K_M, f16/int8 KV, all-Metal; required Metal capabilities and qualified hardware tuples are recorded in validation evidence |
| `metal-hybrid` | **qualified** | The same capability and evidence contract; all-Metal, mixed, or CPU-only placement |
| `cuda` | **qualified** | Recorded Linux x86_64 validation environment, driver 595.84, CUDA Toolkit 12.4.131, authenticated 3B Instruct Q4_K_M, context 4096, and f16/int8 KV tuple only |
| `cuda-hybrid` | **qualified** | The same recorded Linux x86_64 capability/toolchain boundary and artifact; f16/int8 KV at context 4096 with 100/all-GPU, 25/mixed, and 0/CPU-only placement only |

The CPU path remains public even though its primary role is numeric reference.
Vulkan support outside Linux `x86_64` is not implied by build availability.
Homogeneous hybrid endpoints use their underlying numeric paths: all-GPU uses
the selected accelerator and CPU-only uses the CPU reference path. A composition
profile is qualified independently because its mixed placement and crossing
must pass their own real-model campaign.

Runtime admission and qualification are separate. A Vulkan, Metal, or CUDA device may
satisfy the baseline capability contract and use portable fallbacks without
inheriting a production or qualified evidence claim. The exact current
hardware, driver, and operating-system tuples are owned by
[validation evidence](../project-status/validation-evidence.md#current-backend-evidence).
An unavailable row is external verification, not an implicit support or failure
claim.

CUDA embeds compute-7.5 and compute-8.0 PTX images and loads exactly one.
Compute capability 8.0 or newer selects direct K16 matrix instructions for
packed prefill; older admitted devices, or a failed optional capability query,
retain the 7.5-compatible WMMA path. This selection does not broaden device
qualification; both images are tested in the recorded validation environment.

CUDA hybrid is qualified as composition, not as a new numeric backend. It uses
separate RAM and VRAM, an immutable plan, 32-row all-GPU and 4-row mixed
prefill, and exactly one synchronous CPU-to-CUDA residual crossing per mixed
pass. It does not provide prefix-KV reuse, unified memory, multi-GPU execution,
new kernels, a performance comparison, or a production claim. Neighboring
hardware, software, artifacts, model sizes, contexts, KV schemes, and placement
percentages remain outside the evidence boundary.

Supported artifacts are defined in
[supported models and formats](../supported-models-and-formats.md). Current
reviewed results are in the same
[validation evidence](../project-status/validation-evidence.md); measurements
against `llama.cpp` are comparative bottleneck evidence, not product scope.
