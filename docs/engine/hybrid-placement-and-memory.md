<!--
This reference owns immutable hybrid placement, memory budgeting, prefill
ownership, and failure behavior. It does not assign backend support labels.
-->

# Hybrid Placement And Memory

A hybrid plan is selected while the model loads and never changes afterward.
A mixed plan assigns a contiguous layer prefix to CPU and the suffix to the
accelerator. Each layer's KV cache remains with its owner, and activations cross
the boundary once per pass.

## Effective Placement

The effective placement, not merely the Cargo feature name, selects batching
and execution:

| Effective placement | Prefill ownership | Numeric execution |
|---|---:|---|
| CPU standalone or `CpuOnly` | 32 rows | CPU |
| Vulkan standalone | 256 rows normally; 512 with the qualified Matrix2 Q4/Q64 capability set; the bounded-submission profile uses 128 or the deep/long 64-row bound | Vulkan |
| Vulkan-hybrid `AllGpu` | Accounted 32-row fallback; 512 with the eligible, capacity-fitting Matrix2 Q4/Q64 capability set | Vulkan |
| Metal standalone | 64 rows | Metal |
| Metal-hybrid `AllGpu` (`all-metal`) | 32 rows | Metal |
| CUDA standalone | 64 rows | CUDA |
| CUDA-hybrid `AllGpu` (`all-gpu`) | 32 rows | CUDA |
| Vulkan, Metal, or CUDA `Mixed` | 4 rows | CPU prefix, one crossing, accelerator suffix |

Metal stores the immutable `Mixed` fact and applies only the qualified
operation-local per-row matmul and serial-attention exceptions. Vulkan uses the
same kernels for homogeneous and mixed placement.

## Placement Options

`--vram-weights-percent` sets an explicit `0..=100` weight-placement limit.
When absent, hybrid profiles calculate placement from available resources.
Zero selects CPU-only without device initialization; 100 selects the
accelerator-only endpoint. `--vram-reserve-mib` withholds device capacity from
the automatic plan.

`Engine::placement()` exposes the immutable decision and the complete planned
CPU/accelerator breakdown. `Engine::memory()` reports retained model weights
and full-context KV capacity; neither value is process RSS or live allocator
telemetry.

Standalone CUDA may retain an additional lossless representation of Q6 layer
projection weights for prefill batches of at least 16 rows; decode keeps the
original packed weights. Admission accounts for both representations and falls
back to raw weights if the extra storage does not fit, without changing context
or placement. `Engine::memory()` includes any retained companion; CUDA hybrid
does not create it.

Standalone CUDA uses the same 64-row capacity for prefill allocation and checked
scratch admission. CUDA hybrid keeps its independent 32-row all-GPU and 4-row
mixed capacities. Cancellation is checked between graph batches.

Vulkan and CUDA use separate RAM and VRAM. On Linux, the automatic RAM budget
is `floor(MemAvailable × 90 / 100)` from one strict `/proc/meminfo`
`MemAvailable` field, and malformed or overflowing input admits zero host
bytes. The automatic VRAM reserve is the greater of 256 MiB and 5%. CUDA hybrid
probes visible device ordinal 0 only after validating configuration; percentage
zero selects CPU-only without initializing CUDA. A mixed pass performs exactly
one synchronous checked FP32 residual transfer from CPU to CUDA. Metal uses
unified memory derived from physical memory and the recommended working set,
counting weights, KV, scratch, fixed, staging, crossing, and reserve against the
same capacity.

Capability-specific wider routes include their scratch in admission. If an
eligible 512-row Matrix2 plan does not fit, placement is recomputed with the
accounted 32-row fallback before the result is committed. After final
selection, allocation, pipeline, kernel, transfer, command, readback, or decode
errors remain failures. The runtime never reduces context, changes placement,
or retries another backend.

Every hybrid profile uses request-local KV and does not reuse a prefix between
requests. CUDA hybrid adds no unified-memory path, multi-GPU ownership, or new
numeric kernel; it composes the existing CPU and CUDA implementations.

## KV Cache And Sampling

`EngineConfig` selects one KV representation while loading:

- `f16`, the default;
- `int8`, with per-vector quantization metadata.

Sampling supports greedy selection and the public `SamplingParams`. The
terminal always requests greedy sampling. The Web UI keeps Instruct greedy and
uses the qualified Reasoning policy with `temperature=0.7` for a supported
Reasoning profile.

Internal kernel formats do not expand the public GGUF contract. Numeric
qualification is recorded separately in
[validation evidence](../project-status/validation-evidence.md).
