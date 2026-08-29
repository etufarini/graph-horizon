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
| Vulkan or Metal `Mixed` | 4 rows | CPU prefix, one crossing, accelerator suffix |

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

Vulkan uses separate RAM and VRAM. On Linux, the automatic RAM budget is
`floor(MemAvailable × 90 / 100)` and the automatic VRAM reserve is the greater
of 256 MiB and 5%. Metal uses unified memory derived from physical memory and
the recommended working set, counting weights, KV, scratch, fixed, staging,
crossing, and reserve against the same capacity.

Capability-specific wider routes include their scratch in admission. If an
eligible 512-row Matrix2 plan does not fit, placement is recomputed with the
accounted 32-row fallback before the result is committed. After final
selection, allocation, pipeline, kernel, transfer, command, readback, or decode
errors remain failures. The runtime never reduces context, changes placement,
or retries another backend.

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
