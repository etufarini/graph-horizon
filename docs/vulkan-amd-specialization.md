<!--
This document is the persistent checkpoint for the AMD Vulkan specialization
program: immutable baseline identity, device evidence, competitive measurements,
routing audit, experiment decisions, qualification, and quantitative stop state.
-->

# AMD Vulkan Specialization

## Status

The capability inventory, baseline screen, first attribution, and path audit are
complete. An AMD integer-dot Q4_K batched matmul candidate is retained after
hardware parity and competitive measurement. A Q6_K variant was rejected after
the established integration parity suite found a quality regression. A reversible 64-row control first
confirmed that the resets were watchdog failures caused by one long full-graph
submission. The 64-row control is not retained; long-workload qualification of
the Q4-only state remains pending.

Raw local evidence is retained under `target/amd-baseline/` and is intentionally
ignored by Git. Values below are promoted only after checking the tuple and raw
record.

## Immutable start

| Item | Value |
|---|---|
| Date | 2026-08-18 |
| Baseline branch | `main` |
| Baseline commit | `2d7d7f7fee3535a0ee3670ec5f13f37061cefa9c` |
| Merge-base with `main` | `2d7d7f7fee3535a0ee3670ec5f13f37061cefa9c` |
| Starting worktree | clean; `main...origin/main` |
| Investigation branch | `perf/amd-hardware-specialization` |
| Graph Horizon public binary | SHA-256 `8a58d03702bc72b8b6ce5d06c021622f0b1da464b8a04f674b8a749b1e867ce7` |
| Graph Horizon profile binary | SHA-256 `fb65e0fc644e4800949a8fe2512d92877365ac3247ab49d3824fa86a49dd2f8e` |
| llama.cpp reference | local detached worktree at `9bebfcb4b`, build 9826 |
| Push state | nothing pushed |

The start is the merged refactoring/production-cleanup HEAD. Its checked-in
`CURRENT_OPTIMIZATION_STATE.md` records the completed refactoring and qualified
portable/NVIDIA state, so this program does not mix refactoring with AMD work.

## AMD capability inventory

Evidence labels are `queried`, `measured`, `derived`, and `unavailable`.

### Identity and software

| Capability | Value | Evidence |
|---|---|---|
| PCI identity | vendor `0x1002`, device `0x73df`, revision `0xc0`, Navi 22 | queried: `lspci -nnk` |
| Device | AMD Radeon RX 6750 XT, gfx1031 | queried: Vulkan, ROCm |
| Kernel driver | amdgpu, Linux `7.0.0-29-generic` | queried |
| Vulkan driver | Mesa RADV `26.0.3-1ubuntu1` | queried |
| Vulkan device API | 1.4.335, conformance 1.4.0.0 | queried |
| Vulkan loader API | 1.4.341 | queried |
| Compute units | 40; 2 SIMDs/CU; 2 shader engines, 2 arrays/engine | queried: `rocminfo` |
| Maximum GPU clock | 2,880 MHz | queried: `rocminfo` |

Product text is recorded for reproducibility only. Future routing must use
vendor/capability/resource/shape predicates, never this product string.

### Vulkan extensions and arithmetic

RADV advertises 230 device extensions. The compute-relevant advertised set
includes:

```text
VK_AMD_device_coherent_memory
VK_AMD_gcn_shader
VK_AMD_gpu_shader_half_float
VK_AMD_gpu_shader_int16
VK_AMD_shader_ballot
VK_AMD_shader_core_properties
VK_AMD_shader_core_properties2
VK_EXT_calibrated_timestamps
VK_EXT_memory_budget
VK_EXT_pipeline_creation_feedback
VK_EXT_subgroup_size_control
VK_KHR_16bit_storage
VK_KHR_8bit_storage
VK_KHR_buffer_device_address
VK_KHR_performance_query
VK_KHR_pipeline_executable_properties
VK_KHR_shader_float16_int8
VK_KHR_shader_integer_dot_product
VK_KHR_shader_subgroup_extended_types
VK_KHR_workgroup_memory_explicit_layout
```

The complete extension list remains reproducible with `vulkaninfo`; this list
records every extension relevant to the current compute investigation. On the
AMD physical device, `VK_KHR_cooperative_matrix` is **not** advertised. The same
host's llvmpipe device does advertise it, so capability output must not be
mixed between physical devices.

| Capability | Value | Evidence |
|---|---|---|
| FP16 | shader FP16 and 16-bit storage true; fast F16 true | queried |
| INT8 | shader INT8 and 8-bit storage true | queried |
| Integer dot | signed/unsigned scalar 8-bit and packed 4x8 accelerated; mixed signedness false | queried |
| Cooperative/matrix | no cooperative-matrix extension or shapes on AMD; llama.cpp reports `matrix cores: none` | queried |
| Buffer address | supported | queried |
| Explicit LDS layout | scalar, 8-bit, and 16-bit access supported | queried |

### Subgroup and execution limits

| Capability | Value | Evidence |
|---|---|---|
| Vulkan default subgroup | 64 | queried |
| Controllable subgroup range | 32 through 64; compute supports required subgroup size | queried |
| HSA native wavefront | 32 | queried |
| llama.cpp selected Vulkan warp | 32 | queried from startup log |
| Subgroup operations | basic, vote, arithmetic, ballot, shuffle, shuffle-relative, clustered, quad, rotate, rotate-clustered | queried |
| Workgroup invocations | 1,024 | queried |
| Workgroup dimensions | 1,024 × 1,024 × 1,024 | queried |
| Workgroup counts | 4,294,967,295 × 65,535 × 65,535 | queried |
| Maximum waves/CU | 32 | queried: HSA |
| Maximum work-items/CU | 1,024 | queried: HSA |
| LDS/workgroup | 64 KiB | queried |
| Maximum fbarriers/workgroup | 32 | queried |

The Vulkan/HSA subgroup difference is material: a pipeline may default to
wave64 even though RDNA2 supports wave32. It is not evidence that wave32 is
faster; it identifies a controlled measurement axis.

### Memory, alignment, and cache

| Capability | Value | Evidence |
|---|---|---|
| Device-local heap | 11.75 GiB; observed idle budget about 10.6 GiB | queried |
| Host heap | 15.36 GiB | queried |
| Host-visible device-local heap | 256 MiB; observed budget 94.75 MiB | queried |
| Storage-buffer range / max buffer | 4 GiB minus implementation tail | queried |
| Storage/uniform/texel minimum offset alignment | 4 bytes | queried |
| Non-coherent atom size | 64 bytes | queried |
| Cache line | 128 bytes | queried: HSA |
| L1 / L2 / L3 | 16 KiB / 3 MiB / 96 MiB | queried: HSA |
| Baseline active VRAM | 53% during 3B/8K; idle 9% | measured: `rocm-smi` |

### Runtime telemetry and unavailable counters

The sustained 3B/8K Graph Horizon row measured 99% GPU use, 2,630 MHz shader
clock, 135--140 W package power, 54--56 C edge, 86--90 C junction, and 52 C
memory. Idle was about 500 MHz, 8--10 W, and 49--51 C.

RADV exposes pipeline executable and performance-query capabilities, but the
current runtime does not enable/query executable statistics. VGPR, SGPR, LDS
allocation per compiled pipeline, spills, scratch, achieved occupancy, and
resident waves are therefore currently **unavailable**, not estimated. A
minimal feature-gated query is permitted only after the competitive ranking
shows which pipelines need that evidence.

## Initial AMD architecture model

- Effective wave model: hardware and llama.cpp use wave32; Graph Horizon's
  current Vulkan pipelines observe the RADV default subgroup 64 unless a future
  pipeline explicitly requests 32.
- Matrix strategy: no usable AMD cooperative-matrix Vulkan path exists on this
  stack. The viable classes are packed integer dot, subgroup/vector, and
  vectorized scalar/FMA.
- Resource model: 64 KiB LDS and 1,024 work-items/CU bound occupancy. Exact
  VGPR cliffs remain unavailable pending pipeline-executable instrumentation.
- Sustained 99% GPU utilization with only 3--4% reported memory activity on the
  slow portable prefill path is compatible with instruction/dequantization or
  dependency limits, not proof of a DRAM-bandwidth floor.
- The 8K row is stable at full clocks. The 16K failure is a gfx ring timeout,
  not ordinary thermal throttling or VRAM exhaustion.

This model is provisional until focused kernel timestamps and executable
statistics cover the dominant pipelines.

The first retained result confirms the arithmetic model: converting FP16
activations to Q8_1 consumes only 0.18% of candidate prefill GPU time, while
packed integer dot materially reduces Q4_K projection time. A similar Q6_K route
was faster but its Q8 activation error exceeded the established prefill oracle;
speed does not override that quality boundary.

## Competitive tuple

Both runtimes use the authenticated 3B Instruct Q4_K_M artifact (2,147,023,008
bytes, SHA-256 `9ed150d4367e68df0ac8e1540f6ddc65b42d0ee26378329d1ecbca60f93fc5f8`),
the same AMD GPU, Vulkan, full GPU placement, F16 K/V, dense causal attention,
and three measured repetitions after warm-up for prefill. Graph Horizon uses
chat-templated repeated `" a"` text that tokenizes to the exact requested row
count. llama.cpp uses synthetic benchmark token IDs.

Graph Horizon TTFT includes tokenization, logits, first sampling, and public
text emission. llama.cpp pure prefill excludes those host boundaries. Decode
uses 32 generated steps. For stable llama.cpp KV decode, `-d N` constructs and
snapshots an N-token prefix outside the timed section and five repetitions time
only generation from that depth.

## First competitive gap map

`reset` is not a numeric result. Rows without a completed Graph Horizon prefill
also have no comparable decode depth.

| Model | Workload | Graph Horizon | llama.cpp | Ratio | Absolute gap |
|---|---|---:|---:|---:|---:|
| 3B | prefill 128 | 624.35 ms | 158.46 ms | 3.94x | 465.89 ms |
| 3B | prefill 512 | 2,458.54 ms | 220.47 ms | 11.15x | 2,238.07 ms |
| 3B | prefill 2K | 10,264.76 ms | 1,190.97 ms | 8.62x | 9,073.79 ms |
| 3B | prefill 8K | 49,418.34 ms | 6,068.10 ms | 8.14x | 43,350.24 ms |
| 3B | prefill 16K | GPU ring reset | 19,394.49 ms | unavailable | unavailable |
| 3B | prefill 28K | not retried after reset | 56,225.97 ms | unavailable | unavailable |
| 3B | decode KV 128 | 72.95 tok/s | 126.07 tok/s | 1.73x latency | 5.78 ms/token |
| 3B | decode KV 2K | 59.08 tok/s | 40.21 tok/s | 0.68x latency | -7.95 ms/token |
| 3B | decode KV 28K | blocked by Graph Horizon prefill reset | 59.83 tok/s | unavailable | unavailable |
| 8B | prefill 128 | 1,464.95 ms | 295.96 ms | 4.95x | 1,168.99 ms |
| 8B | prefill 512 | GPU ring reset | 508.46 ms | unavailable | unavailable |
| 8B | prefill 2K | GPU ring reset | 2,169.23 ms | unavailable | unavailable |
| 8B | decode KV 128 | 36.61 tok/s | 64.08 tok/s | 1.75x latency | 11.71 ms/token |
| 8B | decode KV 2K | blocked by Graph Horizon prefill reset | 61.97 tok/s | unavailable | unavailable |
| 14B | prefill 128 | GPU ring reset | 435.02 ms | unavailable | unavailable |
| 14B | decode KV 128 | blocked by Graph Horizon prefill reset | 42.21 tok/s | unavailable | unavailable |

Graph Horizon CV is at most 3.1% for the completed rows and at most 0.06% for
prefill from 2K through 8K. llama.cpp prefill CV is at most 2.1% through 8K and
0.20% at 28K. The direct-depth llama decode CVs are 2.58%, 0.56%, and 0.55%.
The surprising 2K decode crossover is retained as measured and needs a 512/8K/
16K crossover sweep; it is not smoothed away.

## Initial attribution and path audit

The profile binary accumulated one warm-up plus three 3B/128 repetitions.
Prefill accounts for 2,466.87 of 2,474.61 GPU ms (99.69%).

| Component | GPU ms | Share | Current AMD path |
|---|---:|---:|---|
| Q | 316.55 | 12.79% | portable Q4_K batch F16 |
| K | 35.79 | 1.45% | portable Q4_K batch F16 |
| V | 54.86 | 2.22% | portable Q6_K batch F16 |
| attention | 11.28 | 0.46% | portable tiled Q8/KV64, wave64 |
| O | 276.09 | 11.16% | portable Q4_K batch F16 |
| gate | 625.45 | 25.27% | portable Q4_K batch F16 |
| up | 502.36 | 20.30% | portable Q4_K batch F16 |
| activation | included in residual/other | below 0.3% | portable |
| down | 612.67 | 24.76% | portable Q6_K batch F16 |
| norm/RoPE/KV/residual/embedding/logits | 31.82 | 1.29% | portable |

The AMD device correctly cannot enter NVIDIA Matrix2/cooperative kernels:
those capability detectors require vendor `0x10de` and NVIDIA extensions. The
generic fallback remains present. For prefill, no already-compatible optimized
matrix path is merely vendor-excluded because AMD exposes no cooperative-matrix
shape. For decode, integer-dot Q4 MMVQ is active and reusable; the 32-wide GQA
path is excluded because pipeline capability code tests the default subgroup
size (64) rather than creating a required-size-32 pipeline.

Decode attribution at 128 accounts for 92.32% of GPU time; its unexplained
7.68% residual must be resolved before claiming 95% decode attribution. The
largest classified shares are Q6 down (23.94%), Q6 logits (24.05%), Q4
gate+up (20.62%), and the remaining projections. Attention is only 2.50% at
KV 128.

## Long-context failure

At 3B/16K, 8B/512, 8B/2K, and 14B/128, the kernel log records the same class of
gfx-ring timeout and reset. The first 3B event included:

```text
ring gfx_0.0.0 timeout
Process bench-public
Starting gfx_0.0.0 ring reset
Ring gfx_0.0.0 reset succeeded
device wedged, but recovered through reset
```

The runs failed with RADV's “context is guilty of a hard recovery.” The device
recovered and returned to idle each time. llama.cpp completed the corresponding
workloads on the same device.

Shape tracing on 14B/128 showed the expected Q4_K/Q6_K projection sequence
across the full layer stack before the reset; no single first-use matrix or
format failed. Source inspection showed that one prefill chunk records
embedding, every layer, and the optional tail into one command buffer, then
waits for one submission. Public Vulkan fixes the chunk capacity at 256 rows.
The observed duration boundary is consistent across models: 3B/128 at about
0.62 seconds and 8B/128 at about 1.46 seconds survive, while a larger complete
graph submission resets.

Control AMD-001 changed only the public Vulkan chunk capacity from 256 to 64.
It made both previously failing controls complete: 14B/128 took 2,430.64 ms and
8B/512 took 6,445.74 ms in one unwarmed repetition. This proves the command-
duration diagnosis; it does not establish a performance win. The durable fix
must select bounded submission work from hardware/resource capabilities and
graph shape, or introduce safe command checkpoints, without product-name
routing.

An unqualified Q4+Q6 prototype made the original 256-row setting complete
14B/128 in 1,291.19 ms and 8B/512 in 2,877.89 ms, but that state was discarded
after Q6 parity failed. These values are diagnostic only. The retained Q4-only
state still needs direct 14B/8B and 3B/16K/28K qualification.

## Retained candidate measurement

The public 3B/128 tuple is directly comparable to the immutable baseline and
llama.cpp screen:

| State | TTFT | Prompt throughput | Decode | CV TTFT | vs baseline | vs llama.cpp |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 624.35 ms | 205.03 tok/s | 72.95 tok/s | 1.16% | 1.00x | 3.94x slower |
| AMD-002 Q4 MMQ, 4-row tile | 258.19 ms | 495.76 tok/s | 72.43 tok/s | 0.14% | 2.42x | 1.63x slower |
| AMD-005 Q4 MMQ, 8-row tile | 223.07 ms | 573.82 tok/s | 73.30 tok/s | 0.03% | 2.80x | 1.41x slower |
| llama.cpp | 158.46 ms | 807.77 tok/s | 126.07 tok/s | 0.92% | 3.94x | 1.00x |

The current profile accounts for 99.12% of prefill GPU time. Across four runs,
Q4 MMQ owns 447.11 ms, portable Q6_K V/down 423.84 ms, Q8 quantization 1.82 ms,
attention 11.25 ms, and all other classified work 24.80 ms. Relative to baseline,
the Q4 projection family improves 3.93x. Decode remains outside the batch route
and is statistically unchanged in the public binary.

Routing requires AMD vendor identity plus the accelerated integer-dot feature,
Q4_K format, a 256-aligned input width, and bounded persistent
scratch. Product names and model IDs are not consulted. NVIDIA continues to
build/use its existing decode MMVQ and qualified Matrix2/cooperative prefill
paths; the AMD batch pipelines are not registered there.

### Scaling and model-size screen

These retained AMD-002 rows used the 4-row MMQ tile; they remain useful evidence
for availability and scaling. AMD-005 qualification currently covers 3B/128,
512, and 2K.

| Model/workload | Baseline | AMD-002 | Speedup | llama.cpp | Remaining ratio |
|---|---:|---:|---:|---:|---:|
| 3B prefill 512 | 2,458.54 ms | 1,054.76 ms | 2.33x | 220.47 ms | 4.78x |
| 3B prefill 2K | 10,264.76 ms | 4,670.49 ms | 2.20x | 1,190.97 ms | 3.92x |
| 3B prefill 8K | 49,418.34 ms | 28,051.50 ms | 1.76x | 6,068.10 ms | 4.62x |
| 3B prefill 16K | reset | 79,592.64 ms | available | 19,394.49 ms | 4.10x |
| 8B prefill 128 | 1,464.95 ms | 674.83 ms | 2.17x | 295.96 ms | 2.28x |
| 8B prefill 512 | reset | 2,913.88 ms | available | 508.46 ms | 5.73x |
| 14B prefill 128 | reset | 1,134.87 ms | available | 435.02 ms | 2.61x |

At 3B, AMD-005 reduces 512-token TTFT from 1,054.76 to 896.51 ms
(17.7%) and 2K from 4,670.49 to 4,029.69 ms (15.9%). The 16K AMD-002 smoke
completed with the original 256-row capacity, confirming that the retained Q4
speed provides watchdog headroom. A repeated long-context qualification remains
too costly until the attention/Q6 bottleneck is reduced.

## Experiment registry

| ID | Target | Hypothesis | Predicted | Measured | Quality | Decision |
|---|---|---|---:|---:|---|---|
| AMD-000 | baseline only | characterize current routes | n/a | 3B/8B/14B gap and reset above | immutable binaries | COMPLETE |
| AMD-001 | Vulkan prefill capacity | 64 rows keep full-graph submissions below the AMD watchdog | failing rows become available; possible submit-overhead regression | 14B/128 2,430.64 ms; 8B/512 6,445.74 ms; both complete | one repetition, no warm-up | REJECTED: diagnostic only |
| AMD-002 | Q4_K batch MMQ | reuse Q8_1 decode arithmetic and packed dot across four prompt rows | about 2.2x end-to-end if Q4 improves 4x | 624.35 → 258.19 ms, 2.42x; Q4 family 3.11x | full Vulkan-hybrid suite; CV 0.14% | RETAINED |
| AMD-003 | Q6_K batch MMQ | signed packed dot removes scalar Q6 FP32 FMAs | about 1.39x incremental if Q6 improves 3x | 258.29 → 233.35 ms, 1.107x; Q6 family 1.33x | four established Q6 prefill parity tests failed by about 0.06--0.07 | REJECTED: quality |
| AMD-004 | Q6 eight-lane rows | halve workgroups/reduction by processing both 128-value segments per lane | at least 1.05x total | 233.35 → 241.01 ms, 3.3% regression | prototype parity passed, parent candidate later failed integration parity | REJECTED |
| AMD-005 | Q4_K MMQ 8-row tile | reuse each packed weight across twice as many prompt rows | about 1.10x total if Q4 improves 20% | 258.19 → 223.07 ms, 1.157x; Q4 family 1.26x | full suite; 128/512/2K CV at most 0.10% | RETAINED |
| AMD-006 | exact Q6_K 64-row tile | halve repeated dequantization with 16 KiB LDS | about 1.07--1.12x total | 223.07 → 237.76 ms, 6.6% regression | focused established Q6 parity passed | REJECTED |
| AMD-007 | Q4_K MMQ 16-row tile | double packed-weight reuse again | about 1.05x total | 223.07 → 251.81 ms, 12.9% regression | focused Q4 parity passed | REJECTED |
| AMD-008 | exact Q6_K packed staging | unpack eight contiguous weights per lane with shared metadata | about 1.20x total | 223.07 → 218.37 ms, 2.15% gain | all established Q6 oracles passed | REJECTED: below threshold |
| AMD-009 | required wave32 for Q6 decode/logits | match RDNA native wave and llama.cpp subgroup choice | about 1.10x decode | 3B +7.2%; 8B below 5% and noisy; 14B +3.2% | Q6 parity passed; cross-model value inconsistent | REJECTED |

## Current ranking and next action

The first candidate ranking is:

| Rank | Opportunity | Measured removable share / value | Maintenance | AMD-family reuse | State |
|---:|---|---|---|---|---|
| 1 | portable Q6_K batch arithmetic | 46.23% of current prefill GPU time; lossy Q8 route is disqualified | medium-high | high | exact-FP16 tile/reuse screen |
| 2 | Q4_K batch MMQ tiling | 48.76% of current prefill GPU time; remaining llama TTFT gap is 64.61 ms | medium | high | 16-row tile only if predicted total gain clears 5% |
| 3 | Q6_K down/logits decode path | 47.99% of classified baseline decode GPU time; 1.92x Amdahl ceiling if removed | medium | high | next decode candidate |
| 4 | bounded prefill command checkpoints | still needed only if 3B/16K or 28K resets after faster kernels | medium | high where watchdogs bound long queues | await direct long-context screen |
| 5 | required-wave32 GQA attention | 2.50% of classified decode time at KV 128; only 1.026x Amdahl ceiling | low-medium | RDNA-focused | defer |

The next action is the retained-candidate cross-model/context sweep, including
3B/16K and 28K. That evidence decides whether command checkpoints remain
necessary and whether Q4/Q6 tile work or decode Q6/logits has the larger global
value.
