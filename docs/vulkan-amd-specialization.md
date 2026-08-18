<!--
This document is the persistent checkpoint for the AMD Vulkan specialization
program: immutable baseline identity, device evidence, competitive measurements,
routing audit, experiment decisions, qualification, and quantitative stop state.
-->

# AMD Vulkan Specialization

## Status

The capability inventory, baseline, attribution, optimization screen, and final
qualification are complete. The retained state combines the AMD integer-dot
Q4_K batched matmul, required-wave32 Q6_K decode/logits, required-wave32 4:1-GQA
decode attention, an exact-order wave64 GQA fallback, and an adaptive 128/64-row
AMD prefill submission bound. The final path recovers 68% of the 3B/28K decode
competitive gap and completes both 3B/28K and 8B/28K without a device reset.

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

### Runtime telemetry and compiler counters

The sustained 3B/8K Graph Horizon row measured 99% GPU use, 2,630 MHz shader
clock, 135--140 W package power, 54--56 C edge, 86--90 C junction, and 52 C
memory. Idle was about 500 MHz, 8--10 W, and 49--51 C.

RADV compiler statistics were captured with `RADV_DEBUG=shader_stats`. The
retained wave32 GQA split uses 64 VGPRs, 108 SGPRs, 5,120 bytes LDS, no scratch,
no spills, and reports 16 subgroups/SIMD; its reduction uses 48 VGPRs, no LDS,
no scratch, and no spills. The exact wave64 split fallback uses 40 VGPRs, no
LDS/scratch/spills, reports 24 subgroups/SIMD, and compiles to 273 instructions;
its reduction uses 16 VGPRs and 65 instructions. Hardware counters for achieved
occupancy, cache hit rate, and DRAM bytes remain **unavailable**; compiler
residency is a limit, not an achieved-occupancy measurement.

## Initial AMD architecture model

- Effective wave model: RADV defaults to wave64 but exposes required subgroup
  sizes 32--64 for compute. The retained AMD decode pipelines explicitly request
  wave32; unrelated pipelines keep their established subgroup behavior.
- Matrix strategy: no usable AMD cooperative-matrix Vulkan path exists on this
  stack. The viable classes are packed integer dot, subgroup/vector, and
  vectorized scalar/FMA.
- Resource model: 64 KiB LDS, 1,024 work-items/CU, and VGPR allocation all bound
  occupancy. The measured wave32 GQA split lands at 64 VGPR and 5 KiB LDS with
  no spill; its 256-thread/eight-wave workgroup is work-item limited to one
  workgroup/CU but exposes the device's 32-wave ceiling across workgroups.
- Sustained 99% GPU utilization with only 3--4% reported memory activity on the
  slow portable prefill path is compatible with instruction/dequantization or
  dependency limits, not proof of a DRAM-bandwidth floor.
- Long rows are stable at 2,630 MHz. Earlier failures are gfx ring timeouts, not
  VRAM exhaustion; bounding long 32+-layer graphs to 64 rows removes the reset.
  The 8B/28K qualification used 80% VRAM and sustained 99% GPU use.

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

Decode attribution at 128 splits 92.41% into timestamped kernels and 7.59% into
the measured command-span residual between those timestamps (dispatch and
global-barrier gaps), covering the complete GPU span. The largest kernel shares
are Q6 down (24.02%), Q6 logits (24.31%), Q4 gate+up (20.39%), and the remaining
projections. Attention is only 2.53% at KV 128, but grows to 79.07% at KV 28K.
AMD-009's Q6 wave32 result was initially rejected in isolation; after GQA
attention became dominant-path qualified, the same small capability plumbing
made it an economical combined path and it was requalified as AMD-029.

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
kernel state later completed 14B/128, 8B/512, 8B/2K, and 3B/16K, but still reset
at 3B/28K. AMD-014's 128-row bound completes those rows and 3B/28K; the later
8B/28K qualification requires AMD-022's 64-row bound.

## Retained candidate measurement

The public 3B/128 tuple is directly comparable to the immutable baseline and
llama.cpp screen:

| State | TTFT | Prompt throughput | Decode | CV TTFT | vs baseline | vs llama.cpp |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 624.35 ms | 205.03 tok/s | 72.95 tok/s | 1.16% | 1.00x | 3.94x slower |
| AMD-002 Q4 MMQ, 4-row tile | 258.19 ms | 495.76 tok/s | 72.43 tok/s | 0.14% | 2.42x | 1.63x slower |
| AMD-005 Q4 MMQ, 8-row tile | 223.07 ms | 573.82 tok/s | 73.30 tok/s | 0.03% | 2.80x | 1.41x slower |
| llama.cpp | 158.46 ms | 807.77 tok/s | 126.07 tok/s | 0.92% | 3.94x | 1.00x |

The AMD-005 checkpoint profile accounts for 99.12% of prefill GPU time. Across four runs,
Q4 MMQ owns 447.11 ms, portable Q6_K V/down 423.84 ms, Q8 quantization 1.82 ms,
attention 11.25 ms, and all other classified work 24.80 ms. Relative to baseline,
the Q4 projection family improves 3.93x. At this pre-decode checkpoint, decode
remains outside the batch route and is statistically unchanged.

Routing requires AMD vendor identity plus the accelerated integer-dot feature,
Q4_K format, a 256-aligned input width, and bounded persistent
scratch. Product names and model IDs are not consulted. NVIDIA continues to
build/use its existing decode MMVQ and qualified Matrix2/cooperative prefill
paths; the AMD batch pipelines are not registered there.

### Quantized dataflow and representation audit

Canonical Q4_K stores 256 weights in 144 bytes: 128 quant bytes, 12 packed
scale/min bytes, and two FP16 block factors. The retained MMQ reads that format
directly, extracts two packed 4-bit vectors per lane/part, and feeds RADV's
accelerated `dotPacked4x8AccSatEXT`. Eight lanes own one output row; WG256 emits
32 rows by eight prompt rows with 8 KiB LDS and one barrier/reduction. Q8_1
activations occupy two 32-bit packed words plus two FP32 factors per eight
values, or 2 bytes/value. Their conversion is only 0.16--0.18% of profiled
prefill GPU time, so activation conversion is not the limiting dataflow.

For the 3B shape, one layer's Q4 Q/K/O/gate/up matrices contain about 47.78 MB
of canonical weights; all 26 layers contain about 1.24 GB. Expanding quant
nibbles to persistent bytes would require at least 272 bytes/block including
unchanged metadata, 88.9% more than canonical and about 1.10 GB additional VRAM
for those matrices alone. It would also replace the retained packed-dot input
with twice the quant traffic, so there is no request-count break-even: load-time
conversion is amortizable, but the extra bytes recur on every weight pass while
the current instruction already consumes packed nibbles directly.

Canonical Q6_K uses 210 bytes/256 weights (128 low-bit bytes, 64 high-bit bytes,
16 signed scales, and one FP16 factor). An int8 expansion is at least 272 bytes,
29.5% more traffic. The Q8-activation packed-dot prototype improved the Q6
family 1.33x but violated four established numeric gates; a persistent repack
does not remove that accumulation/activation-quantization error. The exact
packed-staging alternative preserved quality but improved whole-request time
only 2.15%. Consequently canonical Q4 direct execution is retained, canonical
Q6 remains the portable source, and no AMD execution-native persistent weight
copy is economically or numerically qualified.

Hardware performance counters for actual bytes/s are unavailable. Effective
bandwidth is therefore not invented: byte counts above are derived from shader
loads and block layouts, while elapsed component times and compiler instruction
classes are measured separately.

### Final scaling and model-size qualification

The final rows use the retained 8-row MMQ tile and the 128-row AMD submission
bound. Graphs with at least 32 layers and configured context at least 16K use
64 rows. The 16K, 28K, and 8K rows are one measured repetition; the shorter
rows use one warm-up plus three measured repetitions.

| Model/workload | Baseline | Final | Speedup | llama.cpp | Remaining ratio |
|---|---:|---:|---:|---:|---:|
| 3B prefill 128 | 624.35 ms | 223.07 ms | 2.80x | 158.46 ms | 1.41x |
| 3B prefill 512 | 2,458.54 ms | 908.20 ms | 2.71x | 220.47 ms | 4.12x |
| 3B prefill 2K | 10,264.76 ms | 4,141.48 ms | 2.48x | 1,190.97 ms | 3.48x |
| 3B prefill 8K | 49,418.34 ms | 24,748.24 ms | 2.00x | 6,068.10 ms | 4.08x |
| 3B prefill 16K | reset | 75,379.84 ms | available | 19,394.49 ms | 3.89x |
| 3B prefill 28K | not run after reset | 191,055.22 ms | available | 56,225.97 ms | 3.40x |
| 8B prefill 128 | 1,464.95 ms | 571.13 ms | 2.56x | 295.96 ms | 1.93x |
| 8B prefill 512 | reset | 2,305.96 ms | available | 508.46 ms | 4.54x |
| 8B prefill 2K | reset | 9,821.13 ms | available | 2,169.23 ms | 4.53x |
| 8B prefill 8K | reset | 49,628 ms | available | 11,521.81 ms | 4.31x |
| 8B prefill 16K | reset | 147,040 ms | available | 45,619.86 ms | 3.22x |
| 8B prefill 28K | reset | 352,347 ms | available | 87,818.25 ms | 4.01x |
| 14B prefill 128 | reset | 937.37 ms | available | 435.02 ms | 2.15x |
| 14B prefill 512 | reset | 3,782 ms | available | 1,145.64 ms | 3.30x |
| 14B prefill 2K | reset | 15,800 ms | available | 3,420.10 ms | 4.62x |
| 14B prefill 8K | reset | 75,383 ms | available | 23,830.89 ms | 3.16x |

The shorter repeated final rows have TTFT CV at most 0.26%. The 14B row uses a
2K configured context because its weights plus a 32K F16 KV allocation exceed
this 12 GB device; this does not change its 128-token compute workload. Decode
remains outside the batched Q4 route; its new paths are reported below.

The 256-row final-kernel state completed 16K in 72,927.67 ms but reset at 28K
after a gfx-ring timeout. The retained 128-row policy completes 3B/28K and costs
1.3% at 3B/512, 2.8% at 3B/2K, and 3.4% at 3B/16K; the measured 8B rows varied
slightly faster. A 64-row bound completes 8B/28K where 128 rows resets. Both
are selected by AMD vendor identity plus graph shape, not a product/model
string, and every other Vulkan vendor remains at 256 rows.

Final F16-KV decode uses five complete 32-token repetitions. `Fallback` is the
best pre-AMD-GQA or exact-wave64 row available when the generic workload was
previously blocked by reset. Ratios compare latency, so below 1.0 means Graph
Horizon is faster.

| Model / KV depth | Baseline or fallback tok/s | Final tok/s | Final ms/token | llama.cpp tok/s | llama ms/token | Final latency ratio |
|---|---:|---:|---:|---:|---:|---:|
| 3B / 128 | 77.445 | 78.164 | 12.794 | 126.066 | 7.932 | 1.61x |
| 3B / 2K | 61.801 | 74.907 | 13.350 | 40.215 | 24.866 | 0.54x |
| 3B / 28K | 16.144 | 49.398 | 20.244 | 59.826 | 16.715 | 1.21x |
| 8B / 128 | 36.610 | 39.322 | 25.431 | 64.075 | 15.607 | 1.63x |
| 8B / 2K | 35.966 fallback | 38.259 | 26.138 | 61.972 | 16.136 | 1.62x |
| 8B / 28K | 18.868 fallback | 28.374 | 35.244 | 37.766 | 26.479 | 1.33x |
| 14B / 128 | 23.105 fallback | 23.564 | 42.437 | 42.209 | 23.692 | 1.79x |
| 14B / 512 | unavailable | 23.436 | 42.669 | 34.140 | 29.292 | 1.46x |
| 14B / 2K | unavailable | 23.145 | 43.206 | 41.918 | 23.856 | 1.81x |
| 14B / 8K | unavailable | 21.840 | 45.788 | 36.715 | 27.237 | 1.68x |

All final decode CVs are below 0.45%. The 3B/28K final row recovers 68.0% of
the original competitive throughput gap; 8B/28K is 50.4% faster than the exact
wave64 fallback. The short 8B and 14B ratios remain outside the historical
envelope because attention is a small share there and exact Q6/logits plus Q4
GEMV dominate.

The 14B artifact completes through 8K. A 16,448-token allocation fails during
model initialization because weights, F16 KV, and working storage exceed the
available heap; 16K and 28K are therefore capacity-unavailable rows, not reset
or performance results. No smaller context is substituted for them.

### Retained decode architecture

The production hierarchy remains portable: a device with a native subgroup 32
uses the established split GQA kernel; a compatible AMD device whose default is
64 but which exposes required subgroup-size control builds that kernel as
wave32; a default-wave64 device without that control uses the exact-order
wave64 split; every other shape keeps generic sequential decode attention.
Eligibility is F16 KV, head dimension 128, exact 4:1 GQA, scratch capacity,
workgroup/resource limits, and the reported subgroup contract. No product or
model string participates.

Why the retained GQA path is faster on AMD is quantitative:

- Generic decode assigns one workgroup to each of 32 query heads. At 4:1 GQA it
  rereads the same K and V four times: 16,384 bytes per history position/layer.
  The grouped path assigns one workgroup to each of eight KV heads and reuses
  each load for four queries: 4,096 bytes, a 75% reduction.
- At 28K and 26 layers, generic K/V traffic is about 11.93 GB per generated
  token; grouped traffic is about 2.98 GB. Its 8-part FP32 partial/state traffic
  is only about 133 KiB/layer, or 3.46 MB/token.
- The retained split is WG256, eight wave32 subgroups, eight history splits,
  5,120 bytes LDS, 64 VGPRs, no scratch, and no spills. RADV reports 16
  subgroups/SIMD. The reduce is WG32, 48 VGPRs, no LDS, and no spills.
- On 3B/28K, attention falls from 79.07% of the generic profile to 37.28% of the
  combined final profile. Public decode rises from 16.144 to 49.398 tok/s
  (3.06x); the isolated GQA change accounts for 47.464 tok/s (2.94x).
- Required wave32 Q6_K/logits adds 6.5% at 3B/128 and 6.3% at 3B/2K. It reuses
  the same capability contract and shader sources, so its maintenance cost is
  limited to pipeline creation/routing rather than another numeric kernel.

The exact wave64 fallback provides the same 4:1 K/V reuse without required
subgroup control. It removes LDS entirely, lowers split VGPRs from the generic
64 to 40, has no spills, and compiles to 273 split plus 65 reduce instructions.
It measured 29.762 tok/s at 3B/28K, 84.4% above the generic baseline, but the
available AMD device selects the faster qualified wave32 mapping.

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
| AMD-010 | required wave32 for tiled prefill attention | run the shader's intended two-subgroup/query partition | about 1.08x total at 8K | attention 10,136.55 → 9,980.77 ms; total GPU +0.85% | boundary attention parity exact | REJECTED: below threshold |
| AMD-011 | four-head grouped GQA prefill | load each K/V tile once for all four query heads sharing it | attention about 1.5--2x; total 1.25--1.43x at 8K | attention 10,136.55 → 20,746.94 ms; total GPU 25,228.58 → 37,068.83 ms | boundary attention parity exact | REJECTED: regression |
| AMD-012 | two-head grouped GQA prefill | recover occupancy while retaining two-way K/V reuse | attention at least 1.10x; total at least 1.05x at 8K | attention 10,136.55 → 10,804.51 ms; total GPU 25,228.58 → 26,170.25 ms | boundary attention parity exact | REJECTED: regression |
| AMD-013 | 16-query tiled prefill | halve K/V loads without multi-head accumulator state | attention at least 1.15x; total at least 1.05x at 8K | attention 10,136.55 → 9,076.60 ms; total GPU 25,228.58 → 24,053.72 ms (4.66%) | boundary attention parity exact | REJECTED: below threshold |
| AMD-014 | bounded AMD submissions | 128 rows keep late long-context full-graph commands below the watchdog | complete 28K with less than 5% short-workload cost | 28K reset → 191,055.22 ms; 512/2K/16K cost 1.3%/2.8%/3.4% | full Vulkan and Vulkan-hybrid suites | RETAINED |
| AMD-015 | required-wave32 4:1 GQA decode | reuse each K/V load across four queries with the established eight-split kernel | up to 3x at 28K from 4x lower K/V traffic | 3B 16.144 → 49.398 tok/s; 128/2K 78.279/75.260 tok/s | approved external teacher was not yet run; self-generated sequence proved nondeterministic | SUPERSEDED pending valid oracle |
| AMD-016 | wave32 two-split corrective variant | fewer partials/reduce traffic may improve short decode | small short gain, possible lost long parallelism | no stable advantage over eight splits | invalid self-generated oracle did not decide quality | REJECTED: no advantage |
| AMD-017 | exact-order wave64 grouped GQA | preserve generic wave64 dot/order while removing 4x K/V reads | at least 1.5x at 28K | viable architecture; refined through AMD-020 | bounded attention checks pass | PROVISIONAL |
| AMD-018 | wave64 part/order refinement | match generic wave64 dot and online-partition order exactly | retain most of the traffic win with lower quality risk | no independent public row; converged into AMD-020 | exact arithmetic boundary checks | SUPERSEDED |
| AMD-019 | GQA driver/resource audit | identify occupancy, LDS, spill, and instruction cliffs | explain the mapping delta | wave32 64 VGPR/5 KiB LDS/0 spill; wave64 40 VGPR/0 LDS/0 spill | compiler evidence only | COMPLETE |
| AMD-020 | exact wave64 final + approved teacher | qualify exact grouped GQA with pinned external vectors | at least 1.5x at 28K | 3B 16.144 → 29.762 tok/s (84.4%); all 16 tokens local top-two on 3B/8B/14B | pass at `13f2b28b098623391b1aacfd27995e1c8b7de9a9` | RETAINED fallback |
| AMD-021 | wave64 WG128/eight-WG variant | expose more resident workgroups with half the workgroup size | at least 3% over WG256 if occupancy-limited | 29.762 → 29.801 tok/s (0.13%) | same exact arithmetic | REJECTED: no effect |
| AMD-022 | 64-row deep-graph bound | keep 32+-layer, 16K+ command durations below the RADV watchdog | make 8B/28K available | 128 rows reset; 64 rows completes in 352.1--352.7 s at 80% VRAM | policy boundary tests; repeated completion | RETAINED |
| AMD-023 | approved wave32 GQA requalification | the prior quality rejection used an invalid oracle, so test the fastest mapping directly | recover AMD-015 if approved gate passes | identical 16-token local IDs, top-two pass, zero crossings on 3B/8B/14B | pass | RETAINED |
| AMD-024 | fixed wave32 split/reduce bounds | restore compile-time eight-part loops after a shared dynamic ABI | recover the compiler regression | 3B/2K 70.145 → 70.454 tok/s; push ABI still differed | correctness preserved | INCONCLUSIVE |
| AMD-025 | exact wave32 shader ABI | remove the unused eighth push word | recover prior generated code | no material change: about 70.18 tok/s at 2K | correctness preserved | REJECTED: no effect |
| AMD-026 | exact wave32 pipeline push range | restore the prior 28-byte pipeline range | recover prior generated code | 70.30 tok/s at 2K | correctness preserved | REJECTED: no effect |
| AMD-027 | eight-part scratch-size ablation | test whether doubled fallback scratch perturbs allocation/cache placement | recover the remaining cross-session delta | 70.30 tok/s at 2K, unchanged | correctness preserved | REJECTED: no effect |
| AMD-028 | isolated wave32 GQA bookend | measure the qualified GQA route without Q6 wave32 | about 3x at long KV | 3B/28K 16.144 → 47.464 tok/s (2.94x), CV 0.16% | AMD-023 teacher applies | RETAINED |
| AMD-029 | combined wave32 Q6 + GQA | reuse subgroup control for the earlier low-cost Q6/logits signal | +5--7% short/medium, about +4% long | 3B 73.361 → 78.164 at 128; 70.454 → 74.907 at 2K; historical combined 49.398 at 28K | final external teacher and suites pass | RETAINED |

The grouped-GQA experiments reduce dispatch X from 32 query heads to eight or
16 workgroups per query tile, respectively. The four-head variant requires
33,152 bytes of shared memory and eight FP32 accumulators per invocation; the
two-head variant reduces those to 24,768 bytes and four accumulators. Both are
slower, so this RDNA2/RADV path is sensitive to the added per-workgroup state
and serial head loop despite fewer K/V reads. All grouped-prefill-GQA production
code and routing were removed; raw profiles remain as `amd-011-*` and
`amd-012-*`. The separately qualified decode GQA route is unaffected.

## Final bottleneck ranking

The final combined 3B/28K decode profile accounts for 95.01% of the complete GPU
span. Attention is 37.28%, Q6 down 15.42%, Q6 logits 15.62%, Q4 gate/up 13.07%,
other projections 10.58%, and measured dispatch/barrier residual 4.99%.
Attention's fused split contains equal 2,048-byte K and V reads per history
position across all KV heads; QK, online softmax, and AV share one shader and
cannot be separated by timestamps without changing it. Source/traffic counts
therefore classify those subcomponents as derived, not independently measured.

### Global-stop proof

`Floor` is the fastest qualified mapping actually measured, not a peak-hardware
claim. Remaining removable time includes only in-scope candidates that survived
quality and maintenance constraints.

| Workload | Graph Horizon | llama.cpp | Remaining gap | Dominant final component | AMD practical floor / removable time | Candidate families and terminal reason |
|---|---:|---:|---:|---|---|---|
| 3B prefill 128 | 223.07 ms | 158.46 ms | 64.61 ms | Q4 MMQ + exact Q6 | 218.37 ms tested, 2.15% removable | Q4 16-row regressed 12.9%; exact Q6 staging below 5%; lossy Q6 failed quality |
| 3B prefill 8K | 24.748 s | 6.068 s | 18.680 s | attention 40.18%, Q4 31.87%, Q6 26.71% | 24.054 s profiled, 4.66% removable | grouped 4/2-head attention regressed; Q16 attention below 5%; Q4/Q6 tile space exhausted |
| 3B prefill 28K | 191.055 s | 56.226 s | 134.829 s | attention 70.11%, Q6 down 11.15% | no qualified AMD prefill candidate above 5% | same attention mappings and exact/lossy Q6 alternatives; persistent repack lacks an end-to-end bound |
| 3B decode 128 | 78.164 tok/s | 126.066 tok/s | 4.86 ms/token | Q6 down/logits; attention small | retained Q6 wave32; no exact numeric route above gate | packed/lossy Q6 changes numerical contract; further GQA tuning cannot recover short gap |
| 3B decode 2K | 74.907 tok/s | 40.215 tok/s | Graph Horizon 11.52 ms/token faster | Q6/logits | performance contract exceeded | no competitive gap to recover |
| 3B decode 28K | 49.398 tok/s | 59.826 tok/s | 3.53 ms/token | GQA 37.28%, Q6 down/logits 31.04% | wave32 eight-split is 3.06x generic; about 3.5 ms/token remains | two-split no gain; WG128 wave64 +0.13%; further precision/reorder changes exceed quality risk |
| 8B prefill 28K | 352.347 s | 87.818 s | 264.529 s | same portable long-prefill attention/Q6 families | 64-row policy is capacity floor, not speed win | prefill candidates above all failed regression, quality, or 5% economics |
| 8B decode 28K | 28.374 tok/s | 37.766 tok/s | 8.77 ms/token | long GQA plus larger weight pass | retained wave32 GQA is 50.4% over wave64 fallback | same split/WG alternatives exhausted; remaining weight work is shared generic arithmetic |
| 14B decode 128 | 23.564 tok/s | 42.209 tok/s | 18.75 ms/token | Q6/logits and weight GEMV | wave32 Q6 gives only low-single-digit local value | another vendor kernel cannot clear 10% without changing numeric representation |

The prefill gap remains large, but every measured architectural class with a
credible AMD-family implementation is bounded: packed-dot Q4 is retained;
lossy packed-dot Q6 violates the quality contract; exact Q6 and Q16 attention
are below the 5% low-complexity threshold; larger Q4 tiles and grouped prefill
attention regress. Decode long-context now meets the historical contract on 3B
and 8B. Short larger-model decode is weight/Q6 dominated, where the available
wave-size improvement is retained and further gains require a new numerical or
persistent weight representation outside the qualified hardware-only boundary.
This satisfies global-stop conditions 2, 4, and 5: no remaining AMD-family
candidate clears the economic gate, the remaining large gains require numeric/
representation changes, and the rest is device-specific micro-tuning.

## Final verification and state

- Pure Vulkan: 156 passed, 0 failed, 5 ignored; family integration 5 passed and
  1 ignored; semantic 12 passed and 1 ignored.
- Vulkan-hybrid: 229 passed, 0 failed, 4 ignored; family integration 6 passed
  and 1 ignored; semantic 12 passed and 1 ignored.
- Warning-denied Clippy passes for every target in both profiles; formatting and
  the productive-line/source-structure audit pass.
- Focused attention parity covers base/row boundaries and is exact against
  sequential Vulkan decode; Q4 MMQ tail tests cover five rows and 70 outputs.
- Pinned llama.cpp `13f2b28b098623391b1aacfd27995e1c8b7de9a9`
  teacher qualification passes 3B/8B/14B: exact prompt IDs, 16/16 local top-two,
  zero crossings, matching local greedy IDs, lifecycle and cancellation pass.
- Final production contains AMD-005, AMD-014/022, AMD-020, AMD-023/028, and
  AMD-029 only. Rejected prefill GQA/Q16, dynamic-split, two-split, and WG128
  variants are absent. The temporary decode probe was removed.
- Nothing was pushed. Raw benchmark/profile evidence remains ignored under
  `target/amd-baseline/`.

## Production refactor qualification

### Baseline and scope

- Branch: `perf/amd-hardware-specialization`.
- Pre-refactor HEAD: `5fbadc1`; code-cleanup checkpoint: `1be0d41`.
- Local `main` and merge-base: `2d7d7f7`.
- The requested `perf/systematic-llamacpp-gap` branch was not the checked-out
  repository state. Its production cleanup is already represented by the
  merge-base and `ABLATION_REPORT.md`; this audit therefore covers the complete
  additional AMD delta actually present in the worktree.
- Scope before cleanup: 25 files, 1,351 insertions, 71 deletions against
  `main`; 18 Rust files and four shaders contained production or embedded-test
  changes. Four Rust files contain embedded-test changes; there were no
  dedicated test files in this delta.

### Removed and simplified

- Dead functions, fields, types, shader variants, configuration fields, and
  temporary buffers: none remained in the AMD production delta.
- Obsolete branches and diagnostic infrastructure: none remained. Rejected
  grouped-prefill, Q16, alternate split, and alternate workgroup candidates
  exist only in this historical report and ignored evidence.
- Duplicated routing: the three GQA variants now share one resource-eligibility
  predicate while preserving their distinct subgroup, vendor, and wave-control
  predicates.
- Redundant expressions: AMD Q4 batch eligibility composes the established
  single-row MMVQ predicate rather than repeating its format, alignment, DP4A,
  and scratch checks.
- Pipeline responsibilities: compute-pipeline compilation moved from dispatch
  recording into `pipeline/compiler.rs`; unconditional kernel inventory moved
  beside the kernel ABI. Registry assembly, recording, compilation, and ABI
  ownership are now separate, direct responsibilities, each below 200
  productive lines.
- Documentation: the completed AMD task no longer appears as an active
  investigation in the model-neutral process document.

The refactor commit changes six source files by 217 insertions and 266
deletions, a net reduction of 49 source lines. Against `main`, production source
changed from 22 files / 734 insertions / 57 deletions before cleanup to 23 files
/ 851 insertions / 223 deletions after cleanup. The extra file is the required
compiler responsibility split; the net production delta falls from 677 to 628
lines. No dependency, public API, route, shader ABI, resource lifetime, or error
message changed.

### Retained feature map

| Feature | Production path | Eligibility and fallback | Protection / reason |
|---|---|---|---|
| AMD Q4 MMQ prefill | dispatch, MMVQ host, scratch, registry, Q4 MMQ shader | AMD + integer dot + Q4_K + aligned shape + bounded scratch; portable Q4 batch fallback | CPU oracle/tail test; dominant retained AMD short-prefill gain |
| Bounded AMD prefill | backend selection and Vulkan row policy | AMD plus graph depth/context selects 128 or 64 rows; other devices retain 256 | policy boundary tests; prevents long-graph device loss |
| Required-wave32 Q6/logits | bootstrap feature query, device state, compiler, registry | AMD + supported required subgroup 32; default-subgroup pipeline otherwise | capability tests and full suites; retained short/medium decode gain |
| Required-wave32 4:1 GQA | attention route, shared scratch, split/reduce kernels | F16 KV + head 128 + exact 4:1 + resources + AMD wave32 control; wave64/generic fallback | exact attention oracles; retained long-decode gain |
| Exact wave64 4:1 GQA | attention route, registry, wave64 split/reduce shaders | default subgroup 64 + exact GQA/resource contract; generic fallback otherwise | exact arithmetic tests; portable fallback without subgroup control |
| Inherited Matrix2 Q4/14B paths | unchanged merge-base implementation | existing capability/shape/quantization predicates and fallbacks | previously qualified and unaffected by this AMD-only delta |

### Performance neutrality

Matched binaries from `5fbadc1` and the cleaned source used identical probes,
artifacts, placement, F16 KV, and session state. Short rows used three measured
repetitions; long prefill used the established single exact-length row and long
decode used five samples. CV is a fractional ratio.

| Workload / metric | Before (CV) | After (CV) | Delta |
|---|---:|---:|---:|
| 8B/128 TTFT | 569.85 ms (.0012) | 570.12 ms (.0009) | +0.047% |
| 14B/128 TTFT | 934.69 ms (.0011) | 935.29 ms (.0018) | +0.064% |
| 8B/128 decode | 40.4654 tok/s (.001387) | 40.5373 tok/s (.002626) | +0.178% |
| 8B/28K TTFT | 352,201 ms (single row) | 352,171 ms (single row) | -0.0085% |
| 8B/28K decode | 29.1174 tok/s (.001704) | 29.0804 tok/s (.001379) | -0.127% |
| 3B/28K TTFT | 192,414 ms (single row) | 192,584 ms (single row) | +0.088% |
| 3B/28K decode | 51.6296 tok/s (.001349) | 51.6966 tok/s (.002554) | +0.130% |

Every measured delta is within 0.2%, well inside the 1% neutrality target.

### Quality and validation

- The mission's six-artifact, 128-step gate belongs to the already-completed
  NVIDIA cleanup recorded in `ABLATION_REPORT.md`; its saved 128-token teachers
  are not part of the current repository contract or retained evidence. The
  checked-in AMD protocol instead pins a llama.cpp external teacher for 16
  tokens on the three Instruct artifacts.
- The final 8B and 14B external 16-token rows passed. The first final 3B row
  failed the wrapper's local parity assertion; a diagnostic rerun passed exact
  prompt IDs and all 16 positions. Under the documented no-retry rule, this
  session's 3B terminal result remains **FAIL**, with the later pass recorded
  only as diagnostic evidence.
- A diagnostic extension of the pinned external teacher to 128 tokens missed
  top two at step 126 on both untouched `5fbadc1` and the cleaned code. This is
  paired evidence that the refactor preserved the numerical behavior, but it
  cannot be represented as a passing 128-step qualification.
- Focused AMD Q4 MMQ and attention/numeric oracles pass. No tolerance or oracle
  was changed.
- Root CPU: 166 passed. Engine CPU: 159 passed / 2 ignored; integration 5 passed
  / 1 ignored; semantic 12 passed / 1 ignored.
- Pure Vulkan: 156 passed / 5 ignored; integration 5 passed / 1 ignored;
  semantic 12 passed / 1 ignored. Vulkan-hybrid: 229 passed / 4 ignored;
  integration 6 passed / 1 ignored; semantic 12 passed / 1 ignored.
- Formatting, diff whitespace, warning-denied Vulkan and Vulkan-hybrid Clippy,
  shader/build checks, and documentation/source contract tests pass. Metal
  remains external verification on this Linux host.

### Remaining differences from `main`

| File/component | Required reason |
|---|---|
| Selection and Vulkan row policy | Bounded 128/64-row AMD long-prefill execution with unchanged non-AMD capacity |
| Device bootstrap/state | Query and own AMD vendor, integer-dot, and required-subgroup capabilities |
| Pipeline caps, compiler, ABI, and registry | Build only legal wave32, wave64 GQA, and AMD MMQ pipelines with exact fallback behavior |
| Batched dispatch, MMVQ host/scratch, Q4 shader | Retained canonical-Q4 AMD integer-dot prefill dataflow |
| Attention host/scratch and GQA shaders | Retained wave32 GQA route, exact wave64 fallback, and generic fallback |
| Profile categories and route trace names | Correct attribution of every reachable retained kernel |
| Embedded capability/numeric tests | Protect route boundaries, resource bounds, arithmetic, and tails |
| Optimization state and AMD report | Durable support limits, measured decisions, and rejected-candidate evidence |

### Rejected cleanup candidates

| Candidate | Reason retained |
|---|---|
| Remove the exact wave64 GQA pair | It is the qualified fallback when required-wave32 control is absent |
| Collapse registry, compiler, and recording | It mixes resource ownership, pipeline construction, and command recording and violates the orchestration limit |
| Remove AMD MMQ scratch growth | Eight-row Q4 MMQ requires the retained activation capacity; smaller devices keep the old allocation |
| Merge all GQA predicates | Resource eligibility is shared, but subgroup/vendor controls select materially different pipelines |
| Remove profiling classifications | The supported profiler must classify every reachable kernel without changing disabled-path execution |

### Assessment

Every remaining production-code difference from `main` maps to a retained AMD
optimization, capability/shape/format eligibility, an executable fallback,
resource ownership, or its correctness/attribution coverage. The source and
performance refactor is complete and simpler. Full mission qualification is
not claimed because the requested six-artifact 128-step evidence is not the
current branch's reproducible gate and the no-retry 3B external row failed in
this session; neither issue was introduced by the refactor.
