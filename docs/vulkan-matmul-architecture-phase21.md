<!--
This document owns the Phase 21 Ampere quantized-matmul investigation: the
fresh Phase 20 baseline, physical and competitive model, rejected direct-load
architecture, and the evidence-based stop decision.
-->

# Vulkan Quantized Matmul Architecture — Phase 21

## Outcome

Phase 21 retains **no production change**. The fresh Phase 20 baseline reaches
35,381.53 ms TTFT at 28K. Its seven recurrent matmuls consume 18,398.262 ms,
or 51.67% of profiled GPU prefill and 52.00% of public TTFT. Down projection is
the largest individual matmul at 5,667.921 ms, 15.92% of GPU prefill and 16.02%
of TTFT.

The one new architecture tested was a capability-, shape-, workload-, and
quant-format-gated down-projection path that fed canonical Q4_K/Q6_K blocks to
Matrix2 tensor-load decode callbacks. It preserved M64/N32/K128 and FP32
accumulation while removing the explicit decoded-weight cooperative load. It
did not increase weight reuse: each decoded tile still served 64 prompt rows.

Q4 showed a favorable 128-token dispatch screen, but its decisive 28K half of
down projection regressed from 2,654.326 to 3,478.672 ms (**+31.05%**). Q6 was
pathological at 128: 82.891 versus 2.611 ms/dispatch, with 255 registers/thread
and one resident workgroup. An M128 Q4 refinement was also slower than direct
M64 at 128. The 28K screened GPU request regressed about 2.24%, so the candidate
failed the 3% continuation signal and was removed before broad qualification.

This is **STOP condition C**: no remaining exact matmul candidate has a measured
or resource-credible path to the 5% whole-request keep gate. The earlier,
materially different compressed M128 architecture already reduced traversal
count and regressed 28K Q4 matmuls by 4.43%. Full FP16 down remains below the
economic gate with +1.371 GiB, and lossy native formats remain outside the
numeric contract. All temporary shaders, routing, environment controls, and
resource diagnostics are removed; production source is identical to Phase 20.

The current model is still outside the performance-maturity envelope. The next
high-value step is second-model support, which will reveal whether another
important shape creates a reusable matmul specialization opportunity. It is
not another local sweep on this model.

## Reproducible baseline

| Item | Value |
|---|---|
| Date | 2026-08-16 |
| Phase 21 branch | `perf/vulkan-matmul-architecture-phase21` |
| Clean production baseline | `perf/vulkan-specialization-architecture-phase20` at `e5d78ea` |
| Phase 20 implementation chain | `1c10022`, `65434d2`; reports `09b4b6a`, `e5d78ea` |
| GPU | NVIDIA GeForce RTX 3060 12 GiB, Ampere, PCI device `0x2487` |
| Driver / Vulkan device API | 595.84 / 1.4.329 |
| Model | `Ministral-3-3B-Instruct-2512-Q4_K_M.gguf` |
| Shape | 26 layers, hidden 3072, FFN 9216, Q 4096, K/V 1024, O 4096 -> 3072 |
| Runtime | Vulkan, 32,768 capacity, F16 KV, greedy, 32 decoded tokens |
| MLP policy | retained Phase 12 gate/up native path enabled with `GRAPH_HORIZON_PREFILL_PREDECODE_MLP=1` |
| Prompt | chat-templated `" a"` repetition adjusted to the requested token count |

The initial worktree was clean. Baseline binaries were captured before the
first production edit, and the work proceeded on the dedicated Phase 21 branch.
Nothing was pushed.

Public repetitions were 15/10/5/3/3/3 from 128 through 28K, with 2/2/2/1/1/1
warm-ups. The profiler used a separate feature-gated binary and reports the
stable half of a warm-up-plus-measured aggregate. Public TTFT is CPU wall time
from the public request boundary; profiler GPU time is a separate diagnostic
run and must not be subtracted from it as if the clocks and instrumentation
were identical.

### Public baseline and competitive reference

| Context | TTFT mean ms | Median ms | SD ms | CV | Prompt token/s | Decode token/s | llama.cpp prefill ms | GH/llama |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 96.88 | 96.84 | 0.51 | 0.52% | 1,321.25 | 46.80 | 42.983 | 2.254x |
| 512 | 364.55 | 364.62 | 0.92 | 0.25% | 1,404.47 | 46.44 | 131.722 | 2.768x |
| 2,048 | 1,508.41 | 1,509.80 | 3.75 | 0.25% | 1,357.68 | 46.34 | 541.391 | 2.786x |
| 8,192 | 7,033.01 | 7,043.09 | 23.27 | 0.33% | 1,164.83 | 40.68 | 2,611.573 | 2.693x |
| 16,384 | 16,801.53 | 16,796.32 | 17.38 | 0.10% | 975.14 | 35.06 | 6,435.127 | 2.611x |
| 28,000 | **35,381.53** | 35,385.55 | 18.50 | 0.05% | 791.37 | 29.39 | 14,038.202 | **2.520x** |

The llama.cpp values are the acquired same-device Phase 18 reference. They are
diagnostic rather than a parity requirement and have the documented host timing
boundary difference. Graph Horizon's long-context ratio improves from 2.693x
at 8K to 2.520x at 28K, so the remaining gap grows in seconds but does not show
ratio deterioration across the long-context range.

Telemetry sampled 720 times across the baseline sequence. Application memory
ranged from 373 to 8,646 MiB, graphics clock from 210 to 2,002 MHz, memory clock
from 405 to 7,501 MHz, power from 18.85 to 169.90 W, and temperature from 52 to
71 C. The long runs ended near 1,950–1,980 MHz graphics, 7,501 MHz memory,
159–170 W, and 70 C; there is no observed thermal or clock collapse explaining
the scaling curve.

### GPU-prefill scaling

All values are milliseconds/request from fresh Vulkan timestamps. “Other” is
the profiled residual after attention, seven matmuls, and activation.

| Context | GPU total | Attention | Q | K | V | O | Gate | Up | Activation | Down | Other |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 246.026 | 2.736 | 27.060 | 8.030 | 8.194 | 28.762 | 40.047 | 41.247 | 6.588 | 64.414 | 18.948 |
| 512 | 525.865 | 8.813 | 61.770 | 15.957 | 18.001 | 65.156 | 84.460 | 80.052 | 8.121 | 140.306 | 43.229 |
| 2,048 | 1,591.273 | 94.357 | 184.939 | 52.551 | 53.935 | 196.214 | 247.061 | 244.014 | 16.189 | 427.945 | 74.068 |
| 8,192 | 7,172.865 | 1,410.775 | 716.327 | 199.395 | 212.683 | 744.334 | 956.871 | 951.747 | 66.894 | 1,678.791 | 235.048 |
| 16,384 | 16,964.680 | 5,573.847 | 1,422.200 | 389.856 | 413.413 | 1,470.830 | 1,890.543 | 1,903.194 | 129.876 | 3,332.887 | 438.034 |
| 28,000 | **35,608.897** | **16,261.041** | **2,402.709** | **662.567** | **706.254** | **2,498.686** | **3,234.441** | **3,225.684** | **222.067** | **5,667.921** | **727.527** |

Every main matmul becomes approximately linear in prompt rows once large-M
work dominates. There is no material fixed per-request matmul cost at 28K.
Attention is again the largest single category, but the seven matmuls together
are now larger: 18,398.262 ms versus 16,261.041 ms.

| 28K macro category | GPU ms | GPU prefill | Public TTFT |
|---|---:|---:|---:|
| Attention | 16,261.041 | 45.67% | 45.96% |
| Seven matmuls | 18,398.262 | 51.67% | 52.00% |
| Other, including activation | 949.594 | 2.67% | 2.68% |
| **GPU prefill / CPU wall** | **35,608.897** | **100%** | **35,381.53 ms** |

## Complete 28K matmul attribution

There are 110 row batches per layer: 109 full 256-row batches and one 96-row
tail. Each operation therefore has 2,860 dispatches/request across 26 layers.
The seven rows below account for 100% of recurrent matmul timestamp time, well
above the required 98% attribution threshold. Logits is only about 3.7 ms and
is included in “other.”

| Operation | GPU ms | GPU prefill | Public TTFT | Dispatches | ms/dispatch |
|---|---:|---:|---:|---:|---:|
| Q | 2,402.709 | 6.75% | 6.79% | 2,860 | 0.840108 |
| K | 662.567 | 1.86% | 1.87% | 2,860 | 0.231667 |
| V | 706.254 | 1.98% | 2.00% | 2,860 | 0.246942 |
| O | 2,498.686 | 7.02% | 7.06% | 2,860 | 0.873667 |
| Gate | 3,234.441 | 9.08% | 9.14% | 2,860 | 1.130923 |
| Up | 3,225.684 | 9.06% | 9.12% | 2,860 | 1.127862 |
| Down | **5,667.921** | **15.92%** | **16.02%** | 2,860 | **1.981791** |
| **Total** | **18,398.262** | **51.67%** | **52.00%** | **20,020** | — |

The mixed-format V split is 327.539 ms Q4 and 378.716 ms Q6. The down split is
2,654.326 ms Q4 and 3,013.596 ms Q6. Format times cover 13 layers each.

## Architectural inventory

All prefill matmuls use FP16 inputs and outputs, FP32 accumulation,
M64/N32/K128 Matrix2, a 256-thread workgroup with eight 32-lane subgroups, and
one M64 x N32 output tile (2,048 scalar outputs) per workgroup. M utilization is
99.886% at 28K; N and K tile dimensions are exact. Compressed paths decode a
weight tile once for 64 prompt rows. Native gate/up directly load FP16 weights
but retain the same 64-row ownership.

| Operation | Logical MxNxK | Canonical / execution | Rows/WG / reuse | Registers | Driver shared | Resident WG / occupancy | Useful TFLOP/s |
|---|---|---|---:|---:|---:|---:|---:|
| Q | 28000x4096x3072 | Q4 / Q4 compressed | 64 / 64 | 98 | 33,792 B | 2 / 33.3% | 7.63 |
| K | 28000x1024x3072 | Q4 / Q4 compressed | 64 / 64 | 98 | 33,792 B | 2 / 33.3% | 6.91 |
| V | 28000x1024x3072 | 13 Q4 + 13 Q6 / compressed | 64 / 64 | 98 / 108 | 33,792 B | 2 / 33.3% | 6.48 |
| O | 28000x3072x4096 | Q4 / Q4 compressed | 64 / 64 | 98 | 33,792 B | 2 / 33.3% | 7.34 |
| Gate | 28000x9216x3072 | Q4 / exact FP16 native | 64 / 64 | 99 | 34,304 B | 2 / 33.3% | 12.75 |
| Up | 28000x9216x3072 | Q4 / exact FP16 native | 64 / 64 | 99 | 34,304 B | 2 / 33.3% | 12.78 |
| Down | 28000x3072x9216 | 13 Q4 + 13 Q6 / compressed | 64 / 64 | 98 / 108 | 33,792 B | 2 / 33.3% | **7.27** |

No reliable physical-DRAM, cache-hit, warp-stall, or spill counter is available.
Executable properties report zero stack for baseline Q4/Q6. The driver “local
memory” value is a 64 GiB sentinel and cannot support a spill claim.

## Down physical and compute model

### Canonical layout and logical traffic

The down matrix in each layer is stored row-major by output channel and K block.
A Q4_K block covers 256 values in 144 bytes: 128 payload bytes and 16 bytes of
scale/min metadata. A Q6_K block covers 256 values in 210 bytes: 192 payload
bytes and 18 metadata bytes. The current K128/N32 traversal consumes halves of
canonical blocks, reconstructs FP16 weights into shared memory, and then issues
the cooperative Matrix2 load. There is no separate runtime repack or hidden
persistent execution view for down.

| Traffic | Theoretical unique/request | Shader-visible at M64 | Amplification | Effective logical rate |
|---|---:|---:|---:|---:|
| Quant payload | 460,062,720 B | 201,507,471,360 B | 438x | 35.55 GB/s |
| Metadata | 48,881,664 B | 21,410,168,832 B | 438x | 3.78 GB/s |
| Compressed total | **508,944,384 B** | **222,917,640,192 B** | **438x** | **39.33 GB/s** |
| Activation Matrix2 input | 13.418 GB unique read | 1,289.648 GB logical loads | 96.11x by N tiles | 227.53 GB/s |
| Output | 4.473 GB | 4.473 GB | 1x | 0.79 GB/s |

These are source-level logical bytes, not physical DRAM bytes. Against the
realistic 360 GB/s RTX 3060 device ceiling, compressed logical weight rate is
10.9% and total logical activation rate is 63.2%. Cache reuse makes neither a
physical-bandwidth measurement. The unique compressed-weight floor is only
about 1.4 ms; even treating all 222.918 logical GB as physical gives a 0.619 s
floor, still far below the measured 5.668 s.

The Matrix2 arithmetic is 41.222 useful TFLOP/request. At 5.668 s this is 7.273
TFLOP/s, 11.9% of the acquired 61.1 TFLOP/s dense Matrix2 ceiling. There are
78,713,856 logical Matrix2 multiply-add operations and 1,093,248 output tiles.
Useful M utilization is 99.886%, so tail padding cannot explain the gap.

### Bounded decomposition

The shader overlaps loads, unpack, staging, and Matrix2 issue. Without physical
counters, assigning their joint timestamp to invented isolated buckets would
be false precision. The defensible decomposition is:

| Down component | Evidence / bound |
|---|---|
| Weight global load | 201.507 GB logical payload; physical traffic and cache service time unresolved; <=0.560 s if all logical bytes were served at 360 GB/s |
| Metadata | 21.410 GB logical; the Q4-only free-metadata result scaled from Phase 12 is an approximately 0.145 s upper bound; Q6 is unmeasured |
| Q4/Q6 unpack, scale, conversion | Present in 106/115-load baseline SPIR-V; inseparable from tensor-load issue without counters |
| Activation input | 1,289.648 GB logical Matrix2 loads; only 13.418 GB is the unique down input, already resident from activation |
| Matrix2 staging | 16,384 B shader-declared decoded-weight/activation storage; 33,792 B driver shared/WG |
| Matrix2 compute | 0.675 s optimistic useful-FLOP floor at 61.1 TFLOP/s |
| Reduction / epilogue | No cross-workgroup reduction; FP32 accumulator is narrowed to the existing FP16 output; isolated time unresolved |
| Output write | 4.473 GB/request; <=0.013 s at 360 GB/s |
| Shared traffic / barriers | Decoded weight tile is cooperatively loaded from shared; three static barriers |
| Other | Address calculation, tensor-load issue, synchronization, cache behavior, and occupancy share the unresolved remainder |

The correct classification is **unpack/dequant, tensor-load/shared staging,
instruction issue, synchronization, and occupancy bound**. It is neither a
generic “memory-bound” kernel nor a dense-Matrix2-compute-bound kernel.

### Resource cliff

Baseline Q4/Q6 use 98/108 registers/thread, 33,792 B driver shared/WG, zero
reported stack, two resident workgroups, 16 resident warps, and 33.3% modeled
thread occupancy. Three workgroups require at most 85.33 registers/thread; Q4
must lose 12.67 registers and Q6 22.67. Three current shared allocations use
101,376 of 102,400 B, leaving 1,024 B. Four workgroups require at most 64
registers/thread and 25,600 B shared/WG. The kernel is close to the shared
three-WG cliff but materially over its register threshold; four-WG residency is
not a plausible local cleanup.

## Reuse, fusion, and boundary ceilings

| Group | Current dequantized-weight reuse | Maximum screened ownership | Result / bound |
|---|---:|---:|---|
| Q/K/V/O/down compressed | 64 prompt rows | 128 in the prior FP32 M128 design; llama owns 256 with FP16 accumulation | Prior broad Q4 M128: +4.43% local at 28K; current resource model predicts one WG |
| Gate/up native | 64 prompt rows | 64 retained | No runtime dequant; Phase 12 path is frozen |
| llama.cpp compressed | 256 prompt rows | 256 measured path | Different M256/N128/K64 and FP16 accumulator contract |

The activation product write plus down input read materializes 26.837 GB/request.
At 360 GB/s its impossible-delete ceiling is 74.5 ms, only 0.21% of TTFT. QKV
input sharing removes at most about 8.95 GB plus negligible dispatch work,
roughly 25–30 ms or below 0.1%. Their weight streams remain distinct. Neither
activation/down locality nor QKV fusion can reach the 5% gate, so no mega-kernel
was attempted.

Metadata-only caching is likewise below the gate. Q4 metadata removal contributes
at most about 0.145 s for down, under 0.5% of the request, and leaves payload
unpack, conversion, staging, and Matrix2 issue intact. A lightweight index view
has no credible seconds-scale address/repack cost to remove because canonical
blocks are already ordered by output row and K block.

## Competitive answer

At 28K llama.cpp down takes 1,499.574 ms versus Graph Horizon's 5,667.921 ms.
The comparison explains an architecture opportunity, not a directly portable
implementation:

| Property | Graph Horizon | llama.cpp measured Vulkan path |
|---|---|---|
| Canonical weights | Same Q4_K/Q6_K unique blocks | Same Q4_K/Q6_K unique blocks |
| GPU time | 5.668 s down | 1.500 s down |
| Tile / ownership | M64 x N32 x K128 | token 256 x output 128 x K64 |
| Workgroup | 256 threads, 8 subgroups | 256 threads |
| Accumulator | FP32 | FP16 |
| Compressed traversals | 438 row tiles | 110 row tiles |
| Decode | explicit unpack and shared FP16 staging | Matrix2 tensor-load callback |
| Useful output ownership | 2,048 scalar outputs/WG | 32,768 scalar outputs/WG |
| Q4 resources | 98 registers, 33,792 B, 2 WG | 247 registers, 63,616 B, 1 WG |
| Q6 resources | 108 registers, 33,792 B, 2 WG | 255 registers, 55,296 B, 1 WG |

**Answer:** llama.cpp does not read fewer unique canonical weights. It traverses
and dequantizes each compressed tile for four times as many prompt rows, owns
16 times as many output scalars per workgroup, and feeds Matrix2 through a
direct decode callback. Those mechanisms improve reuse and work ownership.
They are inseparable from its FP16 accumulator mapping; Phase 18 already showed
that selecting llama.cpp's existing FP32 alternative is catastrophically slow.
Phase 21 therefore copied only the direct canonical-feed concept while retaining
FP32 accumulation, then measured it rather than assuming transferability.

For reference, acquired llama.cpp per-operation times are Q 0.757, K 0.213,
V 0.238, O 0.667, gate 1.432, up 1.432, activation 0.236, and down 1.500 s.
Gate/up are not an apples-to-apples representation comparison because Graph
Horizon intentionally pays 2.742 GiB persistent exact-FP16 residency there.

## Candidate ranking before implementation

| Rank | Target / architecture | Current | Practical floor | Removable | Predicted 28K | Portability | Risk |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | Down direct canonical Matrix2 decode callback | 5.668 s | 3.314 s exact-native empirical reference | 2.354 s | **6.65%** | NVIDIA Matrix2; Q4/Q6 and compatible N/K | High compiler/resource sensitivity |
| 2 | Down compressed M128 reuse | 5.668 s | no lower than about 4.434 s from optimistic halving of remaining Q4 decode/staging | <=1.234 s | <=3.49% | Matrix2 and tile compatible | High occupancy risk; below gate |
| 3 | QKV fusion / shared input | 3.772 s affected | about 3.742 s | <=0.030 s | <0.1% | Broad Vulkan | Mega-kernel risk; reject analytically |
| 4 | Activation-to-down locality | 5.890 s pipeline segment | about 5.816 s | <=0.075 s | <=0.21% | Broad Vulkan | Cross-operation coupling; reject analytically |
| 5 | Q4 down metadata view | 2.654 s affected | about 2.509 s | <=0.145 s | <=0.41% | Quant-specific | Leaves dominant work; reject analytically |

The 3.314 s rank-one floor is the acquired exact FP16-native down component,
not a claim that a canonical compressed callback can attain it. It is a strong
screening reference because it keeps FP32 accumulation and identical output
values while removing all runtime unpack/staging. Rank one was the only new,
isolated exact candidate clearing the 5% pre-code gate.

## Prototype: direct canonical Matrix2 feed

### Architecture and routing

The temporary category-K Q4/Q6 shaders kept one operation: quantized matmul.
The host route was restricted to prefill down projection, successful NVIDIA
Matrix2 tensor-addressing pipeline construction, the canonical Q4_K or Q6_K
representation, exact `K=9216` and `N=3072`, and at least one complete 64-row
tile. It used no device marketing name or model name and fell back to the
existing compressed kernel for unsupported shape, format, capability, workload,
or pipeline construction.

The candidate retained M64/N32/K128, WG256, eight subgroups, 64-row dequant
reuse, canonical bytes, FP16 input/output, and FP32 accumulation. A tensor-load
callback decoded canonical blocks and cached scale state in shared memory. It
removed the explicit shared cooperative weight-fragment load, but did not
reduce compressed traversals, weight bytes, metadata bytes, or the three static
barriers. No persistent allocation or startup step was added.

### Compiler and ISA audit

| Variant | SPIR-V words | Loads | Stores | Access chains | Int mul/div | Shifts | Barriers | Tensor loads | Registers | Driver shared | Residency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Q4 baseline | 2,721 | 106 | 70 | 18 | 10 / 3 | 12 | 3 | 1 | 98 | 33,792 B | 2 WG / 33.3% |
| Q4 direct | 2,903 | 106 | 65 | 28 | 10 / 3 | 12 | 3 | 2 | 128 | 39,424 B | 2 WG / 33.3% |
| Q6 baseline | 2,795 | 115 | 70 | 21 | 18 / 4 | 16 | 3 | 1 | 108 | 33,792 B | 2 WG / 33.3% |
| Q6 direct | 3,157 | 116 | 66 | 28 | 17 / 3 | 14 | 3 | 2 | 255 | 47,616 B | 1 WG / 16.7% |

Each form still has one Matrix2 multiply-add and one cooperative store. The
direct form did not reduce static loads or barriers; it increased address chains,
registers, shared memory, and tensor-load sites. Q4 sits exactly at the two-WG
register limit with no headroom. Q6 is limited to one WG by registers and nearly
reaches both the register and per-WG shared limits. Stack remains zero. The
unreliable driver local-memory sentinel changes for Q6 direct, so the evidence
supports resource pathology but not a fabricated byte-accurate spill claim.

### Performance screen

The first 128 screen routed both formats. Q4 improved from 2.344 to 1.228
ms/dispatch, while Q6 regressed from 2.611 to 82.891 ms/dispatch. Q6 direct was
immediately removed. A targeted Q4 M128 refinement then took 2.098 ms/dispatch
at 128, slower than direct M64, and was also removed.

Only Q4 direct M64 proceeded to the 28K component screen; Q6 and every other
operation used the production path:

| 28K component | Baseline ms | Q4-direct screen ms | Delta |
|---|---:|---:|---:|
| Attention | 16,261.041 | 16,205.985 | -0.34% |
| Q | 2,402.709 | 2,408.829 | +0.25% |
| K | 662.567 | 659.732 | -0.43% |
| V | 706.254 | 704.752 | -0.21% |
| O | 2,498.686 | 2,500.061 | +0.06% |
| Gate | 3,234.441 | 3,247.101 | +0.39% |
| Up | 3,225.684 | 3,235.019 | +0.29% |
| Activation | 222.067 | 218.266 | -1.71% |
| Down Q4 | **2,654.326** | **3,478.672** | **+31.05%** |
| Down Q6 | 3,013.596 | 3,015.122 | +0.05% |
| Down total | **5,667.921** | **6,493.793** | **+14.57%** |
| Other, excluding activation | 727.527 | 733.589 | +0.83% |
| **GPU total** | **35,608.897** | **36,407.127** | **+2.24%** |

Q4 direct's logical compressed bytes are unchanged at 90.678 GB for its 13
layers. Effective logical compressed rate falls from 34.16 to 26.07 GB/s, and
useful compute falls from 7.77 to 5.93 TFLOP/s. Matrix tile utilization remains
99.886%; the regression is resource/instruction efficiency, not padded work.

The screening public line was 36,549.01 ms TTFT, directionally consistent with
the component regression but not promoted to a same-session public A/B/A result.
The target itself was already decisively negative and below the required 3%
continuation signal, so spending full 128/512/2K/8K/16K/28K qualification runs
would not alter the reject decision.

### Public performance disposition

| Context | Phase 20 baseline ms | Qualified candidate | llama.cpp ms | Baseline GH/llama |
|---:|---:|---:|---:|---:|
| 128 | 96.88 | — rejected during component screen | 42.983 | 2.254x |
| 512 | 364.55 | — | 131.722 | 2.768x |
| 2,048 | 1,508.41 | — | 541.391 | 2.786x |
| 8,192 | 7,033.01 | — | 2,611.573 | 2.693x |
| 16,384 | 16,801.53 | — | 6,435.127 | 2.611x |
| 28,000 | 35,381.53 | — target GPU component +31.05% | 14,038.202 | 2.520x |

There is intentionally no candidate delta or candidate GH/llama ratio in this
table: Phase 21 does not launder a rejected screen into a final benchmark.

## Correctness and guards

No tolerance, public API, numeric representation, accumulator type, model
metadata rule, attention route, gate/up path, or startup behavior is changed in
the final tree. Candidate shaders and routes were removed before qualification
because performance failed first. Consequently no correctness claim is made
for the discarded prototype beyond successful compilation and execution of the
screen; it is not production code.

The final production sources are byte-for-byte identical to the Phase 20 base.
The Phase 20 acquired evidence therefore remains the applicable correctness
contract: 158 Vulkan tests pass with five ignored, F16 through 28K passes,
INT8 is bit-exact, Q64 boundaries and fallback routing pass, and greedy sequences
match. Final Phase 21 verification is recorded in the repository audit below.

Decode is also source-identical and measured fresh on the public baseline:

| Context | Graph Horizon token/s | llama.cpp token/s | slowdown | <=1% production regression |
|---:|---:|---:|---:|---|
| 128 | 46.80 | 102.780 | 2.196x | pass by source identity |
| 2,048 | 46.34 | 95.064 | 2.051x | pass by source identity |
| 28,000 | 29.39 | 50.016 | 1.702x | pass by source identity |

The Phase 19 attention guard is 16,261.041 ms at 28K. The rejected screen
measured 16,205.985 ms, so no indirect attention regression masked the down
failure. Final attention is the unchanged baseline path.

## Portability and second-model readiness

| Scope | Disposition |
|---|---|
| Portable Vulkan | Existing production compressed matmul and fallbacks remain unchanged |
| NVIDIA-specific | Discarded callback used NVIDIA Matrix2 tensor addressing |
| Ampere-specific | Performance evidence and resource limits exist only for the RTX 3060/Ampere device |
| Shape-specific | Screen required prefill down `K=9216`, `N=3072`, and M>=64; no route remains |
| Quant-specific | Separate canonical Q4_K/Q6_K callbacks were required; both are removed |

Had it passed, a second model would have needed canonical Q4_K/Q6_K weights,
FP16 activations/output, FP32 accumulation, large-M prefill, N/K compatibility
with N32/K128, valid canonical block alignment, NVIDIA Matrix2 tensor addressing,
and successful exact-pipeline construction. Model name and GPU marketing name
were never valid predicates. Because the path failed, these are architectural
eligibility properties only, not a performance-qualified support promise.

## Performance-maturity contract

| Workload | Current GH/llama | Target | Result |
|---|---:|---:|---|
| Decode 128 / 2K / 28K | 2.196x / 2.051x / 1.702x | <=1.3–1.4x | **FAIL** |
| Short prefill 128 | 2.254x | <=1.5x | **FAIL** |
| Short prefill 512 / 2K | 2.768x / 2.786x | <=1.5x | **FAIL** |
| 8K | 2.693x | <=1.5–1.7x | **FAIL** |
| 16K | 2.611x | <=1.5–1.7x | **FAIL** |
| 28K | 2.520x | <=1.5–1.7x | **FAIL** |
| Long-context scaling | 2.693x -> 2.611x -> 2.520x | no material deterioration | **PASS** |

The maturity envelope is not reached, but that does not justify indefinite
micro-tuning. The remaining 28K time is split between exact attention
(16.261 s), distributed matmuls (18.398 s), and less than one second of other
work. Down remains the largest matmul, yet its two credible exact architectures
are now falsified: greater compressed M ownership loses occupancy, and the
direct callback increases compiler/resource cost enough to regress.

## Final verification

- `cargo fmt --all -- --check`: pass.
- CPU workspace `cargo check`: pass.
- Vulkan-profile workspace `cargo check`: pass; this also recompiles the shader
  set after removal of the prototype.
- Strict Vulkan Clippy across all targets: pass, with only the documented
  `manual_is_multiple_of` compatibility lint explicitly allowed.
- Vulkan library suite: **158 passed, 5 ignored, 0 failed**.
- `family_agnostic` excluding `docs_contract`: 4 passed, 1 ignored, 0 failed.
  The excluded check is the acquired repository condition: root
  `VALIDATION.md` is absent, independent of Phase 21.
- `git diff --check`: pass.

## Final decision and repository audit

**STOP.** Retain only this evidence and the documentation index update. No
implementation commit exists because there is no accepted implementation.
All production experiments and temporary diagnostics were removed without
changing history, and no dependency or public interface was added.

The largest remaining bottleneck is the combination of exact dense attention
and FP32-accumulating projection/MLP matmuls; within matmul, down remains
5.668 s. A future matmul candidate must be a genuinely new FP32 large-output
ownership architecture with a resource model that clears 5% whole-request; its
gain is unproven today. Repeating M128, direct callbacks, persistent FP16 down,
metadata caching, fusion, WG/N/K sweeps, or lossy native formats is not such a
candidate.

The strategic next step is **second-model support**. Different important shapes
may justify a capability/shape/quant fast path and will test the portability of
the existing generic Vulkan core plus narrow specializations more effectively
than pursuing another speculative two-percent optimization on this model.
