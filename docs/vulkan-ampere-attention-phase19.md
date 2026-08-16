<!--
This document owns the Phase 19 Ampere Matrix2 attention investigation: its
pre-code architecture, same-session evidence, experiment registry, production
decision, and qualification record.
-->

# Vulkan Ampere Matrix2 Attention — Phase 19

## Investigation state

Phase 19 starts from the clean `perf/vulkan-llamacpp-gap-phase18` branch at
`be42c0d` (`f151ab7` immediately below it); effective production source is
`6cb170f`. The experiment branch is
`perf/vulkan-ampere-attention-phase19`. Nothing is pushed.

The invariant is exact dense causal GQA attention with F16 Q/K/V/output and
FP32 QK, online-softmax state, and AV accumulation. Decode, INT8 KV, weight
representation, public APIs, and numeric tolerances remain unchanged. The
existing Q32/KV64/WG256 Matrix2 attention remains the portable fallback.

## Fresh baseline

The public harness uses a 32,768-token F16-KV capacity, greedy sampling,
`GRAPH_HORIZON_PREFILL_PREDECODE_MLP=1`, and prompts made from `" a"` repeated
`tokens - 3` times. Short rows use two warm-ups; long rows use one. Repetition
counts are 15/10/5/3/3/3. The profiler uses an independent cold sample and a
four-request aggregate; stable GPU values are `(aggregate - cold) / 3`.

| Context | TTFT mean | Median | SD | CV | Stable GPU prefill | Attention |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 96.46 ms | 95.88 ms | 1.82 ms | 1.89% | 61.108 ms | 0.903 ms |
| 512 | 369.33 ms | 369.60 ms | 3.75 ms | 1.02% | 363.127 ms | 10.265 ms |
| 2K | 1,558.17 ms | 1,554.33 ms | 17.49 ms | 1.12% | 1,521.833 ms | 142.134 ms |
| 8K | 7,770.51 ms | 7,777.13 ms | 48.62 ms | 0.63% | 7,698.535 ms | 2,190.898 ms |
| 16K | 19,999.86 ms | 19,966.48 ms | 90.53 ms | 0.45% | 20,267.119 ms | 8,943.556 ms |
| 28K | **45,012.02 ms** | **44,998.84 ms** | **44.90 ms** | **0.10%** | **45,604.555 ms** | **26,082.050 ms** |

The 128 profiler subtraction is clock-ramp-sensitive and is retained only as a
guard. The public 8K/16K/28K rows were stable enough for the candidate screen;
the final decision below uses baseline/candidate/baseline bookends from the
same session.

The source-identical Phase 15 causal shares normalize the fresh 28K attention
total to the pre-prototype attribution below. The focused causal probes later
in this report measure the candidate regions independently and state the
cross-run limitation explicitly.

| 28K attention region | Baseline |
|---|---:|
| Q/K load and QK Matrix2 | 9,657.8 ms |
| Online max/exp/sum and rescale | 3,494.9 ms |
| Probability/V load and AV Matrix2 | 9,128.7 ms |
| Other attention | 3,800.6 ms |
| **Attention total** | **26,082.1 ms** |

## Pre-code architecture specification

The candidate transfers the workgroup-matrix dataflow identified in Phase 18,
not llama.cpp source or the falsified Phase 8 ownership.

```text
Q_TILE = 64
KV_TILE = 64
WG = 128

subgroups/WG:
    4 subgroups of 32 threads

Q rows/subgroup:
    Matrix2 distributes the workgroup-scoped Q64 fragments;
    logical ownership is 16 query rows per subgroup

Matrix2 QK mapping:
    Q[64 x 128] F16 UseA, loaded once before the KV loop
    K^T[128 x 64] F16 UseB, direct tensor load per KV tile
    S[64 x 64] FP32 accumulator

softmax ownership:
    workgroup-scoped FP32 M and L matrices
    row maximum via Matrix2 row reduction
    probability via Matrix2 per-element exp
    row sum via P[64 x 64] * One[64 x 64]

Matrix2 AV mapping:
    P[64 x 64] converted to F16 UseA
    V[64 x 128] F16 UseB, direct tensor load per KV tile
    O[64 x 128] FP32 accumulator

FP32 accumulator ownership:
    O remains live across every KV tile; M and L cover the same prefix
    O is rescaled in Matrix2 state before the next AV operation

shared K tile:
    none; direct Matrix2 tensor load

shared V tile:
    none; direct Matrix2 tensor load

shared Q:
    none; Q is one workgroup-scoped Matrix2 fragment loaded once

barriers/KV iteration:
    zero explicit barriers; Matrix2 operations own workgroup cooperation
```

The output is normalized and written through a Matrix2 per-element operation,
so the prototype also removes the baseline's final FP32 shared round trip and
barrier. There is no global score or probability matrix.

### Difference from the rejected Phase 8 Q64 path

Phase 8 tried Q64 ownership inside the old decomposition. Phase 19 instead
changes WG256 to WG128, uses one flexible workgroup Matrix2 operation for each
QK and AV tile, retains score/max/sum/output in Matrix2 state, removes explicit
score/probability shared arrays, and loads Q only once. Q64 is therefore a
consequence of the new matrix-native dataflow rather than an isolated tile
increase.

## Predicted resource and traffic model

The exact 28K useful interaction count is 326.156 billion. Q64/KV64 executes
the same padded causal work as Q32/KV64, so QK and AV utilization remain
99.237% at 8K and 99.775% at 28K; padding is not the source of the gain.

| Metric | Q32/KV64/WG256 baseline | Q64/KV64/WG128 candidate |
|---|---:|---:|
| Q bytes/request | 1.308 TB | 5.964 GB |
| K bytes/request | 2.615 TB | 1.308 TB |
| V bytes/request | 2.615 TB | 1.308 TB |
| explicit score/probability shared traffic | 5.254 TB | 0 |
| Q/K/V bytes/useful pair | 20.04 B | 8.04 B |
| barriers/KV tile | 2 | 0 |
| explicit shared/WG | 38,272 B | 0 B plus driver-reserved Matrix2 memory |
| useful query rows/WG | 32 | 64 |

The candidate's register count was intentionally not predicted from source.
The pre-prototype gate was no spills and at least one resident WG; two resident
WG would own 128 live query rows/SM versus 64 for the baseline. The following
driver executable-property capture determines registers/thread, total
shared/WG, resident WG, and modeled occupancy.

## Prototype A resource and ISA audit

The driver accepted and routed the exact WG128 pipeline on the RTX 3060. A
temporary `VK_KHR_pipeline_executable_properties` probe was removed after this
capture:

| Metric | Q32/KV64/WG256 | Q64/KV64/WG128 |
|---|---:|---:|
| registers/thread | 124 | 255 |
| registers/WG | 31,744 | 32,640 |
| driver shared/WG | 38,272 B | 48,640 B |
| stack | 0 | 0 |
| register-limited WG/SM | 2 | 2 |
| shared-limited WG/SM | 2 | 2 |
| thread-limited WG/SM | 6 | 12 |
| hardware WG limit | 16 | 16 |
| resident WG/SM | 2 | 2 |
| active subgroups | 16 | 8 |
| modeled thread occupancy | 33.3% | 16.7% |
| useful query rows/SM | 64 | 128 |

The driver's impossible 64-GiB local-memory sentinel is 64 bytes larger for
Q64, while stack remains zero. This is recorded as an unresolvable possible
64-byte local allocation, not as a measured spill count; the available Vulkan
interfaces expose no executed local-load/store or SASS counter. There is no
runtime pathology: two resident workgroups fit with only 256 registers and
5,120 shared bytes of headroom per SM.

Static optimized SPIR-V confirms that the intended dataflow materialized:

| Static category | Q32 | Q64 |
|---|---:|---:|
| module bytes | 12,684 | 9,700 |
| ordinary loads / stores | 134 / 70 | 87 / 49 |
| access chains | 41 | 23 |
| Matrix2 tensor loads | 3 | 3 |
| Matrix2 MMA | 2 | 3 |
| Matrix2 reductions | 0 | 3 |
| Matrix2 per-element operations | 2 | 5 |
| explicit cooperative stores | 2 | 0 |
| control barriers | 4 | 0 |
| branches / conditional branches | 29 / 11 | 7 / 4 |
| FP conversions | 2 | 2 |

The lower occupancy therefore does not explain the speedup. Each resident Q64
WG owns twice the useful query work, Q is loaded once, score/probability never
round-trip through explicit shared arrays, and matrix-native reductions replace
the manual synchronized softmax path.

## Capability and routing contract

Routing requires an NVIDIA device with advertised and enabled
`VK_KHR_cooperative_matrix` plus `VK_NV_cooperative_matrix2`, workgroup
scope, flexible dimensions, row reductions, per-element operations, tensor
addressing, F16 A/B and FP32 C/result, a 128-invocation flexible property whose
granularities divide the QK and AV dimensions, maximum Matrix2 dimension at
least 128, and sufficient device shared memory including the driver-reserved
amount. This capability contract is stronger than a GPU-name match and is the
Ampere gate exercised by the RTX 3060.

Unsupported devices do not build or route the candidate pipeline and allocate
no candidate-specific runtime storage. Non-aligned query tails and all decode
and INT8 attention continue through existing paths.

## Phase 18 prediction

| Item | Value |
|---|---:|
| Phase 18 GH 28K | 46.777 s |
| llama.cpp 28K | 14.038 s |
| competitive gap | 32.739 s |
| predicted attention seconds removed | 9.939 s |
| predicted whole-request improvement | 21.19% |

## Experiment registry

| ID | Architecture | Q/KV/WG | 28K attention | 28K TTFT | Decision |
|---|---|---|---:|---:|---|
| P19-A | workgroup Matrix2 online state | 64/64/128 | -37.77% | -22.41% | **KEEP** |

## Representative prototype screen

The same instrumented binary ran baseline/candidate/baseline bookends. The
only process-level difference was the temporary Q64 route toggle. Each point
used one warm-up and one measured request; attention and GPU totals below are
per request from the two-request profiler aggregate.

| Context | Metric | Baseline A | Q64 | Baseline B | Bookend baseline | Delta |
|---:|---|---:|---:|---:|---:|---:|
| 8K | TTFT | 7,968.23 ms | 7,124.56 ms | 7,973.21 ms | 7,970.72 ms | **-10.62%** |
| 8K | attention | 2,240.242 ms | 1,415.700 ms | 2,240.244 ms | 2,240.243 ms | **-36.81%** |
| 28K | TTFT | 45,799.01 ms | 35,682.65 ms | 46,018.87 ms | 45,908.94 ms | **-22.28%** |
| 28K | GPU prefill | 45,602.211 ms | 35,618.327 ms | 45,805.179 ms | 45,703.695 ms | **-22.07%** |
| 28K | attention | 26,114.037 ms | 16,281.597 ms | 26,211.201 ms | 26,162.619 ms | **-37.77%** |

The final 96-row 28K microbatch is intentionally Q32 because it is not
divisible by 64. This explains why the aggregate profiler presents the last
attention kernel's Q32 label while its maximum dispatch geometry records four
Q64 tiles.

## Causal attention attribution

Temporary source-local variants retained every downstream consumer while
removing one upstream region at a time: `no-QK` supplied zero scores to the live
softmax and AV path, `AV` supplied unit probabilities to the live V load/MMA,
and `skeleton` retained output normalization/store without QK, softmax, or AV.
The variants, their route selector, and their SPIR-V were removed before the
production commit. Each result below is the difference between adjacent live
stages, divided by two profiler requests. The 28K skeleton includes the normal
Q32 final 96-row tail, so the decomposition remains additive rather than
pretending that tail vanished.

Baseline regions use the unchanged Phase 15 causal shares normalized to a
source-aligned Phase 19 baseline total; candidate regions use the Phase 19 live
stage probes. They are not presented as same-dispatch hardware counters—the
Vulkan/NVIDIA interface available here exposes neither SASS nor executed
load/store/spill counters.

| Context | Region | Q32 baseline | Q64 candidate | Delta |
|---:|---|---:|---:|---:|
| 8K | Q/K load + QK | 812.312 ms | 938.903 ms | +15.58% |
| 8K | online softmax/rescale | 312.514 ms | 92.619 ms | -70.36% |
| 8K | probability/V + AV | 782.517 ms | 351.814 ms | -55.04% |
| 8K | other attention | 332.900 ms | 20.923 ms | -93.71% |
| 8K | **total** | **2,240.243 ms** | **1,404.259 ms** | **-37.32%** |
| 16K | Q/K load + QK | 3,256.349 ms | 3,764.452 ms | +15.60% |
| 16K | online softmax/rescale | 1,216.324 ms | 364.036 ms | -70.07% |
| 16K | probability/V + AV | 3,151.709 ms | 1,420.266 ms | -54.94% |
| 16K | other attention | 1,319.175 ms | 42.618 ms | -96.77% |
| 16K | **total** | **8,943.556 ms** | **5,591.371 ms** | **-37.48%** |
| 28K | Q/K load + QK | 9,688.018 ms | 10,864.950 ms | +12.15% |
| 28K | online softmax/rescale | 3,505.791 ms | 1,025.527 ms | -70.75% |
| 28K | probability/V + AV | 9,156.917 ms | 4,172.886 ms | -54.43% |
| 28K | other attention | 3,811.894 ms | 261.041 ms | -93.15% |
| 28K | **total** | **26,162.619 ms** | **16,324.403 ms** | **-37.60%** |

Useful-pair scaling supports the same conclusion. Candidate QK costs
33.70/33.71/33.31 ms per billion useful pairs at 8K/16K/28K, versus
29.09/29.16/29.70 for the baseline. Candidate AV costs only
12.60/12.72/12.79 ms per billion, versus 28.03/28.22/28.07. QK is slightly
slower; the win comes from matrix-native softmax and AV, loading Q once, and
removing score/probability shared round trips and explicit barriers.

## Final same-session qualification

The decision sweep used baseline/Q64/baseline process bookends with identical
public settings. Short contexts used two warm-ups and 15/10/5 repetitions;
8K/16K/28K used one warm-up and three repetitions. All rows generated 32
tokens. CV is the candidate TTFT coefficient of variation.

| Context | Baseline bookend | Q64 TTFT | Delta | CV | llama.cpp | GH/llama before | GH/llama after |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 96.94 ms | 97.13 ms | +0.20% | 0.57% | 42.983 ms | 2.255x | 2.259x |
| 512 | 369.60 ms | 365.71 ms | -1.05% | 0.26% | 131.722 ms | 2.806x | 2.776x |
| 2K | 1,565.17 ms | 1,512.60 ms | -3.36% | 0.23% | 541.391 ms | 2.891x | 2.794x |
| 8K | 7,931.26 ms | 7,068.39 ms | -10.88% | 0.19% | 2,611.573 ms | 3.037x | 2.707x |
| 16K | 20,373.42 ms | 16,865.07 ms | -17.22% | 0.12% | 6,435.127 ms | 3.166x | 2.621x |
| 28K | **45,825.73 ms** | **35,557.68 ms** | **-22.41%** | **0.04%** | **14,038.202 ms** | **3.264x** | **2.533x** |

The only neutral point is 128; benefit starts at 512 and grows monotonically.
No extra runtime crossover threshold is justified: internal prefill batches are
256 rows, alignment already selects the candidate only for complete Q64 tiles,
and the 128 result is within the 1% guard.

Decode throughput is unchanged within the 1% same-session guard:

| Context | Baseline bookend | Q64 | Delta |
|---:|---:|---:|---:|
| 128 | 46.89 tok/s | 46.78 tok/s | -0.23% |
| 512 | 46.30 tok/s | 46.21 tok/s | -0.19% |
| 2K | 46.32 tok/s | 46.30 tok/s | -0.03% |
| 8K | 40.61 tok/s | 40.67 tok/s | +0.15% |
| 16K | 34.97 tok/s | 35.02 tok/s | +0.16% |
| 28K | 29.21 tok/s | 29.20 tok/s | -0.03% |

After deleting every diagnostic variant and route toggle, a rebuilt production
binary independently measured 35,819.52 ms mean at 28K (median 35,845.77 ms,
SD 70.81 ms, CV 0.20%) and 27.97 decode tok/s. This source-clean confirmation
is -21.84% against the decision-sweep baseline bookends and remains inside the
mission's excellent band.

### Whole-request Amdahl shift

A final narrow profiler pair used one warm-up plus one measured 28K request per
route. Values are per request after dividing the two-request aggregates.

| 28K prefill component | Q32 | Q64 | Delta | Q32 share | Q64 share |
|---|---:|---:|---:|---:|---:|
| attention | 26,092.668 ms | 16,344.012 ms | -37.36% | 57.19% | 45.84% |
| seven projection/MLP matmuls | 18,570.759 ms | 18,359.763 ms | -1.14% | 40.70% | 51.49% |
| all other GPU work | 958.703 ms | 952.284 ms | -0.67% | 2.10% | 2.67% |
| **GPU prefill** | **45,622.130 ms** | **35,656.059 ms** | **-21.84%** | **100%** | **100%** |

The matmul movement is ordinary cross-process clock variation, not a candidate
mechanism. It is now the largest measured component at 18.36 seconds; attention
is second at 16.34 seconds, with QK its largest internal region.

### Prediction accuracy and competitive recovery

| Item | Prediction | Final B/C/B | Difference |
|---|---:|---:|---:|
| attention seconds saved | 9.939 s | 9.838 s | -0.101 s (-1.02%) |
| TTFT seconds saved | 9.939 s model | 10.268 s | +0.329 s (+3.31%) |
| whole-request improvement | 21.19% | 22.41% | +1.22 points |

Against the Phase 18 32.739-second GH/llama gap, the candidate recovers 31.36%.
The fresh bookended GH/llama ratio improves from 3.264x to 2.533x (the historical
Phase 18 ratio was 3.332x). The measured result tracks the Phase 18 quantitative
prediction closely enough to validate its comparative model.

## Correctness and regression qualification

- F16 Q64 attention matched sequential Vulkan decode and the CPU oracle at
  2K, 8K, and 28K. The largest Q64 CPU-oracle absolute error at 28K was
  `1.374e-5`; the largest attention/decode error was `1.526e-5`. Logits remained
  inside the unchanged tolerances.
- INT8 KV stayed on the established implementation and was bit-exact at every
  2K/8K/28K long-context check. Boundary tests at 63/64/65, 127/128/129, and
  255/256/257 proved the Q64 route and Q32/generic fallback transitions.
- The known 128-prompt greedy candidate remained exactly
  `2757,10637,2479,1636`. A separate 2K-prompt, 128-token greedy run was exactly
  identical between Q32 and Q64 with decode fallback forced.
- The retained dense format smoke covered F16, Q4_K, Q6_K, and logits. The full
  Vulkan library suite passed 149 tests with 5 ignored; the explicit long-context
  matrix passed all F16 and INT8 rows. The family-agnostic integration suite had
  four passes and one ignore; `docs_contract` alone remains blocked by the
  pre-existing missing root `VALIDATION.md`, unrelated to this branch.
- `cargo fmt --check`, Vulkan `cargo check`, release shader compilation, and
  strict Clippy pass for the changed surface. The repository's unrelated
  `harness/stats.rs` still triggers Rust 1.95's new `manual_is_multiple_of` lint,
  so the strict run allowed that one pre-existing lint explicitly.

## Decision and next bottleneck

**KEEP P19-A.** The production path is capability- and alignment-gated, creates
no candidate storage, builds no candidate pipeline when the exact Matrix2
contract is absent, and preserves Q32, generic, decode, and INT8 fallbacks. It
delivers 22.41% bookended 28K TTFT improvement (21.84% source-clean
confirmation), neutral short-context behavior, and unchanged decode.

No corrective micro-experiment is justified after an excellent result. Phase
19 stops here. The next competitive investigation should target the seven
projection/MLP matmuls, now 51.49% of measured 28K GPU prefill. Inside attention,
QK is the remaining dominant subregion, but it is no longer the whole-request
bottleneck.
