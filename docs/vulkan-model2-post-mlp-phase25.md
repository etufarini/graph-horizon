<!--
This report owns Phase 25 evidence for the Model 2 post-MLP global reprofile:
fresh request and competitive attribution, route eligibility, practical floors,
candidate ranking, and the evidence-gated stop decision. It changes no runtime
support contract or numeric policy.
-->

# Vulkan Model 2 Post-MLP Global Reprofile — Phase 25

## Decision

**STOP without a production prototype.** The fresh Phase 24 production profile
accounts for 100.00% of 8K and 28K GPU prefill and confirms that every hot
Q/K/V/O/gate/up/down shape already uses the expected compressed Matrix2 path.
There is no remaining eligibility bug, unused Model 2 native representation,
hidden host gap, or thermal collapse.

At 28K, fused attention is the largest individual category at 21.433 s, while
gate+up is the largest aggregate domain at 29.062 s. Competitive attribution
alone is misleading: llama.cpp is 12.422 s faster in attention, 22.527 s in
gate+up, and 11.347 s in down, but its matmul advantage retains the acquired
FP16-accumulator numeric difference. The current FP32-compatible gate/up and
down routes remain close to their Phase 23/24 practical floors.

Fresh causal deletion divides attention into 14.488 s Q/K load+QK, 0.986 s
online softmax/rescale, 5.629 s V/AV, and 0.330 s normalization/store residual.
The Q/K load portion already sustains about 316 GB/s logical bandwidth. An
optimistic same-numeric floor that gives both matrix multiplies llama.cpp's
whole-attention effective compute rate is 15.99 s, only 5.44 s removable or
6.93% of public TTFT. Clearing 10% requires moving score scaling before the
FP16 Matrix2 multiply or changing accumulation/representation policy. Neither
is an exact production candidate under the retained numeric contract.

The mission's stop conditions 4, 5, and 8 therefore apply: no remaining exact
candidate has a supported >=10% whole-request prediction, the larger gap needs
new numeric semantics, and qualifying another model now has higher value than
micro-tuning this RTX 3060 path.

## Reproducibility and baseline

| Item | Value |
|---|---|
| Starting branch | `perf/model2-gate-up-phase24` |
| Phase 24 production commit | `b86e2a51cfad16d354d4d0a36828b3ccff5846b1` |
| Phase 24 evidence commit | `8b43fff29d337f12045f354b3ed7bc6cd50dc376` |
| Reviewed baseline HEAD | `a8d323b` (validation-register restoration only) |
| Investigation branch | `perf/model2-post-mlp-phase25` |
| Production change | none |
| GPU / driver | NVIDIA GeForce RTX 3060 12 GiB / 595.84 |
| Nominal memory system | 3 MiB L2, 360 GB/s DRAM, 61.1 TFLOP/s FP16 |
| Model 2 | `Ministral-3-8B-Instruct-2512-Q4_K_M.gguf` |
| Model bytes / SHA-256 | 5,198,911,904 / `33e7a72cf5e6e2cfc2f2847075acc013d68bba023e35310cef86b5cf8fdca761` |
| Prompt | exact chat-templated `" a"` repetition producing the requested count |
| Request | greedy, 32 decoded tokens, capacity 32,768, F16 KV |
| Public repetitions | warmup 1; 5/5/3/3/2/2 at 128/512/2K/8K/16K/28K |
| Profile | warmup 0, one request, Vulkan timestamps, public 256-row batching |
| llama.cpp | commit `9bebfcb4`, build 9826, Vulkan0, F16 KV, flash attention, 3 reps |

The immutable public executable SHA-256 is
`15950a247b79e4c64520eb8580120094e3585b274307318d027c8fe8221831c2`,
exactly the Phase 24 retained binary hash. The fresh timestamp executable is
`62f3967772ad1d8591e7ded613c9dc52c114210137a7e7015bec0ce40b8c4a46`.
Ignored raw evidence is under `target/phase25`.

## Public baseline and bookends

| Context | Reps | TTFT mean / median ms | SD / CV | GPU prefill ms | Prompt / decode tok/s |
|---:|---:|---:|---:|---:|---:|
| 128 | 5 | 275.22 / 275.36 | 1.52 / 0.55% | 287.116 | 465.09 / 22.16 |
| 512 | 5 | 1,063.53 / 1,063.62 | 2.42 / 0.23% | 1,071.268 | 481.42 / 21.94 |
| 2,048 | 3 | 4,329.99 / 4,328.91 | 5.90 / 0.14% | 4,276.804 | 472.98 / 21.40 |
| 8,192 | 3 | 18,640.39 / 18,641.57 | 5.82 / 0.03% | 18,766.991 | 439.48 / 19.74 |
| 16,384 | 2 | 40,751.46 / 40,751.46 | 5.57 / 0.01% | 40,533.214 | 402.05 / 18.02 |
| 28,000 | 2 | 78,438.08 / 78,438.08 | 134.17 / 0.17% | 78,577.628 | 356.97 / 15.90 |

The mandatory restored-baseline bookend used the same immutable executable and
repetition counts for the decision contexts.

| Context | Baseline A | Baseline B | B versus A | Bookend mean |
|---:|---:|---:|---:|---:|
| 8K | 18,640.39 ms | 18,562.54 ms | -0.42% | 18,601.47 ms |
| 28K | 78,438.08 ms | 78,598.70 ms | +0.20% | 78,518.39 ms |

Active public telemetry has 994 samples: VRAM 9,722/9,756/9,763 MiB,
graphics clock 1,807/1,945/1,995 MHz, memory clock fixed at 7,501 MHz,
power 51.1/167.7/170.3 W, and temperature 58/70.6/72 C
(min/mean/max). Bookend-B values are comparable: 1,807/1,947/2,002 MHz,
47.3/168.3/170.1 W, and 46/70.1/72 C. Thermal drift cannot explain the
measurements.

## Full GPU attribution

The independent timestamp runs account for 18,766.989 of 18,766.991 ms at 8K
and 78,577.630 of 78,577.628 ms at 28K; rounding makes the latter exceed the
reported total by 0.002 ms. Attention stages are additive controlled deletions.
The single final RMSNorm invocation is assigned to `other`; the equal-shape
attention and MLP norms split the other normalization invocations evenly.

| Component | 8K s | 8K GPU | 28K s | 28K GPU |
|---|---:|---:|---:|---:|
| attention RMSNorm | 0.033 | 0.17% | 0.106 | 0.14% |
| Q projection | 1.285 | 6.85% | 4.347 | 5.53% |
| K projection | 0.351 | 1.87% | 1.187 | 1.51% |
| V projection | 0.364 | 1.94% | 1.224 | 1.56% |
| RoPE | 0.028 | 0.15% | 0.091 | 0.12% |
| Q/K load + QK | 1.248 | 6.65% | 14.488 | 18.44% |
| softmax / online rescale | 0.093 | 0.50% | 0.986 | 1.25% |
| V load + AV | 0.471 | 2.51% | 5.629 | 7.16% |
| attention normalization/store | 0.026 | 0.14% | 0.330 | 0.42% |
| O projection | 1.286 | 6.85% | 4.359 | 5.55% |
| MLP RMSNorm | 0.033 | 0.17% | 0.106 | 0.14% |
| gate | 4.298 | 22.90% | 14.521 | 18.48% |
| up | 4.288 | 22.85% | 14.541 | 18.50% |
| activation | 0.129 | 0.69% | 0.420 | 0.53% |
| down | 4.643 | 24.74% | 15.670 | 19.94% |
| copies/conversions (KV write) | 0.010 | 0.06% | 0.035 | 0.04% |
| dispatch/control residual | 0.007 | 0.04% | 0.021 | 0.03% |
| residual/embedding/logits/final norm | 0.173 | 0.92% | 0.515 | 0.66% |
| **GPU prefill** | **18.767** | **100.00%** | **78.578** | **100.00%** |

No unaccounted copy or conversion is hidden in `other`. Matrix2's F16 fragment
conversions occur inside the named fused kernels and cannot be timed as
independent dispatches.

Public TTFT above is the CPU-observed request wall. The timestamp profiler's
8K record/submit/wait/descriptor times are 28.077/2.763/18,771.634/2.333 ms;
at 28K they are 96.093/7.802/78,590.288/8.042 ms. GPU wait tracks GPU prefill,
while record, submit, descriptor, and timestamp residual are far below 1%.
There is no separate CPU-side bottleneck or unattributed synchronization gap.

## Macro breakdown

`attention` includes attention norm, RoPE, QK, softmax, AV, and attention
normalization/store. `norm/elementwise` contains MLP norm and residual adds.

| Macro | 8K s | 8K GPU / TTFT | 28K s | 28K GPU / TTFT |
|---|---:|---:|---:|---:|
| attention | 1.899 | 10.12% / 10.19% | 21.631 | 27.53% / 27.58% |
| QKV/O matmul | 3.287 | 17.51% / 17.63% | 11.118 | 14.15% / 14.17% |
| gate/up | 8.586 | 45.75% / 46.06% | 29.062 | 36.98% / 37.05% |
| activation | 0.129 | 0.69% / 0.69% | 0.420 | 0.53% / 0.53% |
| down | 4.643 | 24.74% / 24.91% | 15.670 | 19.94% / 19.98% |
| norm/elementwise | 0.107 | 0.57% / 0.57% | 0.341 | 0.43% / 0.44% |
| other | 0.116 | 0.62% / 0.62% | 0.336 | 0.43% / 0.43% |

## Global eligibility and representation audit

All shapes use prompt rows for `M`. Restrictions are capability-, format-,
shape-, and workload-based. They are performance-qualified allow-list entries,
not model-name or GPU-name routing.

| Operation | `K -> N` / quant | Expected best path | Actual selected path | Restriction classification |
|---|---|---|---|---|
| Q | `4096 -> 4096` / Q4_K | compressed Matrix2 | M64/N32/K128 Q4 Matrix2 | performance-qualified |
| K | `4096 -> 1024` / Q4_K | compressed Matrix2 | M64/N32/K128 Q4 Matrix2 | performance-qualified |
| V | `4096 -> 1024` / Q4_K,Q6_K | compressed Matrix2 | M64/N32/K128 Q4/Q6 Matrix2 | performance-qualified |
| O | `4096 -> 4096` / Q4_K | compressed Matrix2 | M64/N32/K128 Q4 Matrix2 | performance-qualified |
| gate | `4096 -> 14336` / Q4_K | Phase 24 compressed Matrix2 | M64/N32/K128 Q4 Matrix2 | performance-qualified |
| up | `4096 -> 14336` / Q4_K | Phase 24 compressed Matrix2 | M64/N32/K128 Q4 Matrix2 | performance-qualified |
| down | `14336 -> 4096` / Q4_K,Q6_K | Phase 23 compressed Matrix2 | M64/N32/K128 Q4/Q6 Matrix2 | performance-qualified |

The profiler independently records the expected path for every invocation.
Gate/up remain 14.521/14.541 s, down remains on the qualified large-K route,
and no hot shape falls back to the cooperative or scalar kernels.

The newly visible Q/K/V/O family is also already efficiently tiled. At 28K,
M64 utilization is 99.886% (`28000/28032`); N32 and K128 are exact. Logical
bandwidth includes repeated compressed weights, activation tensor loads, and
outputs, and is a shader access model rather than a DRAM-counter claim.

| Projection | 28K s | Logical bytes | Effective logical BW | Useful compute | Matrix2 tile utilization |
|---|---:|---:|---:|---:|---:|
| Q | 4.347 | 1.147 TB | 263.7 GB/s | 7.35 TFLOP/s | 99.886% M; 100% N/K |
| K | 1.187 | 0.287 TB | 241.4 GB/s | 6.73 TFLOP/s | 99.886% M; 100% N/K |
| V, mixed Q4/Q6 | 1.224 | 0.295 TB | 240.7 GB/s | 6.52 TFLOP/s | 99.886% M; 100% N/K |
| O | 4.359 | 1.147 TB | 263.0 GB/s | 7.33 TFLOP/s | 99.886% M; 100% N/K |

Model 2 never satisfied the exact `[3072,9216]` native gate/up predicate, so
Phase 24 made no native allocation obsolete. Native preprocessing and
eliminable persistent VRAM are both zero. Active VRAM remains about 9.76 GiB,
consistent with canonical weights plus 32K F16 KV. The acquired 3.271 s Phase
22 initialization contains zero native preprocessing; there is no startup
cleanup candidate. Model 1's native path is untouched.

## Competitive refresh

Fresh llama.cpp public results are 5,196.070 ms at 8K (SD 4.915 ms, CV 0.09%)
and 24,266.433 ms at 28K (SD 29.126 ms, CV 0.12%). The same logger build then
captured one measured pass after its internal warm-up. Q/O and gate/up share
identical shape families and are divided evenly; K versus Q4 V is divided by
its 34:17 invocation count, with Q6 V retained separately.

| Component, 28K | GH s | llama.cpp s | Absolute gap s | Ratio | Practical removable s |
|---|---:|---:|---:|---:|---:|
| fused attention | 21.433 | 9.011 | 12.422 | 2.38x | 5.44 |
| Q | 4.347 | 1.256 | 3.091 | 3.46x | 0.20 |
| K | 1.187 | 0.355 | 0.832 | 3.34x | 0.05 |
| V | 1.224 | 0.383 | 0.841 | 3.20x | 0.05 |
| O | 4.359 | 1.256 | 3.103 | 3.47x | 0.20 |
| gate | 14.521 | 3.268 | 11.254 | 4.44x | 0.89 |
| up | 14.541 | 3.268 | 11.273 | 4.45x | 0.89 |
| activation | 0.420 | 0.467 | -0.047 | 0.90x | 0 |
| down | 15.670 | 4.322 | 11.347 | 3.63x | 0.55 |
| norm | 0.212 | 0.218 | -0.006 | 0.97x | 0 |
| other | 0.643 | 0.589 | 0.054 | 1.09x | 0.05 |
| **Timestamp total** | **78.578** | **24.393** | **54.185** | **3.22x** | **8.32** |

The public competitive gap is 78.438 - 24.266 = **54.172 s**. Timestamp
components reconstruct 54.185 s, a 0.013 s difference from independent timing
boundaries, and attribute effectively 100% of the gap. The raw gap is 41.57%
gate/up, 22.92% attention, 20.94% down, 14.52% Q/K/V/O, and 0.10% other.

| Context | GH bookend mean | llama.cpp | GH/llama | Target | Status |
|---:|---:|---:|---:|---:|---|
| 8K | 18.601 s | 5.196 s | **3.58x** | <=1.5x | RED |
| 28K | 78.518 s | 24.266 s | **3.24x** | <=1.5–1.7x | RED |

At 28K, 1.7x llama.cpp is 41.253 s and 1.5x is 36.400 s. The current
bookended request still needs 1.90x or 2.16x total speedup respectively; no
single practical component floor in this profile can supply that distance.
Even impossible zero-cost fused attention leaves about 57.09 s, or 2.35x
llama.cpp, confirming that attention alone cannot reach maturity.

## Attention shape, resources, traffic, and llama.cpp comparison

The Phase 19/20 Q64 path is selected for every complete 64-row tile. The final
28K 96-row public batch intentionally uses one Q64 tile and a Q32 tail, which
is why the aggregate category prints the last Q32 kernel name while reporting
`max_groups_y=4`. Eligibility is F16 KV, head dimension 128, 32 Q heads,
8 KV heads, GQA 4, complete Q64 rows, and the full Matrix2 capability contract.

| Metric | Model 1 | Model 2 |
|---|---:|---:|
| layers | 26 | 34 |
| Q / KV heads | 32 / 8 | 32 / 8 |
| GQA / head dimension | 4 / 128 | 4 / 128 |
| Q tile / KV tile / WG | 64 / 64 / 128 | 64 / 64 / 128 |
| subgroups | 4 | 4 |
| registers/thread | 255 | 255, same executable |
| driver shared/WG | 48,640 B | 48,640 B, same executable |
| resident WG / modeled occupancy | 2 / 16.7% | 2 / 16.7% |
| stack | 0 | 0 |
| 28K useful pairs | 326.156B | 426.511B |
| 28K QK+AV useful work | 167.368 TF | 218.374 TF |

Model 2's 21.433 s is the Phase 19 Model 1 cost scaled by 34/26 within ordinary
run variation. The path is not merely correct on Model 2; it has the same shape
utilization and per-pair behavior. There is no Model 2-specific attention route
defect to fix.

| Traffic/throughput metric, 28K | Graph Horizon | llama.cpp |
|---|---:|---:|
| Q logical bytes | 7.799 GB | 7.799 GB |
| K logical bytes | about 1.710 TB | about 1.710 TB |
| V logical bytes | about 1.710 TB | about 1.710 TB |
| Q/K/V bytes per useful pair | 8.04 B | 8.04 B |
| fused effective logical bandwidth | 160 GB/s | 380 GB/s |
| fused effective compute | 10.19 TFLOP/s | 24.23 TFLOP/s |
| QK effective compute | 7.54 TFLOP/s | unavailable inside fused boundary |
| AV effective compute | 19.40 TFLOP/s | unavailable inside fused boundary |

At 28K the source-level logical model is 7.799 GB Q plus about 1.710 TB each K
and V, or 8.04 bytes/useful pair. Graph Horizon sustains 10.19 TFLOP/s over the
whole fused operation; QK and AV sustain 7.54 and 19.40 TFLOP/s respectively.
The no-MMA intermediate probe further divides Q/K load/address at 5.421 s and
the QK Matrix2 multiply at 9.067 s. K traffic alone is therefore about
316 GB/s logical, close to the 360 GB/s nominal ceiling.

llama.cpp executes the same useful pairs with the same Q64/KV64 Matrix2 online
algorithm in 9.011 s, or 24.23 effective TFLOP/s. Its fused boundary does not
expose honest additive QK/softmax/AV timestamps, so none are fabricated. Source
comparison confirms the acquired difference: llama.cpp scales FP32 Q once,
converts that scaled value to its FP16 Matrix2 fragment, and uses the result for
every KV tile. Graph Horizon consumes retained F16 Q and scales the FP32 score
after each QK multiply. Moving that scale before the multiply changes rounding;
it is a numeric-policy experiment, not an exact refactor.

## Experiment registry

All attention probes use the selected F16 Q64/KV64/WG128 path, 34 layers,
32/8 heads, GQA 4, and head dimension 128. Resource figures are captured for
the production executable only; probe variants were temporary and were not
promoted or resource-qualified.

| ID | Hypothesis / live work | 8K attention | 28K attention | Correctness | Decision |
|---|---|---:|---:|---|---|
| P25-B | production baseline | 1.837942 s | 21.433494 s | acquired production path | baseline |
| P25-A1 | remove QK MMA, retain Q/K loads | 1.083554 s | 12.366312 s | intentionally invalid output | causal evidence; restore |
| P25-A2 | also remove Q/K loads | 0.590352 s | 6.945384 s | intentionally invalid output | causal evidence; restore |
| P25-A3 | unit probability, live V/AV | 0.497418 s | 5.959241 s | intentionally invalid output | causal evidence; restore |
| P25-A4 | normalization/store skeleton | 0.026412 s | 0.330262 s | intentionally invalid output | causal evidence; restore |
| P25-C | fresh llama.cpp fused comparison | 0.782503 s | 9.011327 s | authenticated weights/F16 KV | competitive evidence |

Production Q64 resources are 255 registers/thread, 48,640 B driver shared/WG,
two resident workgroups, 16.7% modeled thread occupancy, and zero stack. No
spill counter is exposed; the acquired driver local-memory sentinel is not
misreported as a spill measurement. The selected architecture would have been
an exact post-MMA FP32 QK improvement under the existing capability + shape +
quantization + workload predicate. It was **not selected for implementation**
because its optimistic 6.93% prediction fails the 10% gate.

Portability of that unbuilt direction is NVIDIA-specific, Ampere-tuned,
shape-specific to F16 head-dimension-128 Q64 attention, not weight-quant-specific,
and not model-specific. Unsupported devices and shapes would retain the current
generic/Q32 routes.

## Physical and practical floors

Minimum traffic uses source-level logical bytes divided by nominal 360 GB/s;
minimum compute uses useful work divided by 61.1 TFLOP/s. These are selection
models, not DRAM or executed-instruction counters.

| Domain | Current s | Minimum traffic s | Minimum compute s | Practical floor s | Removable s |
|---|---:|---:|---:|---:|---:|
| fused attention | 21.433 | 9.53 | 3.57 | **15.99** | **5.44** |
| gate+up | 29.062 | 22.30 | 3.66 | **27.28** | **1.78** |
| down | 15.670 | 11.41 | 1.83 | **15.12** | **0.55** |

The attention floor keeps the measured 5.421 s Q/K load, gives QK llama.cpp's
whole-fused 24.23 TFLOP/s rate (4.506 s), bounds AV by its higher 4.751 s
minimum-traffic time, and retains measured softmax/residual cost. It is
deliberately optimistic and still predicts only 6.93% whole-request improvement.

The gate/up floor models an M128 traversal that halves only the 0.984 TB pair's
logical weight traffic while retaining current effective logical bandwidth.
It removes about 1.78 s, not the 10 s suggested by incorrectly reusing Phase
24's obsolete legacy-kernel ablation. Down's 15.12 s floor is the acquired
same-kernel prediction validated in Phase 23; current 15.67 s is within 3.6%.

## Candidate ranking and Amdahl gate

| Rank | Target | Current s | Practical floor s | Removable s | Predicted TTFT | Risk | Portability |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | exact QK/AV execution improvement | 21.433 | 15.99 | 5.44 | **6.93%** | high; compiler/resource sensitive | NVIDIA Matrix2, shape-based, not model-specific |
| 2 | gate/up M128 ownership | 29.062 | 27.28 | 1.78 | 2.27% | high; ~166 registers, one WG | Ampere-tuned Q4 shape family |
| 3 | QKV/O larger-BN FP32 mapping | 11.118 | 10.62 | 0.50 | 0.64% | very high; Phase 21 evidence negative | NVIDIA Matrix2 shape family |
| 4 | down M128 ownership | 15.670 | 15.12 | 0.55 | 0.70% | high; mixed Q4/Q6 resources | Ampere-tuned quant shape |
| 5 | dispatch/control/other | 0.979 | 0.75 | 0.23 | 0.29% | low | portable but immaterial |

For rank 1, affected time is 21.433 s and optimistic local speedup is 1.34x,
removing 5.44 s and predicting 73.08 s TTFT. Reaching the 10% gate requires at
least 7.844 s removed and a 13.589 s attention result, below the same-numeric
practical floor. Rank 2 would predict 76.74 s; the others are smaller.

Selected candidate: **none**. The leading unbuilt hypothesis is exact FP32
post-MMA QK execution for F16 Q/K/V, head dimension 128, Q64/KV64/WG128, under
the existing NVIDIA Matrix2 capability and aligned-prefill predicate. It is
rejected before coding because no concrete architecture closes the required
7.844 s without changing pre-MMA rounding or because its exact bound is below
10% whole-request.

No substantial production prototype was authorized by this ranking. The four
attention executables were causal instrumentation only: production, no QK MMA,
no Q/K load or QK, unit-probability AV, and normalization/store skeleton. They
were intentionally numerically invalid, never considered candidates, and were
restored byte-for-byte before the baseline-B bookends.

No Phase 9–17 local Matrix2 tuning, K/V-hot staging, old Q ownership, simple
multi-Q reuse, split-K, or sparse/windowed direction was reopened. Model 2 has
the same attention shape as Model 1, and the fresh quantitative ceilings do not
provide a new >=10% exact whole-request bound for those closed directions.

## Performance and guards

| Context | Baseline A | Restored baseline B | Production candidate | llama.cpp | Ratio |
|---:|---:|---:|---:|---:|---:|
| 128 | 275.22 ms | not repeated | none | not required | n/a |
| 512 | 1,063.53 ms | not repeated | none | not required | n/a |
| 2K | 4,329.99 ms | not repeated | none | not required | n/a |
| 8K | 18,640.39 ms | 18,562.54 ms | none | 5,196.07 ms | 3.58x bookended |
| 16K | 40,751.46 ms | not repeated | none | not required | n/a |
| 28K | 78,438.08 ms | 78,598.70 ms | none | 24,266.43 ms | 3.24x bookended |

Model 2 decode baseline is 22.16 tok/s at 128, 21.40 at 2K, and 15.90 at 28K.
There is no candidate delta. Model 1 guards are not triggered because final
production source is identical to Phase 24; its acquired <=0.3% guard matrix
remains the exact executable ancestry used here.

Likewise, no final candidate requires a new CPU/generic/optimized, F16/INT8,
Q4/Q6, Matrix2/GQA, logits, 128-step, or 128-token correctness qualification.
Phase 24's full matrix remains acquired evidence. Phase 25 changed no tolerance,
oracle, shader, route, allocation, or public API; its temporary invalid-output
stage probes were deleted and the production diff was empty before bookend B.
Because there is no KEEP and no >=10% improvement, the post-KEEP full-reprofile
trigger does not apply; the baseline attribution in this report is the final
Phase 25 breakdown.

## Verification

| Check | Result |
|---|---|
| `git diff --check` | pass |
| `cargo fmt --all -- --check` | pass |
| release workspace CPU check | pass |
| release workspace Vulkan-profile check | pass |
| CPU `family_agnostic::docs_contract` | pass |
| Vulkan profiler category tests | 3 passed |
| restored production profiler SHA-256 | exactly matches fresh baseline `62f396...4a46` |
| final production source diff | empty |

The initial and pre-documentation worktrees were clean. The only durable Phase
25 files are this report and its documentation-index entry; ignored evidence
contains the immutable binaries and raw measurement logs.

## Maturity and final stop

| Workload | GH/llama | Target | Status |
|---|---:|---:|---|
| short prefill, 8K decision point | 3.58x | <=1.5x | RED |
| long prefill, 28K | 3.24x | <=1.5–1.7x | RED |
| decode | unchanged from acquired Model 2 baseline | <=1.3–1.4x | RED |

The maturity contract is not reached, but further exact competitive tuning is
not economic on this RTX 3060. Raw competitive seconds remain large because
llama.cpp accepts numeric policies that Graph Horizon has deliberately not
adopted. The largest exact remaining bottleneck is attention QK; its practical
whole-request opportunity is below the coding gate.

Final decision: **STOP Phase 25; retain no production experiment.** The branch
contains evidence/documentation only. A future Phase 26 should qualify the next
model/product. Reopening Model 2 should require one of:

1. explicit quality approval for FP16 large-M accumulation or pre-scaled F16 Q;
2. new Vulkan/compiler support that accelerates post-MMA FP32 score scaling;
3. hardware counters or a materially different exact architecture that raises
   the predicted whole-request result above 10% without model-name routing.
