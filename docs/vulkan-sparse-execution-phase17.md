<!--
This report owns Phase 17 evidence for execution efficiency of the fixed Phase
16 sparse-attention mask. It does not reopen mask selection, alter dense-default
semantics, or qualify a candidate before correctness and whole-request gates.
-->

# Vulkan Sparse Execution Phase 17

## Outcome

**STOP — physical execution wall.** The fixed Phase 16 mask cannot reach the
10% 28K whole-request gate on the qualified RTX 3060 implementation.

The exact Phase 16 prototype was reconstructed at commit `f593c5c` on
`perf/vulkan-sparse-execution-phase17`, based on clean Phase 16 evidence commit
`75624fd`. It reproduced 42,695.89 ms TTFT against 45,958.32 ms dense bookends,
or -7.10%. The same sparse layers realize only 68.25% of ideal pair-scaled
savings: measured attention is 3,836.46 ms versus a 2,284.76 ms mathematical
ideal in layers 19--25.

Two execution candidates failed. Ordered historical-global plus contiguous-
window traversal regressed 2.01%; interior/boundary mask specialization gained
only 0.40%, far below the required 3% additional screen. The independent cache-
scheduling direction has at most 0.56 seconds of impossible whole-path upside,
below the 1.33 seconds missing budget.

The decisive lower-bound diagnostic then removed **all required historical
global work** while retaining W4K on layers 19--25. This strictly cheaper,
semantically invalid path still measured 42,363.02 ms, 1,000.53 ms above the
41,362.49 ms gate derived from the reproduction bookends. Restoring any global
QK, softmax, and AV work can only increase it. Phase 17 therefore stops under
physical-wall condition A, not because another mask was tried.

The prototype and every diagnostic were removed by non-destructive revert
`777c67f`. Effective production source is again Phase 12 commit `6cb170f`;
dense exact remains default, no sparse preset exists, and nothing was pushed.

## Baseline and fixed semantics

The worktree started clean on `perf/vulkan-sparse-attention-phase16` at
`75624fd`. The Phase 16, Phase 15, and Phase 12 reports were read in full before
the first edit. Device/model invariants remain RTX 3060 12 GiB, driver 595.84,
authenticated `Ministral-3-3B-Instruct-2512-Q4_K_M.gguf`, F16 KV, 32,768
context capacity, greedy sampling, and opt-in Phase 12 native gate/up weights.

The Phase 17 sparse definition never changed:

| Field | Fixed value |
|---|---|
| Window | 4,096 causal tokens |
| Global structure | every 16th KV64 block |
| Sparse layers | 19--25 |
| Dense layers | 0--18 |
| Context threshold | 16,384 |
| Mask/index buffers | none; arithmetic only |
| Dense default / decode | unchanged |

The critical invariant is that QK and AV visit the same retained keys, global
blocks overlapping the window are processed once, and one FP32 online
`(max, sum, AV)` state spans all retained tiles in ascending key order.

Fresh diagnostics-free dense results were captured before any source edit:

| Context | TTFT mean | SD | CV | Prompt tok/s |
|---:|---:|---:|---:|---:|
| 128 | 96.67 ms | 0.66 ms | 0.69% | 1,324.21 |
| 512 | 367.95 ms | 0.97 ms | 0.26% | 1,391.52 |
| 2K | 1,565.70 ms | 2.18 ms | 0.14% | 1,308.04 |
| 8K | 7,927.40 ms | 21.21 ms | 0.27% | 1,033.38 |
| 16K | 20,412.14 ms | 33.62 ms | 0.16% | 802.66 |
| 28K | **46,048.77 ms** | **72.84 ms** | **0.16%** | **608.05** |

The mandatory reproduction used dense/reference/dense ordering:

| 28K row | TTFT mean | SD | CV |
|---|---:|---:|---:|
| Dense before | 45,847.97 ms | 70.64 ms | 0.15% |
| Phase 16 sparse | 42,695.89 ms | 96.90 ms | 0.23% |
| Dense after | 46,068.67 ms | 117.78 ms | 0.26% |
| Dense bookend mean | **45,958.32 ms** | — | — |

The -7.10% reproduction is 67 ms from Phase 16's acquired 42,628.83 ms and
therefore confirms the expected ~-7.2% regime. Acquired 8K/16K reference rows
remain 7,954.72/20,413.72 ms; the 8K route is dense and the 16K threshold has
no measurable saving. They were not rerun after the 28K physical-wall stop.

| Context | Dense | Phase 16 sparse | Optimized sparse | Delta versus dense |
|---:|---:|---:|---:|---:|
| 8K | 7,859.30 ms acquired | 7,954.72 ms | none retained | +1.21% reference |
| 16K | 20,413.99 ms acquired | 20,413.72 ms | none retained | -0.00% reference |
| 28K | 45,958.32 ms bookends | 42,695.89 ms | none retained | **-7.10% reference** |

The fresh Phase 17 harness reports mean/SD/CV but has no retained per-sample
logger. Phase 16's source-identical raw-sample medians were 7,859.41,
20,443.03, and 45,901.38 ms dense at 8K/16K/28K; its 28K sparse median was
42,656.76 ms. Median and mean therefore support the same decision.

## Missing performance budget

| Quantity | 28K TTFT |
|---|---:|
| Dense bookend mean | 45,958.32 ms |
| Phase 16 sparse | 42,695.89 ms |
| 10% target | **41,362.49 ms** |
| Current saving | 3,262.43 ms |
| Required saving | 4,595.83 ms |
| Remaining saving needed | **1,333.40 ms** |

Direct layer timestamps put current affected-layer attention at 3,836.46 ms.
It would have to fall to at most 2,503.06 ms, a **34.76% local reduction**, to
remove the diagnostics-free missing budget while all other work stayed fixed.

## Logical work and ideal sparse ceiling

Pairs count exact causal tokens. Global pairs exclude keys already retained by
the window. Values below are per attention head and per affected layer unless
the “seven layers” columns are used.

| Context | Dense pairs | Window pairs | Added global pairs | Retained pairs | Retained | Dense, seven layers | Retained, seven layers |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 8K | 33,558,528 | 25,167,872 | 647,296 | 25,815,168 | 76.926% | 234,909,696 | 180,706,176 |
| 16K | 134,225,920 | 58,722,304 | 5,087,616 | 63,809,920 | 47.539% | 939,581,440 | 446,669,440 |
| 28K | 392,014,000 | 106,301,440 | 18,580,224 | 124,881,664 | **31.856%** | 2,744,098,000 | 874,171,648 |

The direct 28K dense affected-layer attention is 7,172.04 ms. Pair scaling
therefore gives an ideal 2,284.76 ms fixed-mask affected time and 4,887.29 ms
ideal saving. That ideal is only 218.30 ms below the 2,503.06 ms requirement;
the mask has almost no practical execution margin.

## Realization and dense-versus-sparse layers

Cold-subtracted Vulkan profiling gives 46,329.46 ms dense GPU prefill with
26,535.30 ms attention. Reference sparse profiling gives 42,774.06 ms GPU
prefill with 23,135.66 ms attention. The latter divides into 19,299.20 ms for
unchanged layers 0--18 and 3,836.46 ms for sparse layers 19--25.

| Layer | Dense attention | Sparse attention | Logical pair reduction | Actual time reduction | Realization efficiency |
|---:|---:|---:|---:|---:|---:|
| 19 | 1,027.28 ms | 549.32 ms | 68.14% | 46.53% | 68.28% |
| 20 | 1,027.06 ms | 547.19 ms | 68.14% | 46.72% | 68.57% |
| 21 | 1,027.48 ms | 549.52 ms | 68.14% | 46.52% | 68.26% |
| 22 | 1,020.02 ms | 548.63 ms | 68.14% | 46.21% | 67.82% |
| 23 | 1,025.17 ms | 546.13 ms | 68.14% | 46.73% | 68.57% |
| 24 | 1,023.34 ms | 546.72 ms | 68.14% | 46.58% | 68.35% |
| 25 | 1,021.69 ms | 548.96 ms | 68.14% | 46.27% | 67.90% |
| **Total** | **7,172.04 ms** | **3,836.46 ms** | **68.14%** | **46.51%** | **68.25%** |

Realization efficiency is `actual saved / ideal pair-scaled saved`. The narrow
67.8--68.6% range proves that no single layer is an anomalous routing target.
Measured sparse affected time exceeds pair-scaled ideal by 1,551.70 ms.

## Sparse attribution and practical floor

The device exposes whole fused-attention timestamps but no instruction sample,
physical-byte, cache-hit, or in-shader timestamp counters. QK, softmax, AV, and
other seconds below therefore normalize the acquired Phase 10/15 causal shares
to the direct Phase 17 totals; they attribute 100% without pretending the
overlapped operations are independently timed.

| Component | Dense direct | Phase 16 sparse model | Fixed-mask model floor |
|---|---:|---:|---:|
| QK / Q+K load | 9,826 ms | 8,567 ms | 8,016 ms |
| Softmax / online state | 3,556 ms | 3,100 ms | 2,900 ms |
| AV / V load | 9,287 ms | 8,097 ms | 7,576 ms |
| Index, barriers, epilogue, other | 3,866 ms | 3,371 ms | 3,153 ms |
| **Attention** | **26,535 ms** | **23,136 ms** | **21,645 ms** |

The model floor keeps 53.85 ms of affected-layer fixed attention work and
scales remaining affected work by the 32.194% visited-tile fraction. It is
already tighter than a pure pair ideal, but it remains optimistic because it
assumes dense per-tile efficiency despite shorter sparse chains.

The decisive measured floor is stronger: W4K-only layers 19--25, with every
periodic global block removed, measured **42,363.02 ms** (65.21 ms SD, 0.15%
CV). It performs strictly less work than the fixed mask yet remains 1,000.53 ms
above the gate. The practical fixed-mask whole-request floor is therefore
**greater than 42,363.02 ms**, limiting gain to less than 7.82% against the
reproduction dense mean. This establishes physical-wall stop condition A.

## Traversal, Matrix2, and control

Counts below are per Q head and affected layer at 28K. Multiplying by 32 heads
and seven layers gives request-level affected work.

| Traversal quantity | Count |
|---|---:|
| Dense-range blocks examined | 191,844 |
| Retained blocks executed | 61,763 |
| Empty scan iterations | 130,081 / 67.81% |
| Window visits | 52,715 |
| Historical-global-only visits | 9,048 |
| Global/window overlaps | 3,356, processed once |
| Full tiles | 60,189 |
| Partial tiles | 1,574 / 2.55% |

Reference construction is an implicit arithmetic dense-range loop with one
workgroup-uniform retain branch. It has no dynamic list, metadata load, mask
buffer, or divergent window/global branch inside a workgroup. Across the seven
affected layers it examines 42,973,056 blocks and executes 13,834,912.

| Matrix2 quantity | Dense | Fixed sparse |
|---|---:|---:|
| Useful-lane utilization | >99.7% acquired | 98.728% |
| QK Matrix2 operations, whole request | 1,276,913,664 | 1,043,808,512 |
| AV Matrix2 operations, whole request | 159,614,208 | 130,476,064 |
| Internal barriers, whole request | 320,684,416 | 262,408,128 |

Sparse partial-tile waste is 1,608,960 of 126,490,624 executed score lanes per
head/layer, or 1.272%. Matrix2 utilization is not the missing 1.33-second
budget. Causal/window divergence is confined to partial boundary tiles; block
selection itself is workgroup uniform.

## Traffic and global reuse

Logical shader-requested bytes are not physical DRAM bytes. Dense K traffic is
2,435.52 GiB across 26 layers; the hybrid requests 1,990.91 GiB. V is identical,
so combined K+V falls from 4,871.04 to 3,981.81 GiB. Pair-minimum K is about
1,988.7 GiB, showing only a small tile-padding gap.

In affected layers alone, K falls from 655.72 to 211.10 GiB. Historical-global-
only K is 30.93 GiB and V another 30.93 GiB. Reading each of 28 global blocks
once per KV head and affected layer would be 24.5 MiB K, a 1,292.6x logical
reread factor. This does not imply 1,292x physical traffic: blocks are naturally
reused by nearby query-head workgroups and the device exposes no cache counter.

Even deleting the entire historical-global-only path would remove at most
14.65% of sparse affected execution, about 0.56 seconds under a favorable
linear bound. That is below the 1.33-second missing budget before retaining the
required global QK/softmax/AV work. Cache-oriented scheduling is rejected
before code on this upper bound; shared staging is not attempted, consistent
with the Phase 10 resource-heavy reuse wall.

## Resources and dispatch

The reference retains Q32/KV64, 256 threads, eight 32-lane subgroups, one
dispatch per attention call, one online state, and no global intermediate.
Shader-declared shared memory remains 20,864 bytes; mask/index/temporary/
persistent sparse bytes remain zero. Dispatch count is unchanged and skipped
blocks bypass their QK/softmax/AV barriers.

Dense executable evidence is 124 registers/thread, 38,272 bytes driver shared
per workgroup, two resident workgroups/SM, and 33.3% modeled occupancy. Exact
sparse registers, spills, and driver-reported shared memory were not exposed by
the retained profiler and are therefore not fabricated. Neither candidate
reached the 3% screen that would justify a temporary executable-properties
probe. The shader's unchanged accumulator/shared arrays make an occupancy
improvement implausible, while the measured lower bound makes it irrelevant to
the final gate.

## Experiments

| ID | Optimization | Attention / TTFT result | TTFT vs Phase 16 | TTFT vs dense | Decision |
|---|---|---:|---:|---:|---|
| P17-REF | exact Phase 16 dense-range arithmetic traversal | 3,836.46 ms affected; 42,695.89 ms TTFT | baseline | -7.10% | reproduce only |
| P17-A | ordered historical-global region then contiguous window | 43,244.36 ms TTFT | +851.51 ms / +2.01% | -5.91% | reject regression; removed |
| P17-C | full interior tiles, mask only oldest-window/causal boundary | 42,221.86 ms one-sample TTFT | -170.99 ms / -0.40% | about -8.13% | reject below 3%; removed |
| P17-B | cache-oriented global scheduling | <=0.56 s impossible upper | below missing budget | cannot reach 10% | reject before code |
| P17-FLOOR | remove every required historical global block | 42,363.02 ms | -332.87 ms / -0.78% | -7.82% | lower bound still fails; diagnostic removed |

P17-A preserved all 61,763 selected 28K blocks in identical ascending order at
8K/16K/28K. P17-C matched the Phase 16 retain predicate for all 126,490,624
executed 28K score lanes, plus the 8K/16K lanes. Their quality semantics were
therefore unchanged by construction; neither survived the performance screen.

## Quality, correctness, and guards

The reference sparse SPIR-V logic is the exact Phase 16 arithmetic prototype.
The acquired CPU/reference sparse versus Vulkan result remains maximum
attention error `1.818e-5` and projected-logit error `2.464e-6`, covering
causal, window, global-block, nonzero-base, and Ministral GQA boundaries.

| Quality metric | Phase 16 reference | Phase 17 optimized |
|---|---:|---:|
| Top-1 | 100% | no retained candidate |
| Top-5 | 92% | no retained candidate |
| Top-10 | 86% | no retained candidate |
| 32-token sequence agreement | 86.25% | no retained candidate |
| Retrieval | 3/3 | no retained candidate |
| Class | strong B | not applicable |

The W4K-only floor deliberately changes semantics and is never treated as a
quality candidate. No full quality suite is rerun after the physical-wall stop,
because no implementation can qualify and all production prototypes are
removed. No tolerance, model format, Q4/Q6 path, F16/INT8 KV path, or sampling
contract changed.

Short/dense and decode guards are restored by source identity. Pre-edit dense
128/512/2K is 96.67/367.95/1,565.70 ms. Final production source is byte-
identical to `6cb170f`, so the final delta is zero by construction. Decode
source never changed; the temporary Phase 16 reference's acquired 128/2K/28K
decode deltas were +0.28%/-1.11%/-0.26%, while final dense production returns
to the exact accepted implementation.

## Final decision and Phase 18

**STOP.** The fixed W4K + global KV64/16 + layers 19--25 mask has acceptable
class-B quality but insufficient execution economics. Empty-loop removal,
boundary specialization, and the complete global-cache upper bound cannot
close the gap. More decisively, removing the global semantics entirely still
does not reach 10%. Another inference-only mask search is not authorized.

The nearest more aggressive Phase 16 frontier points illustrate what model
adaptation would need to recover:

| Frontier point | Predicted 28K TTFT | Predicted gain | Observed quality signal | Adaptation gap |
|---|---:|---:|---|---|
| W4K, layers 19--25, no globals | about 40.87 s | 11.04% | class C; top-10 40%, sequence 68.75% | at least +40 top-10 points and +11.25 sequence points to class B |
| W2K, layers 20--25 | about 40.79 s | 11.23% | class C; top-10 60%, sequence 100% | at least +20 top-10 points plus top-1/retrieval qualification |

The recommended Phase 18 is **sparse-attention model adaptation feasibility**:
fine-tuning, distillation, or architecture-aware training that recovers class-B
rank/retrieval behavior under a mask materially cheaper than the fixed Phase 16
point. Backend optimization of the current mask is closed on this hardware.

## Final repository state

- branch: `perf/vulkan-sparse-execution-phase17`;
- baseline evidence: `75624fd`;
- reconstructed reference: `f593c5c`;
- prototype removal: `777c67f`;
- effective production source: `6cb170f`;
- final sparse implementation: none;
- default: dense exact;
- push/PR: none.

Final format, Clippy, release tests, source-identity, documentation-index, and
clean-worktree results are recorded by the final evidence commit and handoff.
