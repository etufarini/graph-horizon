<!--
This report owns Phase 16 evidence for Vulkan sparse/windowed attention:
the dense baseline, quality contract, reference frontier, candidate economics,
and any later production qualification or stop decision. It does not redefine
the model's default dense semantics.
-->

# Vulkan Sparse Attention Phase 16

## Status

Investigation complete with **STOP** on `perf/vulkan-sparse-attention-phase16`, created
from the required clean Phase 15 evidence commit `34f24c2`. Effective dense
production source is Phase 12 commit `6cb170f`. The sparse production prototype
was rejected and removed; nothing has been pushed.

Phase 16 changes the investigation contract, not the default model contract.
Sparse attention normalizes only over retained causal keys and is therefore a
semantic approximation. Dense Phase 12 remains the oracle and mandatory
fallback. Correctness against the sparse definition is kept separate from
quality divergence against dense model behavior.

## Acquired evidence and invariants

The Phase 12--15 reports and Phase 9--11 attention evidence were read in full
before any source edit. The acquired negative results are not reopened: local
tile tuning, Q/K/V hot-address variants, simple GQA reuse, shared/barrier
cleanup, Q ownership/staging, exact split-K, hierarchical exact merge, and
equivalent dense-exact decompositions remain closed.

The authenticated model is
`Ministral-3-3B-Instruct-2512-Q4_K_M.gguf`, 2,147,023,008 bytes, SHA-256
`9ed150d4367e68df0ac8e1540f6ddc65b42d0ee26378329d1ecbca60f93fc5f8`.
The device remains RTX 3060 12 GiB (`0x2487`), driver 595.84, F16 KV, greedy
sampling, context 32,768, and opt-in Phase 12 native gate/up weights.

The invariant for every simulated or retained sparse path is:

- no key later than its query is ever visible;
- QK and AV visit the same retained blocks;
- online softmax normalizes only over retained keys;
- no global score or mask buffer is materialized;
- dense is used for unselected layers, unsupported shapes/configurations,
  short contexts, explicit dense mode, and the oracle;
- decode remains the existing dense decode implementation, while generated
  sequences are compared because sparse prefill changes prompt KV states;
- no existing tolerance is widened.

## Fresh dense baseline

The diagnostics-free baseline was captured before any source edit. Short rows
use two warm-ups and 15/10 repetitions; 2K uses one warm-up and five
repetitions; 8K/16K/28K use one warm-up and three repetitions.

| Context | TTFT mean | SD | CV | Prompt tok/s |
|---:|---:|---:|---:|---:|
| 128 | 95.79 ms | 0.47 ms | 0.49% | 1,336.27 |
| 512 | 366.50 ms | 0.64 ms | 0.17% | 1,397.01 |
| 2K | 1,561.12 ms | 4.22 ms | 0.27% | 1,311.88 |
| 8K | 7,913.27 ms | 30.59 ms | 0.39% | 1,035.23 |
| 16K | 20,426.03 ms | 41.56 ms | 0.20% | 802.12 |
| 28K | **45,936.04 ms** | **132.40 ms** | **0.29%** | **609.55** |

Raw-sample median confirmation was captured on the restored dense route. These
runs are reported separately rather than silently mixing their means into the
pre-edit table:

| Context | TTFT mean | Median | SD | CV | Prompt tok/s |
|---:|---:|---:|---:|---:|---:|
| 128 | 95.99 ms | 95.90 ms | 0.43 ms | 0.45% | 1,333.49 |
| 512 | 365.52 ms | 365.12 ms | 1.69 ms | 0.46% | 1,400.77 |
| 2K | 1,545.78 ms | 1,545.35 ms | 2.46 ms | 0.16% | 1,324.90 |
| 8K | 7,859.30 ms | 7,859.41 ms | 36.45 ms | 0.46% | 1,042.35 |
| 16K | 20,413.99 ms | 20,443.03 ms | 59.40 ms | 0.29% | 802.59 |
| 28K | 45,886.81 ms | 45,901.38 ms | 36.95 ms | 0.08% | 610.20 |

Active diagnostics-free telemetry averages 99.58% GPU utilization, 1,912 MHz
core, 7,501 MHz memory, 167.43 W, 70.73 C, and reaches 8,677 MiB sampled
device usage. The session is stable enough to discriminate a 3% delta.
Profiler telemetry at 8K/16K/28K reaches 8,538 MiB in every case because the
32,768-token F16 KV allocation is fixed. Active averages are respectively
99.34/99.42/99.54% GPU utilization, 164.85/168.18/168.82 W, and
66.19/71.20/73.05 C.

The independent profiler uses one cold request plus a separate warm-up and
three measured requests. Stable time is `(pair - cold) / 3`; profiler wall is
not mixed with diagnostics-free TTFT.

| Context | GPU prefill | CPU fence wall | Attention | Seven matmuls | Other |
|---:|---:|---:|---:|---:|---:|
| 128 | 104.202 ms | 103.561 ms | 2.145 ms | 95.466 ms | 6.591 ms |
| 512 | 383.249 ms | 383.445 ms | 11.623 ms | 350.685 ms | 20.941 ms |
| 2K | 1,538.853 ms | 1,538.849 ms | 143.557 ms | 1,324.372 ms | 70.924 ms |
| 8K | 7,998.873 ms | 8,003.197 ms | 2,256.910 ms | 5,459.002 ms | 282.961 ms |
| 16K | 20,568.954 ms | 20,577.541 ms | 9,061.904 ms | 10,951.185 ms | 555.865 ms |
| 28K | **45,997.656 ms** | **46,011.357 ms** | **26,382.302 ms** | **18,677.592 ms** | **937.762 ms** |

At 28K, attention is 57.36%, matmul 40.61%, and other 2.04%; profiler coverage is 99.96% and GPU
gaps above 50 us total 1.851 ms across the four-request capture.

Dense attention subregions remain source-identical to Phase 15. Normalizing
its acquired causal shares to the fresh direct total gives approximately
9.770 s QK/load, 3.535 s softmax/state, 9.234 s AV/load, and 3.843 s other.

## Quality contract before sparse implementation

The existing accepted-path context is deliberately much tighter than a
semantic approximation: Phase 12's long F16 qualification reached maximum
attention/logit absolute errors of `1.526e-5`/`4.239e-6`; INT8 KV was
bit-exact in that qualification. Phase 14 demonstrates the danger of a narrow
control: its rejected INT8-weight path kept some repeated-prompt top-1 and
eight-token sequences while diverse prompts or 28K logits failed severely.

The Phase 16 quality record therefore measures, side by side:

- final-layer attention-output max/mean/RMSE error;
- final-layer residual max/mean/RMSE error;
- all-vocabulary final-logit max/mean/RMSE error;
- top-1 agreement and top-5/top-10 overlap;
- 32-token greedy agreement and first divergence for the frontier;
- 128-token generation on any candidate retained after the performance gate;
- error versus 2K/8K/16K/28K context;
- retrieval-marker success for far, middle, and recent placement.

The prompt battery uses repository prompt controls where applicable and adds
deterministically constructed long fixtures: natural language, code,
structured text, repeated text, heterogeneous text, and retrieval-like text
with the same marker placed near the beginning, middle, and recent history.
The inherited sequence `2757,10637,2479,1636` remains a regression control but
cannot qualify sparse semantics by itself.

The following classes are Phase 16 reporting categories, not pre-existing
project requirements. They are fixed before candidate results and do not alter
any test tolerance:

| Class | Aggregate operational definition |
|---|---|
| A — near dense | finite outputs; 100% top-1; mean top-5/top-10 overlap at least 95%; at least 95% greedy-token agreement with no divergence before token 16; every dense-passing retrieval case still passes |
| B — small degradation | finite outputs; top-1 at least 95%; mean top-5/top-10 overlap at least 80%; at least 80% greedy-token agreement; no position class systematically loses retrieval |
| C — material degradation | any aggregate below B while most outputs remain usable, or a repeatable long-range/position failure |
| D — unacceptable | NaN/Inf, causal/correctness failure, top-1 below 75%, greedy agreement below 50%, or broad retrieval collapse |

A production prototype requires class A or a strong class B whose exact
trade-off is explicitly reported, no correctness failure, and at least 10%
predicted 28K whole-request improvement. Dense remains default even then.
Existing accepted-path qualification will be rerun before these provisional
classes are applied to a final candidate.

The accepted dense qualifier was rerun after adding the diagnostic route. F16
remains within `1.526e-5` attention and `4.239e-6` projected-logit maximum
absolute error; INT8 remains bit-exact through the 2K, 8K, and 28K N32/N16
controls. Compiling the dense Matrix2 source independently before and after the
diagnostic edit produces byte-identical SPIR-V (`2ad442...`). The scalar
sparse-definition boundary control (causal window, sink, and periodic global
union) passes at maximum attention error `1.818e-5` and projected-logit error
`2.464e-6`.

## Retained-work and pre-code economics

For a causal window `W`, exact retained pairs are
`W(W+1)/2 + (N-W)W` when `W < N`, otherwise `N(N+1)/2`. Sink and periodic
global blocks count only their union with the causal window. Tile utilization
is mathematical retained pairs divided by executed Q32/KV64 Matrix2 lanes.

| Candidate | 8K retained | 16K retained | 28K retained | 28K Matrix2 utilization | Predicted 28K attention | Predicted TTFT gain | Quality risk | Complexity |
|---|---:|---:|---:|---:|---:|---:|---|---|
| Window 2K | 43.75% | 23.44% | 14.09% | 96.97% | 4.001 s | 48.72% | extreme long-range | low |
| Window 4K | 75.00% | 43.75% | 27.12% | 98.46% | 7.513 s | 41.08% | very high | low |
| Window 8K | 100.00% | 75.00% | 49.95% | 99.23% | 13.671 s | 27.67% | high | low |
| Window 16K | 100.00% | 100.00% | 82.79% | 99.61% | 22.526 s | 8.39% | medium | low; below production gate |
| Window 8K + 64 sinks | 100.00% | 75.39% | 50.28% | 99.23% | 13.759 s | 27.48% | high | low |
| Window 8K + 256 sinks | 100.00% | 76.54% | 51.24% | 99.25% | 14.018 s | 26.92% | high | low |
| Window 4K + every 16th KV64 block | 76.93% | 47.54% | 31.86% | 98.73% | 8.791 s | 38.30% | high, better history coverage | medium |
| Window 8K + every 32nd KV64 block | 100.00% | 75.97% | 51.68% | 99.26% | 14.136 s | 26.66% | medium/high | medium |

The prediction keeps 0.200 s attention fixed, scales the remaining 26.182 s
with retained pairs, and charges 3% of retained pair work for implicit sparse
indexing. Matmul and other request time remain unchanged. It is an Amdahl
screen, not a performance result. Mask storage and persistent/temporary sparse
state are zero because the proposed masks are implicit arithmetic.

Layer-selective W4K screening predicts 9.48% for 6/26 sparse layers, 11.06%
for 7/26, 20.54% for 13/26, and 30.02% for 19/26. Thus a six-layer group is
useful for sensitivity but cannot qualify alone; seven or more equal-cost
layers can cross the production engineering gate if quality supports them.

## Approved diagnostic structure before source changes

The first experiment is quality instrumentation, not a production sparse
kernel. The dense shader is compiled unchanged for normal routing; a test-only
variant of that same one-operation category-K source enables implicit block
masking. No public API, dependency, allocation, model format, default route,
or decode kernel changes.

```text
crates/graph_horizon_engine/
├── build.rs (~75 productive lines for Vulkan compilation; add one variant)
└── src/
    ├── backend/vulkan/
    │   ├── kernels/attention/mod.rs (~145 productive lines; dispatch plus test-only mask control)
    │   ├── pipeline/kernel.rs (~155 productive lines; test-only kernel ABI)
    │   ├── pipeline/mod.rs (~180 productive lines; test-only pipeline construction)
    │   └── shaders/attention/attention_prefill_matrix2.comp
    │       (category K; one attention operation, no line limit)
    └── family/mistral/
        ├── graph/prefill/encode.rs (~70 productive lines; test-only final-row capture excluded)
        ├── graph/prefill/mod.rs (~25 productive lines; test-only capture export excluded)
        └── mod.rs (~85 productive lines excluding tests; authenticated diagnostic stays in tests)
```

The invariant at the shader mutation is that a skipped block contributes to
neither QK nor AV, while every retained score participates in the same online
`(m,l,O)` update. The main implementation risk is an off-by-one mask at causal,
window, or block boundaries. A scalar sparse reference and explicit boundary
tests gate all full-model quality conclusions.

The simulated finalist clears the predeclared class-B screen and the 10%
predicted-gain gate, so the following local production-prototype structure is
approved before its first edit:

```text
crates/graph_horizon_engine/src/backend/vulkan/
├── kernels/attention/
│   ├── mod.rs (~175 productive lines; dispatch and dense fallback)
│   └── policy.rs (~75 productive lines; validated opt-in preset configuration)
├── pipeline/kernel.rs (~155 productive lines; sparse ABI becomes reachable)
├── pipeline/mod.rs (~180 productive lines; capability-gated sparse pipeline)
└── shaders/attention/attention_prefill_matrix2.comp
    (category K; dense and structured-sparse variants of one operation)
```

The new policy file is warranted because environment parsing and preset
validation are a separate responsibility and adding them to dispatch would
cross its 200-line orchestration limit. The production invariant is dense for
an absent/invalid/explicit-dense strategy, contexts below the configured
threshold, unselected layers, unsupported Matrix2 shapes, and every decode.
The smallest candidate is one opt-in `hybrid` strategy, not a general mask API.

## Experiment registry

The quality simulator exercised Q32/KV64 block masks inside the existing
Matrix2 operation. The following summaries aggregate the five 28K prompts;
rank denominators are prompt count times rank width and sequence denominators
are prompt count times 32. Diagnostic wall time is excluded from performance
claims.

### Pure-window frontier

| ID | 28K retained | Predicted TTFT gain | Logit RMSE | Top-1 | Top-10 | Sequence | Retrieval | Class / decision |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| P16-W2K | 14.09% | 48.72% | 2.581 | 1/5 | 3/50 | 16/160 | 0/3 | D, reject |
| P16-W4K | 27.12% | 41.08% | 2.556 | 0/5 | 4/50 | 21/160 | 1/3 | D, reject |
| P16-W8K | 49.95% | 27.67% | 2.615 | 0/5 | 2/50 | 15/160 | 0/3 | D, reject |
| P16-W16K | 82.79% | 8.39% | 2.456 | 1/5 | 13/50 | 14/160 | 0/3 | D and below speed gate, reject |

W2K and W4K are exactly dense through 2K, then both change the first token on
the 8K code fixture. At 16K, W2K/W4K/W8K all change top-1 and agree on 0/32
tokens. W8K/W16K are exactly dense only at contexts they fully cover. At 28K,
every approximate window is class D; even W16K loses all three retrieval
positions while retaining 82.79% of pairs. Pure sliding window is closed.

### Sink and periodic-global frontier

The first global screen uses the 16K structured prompt. All variants recover
top-1 and 9/10 top logits relative to the corresponding pure windows, proving
that a small structured history carries useful information. At 8/32 sequence
agreement, all remain class D under the predeclared below-50% rule.

| ID | Sparse structure | 16K retained | Logit RMSE | Top-5 | Top-10 | Sequence | Class / decision |
|---|---|---:|---:|---:|---:|---:|---|
| P16-W8-S64 | W8K + 64 sinks, all layers | 75.39% | 0.245 | 4/5 | 9/10 | 8/32, div. 3 | D, reject |
| P16-W8-S256 | W8K + 256 sinks, all layers | 76.54% | 0.206 | 4/5 | 9/10 | 8/32, div. 3 | D, reject |
| P16-W4-G16 | W4K + every 16th KV64 block, all layers | 47.54% | 0.257 | 4/5 | 9/10 | 8/32, div. 3 | D, hybrid input only |
| P16-W8-G32 | W8K + every 32nd KV64 block, all layers | 75.97% | 0.240 | 4/5 | 9/10 | 8/32, div. 3 | D, reject |

### Layer sensitivity

The W4K group study applies sparsity to one contiguous layer range at a time.
Affected seconds assume the measured 26.382 s attention total is distributed
uniformly only for screening; the quality result, not that approximation,
selects the next experiment.

| Layers | Affected attention | Predicted gain | Logit RMSE | Top-1 | Top-10 | Sequence | Class |
|---|---:|---:|---:|---:|---:|---:|---|
| 0--5 | 6.088 s | 9.48% | 0.810 | yes | 9/10 | 1/32, div. 1 | D |
| 6--12 | 7.103 s | 11.06% | 2.562 | no | 5/10 | 0/32, div. 0 | D |
| 13--19 | 7.103 s | 11.06% | 1.516 | no | 5/10 | 0/32, div. 0 | D |
| 20--25 | 6.088 s | 9.48% | 1.154 | yes | 6/10 | 32/32 | C; best group, below speed gate |

The uniform W4K all-layer point retains 43.75% of 16K pairs and is class D;
the late-layer-only point retains 87.02% across the model and preserves the
sequence. Layer selection therefore dominates uniform sparsity at comparable
semantic risk, but six late layers cannot meet the performance gate.

### Hybrid frontier

The smallest extensions around the late-layer group were evaluated rather
than sweeping arbitrary schedules. Retained work below is the 28K attention
pair fraction across all 26 layers; the 16K quality columns use the structured
prompt.

| ID | Sparse layers / structure | Dense layers | Retained work | Predicted 28K gain | Logit RMSE | Top-10 | Sequence | Class |
|---|---|---:|---:|---:|---:|---:|---:|---|
| P16-H-W4 | 19--25, W4K | 73.08% | 80.38% | 11.04% | 1.389 | 4/10 | 22/32, div. 18 | C |
| P16-H-W2 | 20--25, W2K | 76.92% | 80.17% | 11.23% | 1.178 | 6/10 | 32/32 | C |
| P16-H-S256 | 19--25, W4K + 256 sinks | 73.08% | 80.80% | 10.80% | 0.192 | 9/10 | 20/32, div. 20 | C |
| P16-H-G16 | 19--25, W4K + every 16th KV64 block | 73.08% | 81.65% | 10.30% | 0.202 | 8/10 | 32/32 | B, prototype |

P16-H-G16 is the only gate-crossing simulated point with class-B quality. Its
28K five-prompt aggregate is:

| Context | Attention RMSE | Layer RMSE | Logit RMSE | Top-1 | Top-5 | Top-10 | Sequence | Retrieval |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2K | 0 | 0 | 0 | 100% | 100% | 100% | 32/32 | n/a |
| 8K | 0 | 0 | 0 | 100% | 100% | 100% | 32/32 | n/a |
| 16K | 0.075 | 0.082 | 0.202 | 100% | 100% | 80% | 32/32 | n/a |
| 28K, five prompts | 0.098 | 0.142 | 0.247 | 100% | 92% | 86% | 138/160 | 3/3 |

At 28K, repetitive text agrees on 30/32 tokens but diverges at token 2;
heterogeneous, beginning-retrieval, and middle-retrieval prompts agree on
32/32; recent retrieval agrees on 12/32 and diverges at token 11. All dense and
candidate retrieval responses contain `ZXQ-7419`. This is strong class B, not
near-dense class A. The planned 128-token subset was not run because the
production prototype subsequently failed the mandatory performance gate.

## Rejected production prototype

The prototype was an explicit opt-in `hybrid` preset: dense below 16K, then
W4K plus every 16th KV64 historical block only in layers 19--25. Unselected
layers, unsupported shapes, invalid/absent configuration, explicit dense mode,
and all decode operations retained Phase 12 dense behavior.

| Kernel property | Prototype value |
|---|---|
| Q/K/V tile | Q32 / K64 / V64 |
| Workgroup | 256 threads; NVIDIA 32-lane subgroups |
| Sparse layers | 19--25 (7/26); 19/26 layers dense |
| Context threshold | 16,384 tokens |
| Matrix2 useful-lane utilization | 98.73% in visited 28K sparse tiles |
| Dispatches / barriers | unchanged dispatch count; skipped blocks bypass their QK/softmax/AV barriers |
| Shared memory | 20,864 bytes, unchanged from dense Matrix2 |
| Mask / index / temporary / persistent bytes | 0 / 0 / 0 / 0; arithmetic mask |
| Registers / occupancy | not captured; prototype failed TTFT gate before low-level qualification |

At 28K the exact tile traversal visits 32.19% as many K/V tiles in each sparse
layer. Logical K reads fall from 2,435.52 GiB dense across 26 layers to
1,990.91 GiB hybrid; V is identical, so combined logical K+V reads fall from
4,871.04 to 3,981.81 GiB. These are shader-requested bytes before cache effects,
not physical DRAM counters. Output/state traffic and the 20,864-byte shared
state are unchanged.

### Diagnostics-free performance

The 128 and 28K rows use mandatory dense-before / candidate / dense-after
ordering. The dense reference is the mean of those bookends. Other dense rows
are the raw-sample confirmation session; below-threshold candidate rows route
to the dense shader by construction.

| Context | Dense mean | Hybrid mean | Hybrid median | Delta | Route |
|---:|---:|---:|---:|---:|---|
| 128 | 96.62 ms | 96.28 ms | 96.21 ms | -0.35% | dense guard |
| 512 | 365.52 ms | 367.04 ms | 366.58 ms | +0.42% | dense guard |
| 2K | 1,545.78 ms | 1,553.60 ms | 1,554.22 ms | +0.51% | dense guard |
| 8K | 7,859.30 ms | 7,954.72 ms | 7,956.92 ms | +1.21% | dense guard |
| 16K | 20,413.99 ms | 20,413.72 ms | 20,421.42 ms | -0.00% | hybrid threshold |
| 28K | **45,946.92 ms** | **42,628.83 ms** | **42,656.76 ms** | **-7.22%** | hybrid |

The 28K dense bookends are 45,886.81 and 46,007.02 ms; hybrid CV is 0.13%.
Median comparison gives 7.19%, the same conclusion. The model predicted
21.646 s attention and 41.262 s TTFT (-10.30%). Holding measured dense
non-attention work fixed implies about 23.064 s candidate attention instead:
roughly 7.1% more time per retained pair than dense. Index/branch cost and lost
amortization consume 3.08 percentage points of whole-request gain. The 16K
flat result shows that the cost is worse before enough blocks can be skipped.

No production reprofile was triggered because actual 28K improvement is below
10%. Consequently there is no candidate QK/softmax/AV breakdown and no claim
that the bottleneck moved. The dense final bottleneck remains attention at
26.382 s (57.36%), ahead of matmul at 18.678 s (40.61%).

Public decode throughput for 128/2K/28K changes by +0.28%/-1.11%/-0.26% in the
same measurement sets. No decode source or routing changed; the isolated 2K
public-delta variance slightly misses the provisional 1% guard and is another
reason not to call the rejected prototype qualified.

## Correctness and memory gates

- The sparse Q32/KV64 Vulkan definition matches its scalar F16 CPU reference at
  maximum attention error `1.818e-5` and projected-logit error `2.464e-6`.
- The reference explicitly covers causal, window, sink, periodic-block, and
  nonzero-base boundaries with Ministral GQA (32 Q heads / 8 KV heads).
- Full-model 2K--28K runs remain deterministic and finite; QK and AV use the
  same visited blocks and softmax normalizes only those keys.
- Q4_K_M model weights and F16 KV are authenticated. INT8 KV stays on its dense
  kernel and the accepted long qualifier remains bit-exact. No Q6 artifact was
  available; because the prototype was rejected before final qualification,
  no unsupported Q6 claim is made.
- The arithmetic mask allocates no temporary or persistent memory. Peak VRAM
  therefore remains the dense 8,538 MiB profiler / 8,677 MiB benchmark sample.

Correctness and quality are separate: sparse CPU/Vulkan agreement proves that
the implementation computes the requested approximation, not that the
approximation preserves dense model behavior.

## Pareto frontier and decision

| Point | 28K retained attention work | Predicted gain | Actual gain | Quality | Decision |
|---|---:|---:|---:|---|---|
| Dense Phase 12 | 100% | 0% | baseline | exact oracle | keep default |
| W16K all layers | 82.79% | 8.39% | not prototyped | D | reject |
| W4K + G16 all layers | 31.86% | 38.30% | simulator only | D at 16K | reject |
| W4K + G16 layers 19--25 | 81.65% | 10.30% | 7.22% | strong B | reject |

**Final decision: STOP.** Three materially different families were evaluated:
pure windows, sink/periodic-global masks, and layer-selective hybrids. Uniform
schemes with large predicted gains hit a class-C/D quality wall. The only
strong class-B hybrid crossed the theoretical gate by just 0.30 percentage
points, then failed the actual 10% performance gate because structured-sparse
overhead erased the margin. It was removed in full. Production source remains
Phase 12 dense exact, with no sparse configuration, shader, pipeline, test hook,
public API, dependency, or default semantic change retained.

The next meaningful step is model adaptation rather than another inference-only
mask permutation: window-aware fine-tuning, architecture-aware retraining,
distillation into a sparse-attention model, a model designed for economical
long context, or context-compression/speculative approaches. None is part of
this phase. Phase 17 should not reopen dense-exact attention or add a more
expensive selector without a new quantitative model that clears both the
quality and 10% whole-request gates.
