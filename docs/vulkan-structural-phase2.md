<!--
This report owns the Vulkan Phase 2 structural Q4-versus-attention investigation:
same-session baselines, practical floors, architecture projections, experiment
registry, retained decisions, and the resumable path to the long-prefill target.
-->

# Vulkan Structural Q4 vs Attention — Phase 2

## Investigation state

The investigation starts from clean production checkpoint `607b14c` and the
acquired competitive-gap evidence at `ebec03e`. The dedicated branch is
`perf/vulkan-q4-attention-floor-phase2`; nothing is pushed. The retained source
change is `36a73a2`, with its feature-scope correction at `ea7ca1a`.

The invariant is the current GGUF weights, exact dense causal attention, F16 KV,
unchanged FP16/FP32 accumulation choices, unchanged quality tolerances, and
capability/shape/quantization routing. Existing NVIDIA, AMD, decode, and generic
fallback paths remain protected.

## Fresh same-session endpoint

The 8B prompt is exactly 28,000 tokens in a 32,768-token F16-KV context. Graph
Horizon uses one warm-up and three measured requests; llama.cpp uses the acquired
matched Vulkan command with three repetitions.

| Endpoint | Mean | SD | CV |
|---|---:|---:|---:|
| Graph Horizon `607b14c` source | 52.103760 s | 25.38 ms | 0.0005 |
| llama.cpp `9bebfcb4`, build 9826 | 24.579387 s | 80.00 ms | 0.0033 |

The live ratio is 2.120x. The 1.7x target is 41.784958 s, requiring 10.318802 s
or 19.80%; the 1.5x target is 36.869081 s, requiring 15.234679 s or 29.24%.
The source-identical acquired attribution is scaled by 1.001959 to this public
denominator:

| Component | Current | Share |
|---|---:|---:|
| direct canonical Q4 | 21.957935 s | 42.14% |
| exact dense attention | 20.479043 s | 39.30% |
| staged canonical Q6 | 8.430484 s | 16.18% |
| other | 1.236418 s | 2.37% |

## Pre-prototype practical-floor ranking

| Frontier | Competitive-gap contribution | Best realistic floor | Removable | Confidence | Required architecture |
|---|---:|---:|---:|---|---|
| Q4 | about 10.11 s | 15.70 s | 6.258 s | medium-low | lower-resource Matrix2 ownership selected by workload shape, then compact execution representation only if feed remains dominant |
| attention | about 11.29 s | 15.334 s | 5.145 s | high | a new exact resource/dataflow mechanism; acquired Q64 local/tile family is closed |
| broader Q6 | about 5.98 s | about 5.00 s | about 3.430 s | low | replacement, not additive, execution-native representation plus a compatible decode path |

The Q4 floor does not assume generic FP16 weights. llama.cpp executes the same
canonical scalar Q4 callback on the same GPU in about 11.85 public-normalized
seconds, establishing that canonical bytes are not by themselves a 21.96-second
floor. Recovering roughly 62% of the observed Q4 difference gives the 15.70 s
realistic screen. This is deliberately less aggressive than comparator parity.

Attention's floor is the acquired same-shader stage-isolation floor scaled to
the live denominator. QK/AV tiled utilization already exceeded 99.7% in valid
historical probes; score/probability shared round trips and per-tile barriers
are already absent. Exact split-K adds partial-state traffic and had less than a
3% whole-request ceiling. No current exact defect invalidates that floor.

Q6 remains open globally but not economically first. Exact FP16 replacement of
the 1,069,547,520 recurrent values needs 1.992 GiB versus 0.817 GiB canonical.
Keeping canonical decode bytes would exceed the safe 12-GiB budget; replacement
requires a new decode path and has unresolved numeric/order risk. A favorable
40% local reduction is only about 3.43 seconds.

## Amdahl projections and target tracker

| Projection | Removable | Predicted TTFT | GH/llama |
|---|---:|---:|---:|
| Q4 only | 6.258 s | 45.846 s | 1.865x |
| attention only | 5.145 s | 46.959 s | 1.910x |
| Q4 plus attention | 11.403 s | 40.701 s | 1.656x |
| Q4 + attention + broader Q6 | 14.833 s | 37.270 s | 1.516x |

```text
8B / 28K

llama.cpp:                  24.579 s
1.7x target:                41.785 s
current:                    52.104 s
remaining reduction:        10.319 s

Q4 realistic removable:     6.258 s
attention removable:        5.145 s
Q6 broader removable:       3.430 s
Q4 + attention floor:      40.701 s
```

This was the mandatory ranking before implementation. Subsequent physical
screens showed that the Q4 estimate was optimistic; the final re-ranking is
recorded below.

## Q4 physical model

The recurrent 8B Q4 set contains 6,345,981,952 values. Canonical storage is
3.324 GiB: 2.955 GiB nibbles plus 0.369 GiB metadata. At 28K it performs 355.375
TFLOP and sustains only 16.18 effective TFLOP/s. The current logical model reads
about 728 GiB of canonical weights (219 traversals), about 893 GiB of
activations, and about 72 GiB of outputs. The one-copy physical minimum is only
3.324 GiB weights plus roughly 69 GiB activations and 72 GiB outputs, but the
comparator is faster despite *more* activation amplification. This closes a
pure bandwidth/wider-load explanation: Matrix feed, accumulator resources, and
work ownership dominate the transferable difference.

| Operation | M x N x K | GPU time | Quantization | Unique canonical weight | Current path |
|---|---|---:|---|---:|---|
| Q | 4096 x 28000 x 4096 | 2.536 s | Q4 all 34 layers | 0.299 GiB | direct Matrix2 |
| K | 1024 x 28000 x 4096 | 1.265 s | Q4 all 34 layers | 0.075 GiB | direct Matrix2 |
| V | 1024 x 28000 x 4096 | 1.228 s | Q4/Q6 split 17/17 | 0.037/0.055 GiB | direct Q4 / staged Q6 |
| O | 4096 x 28000 x 4096 | 2.537 s | Q4 all 34 layers | 0.299 GiB | direct Matrix2 |
| gate | 14336 x 28000 x 4096 | 5.304 s | Q4 all 34 layers | 1.046 GiB | direct Matrix2 |
| up | 14336 x 28000 x 4096 | 5.282 s | Q4 all 34 layers | 1.046 GiB | direct Matrix2 |
| down | 4096 x 28000 x 14336 | 12.235 s | Q4/Q6 split 17/17 | 0.523/0.762 GiB | direct Q4 / staged Q6 |

The Q4 kernel is WG256, M256/N128/K64, 256 output rows and 128 token rows per
workgroup, with an FP16 accumulator. It declares 16,512 bytes of scale/min shared
state and the driver reserves another 8,192 bytes. Current Vulkan interfaces do
not expose executed registers, spills, cache transactions, or Matrix counters
for this pipeline; no values are invented. Tiled arithmetic utilization is
effectively complete for aligned recurrent shapes.

Canonical Q4 decoding performs one nibble extraction, shared scale/min lookup,
FP32 scale/min arithmetic, and FP16 conversion per Matrix operand value. Metadata
is expanded once per K256/output-row tile and synchronized with two barriers.
The direct callback already amortizes one decoded weight across 128 token rows.
Exact FP16 expansion would remove this work but expands 3.555x and cannot fit
the full 8B recurrent set alongside the 4.25-GiB KV allocation. Expanded
metadata alone was historically worth only 2.31% of a smaller whole request.

## Comparator principle and prototype Q4-S01

llama.cpp uses the same canonical Q4 block, scalar callback, scale/min shared
staging, K64 Matrix feed, and FP16 accumulation. Its structural difference is a
three-shape Matrix2 policy: the medium path owns M128/N128 and the large path
M128/N256, whereas Graph Horizon always owns M256/N128. The prior M128/N256
screen retained the same 32K-element accumulator as production and doubled
activation ownership at a 128-token screen; it did not test the lower-resource
M128/N128 point. The M256/N256 screen doubled the accumulator and hit a severe
compiler/resource cliff.

Q4-S01 therefore tests one new mechanism, not a sweep: halve the live Matrix2
accumulator and declared scale state with M128/N128 while preserving canonical
bytes, K64, arithmetic order, callback, WG256, and every route. It doubles
activation traversals and workgroups, so it wins only if accumulator/codegen
pressure is the physical floor. The 8B/128 result decides that question before
a long run.

Required local structure before the production experiment:

```text
crates/graph_horizon_engine/src/backend/vulkan/
├── shaders/matmul/matmul_q4_k_matrix2_f16out.comp
│   (category K, about 210 productive lines): M128/N128 direct Q4 operation
└── kernels/coopmat.rs
    (about 115 productive lines): matching fixed Q4 dispatch grid
```

Invariant: exact canonical Q4 arithmetic and formats, FP16 accumulation/output,
all eligibility, decode, Q6, attention, fallback, and public interfaces. Smallest
change: two tile/grid constants. Main risk: doubled activation traversal and
workgroup count outweigh lower accumulator/shared pressure. The prototype adds
no files, dependency, persistent memory, threshold, or public API.

## Experiment registry

| ID | Frontier | Architecture | Current | Practical floor | Predicted removal | Predicted TTFT | Implementation | Result | Decision |
|---|---|---|---:|---:|---:|---:|---|---|---|
| B2-00 | global | fresh same-session endpoint | 52.104 s | n/a | n/a | n/a | evidence only | stable 2.120x | EVIDENCE |
| Q4-S01 | Q4 | lower-state M128/N128 Matrix2 ownership | 21.958 s | 15.700 s | 6.258 s | 45.846 s | two existing constants | 8B/128 +19.87% | REJECT |
| Q4-S02 | Q4 | direct metadata decode without shared state | 21.958 s | n/a | n/a | n/a | callback-only decode | 8B/2K +9.74% | REJECT |
| Q4-S03 | Q4 | compact persistent execution sidecar | 21.958 s | about 18.7 s | at most 3.3 s | at least 48.8 s | physical/economic model | below large-design gate | CLOSE |
| A-S01 | attention | acquired exact-dataflow floor | 20.479 s | 15.334 s | 5.145 s | 46.959 s | exact mechanism audit | no implementable delta | CLOSE |
| D-S01 | interaction | Matrix2-capable 512-row request ownership | 52.044 s | measured | 3.591 s | 48.453 s | one policy constant/gate | -6.90% at 28K | RETAIN |
| D-S02 | interaction | 1024-row request ownership | 48.453 s | n/a | below gate | n/a | one temporary constant | another -2.15% at 2K, -2.87% at 8K | REJECT |
| Q6-T01 | Q6 | transient exact Q6_K-to-F16 large-M reuse | 7.860 s | n/a | negative | n/a | two private prototype kernels plus reusable scratch | 8B/2K +3.58% | REJECT |

## Post-batch Q6-T01 structure checkpoint

The retained 512-row request ownership reduces the bookended 8B/28K endpoint
from 52.044165 s to 48.452850 s. The remaining 1.7x gap is 6.667892 s. After
the direct-Q4 representation screens and exact-attention dataflow audit closed,
the next distinct large-M reuse class is transient Q6_K expansion: dequantize
one layer tensor into reusable FP16 execution storage, consume it with a wide
Matrix2 matmul, and reuse the same allocation for the next tensor. This is not
the previously rejected persistent expanded-weight representation and does not
change GGUF ownership or decode.

Required local structure for the reversible prototype:

```text
crates/graph_horizon_engine/src/backend/vulkan/
├── shaders/matmul/dequant_q6_k_f16.comp
│   (category K, about 80 productive lines): exact Q6_K-to-F16 conversion
├── shaders/matmul/matmul_f16_matrix2.comp
│   (category K, about 100 productive lines): wide prefill F16 Matrix2 matmul
├── kernels/matmul/prefill/q6_transient.rs
│   (about 90 productive lines): paired conversion and matmul recording
├── kernels/matmul/prefill/mod.rs
│   (about 145 productive lines excluding tests): Q6 route ownership
├── pipeline/kernel.rs
│   (about 150 productive lines): two private kernel ABIs
├── pipeline/mod.rs
│   (about 190 productive lines): Matrix2-gated registration
├── init/setup.rs
│   (about 180 productive lines excluding tests): transactional scratch allocation
├── mem/memory.rs
│   (about 155 productive lines excluding tests): pure-backend preflight accounting
└── mod.rs
    (category I portions plus about 125 productive orchestration lines): scratch ownership
```

Invariant: canonical Q6_K weights remain authoritative; conversion is exact to
the same FP16 operand value, accumulation remains FP32, output remains FP16, decode and every
non-Matrix2 route are unchanged, and one checked allocation is reused only
between barrier-ordered dispatches in the same command buffer. The smallest
change is two private kernels, one format-specific recorder, and one scratch
allocation sized to `2 * embedding_length * feed_forward_length`. Main risks
are conversion/write traffic overwhelming the matrix savings and the extra
scratch violating the long-context VRAM reserve. The prototype passed all four
Q6 CPU-oracle families, including 14B V/down, but 8B/2K measured 2244.41 ms
versus the retained 2166.81 ms (+3.58%). Conversion, write, and expanded-weight
read traffic exceeded the activation-reuse benefit. All production files and
both prototype shaders were removed; only this negative result remains.

## Retained architecture and measured mechanism

The retained change raises request ownership from 256 to 512 rows only when
both the direct-Q4 Matrix2 and exact-Q64 attention pipelines are registered.
AMD remains at its qualified 128/64-row watchdog bounds; partial-capability and
portable fallback devices remain at 256. This changes the interaction between
Q4 and attention rather than either kernel's tile arithmetic.

| Quantity | 256 rows | 512 rows | Change |
|---|---:|---:|---:|
| 8B K-projection workgroups per dispatch | 8 | 16 | 2x launch occupancy |
| 8B Q/O workgroups per dispatch | 32 | 64 | 2x launch occupancy |
| Q64 attention workgroups per layer/batch | 128 | 256 | 2x launch occupancy |
| 28K full-graph submissions | 110 | 55 | -55 |
| request scratch at 8B | baseline | about +75 MiB | still inside qualified VRAM |
| model/KV bytes removed | 0 | 0 | arithmetic and formats unchanged |

The command graph contains 544 dispatches per batch. At 2K, Nsight Systems
observed 320 versus 316 total submissions including fixed load work, and 7189
versus 5013 pipeline-bind calls: the exact 2176 difference is four eliminated
batches times 544 dispatches. Extrapolated to 28K, 55 eliminated batches remove
29,920 dispatch records and their trailing barriers while doubling useful
workgroups in each remaining launch. No shader instruction is removed inside a
workgroup; the gain is launch occupancy, fewer graph-wide barriers/submissions,
and better cache opportunity across twice as many rows.

Instrumented 2K TTFT fell from 2426.61 to 2234.43 ms. The non-instrumented
qualified result fell from 2330.99 to 2166.81 ms (-7.04%), and the bookended
28K endpoint fell by 3.591315 s (-6.90%). A fit across 2K/8K/28K attributes
2.138 s of the 28K gain to a linear projection/MLP term and 1.453 s to a
quadratic attention/cache term; this is an inferred scaling model, not direct
per-kernel timestamps.

## Final endpoint and target tracker

The final 8B/28K mean is 48.452850 s (SD 63.95 ms, CV 0.0013). The matching
same-session llama.cpp mean is 24.579387 s, so the final ratio is 1.971x. The
1.7x endpoint is 41.784958 s; 6.667892 s remains. The retained change realizes
34.80% of the original 10.318802-second target reduction.

```text
llama.cpp:                  24.579387 s
1.7x target:                41.784958 s
baseline, bookended:        52.044165 s
final:                      48.452850 s
measured reduction:          3.591315 s
remaining to 1.7x:           6.667892 s
```

## Final prefill matrix

All Graph Horizon values are TTFT milliseconds. Baselines are the acquired or
same-session bookended 256-row build; llama.cpp uses Vulkan, flash attention,
F16 KV, batch 2048, and microbatch 512.

| Model | Prompt | Baseline | Final | Delta | llama.cpp | Final ratio |
|---|---:|---:|---:|---:|---:|---:|
| 3B | 128 | 76.29 | 76.21 | -0.10% | 42.02 | 1.814x |
| 3B | 512 | 240.13 | 228.42 | -4.88% | 128.79 | 1.774x |
| 3B | 2K | 1014.54 | 961.05 | -5.27% | 531.70 | 1.808x |
| 3B | 8K | 5078.92 | 4802.59 | -5.44% | 2555.15 | 1.880x |
| 3B | 16K | 12951.63 | 12161.48 | -6.10% | 6304.26 | 1.929x |
| 3B | 28K | 28972.89 | 27365.97 | -5.55% | 13731.69 | 1.993x |
| 8B | 128 | 149.30 | 149.44 | +0.09% | 98.43 | 1.518x |
| 8B | 512 | 566.24 | 528.31 | -6.70% | 289.22 | 1.827x |
| 8B | 2K | 2330.99 | 2166.81 | -7.04% | 1180.25 | 1.836x |
| 8B | 8K | 10720.60 | 9995.67 | -6.76% | 5303.49 | 1.885x |
| 8B | 16K | 25156.75 | 23295.37 | -7.40% | 12201.25 | 1.909x |
| 8B | 28K | 52044.17 | 48452.85 | -6.90% | 24579.39 | 1.971x |

The 14B/128 capacity guard also passed: 245.05 ms final versus 245.70 ms
baseline. This is a capability/shape policy, never a model-name route.

## Final decode matrix

The retained policy does not enter decode. Values are tokens/second; the 28K
measurement is a single long-context run, while 128 and 2K use the acquired
matrix.

| Model | KV | Baseline | Final | Delta |
|---|---:|---:|---:|---:|
| 8B | 128 | 34.63 | 35.02 | +1.13% |
| 8B | 2K | 33.42 | 33.59 | +0.51% |
| 8B | 28K | 23.72 | 23.82 | +0.42% |

## Final gap decomposition and re-ranking

Final component seconds are inferred by applying the measured linear/quadratic
fit to the fresh acquired attribution. They sum to within 59 ms of the measured
endpoint. Comparator components are the acquired stage attribution, normalized
to the fresh llama denominator.

| Component | Baseline gap | Final gap | Final seconds | Practical numeric floor | Remaining numeric ceiling |
|---|---:|---:|---:|---:|---:|
| Attention | 11.354 s | 9.901 s | 19.026 s | 15.334 s | 3.692 s, but no audited exact portable mechanism |
| Q4 | 10.208 s | 8.723 s | 20.473 s | about 17.2 s | at most 3.3 s, below the large-design gate |
| Q6 | 5.990 s | 5.420 s | 7.860 s | about 7.0 s | below 0.9 s after the distinct transient class regressed |
| Other | -0.028 s | -0.112 s | 1.152 s | about 1.15 s | none |

These numeric ceilings are not an executable combined plan. Q4's physical
feed probes bounded metadata/shared/barrier work at 199.77 ms and nibble/load/
unpack at 183.22 ms on 2K; exact FP16 replacement of selected Q/K/O tensors
costs about 1.72 GiB for at most 2.47 s at 28K, and compact selected
representations remain below roughly 2.5–3.3 s before decode integration.
Attention already has no global score/probability round trip or per-tile
barrier, is at 255 registers/thread with two workgroups/SM, and exact split-K,
cross-head GQA reuse, phased geometry, and KV-format alternatives are either
traffic-bounded or previously measured below the whole-request gate. Q6-T01
demonstrates that request-transient representation plus a wider FP32 Matrix2
feed is negative even before long context.

Reaching 1.7x now requires combining essentially all remaining optimistic
numeric ceilings, yet no portable implementation survives the resource,
traffic, memory, correctness, and >=10% large-architecture screens. The next
credible class is vendor-specific hardware specialization, a broader numerical
change, or a model-format/API change; each lies outside this mission's portable,
quality-preserving scope.

## Qualification and final state

The retained source passed:

- pure Vulkan workspace/all-target tests: app 166, engine 156 (5 ignored),
  family 5 (1 ignored), semantic 12 (1 ignored);
- Vulkan-hybrid workspace/all-target tests: app 166, engine 229 (4 ignored),
  family 6 (1 ignored), semantic 12 (1 ignored);
- warning-denied Clippy for pure Vulkan and Vulkan-hybrid, plus rustfmt;
- Q64 boundary tests at 63/64/65, 127/128/129, and 255/256/257, and the ignored
  long-context numeric qualification;
- exact greedy token sequence `2757,10637,2479,1636` for the known 8B prompt;
- 3B/8B prefill, 14B capacity, and 8B decode matrices above.

Rejected prototype production code is absent. The worktree retains only the
512-row capability policy and this report. Conceptual complexity increases by
one policy constant and a two-pipeline eligibility conjunction; it avoids a new
kernel, representation, allocation, format, dependency, or public interface.

**PORTABLE VULKAN STRUCTURAL FLOOR REACHED**
