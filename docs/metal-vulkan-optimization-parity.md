<!--
This report is the persistent checkpoint for Vulkan-to-Metal optimization
parity. It owns capability mapping, current Metal measurements, experiment
ranking, retained/rejected decisions, and global-stop evidence; it defines no
runtime API or backend support contract.
-->

# Metal / Vulkan optimization parity

## Mission start

| Field | Value |
|---|---|
| Date | 2026-08-20 Europe/Rome |
| Baseline branch | `main` |
| Baseline SHA | `5e4e830ad1da3ea2ae462f1aad91cf7d82cac374` |
| Merge-base with `main` | `5e4e830ad1da3ea2ae462f1aad91cf7d82cac374` |
| Initial worktree | clean |
| Experiment branch | `perf/metal-vulkan-parity` |
| Repository publication | mission closed locally; branch tip `8deb734` was later pushed and merged into `main` by PR #34 (`87e3e4b`) |

The acquired Metal specialization checkpoint is `0c6b13d`. Later production
changes alter only Metal loader policy plumbing, not Metal shaders or numeric
routing. The fresh 128/512/2K baseline below reproduces that checkpoint, so its
stage-sampled attribution remains applicable until a changed path is measured.

The authenticated initial artifact is `3b-instruct`, 2,147,023,008 bytes,
SHA-256 `9ed150d4367e68df0ac8e1540f6ddc65b42d0ee26378329d1ecbca60f93fc5f8`.

## Acquired history

Retained Vulkan work includes direct canonical Q4 Matrix2 prefill, qualified
Q6 Matrix2 prefill, DP4A Q4 decode, vectorized 4:1 GQA decode, tiled/Matrix2
attention, batched RoPE, final-batch-only logits, retained request KV, larger
qualified prefill submissions, and AMD-specific watchdog/subgroup routes.

Retained Metal work includes direct canonical packed Q4/Q6 decode and prefill,
SIMD-matrix prefill, tiled 4:1 GQA prefill from base zero, segmented and wide
long-context decode, online-softmax attention, shared persistent model buffers,
and capability-based device admission.

Do not repeat the rejected Metal experiments without new evidence: eight- and
sixteen-SIMD projection occupancy variants, FP16 result staging with FP32 SIMD
accumulators, eight-query attention tile reuse, and wide prefill attention by
locally doubling SIMD groups. The rejected Vulkan representation, ownership,
and attention experiments in the existing reports are also acquired evidence.

## Metal device characterization

Device name is reporting-only. Production routing must remain based on
capabilities, formats, resources, and shapes.

| Capability | Value | Evidence class |
|---|---:|---|
| GPU | Apple M4, 10 GPU cores | queried |
| OS | macOS 26.3, build 25D125 | queried |
| Metal | Metal 4; compiler 32023.864 | queried |
| GPU families | Apple7/8/9, Common3, Metal4 | previously queried |
| Unified memory | yes | previously queried |
| SIMD width | 32 | queried from compiled pipeline |
| Max threads/threadgroup | 1,024 | queried |
| Max threadgroup memory | 32,768 bytes | queried |
| Recommended working set | 19,069,665,280 bytes | queried |
| Max buffer length | 14,302,248,960 bytes | queried |
| SIMD-group reductions | supported and exercised | queried / measured |
| SIMD 8x8 matrix multiply | supported and exercised | derived / measured |
| BF16 | supported | queried by comparator |
| Tensor acceleration | unavailable on this pre-M5 GPU | queried by comparator |
| Buffer-view alignment used by runtime | 4 bytes | current contract |
| Occupancy/registers/spills | unavailable without usable capture | unavailable |
| Cache/ALU/stall counters | unavailable without usable capture | unavailable |

## Current execution and eligibility

Each Metal graph chunk creates one command buffer and compute encoder, records
the complete chunk, commits, and waits. Pipelines and model weights are retained
for model lifetime. All current buffers use shared storage on unified memory.

| Operation | Representative role / shape | Expected Metal path | Actual path and predicate | Fallback reason |
|---|---|---|---|---|
| Q4/Q6 decode GEMV | projection, `M=1`, aligned K | packed SIMD decode | two 32-lane groups, four output rows/group, pure Metal | mixed placement uses one-group route |
| Q4/Q6 prefill GEMM | projection/MLP, `M=2..32`, K % 256, N % 32 | packed SIMD-matrix | four SIMD groups, K32, 32 prompt rows x 64 outputs | row 1, mixed, F16/Q5, or unaligned shape |
| Q4/Q6 wide prefill GEMM | projection/MLP, `M=33..64`, K % 256, N % 32 | packed SIMD-matrix with doubled weight reuse | P03 uses eight SIMD groups, K32, 64 prompt rows x 64 outputs | mixed, F16/Q5, or unaligned shape |
| Q4/Q6 prefill above 64 rows | same tensor roles | repeated qualified tiles in one request | unavailable: pure-Metal request capacity is 64 | larger tile exceeds current 32 KiB resource design |
| F16/Q5 prefill | projection/MLP | portable correct path | per-row scalar projection | no cooperative implementation |
| 4:1 GQA F16 prefill | head 128, 64 rows | SIMD-matrix QK/PV with K/V reuse | P09 one 1,024-thread group/KV head; 32-row mode 4 fallback | partial rows, INT8, mixed, other GQA/dim/resource limits |
| other long prefill | base >=512, head 128 | segmented online softmax | mode 3, two SIMD groups/head | unqualified dimension or shorter context |
| short prefill attention | non-mode-4 tuple | one SIMD group/head | mode 1 | shape/format not qualified |
| F16 long decode | row 1, base >=1023, head 128, exact 4:1 GQA | P05d grouped split/reduce | four disjoint history splits/KV head, shared K/V, checked logits scratch | shorter context / mixed / other dim/GQA uses existing modes |
| INT8 long decode | row 1, base >=1023, head 128 | segmented | mode 3, two SIMD groups/head | no wide INT8 qualification |
| medium decode | row 1, base >=511, head 128 | four lane-octet segments | mode 2 | shorter or other dimension |
| RoPE prefill | Q and K rows | batched role dispatch | default trait emits two dispatches per row | Metal has no batched override |
| graph tail/logits | final prompt row only | one norm and logits projection | selected by shared graph on last chunk only | none |
| request KV | K/V for full configured context | retained capacity | P01 retains one successful pure-Metal session allocation | none for pure Metal; hybrid remains request-local |

## Mandatory parity matrix

Status means the same optimization principle is active and qualified, not that
similarly named source exists.

| Capability | Vulkan implementation / eligibility | Metal implementation / eligibility | Representation | Workloads | Metal status | Impact / priority |
|---|---|---|---|---|---|---|
| Quantized Q4 prefill | direct packed Matrix2 on capability + measured shapes; portable/AMD fallbacks | direct packed SIMD-matrix for pure Metal, rows 2..64, K % 256, N % 32 | canonical Q4_K | all prefill projections | PARITY principle; UNDERPERFORMS | P03 improves request TTFT 18.17% at 512 |
| Quantized Q6 prefill | packed Matrix2/coop/portable by capability and shape | same packed SIMD-matrix route as Q4 | canonical Q6_K | V/down mixed rows | PARITY principle; UNDERPERFORMS | included in dominant matrix family; high |
| Short/long prefill ownership | 256 rows portable, 512 when Q4 Matrix2 + Q64 attention, AMD 128/64 | P03 qualifies 64 rows for pure Metal | FP16 intermediates | medium/long prompts | PARTIAL PARITY | 16.07% at 2K and 9.46% at 8K; larger rows remain resource-bound |
| Weight-tile reuse | Matrix paths decode once and reuse across 128/256 prompt rows | P03 decodes each K32 tile once for 64 prompt rows | packed canonical blocks | Q/K/V/O/MLP | METAL UNDERPERFORMS | doubled from 32; further doubling does not fit current result tile |
| Decode GEMV | Q4 DP4A where qualified; exact per-format fallbacks | direct packed Q4/Q6 SIMD decode | canonical Q4_K/Q6_K | all decode projections/logits | METAL HAS BETTER ALTERNATIVE | short decode within 1.22x comparator on 3B; closed pending reprofile |
| GQA decode | dedicated 4:1 split/reduce reuses K/V across query heads | P05d four-split 4:1 F16 route at base >=1,023; existing per-head fallbacks | F16/INT8 KV | especially long decode | PARITY principle for qualified F16 | +9.67% at 2K, +25.40% at 8K, +45.63% at 28K vs P05c |
| Attention prefill | exact fused online softmax; Q32/Q64 Matrix2 and portable tiles | P09 SIMD-matrix Q64 plus fused mode-4/segmented fallbacks | F16/INT8 KV | all prefill | PARITY for qualified F16; PARTIAL other shapes | P09 TTFT -29.35% at 8K and -36.61% at 28K |
| Attention decode | generic plus vectorized GQA split/reduce | P05d split/reduce with shared K/V for qualified 4:1 F16; segmented fallback | F16/INT8 KV | decode | PARITY for qualified F16; PARTIAL other shapes | retained; P05e eight-split scaling closed below gate |
| Persistent attention state | Q64/other fused paths avoid global score/probability tensors | every Metal mode retains online-softmax accumulator/state | registers/threadgroup state | prefill/decode | PARITY | no rewrite |
| Reduced attention traffic/barriers | Q64 avoids global intermediates and per-KV-tile barriers | P09 loads each K/V tile once per KV head/Q64 and retains fused online state | fused | prefill | PARITY for qualified F16 | larger tile exceeds 32 KiB; closed |
| RoPE batching | one dispatch per role/context segment | two dispatches per prompt row through default trait | in-place FP16 Q/K | prefill | MISSING; ECONOMICALLY CLOSED | P02 gained 0.74%/1.88% at 128/512, below gate |
| Final-only logits | shared graph emits tail only on final prompt chunk | same shared graph | FP32 logits | TTFT | PARITY | closed |
| KV append | one batched append per layer/chunk | one batched append per layer/chunk | F16/INT8 KV | prefill | PARITY | prior Metal attribution negligible |
| Request KV retention | single-slot allocation retained; optional exact keyed prefix | P01 enables the same single-slot ownership for pure Metal | backend-native KV buffers | repeated requests / TTFT | PARITY | removes 104--110 ms/request at 3B/context-32768 |
| Prefix reuse | exact caller key + token prefix | P01 enables the same exact key/token-prefix path | KV state | repeated chats | PARITY principle | shared cache tests pass; endpoint gain depends on common prefix |
| Weight loading | persistent device-local canonical blocks | one model-lifetime copy into shared canonical buffers | Q4_K/Q5_K/Q6_K/F16 | model load | METAL HAS BETTER ARCHITECTURAL FIT | unified-memory hot path shows no transfer term; closed |
| Direct canonical Q4 | direct packed callback; no expansion | direct `q4`/`q4x16`; no expansion | canonical Q4_K | prefill/decode | PARITY | closed |
| Direct canonical Q6 | packed routes; staged alternatives rejected where inferior | direct `q6`/`q6x16`; no expansion | canonical Q6_K | prefill/decode | PARITY | audit independently in measurements |
| Packed loads/dequant | packed scalar/vector loads and tile decode | packed byte/ushort loads, per-tile decode into half matrices | canonical blocks | quantized matmul | PARITY principle | local maturity still measured by matrix time |
| Intermediate tensors | fused attention and fused SiLU/mul; no N x N scores | same; FP16 scratch retained per request chunk | FP16 scratch | all | PARITY | no material missing conversion found |
| Command orchestration | one synchronous submit per chunk; larger qualified chunks | one synchronous submit per 64 pure-Metal rows | backend-native commands | prefill/decode | PARTIAL PARITY | P03 halves prefill submissions; Vulkan remains wider |
| Capability routing | features/vendor/shape/format with fallbacks | Apple9/unified memory/resource gate, pipeline width, shape/format/placement | n/a | all | PARTIAL PARITY | 64-row eligibility active; RoPE batching economically closed |

## Scorecard

| Capability | Vulkan | Metal | Status |
|---|---:|---:|---|
| Q4 packed execution | yes | yes | parity principle active |
| Q6 packed execution | yes | yes | parity principle active |
| weight-tile reuse | yes | yes, 64 rows | partial parity after P03 |
| direct canonical Q4 | yes | yes | parity |
| short-prefill specialization | yes | 64-row tile | partial |
| long-prefill specialization | yes | 64-row graph plus SIMD-matrix attention | parity principle after P09 |
| attention traffic optimization | yes | yes, Q64 matrix route | parity principle after P09 |
| persistent attention state | yes | yes | parity |
| GQA/KV reuse prefill | yes | yes | parity |
| GQA/KV reuse decode | yes | yes, qualified 4:1 F16 | parity principle after P05d |
| vectorized decode | yes | yes | partial parity |
| batched RoPE | yes | no | P02 economically closed |
| final-only logits | yes | yes | parity |
| resource retention | yes | model and request KV | parity after P01 |
| shape/capability routing | yes | partial | 64-row eligibility active |

## Fresh Metal baseline

Build: locked Cargo release, pure Metal, full placement, F16 KV, 32,768-token
allocation, one warm-up and three measured repetitions. The prompt is calibrated
to the exact shown token count and 32 completion tokens are requested.

| Model | Prompt | TTFT mean | TTFT CV | Prompt tok/s | Decode tok/s | Status |
|---|---:|---:|---:|---:|---:|---|
| 3B Instruct | 128 | 700.07 ms | 0.32% | 182.84 | 29.94 | fresh |
| 3B Instruct | 512 | 2,569.41 ms | 0.27% | 199.27 | 24.28 | fresh |
| 3B Instruct | 2,048 | 11,339.85 ms | 0.15% | 180.60 | 25.34 | fresh |
| 3B Instruct | 8,192 | 77,048.23 ms | 4.35% | 106.46 | 11.73 | fresh; thermally noisy |
| 3B Instruct | 28,000 | 587,278.11 ms | 0.15% | 47.68 | 4.33 | fresh |

The acquired P00 diagnostics account for 97.7% of 512 and 97.6% of 2K GPU
command time. Before P03, 512 quantized matrices consumed 2,294.89 ms and 92.4%
of sampled kernel time; attention consumed 123.57 ms. At 8K, attention was
31.98 s (45.8%), quantized matrices 36.65 s (52.5%), and command residual
1.34 s. P04 below supersedes those shares for the retained P03 endpoint.

### Static command audit

For a complete 32-row 3B prefill chunk, current source records 2,060 Metal
dispatches: 32 embeddings plus 26 layers times `(14 + 2 * 32)` operations.
RoPE alone is 1,664 dispatches, or 80.8% of the chunk's dispatch count. The
counts below exclude decode and the two final-tail dispatches.

| Prompt | Prefill command buffers / encoders | Current RoPE dispatches | Batched-role RoPE | Dispatches removable |
|---:|---:|---:|---:|---:|
| 128 | 4 | 6,656 | 208 | 6,448 |
| 512 | 16 | 26,624 | 832 | 25,792 |
| 2,048 | 64 | 106,496 | 3,328 | 103,168 |
| 8,192 | 256 | 425,984 | 13,312 | 412,672 |
| 28,000 | 875 | 1,456,000 | 45,500 | 1,410,500 |

This is a static recording count, not a timing claim. It establishes that
batched RoPE removes real command encoding and dispatch work rather than merely
renaming a kernel. GPU and end-to-end A/B measurements still decide retention.

## Initial Amdahl ranking

No candidate was retained from this table alone. P01 measurements now replace
the allocation bound; RoPE timing remains to be measured.

| Rank | Candidate | Affected fraction / evidence | Realistic local speedup | Predicted removable | Decision gate |
|---:|---|---|---:|---:|---|
| 1 | retain Metal request KV and exact keyed prefix | 15.54% at 128; 4.05% at 512 | allocation-only elimination | 104--110 ms/request | RETAINED as P01 |
| 2 | batched Metal RoPE | 2 x rows x layers dispatches become about 2 x layers | measured 47 ms at 512 | 1.88% at 512; about 1.5% stable 2K signal | REJECTED as P02 |
| 3 | 64-row Metal tile reuse and request ownership | 92.4% matrix share at 512 plus half as many weight reads/submits | measured 1.22x at 512 | 0.45 s at 512; 7.62 s at 8K | RETAINED as P03 |
| 4 | vectorized 4:1 GQA Metal decode | attention was 89.44% of 28K decode kernel time | measured split/reduce sequence | decode rises from 4.26 to 11.65 tok/s at 28K across retained steps | RETAINED through P05d |
| 5 | new long-prefill attention architecture | 45.8% at 8K and grows quadratically | prior local reuse only 1.09x attention | prior request result 2.15% | closed until distinct mechanism clears 10% request gate |

## Experiment registry

| ID | Target | Intentional variable | Correctness gate | Result | Decision |
|---|---|---|---|---|---|
| P00 | baseline | current production source | existing Metal suites | full 128--28K matrix above; every TTFT CV below 5% | evidence |
| P01 | request KV lifetime | existing homogeneous cache enabled for pure Metal | 138 unit + 17 integration tests; repeated real generation | 128 -15.54%; 512 -4.05%; no decode regression | RETAINED |
| P02 | RoPE dispatch batching | rows per Q/K dispatch | byte-exact old Metal route + CPU oracle; 139 unit + 17 integration tests | 128 -0.74%; 512 -1.88%; stable 2K signal about -1.5% | REJECTED below gate |
| P03 | 64-row projection/graph tile | weight reuse and pure-Metal row ownership | Q4/Q6 3/32/64-row oracles; 64-row attention oracle; full Metal suite | 128 -18.39%; 512 -18.17%; 2K -16.07%; 8K -9.46% | RETAINED |
| P04 | post-P03 stage attribution | timestamp instrumentation only | phase totals, category sums, diagnostics-free timing authority | required 128/2K/28K points captured; 90.96--99.63% kernel coverage plus measured command residual | EVIDENCE; profiler reverted |
| P05a | grouped GQA decode, 128 threads | four query heads reuse one staged KV tile | 1,024/2,048 GPU serial oracle; route predicate | 128 control flat; 2K decode -16.00% | REJECTED; insufficient history parallelism |
| P05b | grouped GQA decode, 256 threads | eight history streams/head plus four-head KV reuse | serial GPU oracle; full Metal suite; all six real models | 2K +10.59%; 8K +22.56%; 28K +31.92% decode | RETAINED |
| P06 | post-P05b attribution | timestamp instrumentation only | phase totals and explicit grouped-pipeline classification | attention -26.4% at 2K/28K; 97.63% 28K decode coverage | EVIDENCE; profiler reverted |
| P05c | grouped split/reduce decode | restore sixteen history streams/head | serial GPU oracle; full Metal suite; all six real models | vs P05b: 2K +13.02%; 8K +29.74%; 28K +41.34% | RETAINED |
| P07 | post-P05c attribution | timestamp instrumentation only | split and reduce both classified as attention | 2K attention 211.97 ms; 28K 2,958.31 ms; 96.59% coverage | EVIDENCE; profiler reverted |
| P05d | four disjoint history splits | thirty-two aggregate streams/head without duplicate KV reads | serial GPU oracle; replicated 2K/8K; 28K endpoint | vs P05c: 2K +9.67%; 8K +25.40%; 28K +45.63% | RETAINED |
| P05e | eight disjoint history splits | match Vulkan split count; sixty-four aggregate streams/head | serial GPU oracle; P05d A/B/A at 2K/8K; 28K endpoint | vs P05d: 2K -0.82%; 8K +2.51%; 28K -1.29% | REJECTED below gate |
| P08 | post-P05d attribution | timestamp instrumentation only | split/reduce classified as attention | attention 127.48 ms at 2K and 1,890.35 ms at 28K | EVIDENCE; profiler reverted |
| P09 | SIMD-matrix Q64 prefill attention | matrix QK/PV and once-per-KV-head tile loading | 64-row serial GPU oracle; full Metal suite; all six real models | vs P05d TTFT: 2K -7.66%; 8K -29.35%; 28K -36.61% | RETAINED |
| P10 | final retained-source attribution | timestamp instrumentation only | matrix pipeline explicitly classified as prefill attention | attention -39.40% at 2K and -44.37% at 28K vs P08 | EVIDENCE; profiler reverted |

### P01 structure checkpoint: Metal request KV retention

Target: repeated-request TTFT. Intentional variable: allow the existing
single-slot homogeneous session cache to own pure Metal KV state, exactly as it
already owns pure Vulkan state. No kernel, arithmetic, KV layout, allocation
size, first-request path, hybrid placement, dependency, or public API changes.

The local change uses only existing files:

```text
crates/graph_horizon_engine/src/
├── backend/selection.rs                         (~175 productive lines)
├── runtime/homogeneous.rs                       (~100 productive lines)
├── kv_cache/mod.rs                              (~180 productive lines)
├── family/mistral/mod.rs                        (~130 productive lines)
├── family/mistral/generation/mod.rs             (~100 productive lines)
├── family/mistral/generation/cache.rs           (~105 productive lines)
└── api/engine.rs                                (~145 productive lines)
```

Every orchestration file stays below 200 productive lines. The invariant is
that a new request overwrites all reachable causal positions from zero unless
an exact caller-keyed prefix is explicitly reusable; stale tail bytes remain
unreachable. Failure invalidates and frees the slot, and model drop frees it
exactly once. The smallest viable change is compile-time availability for pure
Metal. The main risk is retaining the already-budgeted KV capacity longer than
before or freeing it twice across a failed request; current ownership tests,
Metal allocation tests, sequential generation, and real parity cover that
boundary.

For 3B/F16/context-32768 the two KV buffers total 3,489,660,928 bytes
(`2 * 26 * 32768 * 8 * 128 * 2`). The baseline pays two shared-buffer creation
and destruction operations per request. P01 removes that repeated term after
the first successful request; measured A/B determines its fraction rather than
assuming allocation time from byte size.

P01 changed 36 lines and removed 33 across the seven existing files. Pure
Metal engine tests pass (138 unit, 17 non-ignored integration), as does the full
Metal workspace check. The real-model screen used the frozen P00 binary as both
A bookends and the isolated P01 binary as B:

| Prompt | Baseline A1 | P01 B | Baseline A2 | B vs A mean | Candidate CV | Decode control |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 720.11 ms | 597.86 ms | 695.69 ms | -15.54% | 0.54% | +0.49% |
| 512 | 2,568.62 ms | 2,461.29 ms | 2,561.89 ms | -4.05% | 0.42% | +0.73% |
| 2,048 | 11,394.31 ms | 11,549.94 ms | 12,142.31 ms | -1.86% | 2.12% | +0.10% |

The 2K bookends show a 6.56% thermal trajectory, so that row is a regression
control rather than the allocation-cost estimate. The stable 128/512 brackets
isolate a 104--110 ms fixed term; their mean, 107.00 ms, predicts only 0.94% at
the 2K P00 endpoint. The candidate therefore satisfies the short/medium 3% gate
where the eliminated fixed cost is material and stays inside the 5% control
bound where Amdahl makes it immaterial. First-request work and idle retained
capacity are unchanged from the preflight budget; exact keyed prefix reuse is
an additional semantic capability, not included in these unkeyed timings.

### P02 structure checkpoint: batched Metal RoPE

Target: prefill command recording and dispatch overhead. Intentional variable:
replace Metal's portable per-row Q/K dispatch loop with one Q and one K dispatch
per constant YaRN temperature bucket. The rotation formula, FP16 in-place
layout, role scaling, graph order, barriers, hybrid behavior, and single-token
entry point remain unchanged.

The local change uses only existing files:

```text
crates/graph_horizon_engine/src/backend/metal/
├── backend.rs                 (~250 productive lines; category I, no limit)
├── kernels/rope.rs             (~125 productive lines)
├── shaders/rope.metal          (~15 productive lines; category K, no limit)
└── kernels/mod.rs              (~25 production lines, excluding tests)
```

`backend.rs` remains one category-I `impl Backend` of thin delegators;
segmentation belongs to the focused RoPE dispatch file. `rope.metal` remains
one category-K numeric operation. The invariant is that every row uses absolute
position `base + row`, and query post-scale is constant only inside one
`original_context` bucket; key scale remains one. The smallest viable change is
one row count in the existing constants plus checked host segmentation. The
main risk is crossing a query-temperature bucket or indexing a later row with
the wrong stride. A multi-row GPU test will compare the batched route against
the former per-row Metal route across a bucket boundary before benchmarking.

P02 added a checked batched host route and one row dimension to the existing
shader. Its cross-bucket GPU test was byte-identical to the former per-row Metal
dispatches and within 0.003 of the CPU scalar oracle. All pure-Metal tests pass.
The real-model screen compared retained P01 as A with P01+P02 as B:

| Prompt | P01 A1 | P02 B | P01 A2 | B vs A mean | Candidate CV | Decode control |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 598.68 ms | 598.95 ms | 608.19 ms | -0.74% | 0.62% | -1.42% |
| 512 | 2,519.85 ms | 2,469.98 ms | 2,514.84 ms | -1.88% | 0.15% | +0.33% |
| 2,048 | 11,540.26 ms | 11,372.21 ms | 11,980.52 ms | -3.30% nominal | 0.34% | +0.18% |

The final 2K baseline drifted 3.81% from A1 and had 1.68% within-run CV. The
candidate's 168.05 ms lead over the stable A1 agrees with the 512-derived
dispatch model (about 189 ms), giving a 1.46--1.88% repeatable signal rather
than a repeatable 3% endpoint gain. Projecting the same linear dispatch term to
8K yields about 0.8 seconds against a 77-second endpoint, far below the 5%
long-context gate. P02 is therefore rejected despite removing real work. The
exact candidate is commit `ceede6b`; `3089746` restores retained P01 source.

### P03 structure checkpoint: 64-row Metal weight-tile reuse

Target: the dominant quantized projection family and full-graph submission
ownership. Intentional variable: add a separate 64-row SIMD-matrix kernel for
pure Metal so one canonical K32 weight tile serves twice as many prompt rows,
and submit the corresponding 64-row graph chunk. The existing 32-row kernel is
unchanged for its route; mixed Metal, Vulkan, CPU, formats, arithmetic precision,
weight representation, decode, and public APIs remain unchanged.

The local change uses only existing files:

```text
crates/graph_horizon_engine/src/
├── backend/selection.rs                         (~180 productive lines)
├── backend/metal/mod.rs                         (~105 productive lines)
├── backend/metal/device.rs                      (~145 productive lines)
├── backend/metal/pipeline.rs                    (~185 productive lines)
├── backend/metal/mem/budget.rs                  (~180 productive lines)
├── backend/metal/kernels/matmul.rs              (~180 productive lines)
├── backend/metal/kernels/attention.rs           (~150 productive lines)
├── backend/metal/shaders/matmul.metal           (category K, no limit)
└── backend/metal/kernels/mod.rs                 (~25 production lines, excluding tests)
```

Every orchestration file remains below 200 productive lines. The projection
shader remains one category-K family of per-format variants of one operation.
The invariant is that each 64-row threadgroup uses 24,576 bytes (4 KiB weights,
4 KiB activations, 16 KiB FP32 results), below the queried 32,768-byte limit;
the existing 32-row route keeps its 14 KiB allocation and occupancy. Only the
first 128 of 256 threads dequantize the shared weight tile, all eight SIMD groups
multiply distinct 16-row tiles, and every thread reaches both barriers. The
smallest viable change is one additional pipeline function plus a pure-Metal
64-row capacity constant. The main risks are threadgroup indexing, memory-budget
under-accounting, partial final batches, and losing the qualified 4:1 GQA
attention route at 64 rows. Exact Q4/Q6 projection oracles, 32/64-row attention
oracles, allocation-policy tests, the full Metal suite, and real A/B/A timing
guard those boundaries. Retention requires at least 3% at 512/2K, no short or
decode control worse than 5%, and at least 5% before claiming a long-context win.

The first successful real run exposed one admission invariant that was not in
the initial tree: standalone Metal must guarantee the wide kernel's 256 threads
and 24 KiB threadgroup memory before selecting 64 rows. `device.rs` is therefore
added above before that local fix. The pipeline registry also checks the
compiled function's own 256-thread capacity. Metal-hybrid retains its existing
128-thread/16 KiB device floor because it never selects the wide route.

P03 adds a separate 256-thread wide pipeline; the retained 128-thread pipeline
and mixed-placement route are unchanged. The wide Q4/Q6 projection matches the
sequential Metal oracle at 64 rows, including a partial output tile. Both 32-
and 64-row mode-4 attention routing tests pass, the 64-row GPU result matches
serial attention at base zero and 512, and the full pure-Metal suite passes (138
unit plus 17 non-ignored integration tests). Real-model A/B/A used retained P01
as both bookends:

| Prompt | P01 A1 | P03 B | P01 A2 | B vs A mean | Candidate CV | Decode control |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 595.82 ms | 486.11 ms | 595.48 ms | -18.39% | 0.53% | -0.34% |
| 512 | 2,467.24 ms | 2,015.26 ms | 2,458.40 ms | -18.17% | 0.12% | +0.16% |
| 2,048 | 11,300.68 ms | 9,527.98 ms | 11,403.43 ms | -16.07% | 0.18% | -0.23% |
| 8,192 | 78,705.77 ms | 72,874.32 ms | 82,278.13 ms | -9.46% | 0.99% | +1.21% |

The 512-to-2K removable time grows from 447.56 to 1,824.08 ms, confirming the
intended linear projection/weight-read term. At 8K it reaches 7,617.63 ms while
quadratic attention dilutes the endpoint fraction, yet the whole-request gain
still clears the 5% long-context gate. Every decode control remains within
1.21%, as expected because P03 changes no decode route. Relative to the fresh
P00 endpoints, retained P01+P03 reduce 128/512/2K/8K TTFT by
30.56%/21.57%/15.98%/5.42%; same-session P03 bookends are the authoritative
candidate effect where thermal drift is material.

### P04 structure checkpoint: temporary post-P03 attribution

Target: restore at least 95% GPU-time attribution after the material P03
change. Intentional variable: reintroduce the repository's prior stage-boundary
timestamp profiler behind a non-production `metal-profile` feature, extended
only to classify `MatmulBatchedWide` as prefill matmul. Numeric kernels,
production Metal builds, routing, resources, and public APIs remain unchanged.

```text
crates/graph_horizon_engine/src/backend/metal/exec/
├── profile/mod.rs              (~200 productive lines)
├── profile/category.rs         (~160 productive lines)
├── profile/report.rs            (~50 productive lines)
└── profile/summary.rs           (~75 productive lines)

Existing orchestration touched by the temporary feature:
├── Cargo.toml                                   (~55 productive lines)
├── crates/graph_horizon_engine/Cargo.toml       (~55 productive lines)
├── backend/metal/device.rs                      (~170 productive lines)
├── backend/metal/exec/encoder.rs                (~130 productive lines)
├── backend/metal/exec/dispatch.rs               (~145 productive lines)
├── backend/metal/exec/mod.rs                     (~10 productive lines)
└── backend/metal/pipeline.rs                    (~190 productive lines)
```

The new directory is warranted because sampling ownership, category mapping,
aggregation, and reporting are four separate files; every orchestration file
stays at or below 200 productive lines. The invariant is that the profiling
feature changes only encoder boundaries and timestamp collection, while the
diagnostics-free P03 binary remains the performance authority. Main risks are
sample-capacity overflow, category drift from the seven-matmul layer order, and
profiling perturbation. The existing checked 4,096-sample bound, explicit wide
kernel classification, phase totals, and comparison against diagnostics-free
TTFT guard those risks. The profiler will be reverted after recording the
attribution checkpoint.

P04 is isolated as commit `38de361` and removed by `d8cb37d`; neither commit
changes a numeric kernel. The profiler timestamps command boundaries and every
adjacent same-category compute pass. `Kernel coverage` below is the sum of
those passes divided by the enclosing GPU command interval. The complementary
measured interval is command/encoder overhead, so kernel stages plus that
orchestration bucket cover 100% of GPU time even where short requests do not
reach 95% kernel-only coverage.

| Position | Phase | GPU total | Kernel coverage | Attention | Projections | MLP | Logits | Other kernels | Command / encoder |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | prefill | 582.72 ms | 90.96% | 1.50% | 28.63% | 66.36% | 0.59% | 2.92% | 9.04% |
| 128 | decode | 1,126.81 ms | 88.43% | 26.11% | 16.69% | 42.42% | 10.15% | 4.63% | 11.57% |
| 2,048 | prefill | 9,896.28 ms | 98.07% | 19.82% | 22.34% | 54.69% | 0.03% | 3.12% | 1.93% |
| 2,048 | decode | 1,319.85 ms | 90.52% | 37.89% | 14.45% | 35.37% | 8.62% | 3.67% | 9.48% |
| 28,000 | prefill | 521,921.97 ms | 99.63% | 76.58% | 6.67% | 15.77% | 0.00% | 0.98% | 0.37% |
| 28,000 | decode | 7,314.71 ms | 98.16% | 89.44% | 2.50% | 5.92% | 1.43% | 0.71% | 1.84% |

Category percentages are shares of sampled kernel time; the final column is a
share of enclosing command time. Supporting 512 and 8K traces show the same
crossover: prefill attention grows from 5.38% to 50.23%, while decode attention
grows from 38.12% to 70.68%. At 28K, decode attention alone consumes 6,422.10
ms of the 7,314.71 ms GPU interval. Halving that stage predicts a 1.78x command
speedup, nearly identical to the acquired 1.84x Vulkan/Metal comparator ratio.
This is direct evidence for a grouped 4:1 GQA decode experiment, not merely a
source-parity argument.

### P05 structure checkpoint: grouped Metal GQA decode

Target: the dominant long-context decode-attention traffic. Intentional
variable: for pure-Metal F16 decode with head dimension 128 and exact 4:1 GQA,
replace one mode-5 threadgroup per query head with one mode-6 threadgroup per
KV head. Its four SIMD groups own the four query heads and cooperatively stage
one 32-token K/V tile, reducing global K/V reads fourfold. The existing serial,
medium, segmented, INT8, mixed-placement, and prefill modes remain exact
fallbacks.

```text
crates/graph_horizon_engine/src/backend/metal/
├── kernels/attention.rs         (~160 productive lines)
├── shaders/attention.metal      (category K, no line limit)
└── kernels/mod.rs               (~25 production lines, excluding tests)
```

No new file, pipeline, scratch buffer, or durable abstraction is warranted.
`attention.rs` remains the single operation's shape/capability dispatch and
below 200 productive lines. `attention.metal` remains a category-K family for
one numeric operation. The invariant is that all 128 threads reach both tile
barriers; each KV head maps to exactly query heads `4*kh..4*kh+3`; tiles never
read past configured context; and only positions through the current causal
position contribute. The shared tile is exactly 16,384 bytes, matching the
already-qualified mode-4 allocation.

The smallest viable change is one new mode in the existing function plus its
exact routing predicate. The main risks are reduced history parallelism,
changed FP32 summation order, and a barrier or tail-tile indexing error. A GPU
comparison against the forced serial Metal route at the 1,024 boundary and a
later context must pass before real-model A/B/A. Retention requires at least
10% long-decode improvement, no more than 5% short/control regression, finite
outputs, and no generation drift beyond the repository's approved numeric
tolerance.

P05a is exact candidate commit `699ce10` and is removed by `e878459`. Its GPU
oracle passes at 1,024 and 2,048. The 128 A/B/A control is flat: 29.04 tok/s
against 29.01 tok/s baseline bookends. At 2K, however, decode falls to 21.16
tok/s from a 25.19 tok/s bookend mean, a 16.00% regression; candidate CV is
0.20%. TTFT remains within 0.2%, confirming that only decode changed. The
candidate traded sixteen independent history streams per query head for one.
The fourfold reduction in global KV traffic does not repay that lost latency
hiding at 2K, so the stop rule rejects it before expensive 8K/28K runs.

### P05b structure amendment: parallel grouped Metal GQA decode

The P05a result reveals an in-scope resource change that was not knowable
before measurement. P05b keeps the same cross-head reuse but uses eight SIMD
groups: two per query head, each with four lane-octet history streams. This
retains eight streams per head while still loading every global K/V tile once.
A separate pure-Metal pipeline contains 16 KiB of staged K/V plus 4 KiB of
FP32 partial outputs and 64 bytes of online-softmax state (20,544 bytes total).
It avoids raising the resource floor of the existing attention pipeline or the
Metal-hybrid fallback.

```text
crates/graph_horizon_engine/src/backend/metal/
├── pipeline.rs                  (~195 productive lines)
├── kernels/attention.rs         (~175 productive lines)
├── shaders/attention.metal      (category K, no line limit)
└── kernels/mod.rs               (~25 production lines, excluding tests)
```

All orchestration remains below 200 productive lines. The new pipeline is
compiled and loaded only for the standalone `metal` feature, whose P03 device
contract already guarantees 256 threads and 24 KiB. Its exact compiled SIMD
width, thread capacity, and static-memory size are checked before routing.
Metal-hybrid and all old attention modes remain unchanged. The additional
invariant is that each pair of SIMD groups writes disjoint partial state for
one query head, all 256 threads reach the merge barrier, and only the first
group in each pair writes that head's output. The main risks are partial-state
indexing and a remaining twofold loss of history parallelism. The same oracle
and 2K screen gate P05b before any long run.

P05b is retained as exact commit `8408e92`. Its grouped result matches the
forced serial Metal oracle at positions 1,024 and 2,048 within 0.02. The full
Metal engine suite passes (139 unit and 17 non-ignored integration tests), and
CPU plus Metal-hybrid compile controls pass with only the two acquired CPU SIMD
warnings. All six 3B/8B/14B Instruct/Reasoning artifacts generate successfully
beyond the route threshold. The broader root suite passes 165 of 166 tests;
the unrelated remaining fixture hardcodes absent macOS path
`/usr/bin/sha256sum`.

| Position | Baseline A1 | P05b B | Baseline A2 | B vs A mean | Candidate decode CV | TTFT control |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 28.98 tok/s | 29.04 tok/s | 29.03 tok/s | +0.12% | 0.18% | -0.11% |
| 2,048 | 25.21 tok/s | 27.98 tok/s | 25.30 tok/s | +10.79% | 0.14% | within 1.1% |
| 8,192 | 12.08 tok/s | 14.48 tok/s | 11.55 tok/s | +22.56% | 0.32% | thermal only; route unchanged |
| 28,000 | 4.26 tok/s | 5.62 tok/s | 4.26 tok/s | +31.92% | established by 2K/8K reps | thermal only; route unchanged |

P06 reuses the isolated profiler as commit `4e509b3` and removes it with
`62be3ef`. At 2K, grouped attention falls from 452.67 to 333.34 ms (-26.36%)
and from 37.89% to 31.08% of decode kernel time. At 28K it falls from 6,422.10
to 4,723.57 ms (-26.45%); prefill shares remain unchanged, while decode kernel
coverage is 97.63%. Attention still owns 86.49% of sampled kernel time and
84.45% of the enclosing GPU command interval, so the global stop is not yet
met.

### P05c structure amendment: full-parallel grouped split/reduce

Target: the remaining twofold history-parallelism gap in P05b. Intentional
variable: use one 512-thread split threadgroup per KV head, four SIMD groups
per query head, and the same shared 32-token KV tile. Each group writes one
FP32 online-softmax partial into an unused view of the persistent logits buffer;
a 32-thread reduction pipeline combines four partials per query head. This
restores the mode-5 sixteen history streams per head while retaining fourfold
global KV reuse. P05b remains the exact fallback when SIMD32 or 512 threads are
unavailable; Metal-hybrid, INT8, prefill, public APIs, and allocations remain
unchanged.

```text
crates/graph_horizon_engine/src/backend/metal/
├── backend.rs                   (~250 productive lines; category I, no limit)
├── pipeline.rs                  (~200 productive lines)
├── kernels/attention.rs         (~200 productive lines)
├── shaders/attention.metal      (category K, no line limit)
└── kernels/mod.rs               (~25 production lines, excluding tests)
```

No new file is warranted: both pipelines are variants of causal attention,
and their narrow dispatch remains in the attention operation module. Every
orchestration file stays at or below 200 productive lines; `backend.rs` remains
one category-I delegator block. The logits alias requires 65,536 bytes of FP32
partials plus 1,024 bytes of state for the qualified 32-head shape, is checked
against the real buffer before encoding, and is fully consumed before final
logits overwrite it. Default Metal hazard tracking orders split writes before
reduction reads in the same encoder.

The invariants are that all 512 split threads reach both tile barriers, the
four groups for each query head write disjoint partials, the reduction reads
exactly those four states, and fallback routing occurs before any encoder
mutation when resources do not qualify. Main risks are global-partial indexing,
the extra dispatch, and 512-thread occupancy. The forced-serial oracle and 2K
economic screen gate any longer run.

P05c is retained as `2a2ff0c`; `70e1715` keeps its capability-only field out
of Metal-hybrid builds. It passes the same 1,024/2,048 serial GPU oracle, the
full Metal engine suite, CPU/hybrid controls, and generation on all six real
model variants. The root-suite exception remains the unrelated macOS checksum
fixture described above.

| Position | P05b A1 | P05c B | P05b A2 | B vs A mean | Candidate decode CV |
|---:|---:|---:|---:|---:|---:|
| 2,048 | 28.02 tok/s | 31.59 tok/s | 27.88 tok/s | +13.02% | 0.40% |
| 8,192 | 14.52 tok/s | 18.65 tok/s | 14.23 tok/s | +29.74% | 0.42% |
| 28,000 | 5.61 tok/s | 8.00 tok/s | 5.71 tok/s | +41.34% | established by 2K/8K reps |

P07 is isolated as `24f0df0` and removed by `450d3db`. At 2K, P05c attention
is 211.97 ms and 22.53% of kernel time. At 28K it is 2,958.31 ms, 79.48% of
kernel time and 76.76% of the 3,853.91 ms GPU command interval; total coverage
is 96.59%. Prefill remains unchanged. The remaining attention fraction still
admits a 1.62x request speedup if a distinct mechanism can halve it, so the
global stop is not met.

### P05d structure amendment: disjoint multi-split history

Target: remaining long-decode history parallelism. Intentional variable: keep
the split/reduce pipelines and four-head KV reuse, but dispatch four disjoint
history splits per KV head. Each 256-thread group owns every fourth 32-token
tile and provides two SIMD groups per query head; the reduction combines eight
partials per head. This doubles aggregate history streams from sixteen to
thirty-two without rereading any global KV tile. P05c/P05b behavior remains the
fallback until measurements qualify the new threshold.

```text
crates/graph_horizon_engine/src/backend/metal/
├── kernels/attention.rs         (~200 productive lines)
├── shaders/attention.metal      (category K, no line limit)
└── kernels/mod.rs               (~25 production lines, excluding tests)
```

No new file or pipeline is warranted. The invariant is that tile number `n`
belongs only to split `n % 4`, every token through the causal position appears
in exactly one partial, part index `head*8 + split*2 + band` is collision-free,
and all 256 threads reach both barriers. The larger scratch alias is still only
132,096 bytes at 32 query heads and remains checked before encoding. Main risks
are incorrect tile coverage, reduction indexing, and extra-dispatch overhead at
short contexts. The serial oracle and 2K screen gate longer runs; if 2K loses
but 8K is plausible, routing must use an evidence-based longer threshold.

P05d is retained as exact commit `5c6b001`. Its serial oracle passes at both
1,024 and 2,048. Against P05c bookends it is effectively neutral-to-positive
at the route's shorter end and scales strongly with history length:

| Position | P05c A1 | P05d B | P05c A2 | B vs A mean | Candidate decode CV |
|---:|---:|---:|---:|---:|---:|
| 2,048 | 31.54 tok/s | 34.47 tok/s | 31.32 tok/s | +9.67% | 0.06% |
| 8,192 | 19.14 tok/s | 23.80 tok/s | 18.82 tok/s | +25.40% | 1.12% |
| 28,000 | 8.00 tok/s | 11.65 tok/s | established P05c endpoint | +45.63% | established by 2K/8K reps |

P08 is isolated as `3cd2c4f` and removed by `0dacf82`. At 2K, attention is
127.48 ms, down 39.86% from P05c, and only 14.75% of decode kernel time. At
28K it is 1,890.35 ms, down 36.10%; it remains 70.57% of kernel time and
67.19% of the 2,813.39 ms GPU command interval. That residual justified the
final eight-split endpoint rather than stopping at the 8K screen.

### P05e structure amendment: Vulkan-parity split count

Target: the last explicit split-count gap with Vulkan grouped decode.
Intentional variable: extend P05d from four to eight disjoint history splits
per KV head, retaining 256 threads, two SIMD groups per query head, the same
32-token shared KV tile, and the same reduction pipeline. The reduction
combines sixteen partials per query head, providing sixty-four aggregate
history streams without rereading global KV. P05d and earlier routes remain
available as exact comparison binaries; production keeps only the measured
winner.

```text
crates/graph_horizon_engine/src/backend/metal/
├── kernels/attention.rs         (~200 productive lines)
└── shaders/attention.metal      (category K, no line limit)
```

No new file or pipeline is warranted. Tile number `n` belongs only to split
`n % 8`; every causal token appears in exactly one partial; part index
`head*16 + split*2 + band` is collision-free; and all 256 threads reach both
barriers. At 32 query heads the checked scratch alias grows to 266,240 bytes
and is still consumed before logits overwrite the buffer. The main risks are
reduction cost, dispatch saturation, and a context-dependent crossover. The
same serial oracle gates performance measurement, followed by P05d A/B/A at
2K and only the longer contexts justified by that screen.

P05e is isolated as `a428d17` and removed by `01510ac`. It passes the same
1,024/2,048 serial GPU oracle, but split-count scaling is already saturated:

| Position | P05d A1 | P05e B | P05d A2 | B vs A mean | Candidate decode CV |
|---:|---:|---:|---:|---:|---:|
| 2,048 | 34.48 tok/s | 33.80 tok/s | 33.68 tok/s | -0.82% | 0.40% |
| 8,192 | 23.79 tok/s | 24.46 tok/s | 23.93 tok/s | +2.51% | 0.28% |
| 28,000 | 11.65 tok/s | 11.50 tok/s | established P05d endpoint | -1.29% | established by 2K/8K reps |

All outcomes are below the 10% retention gate. Four splits are the measured
winner on this Apple M4 despite Vulkan's eight-split implementation;
optimization-principle parity does not require copying a device-specific split
count.

### P09 structure amendment: SIMD-matrix long-prefill attention

Target: the final 75.90% long-prefill attention kernel share. Intentional
variable: add a separate complete-Q64 F16 4:1 GQA kernel in which one
1,024-thread group owns one KV head and all sixty-four query rows. Its thirty-two
SIMD groups each own eight rows of one query head, use 8x8 SIMD-matrix products
for QK and probability-by-V, and share each eight-token K/V tile across all four
query heads. FP32 matrix accumulators and per-row online maximum/sum state remain
resident; probabilities alone narrow to FP16, matching the acquired Vulkan
Matrix2 principle.

```text
crates/graph_horizon_engine/src/backend/metal/
├── pipeline.rs                  (~200 productive lines)
├── kernels/attention.rs         (~200 productive lines)
└── shaders/attention.metal      (category K, no line limit)
```

No new file is warranted because this is one causal-attention kernel variant
and its narrow dispatch. Existing mode 4 is the exact fallback. Eligibility
requires pure Metal, F16 KV, head dimension 128, exact 4:1 GQA, sixty-four
complete rows, SIMD32, 1,024 compiled threads, and no more than the device's
32 KiB threadgroup limit. The 28 KiB static layout contains shared K/V,
per-SIMD-group score/probability tiles, and diagonal row scales.

The invariants are that all 1,024 threads reach every group barrier, causal
masking admits exactly positions through `base + row`, every K/V token is loaded
once per KV head/tile, and output accumulator rescaling uses the same new
online maximum as probability normalization. Main risks are occupancy/resource
rejection, matrix layout transposition, causal-edge indexing, and FP16
probability error. The existing 64-row serial oracle gates a 2K screen; a gain
below 10% closes the mechanism without a long run.

The completed candidate made `kernels/attention.rs` 223 productive lines, which
the structural test rejects. The local, behavior-preserving correction is to
express the now-real decode/prefill domain split as a folder before final
retention:

```text
crates/graph_horizon_engine/src/backend/metal/kernels/attention/
├── mod.rs                      (~25 productive lines)
├── decode.rs                   (~75 productive lines)
└── prefill.rs                 (~145 productive lines, excluding tests)
```

`mod.rs` owns only shared constant packing and the unchanged exports; decode
and prefill each own their narrow dispatch policy. All orchestration files are
again below 200 productive lines, and callers retain the same module surface.

P09 is retained as `98f1dbf`; `de54882` changes its capability check from an
exact source-level allocation to an upper bound because the Metal compiler
legally overlays lifetimes and reports 24,576 static bytes. The actual route
matches serial Metal at 64 rows. The full Metal engine suite passes (139 unit
and 17 non-ignored integration tests), the structural line-limit test passes,
CPU/Metal-hybrid controls pass with only the two acquired CPU `simd` warnings,
and all six 3B/8B/14B Instruct/Reasoning models generate beyond both optimized
thresholds.

| Position | P05d A1 | P09 B | P05d A2 | B vs A mean | Candidate TTFT CV |
|---:|---:|---:|---:|---:|---:|
| 2,048 | 9,712.20 ms | 8,968.11 ms | 9,712.45 ms | -7.66% | 0.06% |
| 8,192 | 72,317.93 ms | 51,194.24 ms | 72,603.34 ms | -29.35% | 2.89% |
| 28,000 | 524,294.81 ms | 332,347.45 ms | established P05d endpoint | -36.61% | established by 2K/8K reps |

### Final retained endpoint

The table combines diagnostics-free measurements from the final retained
binary. Percentage changes use the fresh P00 baseline; the 28K decode value is
an observed control because P09 changes prefill only.

| Prompt | Final TTFT | TTFT vs P00 | Prompt tok/s | Prompt rate vs P00 | Decode tok/s | Decode rate vs P00 |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 498.15 ms | -28.84% | 256.95 | +40.53% | 29.02 | -3.07% |
| 512 | 2,054.43 ms | -20.04% | 249.22 | +25.07% | 24.36 | +0.33% |
| 2,048 | 8,968.11 ms | -20.92% | 228.36 | +26.45% | 34.30 | +35.36% |
| 8,192 | 51,194.24 ms | -33.56% | 160.11 | +50.39% | 24.67 | +110.32% |
| 28,000 | 332,347.45 ms | -43.41% | 84.25 | +76.70% | 12.83 | +196.30% |

P10 is isolated as `fc256ca` and removed by `f2a47a9`. At 2K, P09 prefill
attention is 1,177.30 ms, down 39.40% from P08, and 13.10% of kernel time. At
28K it is 217,109.37 ms, down 44.37% from P08 and 66.70% of kernel time;
kernel coverage is 99.40%, and attention is 66.30% of the 327,464.07 ms GPU
command interval. Decode remains P05d: 127.74 ms attention at 2K and 1,605.21
ms at 28K in this thermally favorable trace.

### Global stop

The stop condition is met. Qualified Q64 prefill now uses SIMD-matrix QK/PV,
keeps FP32 online-softmax/output state resident, and loads every eight-token KV
tile once per KV head across all sixty-four rows and four query heads. A larger
tile cannot retain the required score/probability/scale and K/V state within
the 32 KiB device contract; the queried M4 exposes no tensor path beyond the
SIMD-matrix unit already used. Decode history parallelism was measured at one,
two, four, and eight split equivalents; four wins, while eight is -1.29% at
28K. Wider projection ownership, batched RoPE, and wider local prefill reuse
were also measured and either retained or closed below their gates. Remaining
attention time is the irreducible quadratic arithmetic/traffic of the current
exact causal algorithm, not an identified missing Vulkan optimization
principle. No distinct untried candidate has evidence for a >=10% request gain.

## Completed state

The final branch contains only retained P01/P03/P05d/P09 production changes,
their required structural split, experiment history, and this report. Temporary
profilers and rejected production candidates are reverted. The mission closed
before publication; the retained branch was subsequently pushed and merged into
`main` by PR #34.
