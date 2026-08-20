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
| Push state | nothing pushed |

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
| 4:1 GQA F16 prefill | head 128, 32 or 64 rows | tiled K/V reuse | mode 4, base zero onward, pure Metal | partial rows, INT8, mixed, other GQA/dim |
| other long prefill | base >=512, head 128 | segmented online softmax | mode 3, two SIMD groups/head | unqualified dimension or shorter context |
| short prefill attention | non-mode-4 tuple | one SIMD group/head | mode 1 | shape/format not qualified |
| F16 long decode | row 1, base >=1023, head 128 | wide segmented | mode 5, four SIMD groups/head | shorter context / mixed / other dim |
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
| GQA decode | dedicated 4:1 split/reduce reuses K/V across query heads | per-query-head mode 1/2/3/5; no cross-head K/V reuse | F16/INT8 KV | especially long decode | MISSING METAL IMPLEMENTATION | 28K comparator ratio 1.84x; medium-high |
| Attention prefill | exact fused online softmax; Q32/Q64 Matrix2 and portable tiles | fused online softmax; mode 4 reuses a K/V tile across 4 query rows and four GQA heads | F16/INT8 KV | all prefill | PARITY principle; UNDERPERFORMS long | 45.8% at 8K, grows quadratically; high but prior local tiles closed |
| Attention decode | generic plus vectorized GQA split/reduce | segmented persistent partial state per query head | F16/INT8 KV | decode | PARTIAL PARITY | missing cross-head reuse; medium-high |
| Persistent attention state | Q64/other fused paths avoid global score/probability tensors | every Metal mode retains online-softmax accumulator/state | registers/threadgroup state | prefill/decode | PARITY | no rewrite |
| Reduced attention traffic/barriers | Q64 avoids global intermediates and per-KV-tile barriers | mode 4 uses shared K/V tiles and two barriers/tile; other modes stream directly | fused | prefill | METAL UNDERPERFORMS | prior 8-row tile gained only 2.15% request; closed locally |
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
| long-prefill specialization | yes | 64-row graph and attention tile | partial after P03 |
| attention traffic optimization | yes | yes | Metal underperforms long |
| persistent attention state | yes | yes | parity |
| GQA/KV reuse prefill | yes | yes | parity |
| GQA/KV reuse decode | yes | no | missing |
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
| 4 | vectorized 4:1 GQA Metal decode | attention is 89.44% of 28K decode kernel time; long ratio reaches 1.84x | 2x attention gives about 1.78x command speedup | about 3.2 s/token at 28K | next production candidate |
| 5 | new long-prefill attention architecture | 45.8% at 8K and grows quadratically | prior local reuse only 1.09x attention | prior request result 2.15% | closed until distinct mechanism clears 10% request gate |

## Experiment registry

| ID | Target | Intentional variable | Correctness gate | Result | Decision |
|---|---|---|---|---|---|
| P00 | baseline | current production source | existing Metal suites | full 128--28K matrix above; every TTFT CV below 5% | evidence |
| P01 | request KV lifetime | existing homogeneous cache enabled for pure Metal | 138 unit + 17 integration tests; repeated real generation | 128 -15.54%; 512 -4.05%; no decode regression | RETAINED |
| P02 | RoPE dispatch batching | rows per Q/K dispatch | byte-exact old Metal route + CPU oracle; 139 unit + 17 integration tests | 128 -0.74%; 512 -1.88%; stable 2K signal about -1.5% | REJECTED below gate |
| P03 | 64-row projection/graph tile | weight reuse and pure-Metal row ownership | Q4/Q6 3/32/64-row oracles; 64-row attention oracle; full Metal suite | 128 -18.39%; 512 -18.17%; 2K -16.07%; 8K -9.46% | RETAINED |
| P04 | post-P03 stage attribution | timestamp instrumentation only | phase totals, category sums, diagnostics-free timing authority | required 128/2K/28K points captured; 90.96--99.63% kernel coverage plus measured command residual | EVIDENCE; profiler reverted |

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

## Current next action

Implement and numerically qualify P05, then screen it at short, 2K, 8K, and
28K positions. Retain it only if long decode clears 10% with no material short
regression or correctness drift.
