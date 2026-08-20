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
| Q4/Q6 prefill above 32 rows | same tensor roles | repeated qualified tiles in one request | unavailable: request capacity and kernel are fixed at 32 | missing row-tile dispatch/routing |
| F16/Q5 prefill | projection/MLP | portable correct path | per-row scalar projection | no cooperative implementation |
| 4:1 GQA F16 prefill | head 128, 32 rows | tiled K/V reuse | mode 4, base zero onward, pure Metal | partial rows, INT8, mixed, other GQA/dim |
| other long prefill | base >=512, head 128 | segmented online softmax | mode 3, two SIMD groups/head | unqualified dimension or shorter context |
| short prefill attention | non-mode-4 tuple | one SIMD group/head | mode 1 | shape/format not qualified |
| F16 long decode | row 1, base >=1023, head 128 | wide segmented | mode 5, four SIMD groups/head | shorter context / mixed / other dim |
| INT8 long decode | row 1, base >=1023, head 128 | segmented | mode 3, two SIMD groups/head | no wide INT8 qualification |
| medium decode | row 1, base >=511, head 128 | four lane-octet segments | mode 2 | shorter or other dimension |
| RoPE prefill | Q and K rows | batched role dispatch | default trait emits two dispatches per row | Metal has no batched override |
| graph tail/logits | final prompt row only | one norm and logits projection | selected by shared graph on last chunk only | none |
| request KV | K/V for full configured context | retained capacity | allocated and freed for every Metal request | cache is Vulkan-gated |

## Mandatory parity matrix

Status means the same optimization principle is active and qualified, not that
similarly named source exists.

| Capability | Vulkan implementation / eligibility | Metal implementation / eligibility | Representation | Workloads | Metal status | Impact / priority |
|---|---|---|---|---|---|---|
| Quantized Q4 prefill | direct packed Matrix2 on capability + measured shapes; portable/AMD fallbacks | direct packed SIMD-matrix for pure Metal, rows 2..32, K % 256, N % 32 | canonical Q4_K | all prefill projections | PARITY principle; UNDERPERFORMS | 92.4% of sampled 512 kernels after M02; high |
| Quantized Q6 prefill | packed Matrix2/coop/portable by capability and shape | same packed SIMD-matrix route as Q4 | canonical Q6_K | V/down mixed rows | PARITY principle; UNDERPERFORMS | included in dominant matrix family; high |
| Short/long prefill ownership | 256 rows portable, 512 when Q4 Matrix2 + Q64 attention, AMD 128/64 | fixed 32 rows | FP16 intermediates | medium/long prompts | MISSING METAL ROUTING/IMPLEMENTATION | submissions and fast-path launch occupancy; high |
| Weight-tile reuse | Matrix paths decode once and reuse across 128/256 prompt rows | K32 tile decoded once for 32 prompt rows | packed canonical blocks | Q/K/V/O/MLP | METAL UNDERPERFORMS | lower reuse factor; high after routing screens |
| Decode GEMV | Q4 DP4A where qualified; exact per-format fallbacks | direct packed Q4/Q6 SIMD decode | canonical Q4_K/Q6_K | all decode projections/logits | METAL HAS BETTER ALTERNATIVE | short decode within 1.22x comparator on 3B; closed pending reprofile |
| GQA decode | dedicated 4:1 split/reduce reuses K/V across query heads | per-query-head mode 1/2/3/5; no cross-head K/V reuse | F16/INT8 KV | especially long decode | MISSING METAL IMPLEMENTATION | 28K comparator ratio 1.84x; medium-high |
| Attention prefill | exact fused online softmax; Q32/Q64 Matrix2 and portable tiles | fused online softmax; mode 4 reuses a K/V tile across 4 query rows and four GQA heads | F16/INT8 KV | all prefill | PARITY principle; UNDERPERFORMS long | 45.8% at 8K, grows quadratically; high but prior local tiles closed |
| Attention decode | generic plus vectorized GQA split/reduce | segmented persistent partial state per query head | F16/INT8 KV | decode | PARTIAL PARITY | missing cross-head reuse; medium-high |
| Persistent attention state | Q64/other fused paths avoid global score/probability tensors | every Metal mode retains online-softmax accumulator/state | registers/threadgroup state | prefill/decode | PARITY | no rewrite |
| Reduced attention traffic/barriers | Q64 avoids global intermediates and per-KV-tile barriers | mode 4 uses shared K/V tiles and two barriers/tile; other modes stream directly | fused | prefill | METAL UNDERPERFORMS | prior 8-row tile gained only 2.15% request; closed locally |
| RoPE batching | one dispatch per role/context segment | two dispatches per prompt row through default trait | in-place FP16 Q/K | prefill | MISSING METAL IMPLEMENTATION | thousands of dispatches at long prompts; high, compact |
| Final-only logits | shared graph emits tail only on final prompt chunk | same shared graph | FP32 logits | TTFT | PARITY | closed |
| KV append | one batched append per layer/chunk | one batched append per layer/chunk | F16/INT8 KV | prefill | PARITY | prior Metal attribution negligible |
| Request KV retention | single-slot allocation retained; optional exact keyed prefix | new allocation and destruction per request | backend-native KV buffers | repeated requests / TTFT | MISSING METAL IMPLEMENTATION | must isolate allocation cost; high by priority rule |
| Prefix reuse | exact caller key + token prefix | public call falls back to uncached generation | KV state | repeated chats | MISSING METAL IMPLEMENTATION | potentially request-scale; requires same session cache |
| Weight loading | persistent device-local canonical blocks | one model-lifetime copy into shared canonical buffers | Q4_K/Q5_K/Q6_K/F16 | model load | METAL HAS BETTER ARCHITECTURAL FIT | unified-memory hot path shows no transfer term; closed |
| Direct canonical Q4 | direct packed callback; no expansion | direct `q4`/`q4x16`; no expansion | canonical Q4_K | prefill/decode | PARITY | closed |
| Direct canonical Q6 | packed routes; staged alternatives rejected where inferior | direct `q6`/`q6x16`; no expansion | canonical Q6_K | prefill/decode | PARITY | audit independently in measurements |
| Packed loads/dequant | packed scalar/vector loads and tile decode | packed byte/ushort loads, per-tile decode into half matrices | canonical blocks | quantized matmul | PARITY principle | local maturity still measured by matrix time |
| Intermediate tensors | fused attention and fused SiLU/mul; no N x N scores | same; FP16 scratch retained per request chunk | FP16 scratch | all | PARITY | no material missing conversion found |
| Command orchestration | one synchronous submit per chunk; larger qualified chunks | one synchronous submit per fixed 32 rows | backend-native commands | prefill/decode | METAL UNDERPERFORMS prefill | coupled to row-ownership gap |
| Capability routing | features/vendor/shape/format with fallbacks | Apple9/unified memory/resource gate, pipeline width, shape/format/placement | n/a | all | PARTIAL PARITY | operation gates exist; row ownership and RoPE absent |

## Scorecard

| Capability | Vulkan | Metal | Status |
|---|---:|---:|---|
| Q4 packed execution | yes | yes | parity principle active |
| Q6 packed execution | yes | yes | parity principle active |
| weight-tile reuse | yes | yes, 32 rows | Metal underperforms |
| direct canonical Q4 | yes | yes | parity |
| short-prefill specialization | yes | 32-row tile only | partial |
| long-prefill specialization | yes | attention only | missing larger ownership |
| attention traffic optimization | yes | yes | Metal underperforms long |
| persistent attention state | yes | yes | parity |
| GQA/KV reuse prefill | yes | yes | parity |
| GQA/KV reuse decode | yes | no | missing |
| vectorized decode | yes | yes | partial parity |
| batched RoPE | yes | no | missing |
| final-only logits | yes | yes | parity |
| resource retention | yes | model only | request KV missing |
| shape/capability routing | yes | partial | missing row/batch eligibility |

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

The acquired diagnostics at the unchanged numeric/routing checkpoint account
for 97.7% of 512 and 97.6% of 2K GPU command time. After the retained early
tiled-attention route, 512 quantized matrices consume 2,294.89 ms and 92.4% of
sampled kernel time; attention consumes 123.57 ms. At 8K, attention is 31.98 s
(45.8%), quantized matrices 36.65 s (52.5%), and command residual 1.34 s.

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

No candidate is retained from this table alone. Allocation and RoPE timing will
replace the conservative bounds before implementation decisions.

| Rank | Candidate | Affected fraction / evidence | Realistic local speedup | Predicted removable | Decision gate |
|---:|---|---|---:|---:|---|
| 1 | retain Metal request KV and exact keyed prefix | allocation is currently once/request; direct timing pending | allocation-only elimination | pending | compact parity gap; keep only above noise |
| 2 | batched Metal RoPE | 2 x rows x layers dispatches become about 2 x layers | dispatch/encoding term pending | bounded by non-matrix/attention + residual | compact parity gap; screen 128/512 |
| 3 | larger Metal prefill request ownership | 32-row qualified tiles; Vulkan gained 5--7% from larger ownership | 1.05--1.10x request estimate | about 0.12--0.26 s at 512 if estimate holds | requires >=5% and stable controls |
| 4 | vectorized 4:1 GQA Metal decode | missing K/V reuse; long ratio reaches 1.84x | attribution pending | pending | implement only after decode attribution |
| 5 | new long-prefill attention architecture | 45.8% at 8K and grows quadratically | prior local reuse only 1.09x attention | prior request result 2.15% | closed until distinct mechanism clears 10% request gate |

## Experiment registry

| ID | Target | Intentional variable | Correctness gate | Result | Decision |
|---|---|---|---|---|---|
| P00 | baseline | current production source | existing Metal suites | full 128--28K matrix above; every TTFT CV below 5% | evidence |

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

## Current next action

Freeze the fresh baseline checkpoint, isolate Metal KV allocation and RoPE
command cost, calculate Amdahl values from those measurements, then test the
highest-ranked compact parity gap. Reprofile after every retained whole-request
gain.
