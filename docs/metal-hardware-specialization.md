<!--
This report is the persistent evidence record for Apple/Metal specialization.
It owns device characterization, competitive measurements, experiments, and
qualification status; it defines no runtime API or backend support contract.
-->

# Metal hardware specialization

## Mission start

| Field | Value |
|---|---|
| Baseline branch | `main` |
| Baseline commit | `910a11689c3804b58f5216f2dbc40c60eabfa90c` |
| Merge-base with `main` | `910a11689c3804b58f5216f2dbc40c60eabfa90c` |
| Initial worktree | one untracked Finder `.DS_Store`; moved to `/tmp/codex-graph-horizon-metal-start/repository.DS_Store` |
| Experiment branch | `perf/metal-hardware-specialization` |
| Clean gate | passed before build, measurement, or production edit |
| Push state | nothing pushed |

The starting revision is the clean production merge of the qualified AMD
specialization. The existing Metal long-context decode result in
[`metal-long-context-investigation.md`](metal-long-context-investigation.md)
is retained history, not reimplemented here.

## Qualified artifacts and oracle

All files were opened read-only and matched `support/models.tsv` on 2026-08-18.

| Model | Bytes | SHA-256 |
|---|---:|---|
| 3B Instruct Q4_K_M | 2,147,023,008 | `9ed150d4367e68df0ac8e1540f6ddc65b42d0ee26378329d1ecbca60f93fc5f8` |
| 8B Instruct Q4_K_M | 5,198,911,904 | `33e7a72cf5e6e2cfc2f2847075acc013d68bba023e35310cef86b5cf8fdca761` |
| 14B Instruct Q4_K_M | 8,239,593,024 | `824e0f3373e69b84f2cae46fdcb9bd1ebc6ab3bfc7acc125d818b7b8178cc613` |

The llama.cpp Metal comparator is the local checkout at commit `13f2b28b0`
(build 9973), rebuilt in ignored `build-metal` with `GGML_METAL=ON` and the
embedded Metal library. The previous local binary had `GGML_METAL=OFF` and is
not used. The comparator reports Metal, Accelerate, full GPU offload, shared
buffers, and residency sets. It is not modified or pushed.

## Device capability report

Device name is recorded only for reporting. Production eligibility must use
capabilities and resource limits.

| Capability | Value | Evidence |
|---|---:|---|
| Device | Apple M4, 10 GPU cores | queried: `MTLDevice` / system inventory |
| OS | macOS 26.3, build 25D125 | queried |
| Metal support | Metal 4 | queried: system inventory |
| Metal compiler | Apple metal 32023.864 | queried: `xcrun metal --version` |
| GPU families | Apple7, Apple8, Apple9, Common3, Metal4 | queried: `supportsFamily` |
| Unified memory | yes | queried: `hasUnifiedMemory` |
| SIMD-group width | 32 | queried: compiled compute pipeline |
| Maximum threads/threadgroup | 1,024 | queried: device and pipeline |
| Maximum threadgroup memory | 32,768 bytes | queried: `MTLDevice` |
| Recommended working set | 19,069,665,280 bytes | queried: `MTLDevice` |
| Maximum buffer length | 14,302,248,960 bytes | queried: `MTLDevice` |
| Argument-buffer tier | tier 2 (`rawValue=1`) | queried: `MTLDevice` |
| Read/write texture tier | tier 2 (`rawValue=2`) | queried: `MTLDevice` |
| Function pointers / dynamic libraries | supported | queried: `MTLDevice` |
| SIMD-group reduction | supported | queried by llama.cpp capability probe; exercised by current shaders |
| SIMD-group 8x8 matrix multiply | supported | derived from successful pipeline compilation and current execution |
| Counter sampling at dispatch boundary | unavailable | queried: `MTLDevice` |
| Counter sampling at encoder stage boundary | supported | queried: `MTLDevice` |
| BF16 arithmetic | supported | queried by llama.cpp capability probe |
| Tensor acceleration | unavailable on pre-M5/pre-A19 | queried by llama.cpp capability probe |
| Buffer-view alignment used by runtime | 4 bytes | current runtime contract; not a device-wide optimum claim |
| Occupancy, register count, spills | unavailable before GPU capture | unavailable |
| Cache bandwidth, ALU utilization, stalls | unavailable before GPU capture | unavailable |

The effective execution width is 32 lanes. Current projection dispatch uses two
SIMD groups for quantized decode and four for cooperative prefill. Current
long-context attention uses two or four SIMD groups depending on shape, KV
format, and context. Resource-cliff and residency conclusions remain pending
measured variants or profiler evidence.

## Current Metal architecture and routing

- Correct generic execution remains available through CPU and portable backend
  semantics. Metal fast paths are optional operation routes.
- Device acquisition requires unified memory, Apple9-family support, at least
  128 threads per threadgroup, and at least 16 KiB threadgroup memory. Product
  names are reporting-only. Pipeline compilation and operation-local SIMD-width
  checks retain the canonical failure/fallback boundary.
- Every Metal allocation currently uses `StorageModeShared`. Model weights are
  copied once from the read-only GGUF mapping into shared Metal buffers.
- CPU readback occurs only after synchronous command completion. There is no
  managed-resource synchronize or discrete-GPU staging copy in the Metal path.
- Each graph batch/token creates one command buffer and compute encoder, records
  the full graph, commits, then waits for completion. Pipeline states are loaded
  once for model lifetime.
- Quantized decode projection uses two 32-lane SIMD groups and emits four output
  rows/threadgroup on pure Metal. Q4_K and Q6_K are consumed canonically.
- Batched Q4_K/Q6_K projection uses four SIMD groups, 6 KiB threadgroup memory
  (`weights[64x32]` plus `acts[32x32]`), SIMD-group 8x8 matrix multiply, and a
  32-row by 64-column logical tile. Eligibility is rows 2..32, K divisible by
  256, N divisible by 32, quantized weights, and non-mixed placement.
- F16/Q5_K batched projection falls back to one scalar/per-row dispatch per row.
- Attention routing uses serial, one-SIMD-per-head, segmented, four-SIMD decode,
  and a 32-row 4:1 GQA tile. Eligibility depends on head dimension 128, SIMD
  width 32, F16/INT8 KV, placement, GQA relation, row count, and context.

## Unified-memory audit: initial state

| Transfer/resource | Lifetime | Current behavior | Initial classification |
|---|---|---|---|
| Canonical model mapping -> Metal weights | model | one CPU memcpy into shared buffer | required upload candidate; private-storage alternative must include copy/sync economics |
| Input/scratch/KV/output buffers | context/request | shared CPU/GPU-visible allocations | no discrete staging copy |
| Logits/top-k/argmax readback | token/request | blocking submit then direct shared-memory read | synchronization cost requires attribution |
| Command buffer / compute encoder | prefill batch or token | recreated, committed, waited synchronously | possible short-workload fixed cost |
| Pipeline registry | model | created once | appropriate lifetime |
| KV buffers | context | allocated for configured context and reused by request cache | appropriate candidate; verify zeroing/reuse cost |

Shared storage is not presumed optimal for GPU-resident weights. A private
weight experiment is admissible only after profiling shows enough removable
memory cost to pay for its load-time blit and retained canonical fallback.

## Baseline method

- Graph Horizon: release build, Metal feature, context allocation 32,768, F16
  KV, full Metal placement, 32 generated tokens, one warm-up and three measured
  repetitions. `N-4` repetitions of `a ` produce exactly `N` runtime prompt
  tokens under the project chat template.
- llama.cpp: same authenticated file, pinned commit, full Metal offload, F16 K/V,
  flash attention disabled, batch and microbatch 32, one warm-up and three
  measured requests. The native completion endpoint receives an explicit
  `N`-token ID array, avoiding tokenizer/chat-template differences.
- Both ran in the same AC-powered session with no thermal or performance warning.
- Graph Horizon TTFT includes tokenization, first sampling, and first public
  delta. llama `prompt_ms` isolates prompt evaluation. Decode compares public
  delta throughput against llama model-token throughput, so it is directionally
  competitive but not an exact event-boundary identity.
- Final retained-candidate measurements will use diagnostics-free A/B/A or
  bookends. No profiler result is mixed into this baseline.

## Initial competitive gap map

Latency ratio is Graph Horizon latency divided by llama.cpp latency. Decode
latency ratio is equivalently llama tok/s divided by Graph Horizon tok/s.

| Model | Workload | Ours | llama Metal | Ratio | Absolute gap |
|---|---|---:|---:|---:|---:|
| 3B | prefill 128 | 914.88 ms | 323.55 ms | 2.83x | 591.33 ms |
| 3B | prefill 512 | 6,188.66 ms | 1,326.42 ms | 4.67x | 4,862.24 ms |
| 3B | prefill 2K | 15,037.83 ms | 5,767.88 ms | 2.61x | 9,269.95 ms |
| 3B | decode KV128 | 26.30 tok/s | 35.72 tok/s | 1.36x | 10.05 ms/token |
| 3B | decode KV512 | 22.91 tok/s | 30.45 tok/s | 1.33x | 10.81 ms/token |
| 3B | decode KV2K | 25.31 tok/s | 27.56 tok/s | 1.09x | 3.22 ms/token |

Measured CVs are 0.16--0.29% for the retained Graph Horizon prefill points,
apart from 5.36% decode CV at KV512. llama prompt samples are similarly stable.
The non-monotonic 512-token valley is too large to explain with variance and is
the first attribution target. This table remains the pre-change evidence; the
final retained matrix is reported separately below.

## Attribution and ranking checkpoint

| Model | Workload | Dominant component | Component time | Realistically removable |
|---|---|---|---:|---:|
| 3B | prefill 128 | quantized Q/K/V/O + MLP | 587.08 ms | routing target remains attention because matrices scale normally |
| 3B | prefill 512 | attention | 3,709.19 ms | about 3,092 ms from existing tiled-path behavior |
| 3B | prefill 2K | quantized Q/K/V/O + MLP | 9,124.18 ms | separate matrix candidate pending after routing win |
| 3B | decode KV128 | quantized projections + logits, then attention | 727.15 ms / 259.84 ms over 31 steps | decode already inside historical 1.4x envelope |

The profiler directly accounts for 89.6%, 97.7%, and 97.6% of GPU command time
at prefill 128, 512, and 2K respectively. The remainder is reported as sampled
encoder-transition/command residual. At the prioritized 512/2K workloads the
95% attribution gate is met.

The external Metal System Trace attempt was rejected: a 512-token diagnostic
run remained active for minutes, produced an incomplete 1 GB trace, and could
not export its template tables. The trace was moved to the temporary recovery
directory and is not performance evidence. Device queries then selected the
smallest usable in-runtime design:

```text
metal/exec/profile/
├── mod.rs       (~180 productive lines: sampled-pass ownership)
├── category.rs  (~115 productive lines: phase/operation classification)
├── summary.rs   (~75 productive lines: marks and totals aggregation)
└── report.rs    (~80 productive lines: diagnostic output)
```

Feature-manifest wiring and focused changes to `device.rs`, `encoder.rs`,
`dispatch.rs`, and `pipeline.rs` remained below 200 productive lines per
orchestration file. The diagnostic feature used stage-sampled compute passes
inside the original command buffer because this Apple9 device does not expose
dispatch-boundary sampling. Ordinary `metal` builds retained the single-encoder
path at compile time.

The first 32-row request proved that an 8,192-sample buffer is rejected by the
device while 4,096 is accepted. The investigation diagnostic therefore used
one sampled pass for each contiguous run of the same category and records the
dispatch count inside the run. This preserves operation attribution without
sampling thousands of row-wise embedding/RoPE dispatches separately.

At 512, all 16 batches remain below the existing `base >= 512` tiled-attention
eligibility gate. Attention consumes 3,709.19 ms (59.3% of GPU command time),
quantized Q/K/V/O and MLP consume 2,318.56 ms, and the remaining sampled work is
88.51 ms. At 2K, attention is 5,560.46 ms while quantized matrix work is
9,124.18 ms. The first 16 slow batches therefore consume about 3.71 s, while
the next 48 already-tiled batches add only about 1.85 s.

For the 512 routing candidate:

```text
T = 6.189 s
f_attention = 0.593 of GPU command time
practical attention floor ~= 0.617 s (16 x measured later-tiled batch mean)
S_local ~= 6.0x
removable_time ~= 3.092 s
predicted whole-request speedup ~= 2.0x
competitive-gap recovery ~= 63.6%
```

This clears the strong specialization gate without a new shader. The next
prototype removes only the context restriction from the already-qualified
32-row, F16, head-dimension-128, exact-4:1-GQA tiled route. Partial rows, INT8,
mixed placement, other dimensions, and other GQA ratios retain their fallbacks.

### M02 retained result

The candidate changed only that routing restriction. Its 512-token post-change
profile measured 123.57 ms of attention, down 96.7% from 3,709.19 ms. Total
prefill GPU time fell from 6,258.92 ms to 2,616.82 ms. Quantized Q/K/V/O and
MLP work remained stable at 2,294.89 ms and now accounts for 92.4% of sampled
kernel time. The profiler accounts for 95.0% after rounding; the unsampled
131.98 ms command residual is retained explicitly.

Diagnostics-free measurements retained the win at every calibrated short
prompt:

| Prompt | Baseline TTFT | M02 TTFT | Reduction | M02 / llama |
|---:|---:|---:|---:|---:|
| 128 | 914.88 ms | 677.26 ms | 26.0% | 2.09x |
| 512 | 6,188.66 ms | 2,554.70 ms | 58.7% | 1.93x |
| 2,048 | 15,037.83 ms | 11,449.03 ms | 23.9% | 1.99x |

The stricter 512-token baseline/candidate/baseline bracket measured 6,129.29,
2,546.25, and 6,120.03 ms respectively, with TTFT CV at or below 0.16%. That is
a 2.40x whole-request speedup and recovers 74.7% of the original competitive
gap, exceeding the 2.0x Amdahl estimate. Decode throughput did not regress in
the bracket. An earlier bracket was rejected because its detached target had
accidentally been compiled from the candidate source tree; identical timings
exposed the error, and none of those mislabeled results are retained.

## Final retained competitive matrix

Measurements continued across midnight on 2026-08-18/19 in the same
AC-powered session. Every row uses a diagnostics-free release binary, one
warm-up, and three measured repetitions. llama prompt caching is disabled and
every response reports the full requested `prompt_n`.

| Model | F16 prefill | Ours TTFT | Ours CV | llama prompt | llama CV | Ratio |
|---|---:|---:|---:|---:|---:|---:|
| 3B | 128 | 677.26 ms | 0.42% | 323.55 ms | stable | 2.09x |
| 3B | 512 | 2,554.70 ms | 0.28% | 1,326.42 ms | stable | 1.93x |
| 3B | 2,048 | 11,449.03 ms | 0.12% | 5,767.88 ms | stable | 1.99x |
| 3B | 8,192 | 71,100.02 ms | 3.71% | 35,175.63 ms | 4.72% | 2.02x |
| 3B | 16,384 | 229,641.64 ms | 0.45% | 101,427.38 ms | 0.65% | 2.26x |
| 3B | 28,000 | 562,828.73 ms | 2.07% | 238,239.39 ms | 0.56% | 2.36x |
| 8B | 128 | 1,655.63 ms | 0.55% | 766.44 ms | 0.71% | 2.16x |
| 8B | 2,048 | 26,684.81 ms | 0.40% | 15,426.68 ms | 2.65% | 1.73x |
| 14B | 128 | 2,523.51 ms | 0.15% | 1,221.35 ms | 0.88% | 2.07x |
| 14B | 2,048 | 45,328.53 ms | 3.93% | 24,350.93 ms | 3.31% | 1.86x |

| Model | F16 decode history | Ours | llama | Latency ratio |
|---|---:|---:|---:|---:|
| 3B | 128 | 29.28 tok/s | 35.72 tok/s | 1.22x |
| 3B | 2,048 | 25.40 tok/s | 27.56 tok/s | 1.09x |
| 3B | 8,192 | 12.15 tok/s | 19.89 tok/s | 1.64x |
| 3B | 16,384 | 6.90 tok/s | 12.27 tok/s | 1.78x |
| 3B | 28,000 | 4.40 tok/s | 8.11 tok/s | 1.84x |
| 8B | 128 | 15.53 tok/s | 17.94 tok/s | 1.16x |
| 8B | 2,048 | 13.76 tok/s | 15.68 tok/s | 1.14x |
| 14B | 128 | 10.43 tok/s | 12.38 tok/s | 1.19x |
| 14B | 2,048 | 9.01 tok/s | 10.81 tok/s | 1.20x |

The long 3B points validate the expected crossover. At 8K the profiler accounts
for 98.1% of GPU command time: attention is 31.98 s (45.8% of kernel time),
quantized Q/K/V/O and MLP matrices are 36.65 s (52.5%), and command residual is
1.34 s. Attention grows quadratically while matrix work grows linearly, so it
is the largest remaining 16K/28K bottleneck. The decode ratios remain inside
2x through KV28K even though absolute interactive throughput is low.

### INT8 KV matrix

| 3B prompt/history | TTFT | TTFT CV | Decode | F16 TTFT ratio |
|---:|---:|---:|---:|---:|
| 128 | 832.53 ms | 0.26% | 30.29 tok/s | 1.23x |
| 2,048 | 30,181.98 ms | 0.06% | 19.83 tok/s | 2.64x |
| 28,000 | 2,701,790.70 ms | 0.34% | 2.83 tok/s | 4.80x |

INT8 remains correct but intentionally does not enter M02's exact F16 tile.
The stable 45-minute/request 28K result makes the route gap algorithmic rather
than noisy. An equivalent llama row is unavailable under the fixed comparison
contract: this pinned llama.cpp rejects quantized V cache unless flash attention
is enabled. Enabling it would change the attention algorithm, so no mismatched
ratio is published.

## Final unified-memory conclusion

No hidden host/device copy appears in the hot path. Weights are uploaded once;
scratch, KV, and outputs remain shared; CPU reads occur only after the existing
command completion. At 8K, CPU record time is 0.33 s, submit is 0.002 s, and the
sampled command residual is 1.34 s against 71.14 s GPU time. Matrix and
attention kernels account for the competitive gap, not transfer plumbing.

A private-weight experiment was therefore not run: it would add a model-load
blit and a second storage policy without a measured removable transfer term.
The retained shared allocation is the simplest evidence-supported choice. The
only durable routing cleanup replaces the product-name gate with unified-memory,
Apple9-family, 128-thread, and 16-KiB threadgroup-resource capabilities.

## Experiment registry

| ID | Target | Hypothesis / architecture | Predicted | Measured | Decision |
|---|---|---|---:|---:|---|
| M00 | baseline | pinned same-device competitive map | n/a | rows above | KEEP evidence |
| M01 | attribution | stage-sampled contiguous operation passes | >=95% at prioritized prefill | 97.7% at 512; 97.6% at 2K | KEEP for investigation; REMOVE after attribution |
| M02 | early tiled GQA prefill | use the existing exact F16 tile from base zero | about 2.0x request; 63.6% gap recovery | 2.40x request; 74.7% gap recovery | KEEP production route |
| M03a | batched quantized matrix occupancy | eight SIMD groups, four accumulators/group, unchanged K32/output tile | 1.1--1.4x matrix family | 1.8% request at 128; 3.5% at 512 | REJECT modest |
| M03b | batched quantized matrix occupancy | sixteen SIMD groups, two accumulators/group, unchanged K32/output tile | exceed M03a | 15.9% regression at 128; 16.7% at 512 | REJECT regression |
| M04 | batched result staging | narrow SIMD FP32 matrices into F16 threadgroup storage, reducing the tile from 14 to 10 KiB | 1.2--1.5x matrix family if three groups reside | Metal requires matching matrix/store types; explicit storage conversion crashes compiler | REJECT infeasible |
| M05 | tiled prefill attention reuse | eight query rows per shared K/V tile instead of four; identical per-head online softmax | 1.15--1.35x attention; >0.75 s at 8K | 1.09x attention; 2.15% bracketed request gain | REJECT modest |

M03 isolated accumulator pressure from tile shape and dequantization. Doubling
from four to eight SIMD groups produced only a 1.8--3.5% request improvement;
doubling again regressed 15.9--16.7%. The weak midpoint is not retained because
it is below the specialization threshold and sits next to a sharp resource
cliff. Production code is restored to M02; the next matrix experiment must
change data movement or useful work rather than threadgroup size alone.

M04 failed before runtime and made no production change. Metal's SIMD-group
store requires identical accumulator and destination element types. An explicit
FP32-to-F16 matrix-storage conversion then terminated the Apple Metal compiler
itself. Retaining FP32 accumulation therefore requires the current FP32 staging
tile unless the graph's output representation changes, which is outside this
candidate's narrow scope.

M05 passed the serial-attention numeric oracle and was neutral within 1.3% at
128/512. Its first 8K screen was 66.09 s, but profiling measured only an 8.5%
attention reduction (31.98 to 29.26 s) and a 2.5% command reduction. The heated
8K M02/M05/M02 bracket measured 73.90/73.79/76.92 s, so M05 improved only 2.15%
against the bookend mean. The initial screen is retained as an outlier, not a
claim. Production code is restored to M02.

## Qualification status

At the specialization checkpoint, M02 passed the complete ordinary Metal suite
(136 passed, 2 ignored), the complete Metal-hybrid suite (202 passed, 2
ignored), diagnostics-free and profile-feature clippy with warnings denied,
and formatting. The production cleanup below subsequently removes the
investigation-only profile feature. M02's focused
numeric oracle compares tiled and serial attention at both base zero and base
512. Authenticated 3B teacher-forced lifecycle parity matches the pinned
llama.cpp oracle for both F16 and INT8 KV, including all 16 completion token
IDs. INT8, mixed placement, partial rows, and non-exact GQA remain on their
previous routes.

The performance stop condition is met by evidence, not by claiming universal
parity. M03 occupancy variants were modest or regressive, M04 could not retain
FP32 accumulation with smaller staging under Metal's type rules, and M05's
bracketed long-context gain was only 2.15%. These are three consecutive major
hypotheses without a meaningful retainable result. Production is restored to
M02 plus the capability gate.

Final F16 prefill is inside 2x at 512, 2K, and both 8B/14B 2K rows; it remains
2.02--2.36x at 3B 8K--28K and 2.07--2.16x at 8B/14B 128. Decode remains inside
1.85x everywhere measured. The largest remaining bottleneck is tiled F16
long-context attention, followed by quantized matrices at shorter contexts;
INT8 long prefill is a separate non-tiled route gap. No further candidate is
retained because the explicit consecutive-hypothesis stop rule has fired.

## Production cleanup qualification

### Baseline and scope

| Field | Value |
|---|---|
| Cleanup branch | `perf/metal-hardware-specialization` |
| Pre-cleanup HEAD | `fb39596ef33948d8c7670f9f6d4f2e3a5e9c6483` |
| Code-cleanup checkpoint | `91df4f5` |
| Capability-cleanup checkpoint | `d991225` |
| Local `main` and merge-base | `910a11689c3804b58f5216f2dbc40c60eabfa90c` |
| Initial worktree | clean |
| Push state | nothing pushed |

The complete pre-cleanup delta contained 13 changed source/manifest files and
this report. Eleven Rust files and two manifests contained production,
diagnostic, or embedded-test changes; four Rust files contained changed tests;
no shader, benchmark, fixture, dependency, or public API changed. The files
were classified before editing as device admission, command recording,
attention routing, embedded numeric/capability tests, investigation-only
profiling, and historical evidence.

### Removed

- Two experiment-only feature flags and four profiling modules were deleted.
- Seven profiling types, 18 classification variants, 24 state fields, and 13
  profiler-module functions were removed together with their host integration
  hooks.
- The counter sample buffer, aggregate mutex, per-command mark vector, active
  sampled encoder, phase/matmul classification state, CPU timers, timestamp
  readback, and stderr report disappeared from production.
- The profiling-only command-encoder wrapper, conditional encoder ownership,
  `Device` drop report, and `Kernel` comparison/debug derives disappeared. The
  ordinary single-encoder implementation now matches `main` exactly.
- No rejected occupancy, result-staging, or alternate-attention shader survived
  into the pre-cleanup production tree; those candidates remain historical
  evidence only.

No live production function, shader, buffer binding, push constant, descriptor,
quantization path, fallback, or public API was removed.

### Simplified

- Device admission remains capability-based and now has one shared predicate
  for runtime probing and boundary tests. It requires unified memory, Apple9,
  at least 128 one-dimensional threads per threadgroup, and at least 16 KiB of
  threadgroup memory. The latter limits cover the live 128-thread attention
  dispatch and 16-KiB attention threadgroup state.
- The retained exact F16, 32-row, head-dimension-128, width-32, 4:1-GQA tile is
  eligible from base zero. INT8, mixed placement, partial rows, other head
  dimensions, other SIMD widths, and other GQA ratios retain their prior
  routes.
- The tiled/serial oracle retains both base-zero and base-512 cases while
  calculating its invariant buffer size once.
- Weight execution, decode routing, resource lifetimes, backend interfaces,
  and shader ABI required no cleanup change.

### Retained feature map

| Retained feature | Files / route | Protection and reason |
|---|---|---|
| Capability-based Metal admission | `metal/device.rs` -> `Device::qualified` | Boundary test rejects each missing capability; removes exact product-name routing while preserving the shader resource contract |
| Early tiled GQA prefill | `metal/kernels/attention.rs` -> mode 4 | Geometry boundary test plus serial/tiled numeric oracle at base zero and 512; retained M02 performance result |
| Base-zero numeric coverage | `metal/kernels/mod.rs` | Proves the newly admitted tile against the established serial path without changing tolerances |

### Same-source preservation

The untouched candidate and cleanup checkpoint used the same M4, release mode,
authenticated 3B Instruct Q4_K_M artifact, 32,768-token context allocation,
F16 KV, 128 prompt tokens, 32 requested tokens, one warm-up, and three measured
repetitions.

| Metric | Before (CV) | After (CV) | Delta |
|---|---:|---:|---:|
| TTFT | 671.97 ms (.0021) | 672.99 ms (.0053) | +0.15% |
| Decode public deltas | 29.67/s (.0028) | 29.75/s (.0038) | +0.27% |

Both changes are below the 1% neutrality threshold and favor neither a new
route nor an algorithm change. **SAME-SOURCE PERFORMANCE PRESERVATION: PASS.**

The pinned `13f2b28b0` CPU llama.cpp protocol was run before and after cleanup
on the authenticated 3B Metal/F16 row. Both runs produced identical 16-token
oracle and local-greedy sequences, and every teacher token passed the local
top-two gate. **SAME-SOURCE QUALITY PRESERVATION: PASS.** No tolerance, input,
artifact, sampling parameter, or generation protocol changed.

Final local validation passes formatting, diff whitespace, and warning-denied
CPU, Metal, and Metal-hybrid Clippy. The workspace root passes 166 tests under
each profile. Engine results are CPU 156 passed / 2 ignored, Metal 136 passed /
2 ignored, and Metal-hybrid 202 passed / 2 ignored. Each integration target
passes 5 / ignores 1; each semantic target passes 12 / ignores 1. Documentation
and source contracts, shader pipeline loading, the base-zero attention oracle,
and the authenticated teacher row pass. The removed profile feature is not a
production qualification target.

### Historical unavailable fixture

The older six-artifact, 128-step teacher gate remains **HISTORICAL FIXTURE
UNAVAILABLE**. As already exhaustively recorded in the merged AMD cleanup,
repository history retains its generating protocol and artifact identities but
not the six exact teacher sequences or fixture hashes. That audit covered
tracked fixtures, tests, documentation, all refs and tags, available artifacts,
and local worktrees. Reconstructing the teacher would be non-authoritative, so
this cleanup does not repeat the search or alter production code for it.

A future deterministic fixture should persist the model hash, exact input and
teacher tokens, expected top-two data, step count, protocol version, tolerances,
reference configuration, and fixture hash.

### Production delta

Source/manifests including embedded tests changed from 13 files with 683
insertions and 97 deletions against `main` (net +586) before cleanup to three
Rust files with 100 insertions and 66 deletions (net +34). The cleanup checkpoints
remove 552 net lines. These counts are diagnostic; deletion was retained
only where it reduced production complexity.

### Remaining differences from `main`

| Component | Required justification |
|---|---|
| Metal device admission and boundary test | Capability-based supported-device contract and exact resource minima |
| Attention geometry and route test | Retained M02 base-zero tiled-GQA eligibility with unchanged fallbacks |
| Tiled/serial numeric oracle | Correctness coverage for both newly admitted and prior context bases |
| This report | Durable device, experiment, qualification, cleanup, and missing-fixture evidence |

### Rejected cleanup candidates

| Candidate | Reason retained |
|---|---|
| Remove Apple9/resource admission | Current shaders require the family and measured thread/threadgroup resources; unsupported devices must fail before pipeline use |
| Restore the base-512 tile restriction | Would delete the retained M02 optimization and its measured 2.40x 512-token result |
| Remove `base` from attention geometry | Live serial, segmented, and wide-decode fallbacks still select by context |
| Delete the eight-argument geometry lint boundary | Bundling independent routing inputs into a temporary struct adds a non-production abstraction; the narrow private function remains clearer |
| Remove base-512 numeric coverage | It protects the previously qualified context route while base-zero covers the new eligibility |

### External qualification status

| Qualification | Status | Reason |
|---|---|---|
| Hardware-independent build/test gates | PASS | Formatting, source contracts, and warning-denied local checks pass |
| Local Metal static, numeric, and runtime gates | PASS | Pure/hybrid builds and suites, shader pipeline loading, focused numeric tests, and teacher parity pass on the qualified M4 |
| Local Metal cleanup performance | PASS | Same-session TTFT and decode deltas are within 0.3% |
| Six-artifact historical 128-step teacher | HISTORICAL FIXTURE UNAVAILABLE | Exact teachers and fixture hashes were never persisted |
| NVIDIA/AMD cleanup performance | NOT APPLICABLE | This delta modifies only Metal-specific source and embedded tests; no shared runtime path changed |

### Final assessment

Every remaining production difference from `main` is required by the retained
capability admission, M02 attention route, or its boundary/numeric coverage.
All changed production files were audited; investigation scaffolding, temporary
state, profiling resources, and experiment-only configuration are absent.
Locally reproducible behavior, quality, and performance are preserved, while
the unavailable historical fixture remains a provenance issue rather than a
cleanup blocker. The branch is integration-ready with no temporary probes and
nothing pushed.
