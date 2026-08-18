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
- Device acquisition currently requires the product string `Apple M4`, unified
  memory, and Apple9. The string gate conflicts with this mission and must be
  replaced by a capability/resource gate before final qualification.
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
the first attribution target. 8K/16K/28K and 8B/14B rows remain pending.

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
`dispatch.rs`, and `pipeline.rs` remain below 200 productive lines per
orchestration file. The diagnostic feature uses stage-sampled compute passes
inside the original command buffer because this Apple9 device does not expose
dispatch-boundary sampling. Ordinary `metal` builds retain the single-encoder
path at compile time.

The first 32-row request proved that an 8,192-sample buffer is rejected by the
device while 4,096 is accepted. The retained diagnostic design therefore uses
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

## Experiment registry

| ID | Target | Hypothesis / architecture | Predicted | Measured | Decision |
|---|---|---|---:|---:|---|
| M00 | baseline | pinned same-device competitive map | n/a | rows above | KEEP evidence |
| M01 | attribution | stage-sampled contiguous operation passes | >=95% at prioritized prefill | 97.7% at 512; 97.6% at 2K | KEEP diagnostic feature |
| M02 | early tiled GQA prefill | use the existing exact F16 tile from base zero | about 2.0x request; 63.6% gap recovery | 2.40x request; 74.7% gap recovery | KEEP production route |

## Qualification status

M02 passes the complete ordinary Metal suite (136 passed, 2 ignored), the
complete Metal-hybrid suite (202 passed, 2 ignored), diagnostics-free and
profile-feature clippy with warnings denied, and formatting. Its focused
numeric oracle compares tiled and serial attention at both base zero and base
512. Authenticated 3B teacher-forced lifecycle parity matches the pinned
llama.cpp oracle for both F16 and INT8 KV, including all 16 completion token
IDs. INT8, mixed placement, partial rows, and non-exact GQA remain on their
previous routes.

The global stop conditions are not met. At short prefill the remaining 128-token
ratio is 2.09x, and the long-context/model-size matrix remains incomplete. The
post-M02 top measured family is quantized projection and MLP matrix work; the
ranking will be rebuilt across longer contexts before selecting M03.
