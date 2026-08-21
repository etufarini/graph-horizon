<!--
This checkpoint owns the reproducible evidence for the Apple Metal long-context
decode investigation: authenticated inputs, measurements, decisions, and the
remaining bottleneck. It does not define the runtime support contract.
-->

# Metal Long-Context Decode Optimization

Status: active investigation on `perf/metal-long-context-decode`.

The production baseline is `87e3e4b1bb17e811caac030f69261bd0498284ec`.
The only retained change so far is richer benchmark reporting; the profiler is
temporary diagnostic code and must be removed before final qualification.

## Environment And Inputs

- Device: MacBook Air `Mac16,13`, Apple M4 with 10 GPU cores and 24 GB unified
  memory, Metal 4, macOS 26.3 (`25D125`), connected to AC power.
- Model: `Ministral-3-14B-Instruct-2512-Q4_K_M.gguf`, 8,239,593,024 bytes,
  SHA-256 `824e0f3373e69b84f2cae46fdcb9bd1ebc6ab3bfc7acc125d818b7b8178cc613`.
- Model shape: 40 blocks, hidden width 5,120, 32 query heads, 8 KV heads,
  head dimension 128, and 16,384-wide feed-forward layers.
- KV configuration: F16, 32,768-token allocation. The full allocation is
  10.74 GB; 28,000 populated positions contain 9.18 GB of K/V data.
- Comparator: pinned llama.cpp checkout
  `2b63e0610bbc2be990ae1360d5256efcdc3f9efb`.

The synthetic prompt is the string `"a "` repeated until the tokenizer reports
the requested prompt count. Commands use a release Metal build and disable
benchmark warm-up unless a run explicitly says otherwise.

## Baseline Curve

Each point below is one 128-token decode in a single ascending-context session.
`model tok/s` uses the engine's terminal decode duration. Interval statistics
come from adjacent public decode events; completion and interval counts matched,
so those intervals are model-token evidence for these runs.

| Prompt tokens | Model tok/s | Mean interval | Interval CV | Beginning / middle / end tok/s | TTFT |
|---:|---:|---:|---:|---:|---:|
| 128 | 10.12 | 99.58 ms | 3.39% | 10.45 / 10.05 / 9.66 | 3.34 s |
| 2,048 | 11.33 | 88.92 ms | 0.68% | 11.25 / 11.26 / 11.24 | 23.27 s |
| 8,192 | 7.62 | 132.25 ms | 13.45% | 7.90 / 7.52 / 7.28 | 123.42 s |
| 15,434 | 5.94 | 169.71 ms | 11.12% | 6.09 / 5.83 / 5.75 | 313.26 s |
| 28,000 | 4.56 | 220.96 ms | 9.77% | 4.70 / 4.44 / 4.45 | 742.12 s |

A separate 1,156-token sustained decode at 15,434 prompt tokens measured 5.83
public tok/s. GPU utilization remained 99%, renderer/tiler utilization settled
near 79--82%, memory footprint was about 13.7 GB, and macOS reported no thermal
or performance warning. The declining thirds and fanless device make thermal
state part of any same-session comparison, but do not explain away the strong
context-length slope.

A diagnostic linear fit to the interval means is
`T(N) = 91.487 ms + 0.0047207 ms * N`. Its fitted KV-dependent fraction is
44.3% at 15,434 tokens and 59.1% at 28,000 tokens. The 128- and 2,048-token
points show clock-ramp effects, so this fit is attribution evidence, not a
latency predictor.

## 15K Decode Attribution

A feature-gated Metal trace measured 31 decode command buffers. Profiler
boundaries perturb the endpoint, so category shares are used for prioritization
while the uninstrumented curve above remains the performance baseline.

| Category | Per token | Share of classified kernel time |
|---|---:|---:|
| Grouped attention | 66.16 ms | 39.2% |
| Gate projection | 25.99 ms | 15.4% |
| Up projection | 21.70 ms | 12.8% |
| Down projection | 25.90 ms | 15.3% |
| Q, K, V, and O projections | 17.98 ms | 10.6% |
| Logits | 6.36 ms | 3.8% |
| Norm, RoPE, KV write, and elementwise | 4.78 ms | 2.8% |

Classified kernels consumed 168.92 ms/token and the full GPU interval consumed
178.64 ms/token. CPU recording plus submission was 1.68 ms/token; CPU wait was
179.41 ms/token. Sampling was a separate 1.63 ms/token. The GPU critical path,
not host dispatch, is the active limit.

All seven weight projections plus logits took 97.92 ms/token and stream at
about 80.2 GB/s in the hot trace. Their exact weight traffic is 7.852 GB/token.
At a nominal 120 GB/s the all-weight floor alone is 65.43 ms, before attention
and non-projection work. Previous same-format GEMV specialization and residency
experiments did not reduce this traffic and were rejected.

F16 K/V reads have an exact lower bound of 163,840 bytes per history position
per generated token: 2.529 GB at 15,434 positions and 4.588 GB at 28,000. The
retained four-query-head grouped kernel reads each K/V tile once per layer and
uses four disjoint history splits, so it is already near 1x exact K/V traffic.
Its 15K attention time implies only about 38.2 GB/s effective K/V bandwidth,
including dot products, softmax, value accumulation, barriers, and reduction.
This rules out a simple redundant-read explanation and does not yet justify a
lossy KV representation.

## Candidate Ledger

### Retained Support

- The benchmark now reports completion-token throughput, adjacent-transition
  mean/median/standard deviation/CV, and beginning/middle/end thirds. This
  separates model work from public event semantics and exposes within-run drift.

### Rejected In This Investigation

| Candidate | Correctness | Measurement | Decision |
|---|---|---|---|
| SIMD-matrix 8x8 grouped decode, four history splits | GPU-oracle checks passed at 1K and 2K within 0.02 | 8K A/B/A: candidate 7.26 tok/s versus 7.07 mean baseline, +2.8% | Reverted: below the 5% retention gate and adds reduction-order complexity |
| 64-position shared-memory tile | GPU-oracle checks passed at 1K and 2K | 8K A/B: 6.78 versus 7.25 tok/s, -6.5% | Reverted: 32 KiB threadgroup memory reduced occupancy more than fewer barriers helped |
| Four-position block online softmax | GPU-oracle checks passed at 1K and 2K within 0.02; all 144 engine tests passed | 8K hot boundaries: candidate end 7.36 vs next baseline begin 7.46 tok/s; baseline end 6.98 vs next candidate begin 6.92 tok/s | Reverted: reciprocal transition evidence is about -1%; lower arithmetic was offset by added register state |

Historical results also close the obvious alternatives: a grouped kernel without
history splits regressed 16% at 2K; eight splits gained 2.5% at 8K and regressed
1.3% at 28K; per-format Q4/Q6 GEMV specialization was flat; and explicit
residency hints regressed the comparator by 3.6%.

## Current Decision

Attention is still the largest context-sensitive opportunity, but three local
kernel changes failed the retention gate and exact K/V traffic is already minimal.
The structural comparison with llama.cpp's vector flash-attention path found
that llama.cpp uses 32 workgroups per query head, one to four SIMD groups per
workgroup, a 32-position block softmax, and a second global reduction. That
layout reloads shared GQA K/V for each query head, whereas this backend uses
four workgroups per KV head and stages each tile once for its four query heads.
The tested block-softmax transfer did not improve the grouped kernel because its
extra score and weight arrays increased register state. More history splits and
larger tiles are already closed by measured regressions, so the exact F16
attention-kernel frontier currently has no candidate above the economic gate.

After attention, weight streaming is the dominant fixed floor. Any projection
candidate must remove material weight reads or whole passes; dispatch-only or
activation-only fusion cannot meet the end-to-end gate on this model.
