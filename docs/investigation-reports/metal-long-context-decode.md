<!--
This checkpoint owns the reproducible evidence for the Apple Metal long-context
decode investigation: authenticated inputs, measurements, decisions, and the
remaining bottleneck. It does not define the runtime support contract.
-->

# Metal Long-Context Decode Optimization

Status: completed investigation on `perf/metal-long-context-decode`.

Repository status: the branch was later pushed with final tip `f0c3120` and
merged into `main` by PR #36 (`ae69781`).

The production baseline is `87e3e4b1bb17e811caac030f69261bd0498284ec`.
No production kernel change is retained. Richer benchmark reporting is commit
`0899b74`; the final attribution profiler was removed by `49499e1`. The last production
candidate `54c1621` was removed non-destructively by `c4a49f2` after sustained
generation failed its retention gate. Final production-equivalent source before
this report is therefore `c4a49f2`. At report close the checkpoint used the
symbolic branch tip `refs/heads/perf/metal-long-context-decode`; its later pushed
tip is `f0c3120`.

## Environment And Inputs

- Device: Apple silicon validation host with 10 GPU cores and 24 GB unified
  memory, Metal 4, macOS 26.3 (`25D125`), connected to AC power.
- Model: `Ministral-3-14B-Instruct-2512-Q4_K_M.gguf`, 8,239,593,024 bytes,
  SHA-256 `824e0f3373e69b84f2cae46fdcb9bd1ebc6ab3bfc7acc125d818b7b8178cc613`.
- Model shape: 40 blocks, hidden width 5,120, 32 query heads, 8 KV heads,
  head dimension 128, and 16,384-wide feed-forward layers.
- KV configuration: F16, 32,768-token allocation. The K and V buffers total
  5.37 GB logically; 28,000 populated positions contain 4.59 GB of K/V data.
- Comparator: pinned llama.cpp checkout
  `2b63e0610bbc2be990ae1360d5256efcdc3f9efb`.

### Apple Metal capability audit

Device names below are evidence labels only. Production routing uses family,
shape, format, placement, and resource predicates.

| Property | Queried value |
|---|---:|
| GPU families | Apple7 / Apple8 / Apple9, Common3, Metal3 / Metal4 |
| Unified memory | yes |
| Device maximum threads/threadgroup | 1,024 |
| Device maximum threadgroup memory | 32,768 bytes |
| Pipeline `threadExecutionWidth` | 32 for every retained pipeline |
| Pipeline maximum threads/threadgroup | 1,024 for every retained pipeline |
| Grouped split attention static threadgroup memory | 16,384 bytes |
| Grouped unsplit attention static threadgroup memory | 20,544 bytes |
| Recommended maximum working set | 19,069,665,280 bytes |
| Maximum Metal buffer | 14,302,248,960 bytes |
| Shared heap-buffer alignment query | 256 bytes |
| Runtime sub-buffer alignment contract | 4 bytes |
| Dispatch-boundary counters | unavailable |
| Stage-boundary timestamp counters | available and used |

SIMD reductions/shuffles, 8x8 SIMD matrices, F16/F32 and byte/word integer
arithmetic are compiled and exercised. The Metal 4 cooperative-tensor
projection also compiles and executes on this device. All runtime buffers use
shared storage: weights are copied once at model load; weights, scratch, K/V,
and outputs remain in unified memory thereafter. Register count, occupancy,
spill, cache-miss, ALU-stall, and per-process power/clock counters were not
exposed by the available stage sampler. `powermetrics` required superuser
access and was not used. OS telemetry reported no thermal or performance
warning and no GPU recovery throughout the qualified runs.

The synthetic prompt is the string `"a "` repeated until the tokenizer reports
the requested prompt count. Commands use a release Metal build and disable
benchmark warm-up unless a run explicitly says otherwise.

The reproducible Graph Horizon command shape is:

```sh
cargo build --locked --release --no-default-features --features metal --example bench
target/release/examples/bench <model.gguf> --context 32768 --kv f16 \
  --prompt <exact-N-token-synthetic-prompt> --max-tokens 128 --warmup 0 --reps 1
```

The retained capacity planner and temporary attribution build were invoked as:

```sh
cargo run --locked --release --no-default-features --features metal-hybrid \
  --example profile -- <model.gguf> 32768 f16
cargo build --locked --release --no-default-features \
  --features metal,metal-profile --example bench
target/release/examples/bench <model.gguf> --context 32768 --kv f16 \
  --prompt <exact-N-token-synthetic-prompt> --max-tokens 32 --warmup 0 --reps 1
```

`metal-profile` existed only on the investigation branch while collecting the
stage timestamps. Its commits and final removal remain in ordinary Git history;
the feature and profiler source are absent from the handoff tree.

The comparator used `llama-server` with `-c 32768 -ngl 99 -fa on -b 64 -ub 64
-ctk f16 -ctv f16`. Each `/completion` request disabled prompt-cache reuse and
supplied exact token IDs `[1, 3, 1097]`, token `1261` repeated `N - 5` times,
then `[1032, 4]`, with 128 predicted tokens and temperature zero.

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

| Final scaling model | Value |
|---|---:|
| Fixed term `A` | 91.487 ms/token |
| Context slope `B` | 0.0047207 ms/token/position |
| KV-dependent fraction at 15,434 | 44.3% |
| KV-dependent fraction at 28,000 | 59.1% |

### Independent baseline audit

A second diagnostics-free ascending run recorded the distribution fields that
were omitted from the first checkpoint. It used the same binary, prompt,
32,768-token allocation, and 128-token generation. This is a reproducibility
audit, not a replacement for the first curve: the fanless machine was in a
different accumulated thermal state.

| KV | Model tok/s | Public tok/s | Mean | Median | Stddev | CV | TTFT |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 10.50 | 10.42 | 95.96 ms | 96.13 ms | 3.20 ms | 3.33% | 1.91 s |
| 2,048 | 12.02 | 11.93 | 83.85 ms | 83.69 ms | 0.63 ms | 0.75% | 20.83 s |
| 8,192 | 8.59 | 8.53 | 117.30 ms | 116.90 ms | 12.77 ms | 10.89% | 108.62 s |
| 15,434 | 6.78 | 6.73 | 148.61 ms | 147.30 ms | 17.72 ms | 11.92% | 309.92 s |
| 28,000 | 5.01 | 4.97 | 201.33 ms | 203.33 ms | 14.31 ms | 7.11% | 766.97 s |

The 2K point meets the requested 1% CV target. The longer points do not:
begin/middle/end drift on a fanless GPU is material, so reporting the variance
and using same-session A/B/A controls is more defensible than shortening or
filtering the workload. During the run, sampled utilization reached 99% device
and 94--99% renderer/tiler at 15K--28K; an earlier 2K sample was 79% device and
59% renderer/tiler during ramp. Some short points completed between system
polls, so their utilization is recorded as unavailable. Numeric clock, power,
and temperature readings were unavailable without privileged telemetry.

## Vector-Staging Result — Rejected

The candidate copied each staged K/V tile as aligned `half4` vectors
instead of scalar halves. Each thread executes four load/store/address steps per
tile rather than sixteen. Bytes, threadgroup layout, four history splits,
online-softmax arithmetic, reduction order, and eligibility are unchanged.

| Qualification | Baseline | Candidate | Delta |
|---|---:|---:|---:|
| 128, same-session guard | 10.13 tok/s | 10.13 tok/s | neutral; route is ineligible |
| 2K, hot A/B/A | 10.39 tok/s control mean | 10.39 tok/s | neutral |
| 8K, hot A/B/A | 6.805 tok/s control mean | 7.19 tok/s | +5.7% |
| 15,434, candidate/baseline/candidate | 5.55 tok/s | 5.92 tok/s candidate mean | +6.7% |
| 28K sustained endpoint | 4.56 tok/s | 4.86 tok/s | +6.6% |

The 8K component reprofile measured attention falling from 31.07 to 21.68
ms/token, a 30.2% local reduction. Profiled GPU plus sampling time fell from
126.23 to 117.81 ms/token, or 6.7%. At 28K, the diagnostics-free public interval
fell from 220.96 to 207.55 ms/token (6.1%) even though final TTFT rose from
742.12 to 805.56 seconds under the hotter sustained session. Prefill uses an
unchanged kernel; this TTFT movement is the thermal guard, not a code effect.

The required 15,434-prompt, 1,156-token generation rejected the candidate. It
measured 5.82 tok/s, 171.91 ms mean intervals, and 6.13 / 5.70 / 5.65 tok/s
thirds, versus the fresh baseline's 5.83 tok/s over the same 1,156 tokens.
Therefore the local and short-endpoint gain did not survive the primary sustained
workload. The candidate is not production code despite its useful mechanism.

## Decode Attribution

A first feature-gated Metal trace measured 31 decode command buffers. Profiler
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

The full baseline context matrix below includes GPU interval residual explicitly,
so each row accounts for 100% of the traced GPU interval plus sampling. “Small”
is the measured sum of normalization, RoPE, KV append, elementwise, and embedding.

| KV | Attention | Seven projections | Logits | Small | GPU residual | Sampling | Reconstructed total |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 12.17 | 71.07 | 5.16 | 2.16 | 5.99 | 0.98 | 97.53 ms |
| 2,048 | 6.20 | 70.97 | 5.17 | 2.19 | 5.78 | 0.98 | 91.28 ms |
| 8,192 | 31.07 | 76.46 | 5.62 | 2.97 | 8.62 | 1.49 | 126.23 ms |
| 15,434 | 66.16 | 91.56 | 6.36 | 4.83 | 9.72 | 1.63 | 180.27 ms |
| 28,000 | 108.02 | 81.68 | 5.76 | 3.20 | 9.46 | 1.47 | 209.59 ms |

The completion audit repeated the stage trace with attention scan/reduction and
activation/residual boundaries split. Every row below is milliseconds per
public token transition. `GPU envelope` is the unsampled interval between the
sampled stage runs and the command-buffer envelope; `host/event` is the
remainder needed to reconcile the profiled forward plus sampling GPU time to
the public adjacent-delta interval. Thus every row accounts for 100% of token
wall time, even where the profiler itself adds encoder-transition overhead.

| KV | Attention scan | Attention reduce | Q | K | V | O | Gate | Up | Down | Logits | Norm | RoPE | KV append | Activation | Residual | Embed | GPU envelope | Sampling | Host/event | Total |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 12.95 | 0.00 | 4.94 | 1.53 | 1.80 | 5.14 | 18.56 | 18.63 | 22.85 | 5.35 | 1.08 | 0.36 | 0.66 | 0.14 | 0.56 | 0.01 | 7.84 | 0.99 | 1.87 | 105.24 |
| 2,048 | 6.11 | 0.24 | 4.87 | 1.47 | 1.70 | 5.22 | 18.23 | 18.77 | 22.91 | 5.48 | 1.15 | 0.20 | 0.65 | 0.16 | 0.52 | 0.01 | 7.99 | 0.99 | 1.99 | 98.65 |
| 8,192 | 26.04 | 0.50 | 5.01 | 1.49 | 1.77 | 5.59 | 18.83 | 18.73 | 23.16 | 5.40 | 1.22 | 0.29 | 0.76 | 0.23 | 0.79 | 0.01 | 7.04 | 1.23 | 6.78 | 124.86 |
| 15,434 | 48.95 | 0.24 | 5.09 | 1.42 | 1.75 | 5.28 | 19.45 | 18.43 | 22.65 | 5.36 | 1.00 | 0.20 | 0.80 | 0.20 | 0.41 | 0.01 | 8.68 | 1.23 | 3.77 | 144.91 |
| 28,000 | 89.11 | 0.17 | 4.93 | 1.48 | 1.67 | 5.33 | 19.29 | 18.62 | 22.58 | 5.26 | 1.15 | 0.22 | 0.78 | 0.13 | 0.30 | 0.01 | 8.93 | 1.21 | 3.33 | 184.49 |

The required 15K component budget is the corresponding row normalized to its
144.91 ms public-token total:

| 15K component | ms/token | Share |
|---|---:|---:|
| Attention scan | 48.95 | 33.78% |
| Attention reduction | 0.24 | 0.17% |
| Q / K / V / O | 5.09 / 1.42 / 1.75 / 5.28 | 3.51% / 0.98% / 1.21% / 3.64% |
| Gate / up / down | 19.45 / 18.43 / 22.65 | 13.42% / 12.72% / 15.63% |
| Logits | 5.36 | 3.70% |
| Norm / RoPE / KV append | 1.00 / 0.20 / 0.80 | 0.69% / 0.14% / 0.55% |
| Activation / residual / embedding | 0.20 / 0.41 / 0.01 | 0.14% / 0.28% / 0.01% |
| GPU envelope | 8.68 | 5.99% |
| Sampling | 1.23 | 0.85% |
| Host/event | 3.77 | 2.60% |

Baseline resource fields are summarized separately because profiler GPU time
and diagnostics-free wall time come from different runs and must not be
subtracted from each other.

| KV | Diagnostics-free CPU wall | Profiled forward + sampling GPU | Metal allocation | Utilization sample | Clock / power / numeric temp |
|---:|---:|---:|---:|---|---|
| 128 | 95.96 ms | 103.37 ms | 13.605 GB | unavailable between polls | unavailable |
| 2,048 | 83.85 ms | 96.66 ms | 13.605 GB | 79% device, 59% renderer/tiler ramp | unavailable |
| 8,192 | 117.30 ms | 118.08 ms | 13.605 GB | unavailable between polls | unavailable |
| 15,434 | 148.61 ms | 141.14 ms | 13.605 GB | 99% device, 94% renderer/tiler | unavailable |
| 28,000 | 201.33 ms | 181.16 ms | 13.605 GB | 99% device, 96--99% renderer/tiler | unavailable |

At 15K, the forward GPU interval is 139.91 ms, GPU sampling is 1.23 ms,
and the public interval is 144.91 ms. The scan and reduction are separate
dispatches. Layer and final normalization share one identical 5,120-wide
pipeline: 80 layer invocations and one final invocation per forward imply
approximately 0.99 ms layer normalization and 0.01 ms final normalization at
15K. KV management performs checked offset/view arithmetic during the measured
host recording term; it allocates, resizes, maps, and copies zero bytes per
token. There are no explicit Metal memory barriers. Dependencies are the
ordered dispatch stream inside one compute encoder.

The stage sampler cannot timestamp instructions inside a fused shader. The
following mandatory constituents are therefore separated by exact traffic and
operation counts rather than falsely assigned additive milliseconds:

- `attention scan` jointly owns K-cache load, QK, online softmax, V-cache load,
  and AV. K and V are staged together before those operations overlap.
- Each history position at 15,434 contributes 1.264 GB K and 1.264 GB V,
  5.057 GFLOP QK, 5.057 GFLOP AV, and 19.76 million online-softmax scores.
- The separate eight-part reduction is directly measured at 0.24 ms/token.
- Quantized load, metadata decode, integer unpack, conversion, dot product, and
  SIMD reduction are fused inside each projection and are audited below.

No broad category is omitted: fused suboperations retain exact byte/work
ownership, while measured stage rows plus the two explicit residuals reconstruct
the public wall clock.

At 128 the unsplit attention route explains the non-monotonic curve. At 2K the
four-split route is already active and lowers attention despite longer history.
At 28K attention is 51.5% of reconstructed token time. The split profiler
perturbs endpoint timing most at 8K/15K, so uninstrumented intervals remain the
performance contract.

In the first, hotter trace, all seven weight projections plus logits took 97.92 ms/token and streamed at
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

At 15,434 positions, K reads and V reads are 1.2645 GB each. QK and AV each
perform about 5.06 GFLOP/token across 40 layers; the online softmax processes
19.76 million query-head/history scores. One-way partial/state traffic is about
5.4 MB/token, or about 11 MB including the reducer read. These constituents
are fused inside the measured 66.16 ms kernel
and cannot be timed independently without changing its execution, but their
exact byte and operation counts bound the roofline: 21.1 ms at a theoretical
120 GB/s for K/V alone, approximately 31.6 ms at the observed 80 GB/s hot
streaming rate, before dot products, exponentials, barriers, and reduction.

The refined trace makes the scaling classification sharper. At 2K, 8K, 15K,
and 28K the exact grouped route measures 52.8, 50.6, 51.4, and 51.4 GB/s of
logical K/V traffic, and 211, 202, 206, and 206 GFLOP/s respectively. Its
arithmetic intensity is 4 FLOP per K/V byte. The 128 fallback instead reloads
shared GQA K/V once per query head: 83.9 MB actual versus 21.0 MB theoretical
minimum, 4x amplification, about 6.5 GB/s and 6.5 GFLOP/s. At 2K and above,
four disjoint history workgroups per KV head each emit two band partials, so
the reducer combines eight partial states per query head while global K/V
traffic remains exactly 1x.

| KV | K minimum | V minimum | Actual K+V | Other device traffic | QK+AV work | Intensity | Attention time | Effective BW | Effective compute |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 10.49 MB | 10.49 MB | 83.89 MB | below 1 MB | 0.084 GF | 1.0 FLOP/B | 12.95 ms | 6.5 GB/s | 6.5 GF/s |
| 2,048 | 167.77 MB | 167.77 MB | 335.54 MB | about 11 MB | 1.342 GF | 4.0 FLOP/B | 6.35 ms | 52.8 GB/s | 211 GF/s |
| 8,192 | 671.09 MB | 671.09 MB | 1.342 GB | about 11 MB | 5.369 GF | 4.0 FLOP/B | 26.54 ms | 50.6 GB/s | 202 GF/s |
| 15,434 | 1.264 GB | 1.264 GB | 2.529 GB | about 11 MB | 10.115 GF | 4.0 FLOP/B | 49.19 ms | 51.4 GB/s | 206 GF/s |
| 28,000 | 2.294 GB | 2.294 GB | 4.588 GB | about 11 MB | 18.350 GF | 4.0 FLOP/B | 89.29 ms | 51.4 GB/s | 206 GF/s |

`Other` is principally partial/state write plus reduction read and output; Q
reads are approximately 1.3 MB/token on the split route. The exact F16
K/V-only physical floor is 21.1 ms at 15K and 38.2 ms at 28K at 120 GB/s.
The practical measured exact-path envelope is about 46 ms at 15K after adding
QK/softmax/AV, partial reduction, and synchronization. The stable 51.4 GB/s
and 206 GF/s at 15K/28K show a streaming-plus-arithmetic kernel, not a pure
copy ceiling.

## Routing, Residency, And Capacity

| Operation | Shape per layer | Storage | Actual route |
|---|---|---|---|
| Q | 5,120 → 4,096 | Q4_K | packed two-row SIMD GEMV |
| K | 5,120 → 1,024 | Q4_K | packed two-row SIMD GEMV |
| V | 5,120 → 1,024 | 20 Q4_K + 20 Q6_K layers | matching packed SIMD GEMV |
| O | 4,096 → 5,120 | Q4_K | packed two-row SIMD GEMV |
| Gate / up | 5,120 → 16,384 | Q4_K | packed two-row SIMD GEMV |
| Down | 16,384 → 5,120 | 20 Q4_K + 20 Q6_K layers | matching packed SIMD GEMV |
| Attention ≥1,023 | dim 128, 32 Q / 8 KV heads | F16 KV | four history workgroups/KV head, eight partials/query head + reduce |
| Attention fallback | other shapes, schemes, placement, or shorter KV | F16/INT8 | generic exact Metal attention |

For every projection, expected and actual decode paths agree: `M=1`, F16
activation/output (FP32 logits), canonical Q4_K or Q6_K, pure-Metal placement,
K divisible by 256, and pipeline SIMD width 32 select the two-SIMD packed
kernel. Mixed placement retains the qualified one-SIMD fallback; F16/Q5 or
unaligned shapes retain the generic format path.

| KV | Datatype / shape | Expected and actual attention route | Predicate / fallback reason |
|---:|---|---|---|
| 128 | F16, dim 128, 32Q/8KV, GQA 4:1 | generic mode 1, one SIMD/query head | below medium and split crossovers; 4x shared-K/V reads |
| 2,048 | same | grouped split + reduce | pure Metal, base >=1,023, F16, dim 128, exact 4:1, SIMD32, 256 threads, 16 KiB TGM, scratch fits |
| 8,192 | same | grouped split + reduce | same predicate; no context-dependent fallback |
| 15,434 | same | grouped split + reduce | same predicate; no context-dependent fallback |
| 28,000 | same | grouped split + reduce | same predicate; no context-dependent fallback |

INT8, mixed placement, another head dimension/GQA ratio, insufficient pipeline
resources, or insufficient scratch uses the generic exact route. The measured
crossover is therefore capability- and shape-based at position 1,023; no
product or model string participates.

All major decode work is GPU resident. Model weights are persistent canonical
packed buffers; K/V is request-persistent; pipelines and execution buffers are
model/runtime lifetime; no per-token allocation, map/unmap, resize, host/device
copy, or CPU fallback was observed. The split route requires pure Metal, F16 KV,
head dimension 128, exact 4:1 GQA, context at least 1,023, SIMD width 32, 256
threads, 16 KiB threadgroup memory, and sufficient scratch. Failure of any
predicate retains the generic path.

One long-context forward records 683 dispatches into one command buffer: 40
instances of every layer projection, 40 split and 40 reduce attention
dispatches, all small layer operations, embedding, final norm, and logits.
Greedy argmax uses one additional dispatch and command buffer. At KV128 the
unsplit attention route makes the forward count 643. Production therefore uses
two queue submissions and two synchronous waits per emitted token, zero
explicit barriers, and zero per-token buffer or pipeline allocations. The CPU
reads four shared bytes for argmax; no blit or CPU-to-GPU copy occurs.

| Lifetime | Resources and steady-state behavior |
|---|---|
| One-time/process | Embedded Metal library load and device discovery |
| Per model/runtime | Pipelines, packed weights, scratch, logits, reduction, and staging buffers |
| Per configured context | Capacity plan and fixed-size 32,768-position K/V allocation |
| Per request | Logical K/V contents and generation state; storage is reused without resize or recreation |
| Per token | Forward and sampling command-buffer/encoder objects only; no data-buffer, descriptor, pipeline, temporary-storage, map/unmap, or KV allocation |

| Populated KV | F16 K+V bytes | Model + populated KV | Observed process state |
|---:|---:|---:|---|
| 2,048 | 0.336 GB | 8.58 GB | GPU resident |
| 8,192 | 1.342 GB | 9.58 GB | GPU resident |
| 15,434 | 2.529 GB | 10.77 GB | about 13.7 GB peak footprint |
| 28,000 | 4.588 GB | 12.83 GB | about 16.6 GB maximum RSS, no Metal fallback |

The table above is populated-data pressure, not allocation size. The baseline
always configures 32,768 positions, so its Metal allocation is constant at
13,605,224,448 bytes: 8,230,348,800 weights, 5,368,709,120 K/V, 5,609,472
scratch, 540,672 fixed logits/reduction storage, and 16,384 staging. There is
no temporary per-token allocation. The planner additionally withholds
953,483,264 bytes as a capacity guard; that reserve is not an allocation.

| Capacity at N | Model weights | Populated K/V | Persistent execution | Populated pressure | 32K allocated pressure |
|---:|---:|---:|---:|---:|---:|
| 2,048 | 8.230 GB | 0.336 GB | 6.167 MB | 8.572 GB | 13.605 GB |
| 8,192 | 8.230 GB | 1.342 GB | 6.167 MB | 9.579 GB | 13.605 GB |
| 15,434 | 8.230 GB | 2.529 GB | 6.167 MB | 10.765 GB | 13.605 GB |
| 28,000 | 8.230 GB | 4.588 GB | 6.167 MB | 12.824 GB | 13.605 GB |

System-wide swap counters were nonzero and cannot be attributed to this
process. The defensible finding is narrower: context growth caused no Metal
placement change, CPU fallback, allocation churn, per-token host/device copy,
or route change after the 1,023-position crossover. The device stayed below
its queried working-set limit and reported zero GPU recoveries.

## Projection Inventory And Floors

Canonical packed bytes are read once per generated token. Q4_K blocks include
144 bytes per 256 weights; Q6_K blocks include 210. Metadata is already inside
those totals, so traffic amplification is 1x.

| Operation | Bytes/token | Refined 15K time | Effective bandwidth |
|---|---:|---:|---:|
| Q | 0.472 GB | 5.09 ms | 92.6 GB/s |
| K | 0.118 GB | 1.42 ms | 83.0 GB/s |
| V | 0.145 GB | 1.75 ms | 83.0 GB/s |
| O | 0.472 GB | 5.28 ms | 89.4 GB/s |
| Gate | 1.887 GB | 19.45 ms | 97.0 GB/s |
| Up | 1.887 GB | 18.43 ms | 102.4 GB/s |
| Down | 2.320 GB | 22.65 ms | 102.4 GB/s |
| Logits | 0.551 GB | 5.36 ms | 102.7 GB/s |

Input/output activation traffic is about 0.10% of the 7.852 GB packed-weight
stream. The refined 15K trace totals 79.43 ms at 98.9 GB/s; the fastest
individual rows reach about 103 GB/s, giving a 76.2 ms practical aggregate
floor. The 65.43 ms 120-GB/s bound is
physical, not an attainable whole-projection target. The current kernels match
llama.cpp's two-row Q4_K/Q6_K arithmetic and two-SIMD-group geometry. Earlier
per-format specialization was flat, so fusion that only removes activation or
dispatch traffic cannot meet the 5% whole-token gate.

The complete `M=1` inventory follows, ordered by plausible removable time.
`Metadata` is part of packed bytes:
Q4_K uses 128 payload + 16 metadata bytes per 256 weights; Q6_K uses 192 + 18.
Activation is the logical F16 input traffic; output is F16 except FP32 logits.
Every layer role has 40 dispatches/token; logits has one. Reduction is one SIMD
sum per output row after lanes share input blocks.

| Role | M/N/K | Quantization | Packed bytes | Metadata | Activation | Output | Dispatches | GPU time |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Gate | 1 / 16,384 / 5,120 | Q4_K | 1,887.437 MB | 209.715 MB | 409.600 KB | 1.311 MB | 40 | 19.45 ms |
| O | 1 / 5,120 / 4,096 | Q4_K | 471.859 MB | 52.429 MB | 327.680 KB | 409.600 KB | 40 | 5.28 ms |
| Q | 1 / 4,096 / 5,120 | Q4_K | 471.859 MB | 52.429 MB | 409.600 KB | 327.680 KB | 40 | 5.09 ms |
| V | 1 / 1,024 / 5,120 | 20 Q4_K + 20 Q6_K | 144.998 MB | 13.926 MB | 409.600 KB | 81.920 KB | 40 | 1.75 ms |
| K | 1 / 1,024 / 5,120 | Q4_K | 117.965 MB | 13.107 MB | 409.600 KB | 81.920 KB | 40 | 1.42 ms |
| Down | 1 / 5,120 / 16,384 | 20 Q4_K + 20 Q6_K | 2,319.974 MB | 222.822 MB | 1.311 MB | 409.600 KB | 40 | 22.65 ms |
| Up | 1 / 16,384 / 5,120 | Q4_K | 1,887.437 MB | 209.715 MB | 409.600 KB | 1.311 MB | 40 | 18.43 ms |
| Logits | 1 / 131,072 / 5,120 | Q6_K | 550.502 MB | 47.186 MB | 10.240 KB | 524.288 KB | 1 | 5.36 ms |

Source-required packed traffic equals the mathematical packed minimum for every
row: 1.00x logical amplification. Hardware transaction counters are
unavailable, so no finer physical-transaction claim is made. At the measured
103 GB/s practical ceiling, plausible removable time sorted by role is gate
1.13 ms, O 0.69, Q 0.51, V 0.34, K 0.28, down 0.13, up 0.10, and logits 0.02:
about 3.2 ms/token total.

| Role | Minimum / actual | Amplification | Measured | 120-GB/s physical floor | 103-GB/s practical floor |
|---|---:|---:|---:|---:|---:|
| Gate | 1.887 / 1.887 GB | 1.00x | 19.45 ms | 15.73 ms | 18.32 ms |
| O | 0.472 / 0.472 GB | 1.00x | 5.28 ms | 3.93 ms | 4.58 ms |
| Q | 0.472 / 0.472 GB | 1.00x | 5.09 ms | 3.93 ms | 4.58 ms |
| V | 0.145 / 0.145 GB | 1.00x | 1.75 ms | 1.21 ms | 1.41 ms |
| K | 0.118 / 0.118 GB | 1.00x | 1.42 ms | 0.98 ms | 1.15 ms |
| Down | 2.320 / 2.320 GB | 1.00x | 22.65 ms | 19.33 ms | 22.52 ms |
| Up | 1.887 / 1.887 GB | 1.00x | 18.43 ms | 15.73 ms | 18.32 ms |
| Logits | 0.551 / 0.551 GB | 1.00x | 5.36 ms | 4.59 ms | 5.34 ms |

### Independent Q4_K and Q6_K audit

Q4_K reads each 144-byte block once. It decodes two F16 block scales and 12
packed 6-bit scale/min fields, unpacks 4-bit nibbles, converts F16 activations
and scales to FP32, accumulates two output rows per SIMD group, and performs a
`simd_sum`. Q6_K reads each 210-byte block once, combines 128 low-nibble bytes
with 64 high-two-bit bytes, applies 16 signed subscales and one F16 block scale,
then uses the same FP32 accumulation/reduction boundary. Neither path writes a
dequantized intermediate.

The load, metadata, scale-decode, unpack, conversion, dot, and reduction
instructions are fused and cannot receive honest independent milliseconds from
stage-boundary timestamps. Evidence instead classifies the large Q4 and mixed
Q4/Q6 roles as primarily weight-bandwidth limited: they sustain 97--103 GB/s,
activation traffic is negligible, and a prior per-format specialization was
flat. K/V projection rows are underfilled/launch-sensitive at 83 GB/s. Both
paths use two SIMD groups and zero threadgroup memory. Register/spill counters
were unavailable; the block-softmax and 32-KiB attention experiments provide
separate empirical register and occupancy cliffs, not GEMV spill measurements.

Q/K/V fusion can eliminate at most two repeated 5,120-element F16 activation
reads per layer, 0.819 MB/token. Gate/up fusion can eliminate one, 0.410
MB/token. Together that is only 0.012 ms at 100 GB/s; all 7.852 GB of packed
weights, metadata decode, unpack, and dot products remain. Even granting a
linear share of the 1.68 ms production record/submit term for 120 removable
dispatches gives less than 0.31 ms/token, far below the economic gate.

No material production GPU idle gap was established. A token uses one ordered
forward encoder without explicit barriers, utilization is 99% at long context,
and CPU wait tracks the GPU interval. The 7--9 ms profile envelope includes
stage-sampler pass transitions inserted only for attribution, so labeling it as
removable production idle would be incorrect.

## Competitive Reference

Warmed llama.cpp results use its own `predicted_ms / predicted_n` counter.
Each Graph Horizon row is its diagnostics-free same-session control; different
rows have different accumulated thermal state, so this is a competitive matrix,
not another scaling curve.

| KV | Graph Horizon | llama.cpp | Ratio ours/llama | Absolute gap | Position |
|---:|---:|---:|---:|---:|---|
| 128 | 95.24 ms audit control | 78.39 ms | 1.215x | +16.85 ms | 21.5% slower |
| 2,048 | 83.19 ms audit control | 84.65 ms | 0.983x | -1.46 ms | 1.7% faster |
| 8,192 | 137.74 ms hot control | 144.94 ms | 0.950x | -7.20 ms | 5.0% faster |
| 15,434 | 169.71 ms | 163.16 ms | 1.040x | +6.55 ms | 4.0% slower |
| 28,000 | 220.96 ms | 203.83 ms | 1.084x | +17.13 ms | 8.4% slower |

llama.cpp's vector decode uses 32 workgroups per query head and reloads shared
GQA K/V for the four query heads. Graph Horizon uses four workgroups per KV head
and stages one tile for all four heads. That explains Graph Horizon's stronger
8K attention scaling; llama.cpp retains a roughly 15--20 ms fixed-path advantage.
The rejected staging experiment briefly narrowed the 15K/28K endpoint gap
without adopting fourfold K/V traffic, but did not pass sustained generation.

## Candidate Ledger

The first three candidates were one-kernel reversible screens, so their
economic gate allowed a sub-5% prediction. The fourth was the only candidate
with a whole-token prediction above the moderate-redesign gate.

| Candidate | T / component | f | Physical floor | Practical floor / `S_local` | Removable | Predicted final | Predicted gain | Measured |
|---|---|---:|---:|---|---:|---:|---:|---:|
| 8x8 SIMD-matrix attention | 126.23 / 31.07 ms at 8K | 24.6% | 11.18 ms K/V-only | 24.86 ms / 1.25x | 6.21 ms | 120.02 ms, 8.33 tok/s | 4.9% | +2.8% endpoint |
| 64-position tile | 126.23 / 31.07 ms at 8K | 24.6% | 11.18 ms K/V-only | 27.02 ms / 1.15x | 4.05 ms | 122.18 ms, 8.18 tok/s | 3.2% | -6.5% |
| Four-position softmax block | 126.23 / 31.07 ms at 8K | 24.6% | 11.18 ms K/V-only | 28.25 ms / 1.10x | 2.82 ms | 123.41 ms, 8.10 tok/s | 2.2% | about -1% reciprocal boundaries |
| Aligned `half4` staging | 169.71 / 66.16 ms at 15K | 39.0% | 21.1 ms K/V-only | 50.89 ms / 1.30x | 15.27 ms | 154.44 ms, 6.47 tok/s | 9.0% | +6.7% short, 0% sustained |

Each row uses `1 / ((1-f) + f/S_local)` equivalently through its removable
milliseconds. The matrix/tile/block floors were pre-screen estimates for the
single changed mechanism, not theoretical hardware floors. Correctness passed
before performance screening in every case; measurement then selected reject.

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
| Aligned `half4` K/V staging | GPU-oracle checks passed at 1K and 2K; 8K attention locally -30.2% | 8K +5.7%, 15K short A/B +6.7%, 28K +6.6%, but 15K/1,156 candidate 5.82 vs baseline 5.83 tok/s | Reverted: required sustained primary workload was neutral |

Historical results also close the obvious alternatives: a grouped kernel without
history splits regressed 16% at 2K; eight splits gained 2.5% at 8K and regressed
1.3% at 28K; per-format Q4/Q6 GEMV specialization was flat; and explicit
residency hints regressed the comparator by 3.6%.

## Practical Floors And Final Decision

| Component at refined 15K | Current | Physical floor | Practical floor | Plausible removable | Closure |
|---|---:|---:|---:|---:|---|
| Attention scan + reduce | 49.19 ms | 21.1 ms K/V-only | about 46 ms exact-path envelope | about 3.2 ms in a short trace; 0 ms sustained proven | `half4`, larger tile, more splits, matrix, and block-softmax variants rejected |
| Seven projections + logits | 79.43 ms | 65.43 ms at 120 GB/s | 76.2 ms at 103 GB/s | about 3.2 ms | exact 1x packed traffic; canonical comparator-class arithmetic |
| Norm/RoPE/KV/activation/residual/embed | 2.61 ms | not useful independently | about 2 ms | below 0.7 ms | any fusion is below 0.5% token time |
| GPU envelope | 8.68 ms | 0 ms theoretical | about 6 ms observed | below 2.7 ms | mostly profiler pass/command envelope; production has one encoder and no explicit barriers |
| Sampling | 1.23 ms | one reduction/read | about 0.99 ms | 0.24 ms | below economic gate |
| Host/event remainder | 3.77 ms | 0 ms theoretical | about 2 ms short-context | below 1.8 ms | includes public event and profiler recording perturbation; CPU does not outrun GPU critical path |

The final `half4` candidate's Amdahl estimate used `T=169.71 ms`, attention fraction
`f=0.390`, and a realistic local speedup of `1.30`: 15.27 ms removable,
154.44 ms predicted, 6.47 tok/s, and 9.0% predicted whole-token gain. Measured
hot 15K short A/B was 6.7%, but the predicted benefit did not materialize over
the required 1,156-token generation. This is why the code was removed despite
the component-level reduction.

No remaining exact-representation candidate has an economically supported
whole-token gain. Lossy KV is not opened because effective attention bandwidth
is well below the physical ceiling and the baseline exact path already preserves
the 1x traffic minimum. Weight representation changes would alter the durable
model/quality contract and are outside the evidence-supported frontier.

The representation gates are quantitative. Replacing every Q6_K role in this
model with Q4_K would save 632.6 MB/token, only about 6.33 ms at 100 GB/s or
4.4% of the refined token, before quality cost. Generic INT8 KV is not a hidden
F16 win: because it is per-query-head rather than grouped, its payload plus
metadata moves about 2.06x the grouped F16 K/V bytes. An automatic grouped-INT8
route would be a lossy numerical policy and does not preserve the primary F16
quality contract.

The allowed design levels are closed as follows: routing selects the intended
path; resource lifetime/control is under 2 ms production CPU recording and has
no per-token allocation; kernel architecture, split count, tiles, matrix use,
softmax batching, and vector staging were measured; exact dataflow already
loads K/V and weights once; representation and numerical changes fail their
quality/economic gates; Apple-family specialization is already capability
based; and an exact-device product-string route has no quantitative basis.

The production final decode matrix therefore equals the baseline matrix:

| KV | Baseline tok/s | Final tok/s | Baseline ms/token | Final ms/token | Delta |
|---:|---:|---:|---:|---:|---:|
| 128 | 10.12 | 10.12 | 98.81 | 98.81 | 0 |
| 2,048 | 11.33 | 11.33 | 88.26 | 88.26 | 0 |
| 8,192 | 7.62 | 7.62 | 131.23 | 131.23 | 0 |
| 15,434 | 5.94 | 5.94 | 168.35 | 168.35 | 0 |
| 28,000 | 4.56 | 4.56 | 219.30 | 219.30 | 0 |

The diagnostics-free public interval means are the baseline table near the top;
this final table uses model completion duration. Since no production inference
change is retained, baseline and final are the same executable inference path,
not two thermally incomparable runs.

The primary sustained request gives the explicit end-to-end impact. The final
qualification reran the same 15,434-prompt/1,156-token workload on the restored
production source and captured all required generation fields:

| Final sustained field | Measurement |
|---|---:|
| TTFT | 233.177 s |
| Model / public generation throughput | 7.19 / 7.18 tok/s |
| Adjacent interval mean / median | 139.20 / 137.68 ms |
| Adjacent interval standard deviation / CV | 8.32 ms / 5.98% |
| Beginning / middle / end | 7.46 / 7.13 / 6.97 tok/s |
| Approximate request wall, TTFT through last public delta | 394.04 s |
| Whole process wall, including model initialization | 395.44 s |

The benchmark did not retain an independent prefill-only timer; TTFT is the
honest upper bound for tokenization, prefill, and first-token sampling. Request
wall is reconstructed from TTFT plus 1,155 public intervals and is approximate
because the printed throughput is rounded. Whole-process wall is the external
`time` measurement and must not be mislabeled as request-only time.

| 15,434 prompt + 1,156 generated | Historical baseline | Final requalification | Observed delta | Code-attributed delta |
|---|---:|---:|---:|---:|
| Prefill-only | not independently recorded | not independently recorded | no claim | 0; same path |
| TTFT upper bound | 261.976 s | 233.177 s | -28.799 s | 0; same path |
| Public decode span | about 198.11 s | about 160.86 s | about -37.25 s | 0; same path |
| Total request wall | about 460.09 s | about 394.04 s | about -66.05 s | 0; same path |
| Whole process wall | 472.230 s | 395.440 s | -76.790 s | 0; same path |
| Public generation throughput | 5.83 tok/s | 7.18 tok/s | +23.2% | 0; same path |

The observed before/after difference is run-state variation, not an optimization
gain: the inference source is identical, the historical baseline did not retain
its interval distribution, and this fanless device exhibits large thermal and
clock-ramp effects. The code-attributed final remains the baseline matrix above.

The rejected `half4` candidate measured 5.82 tok/s and 280.96 s TTFT on its
sustained workload. It did not improve the primary decode metric, and its hotter
TTFT cannot be attributed to an unchanged prefill kernel. No candidate therefore
earns an end-to-end improvement claim.

The independent audit also supplies the retained prefill/TTFT regression guard.
These are observations of the final production-equivalent path; source identity
provides the baseline/final equality while the values bound future regressions.

| Prompt tokens | Final-path TTFT | Baseline/final code delta |
|---:|---:|---:|
| 128 | 1.91 s | none |
| 2,048 | 20.83 s | none |
| 8,192 | 108.62 s | none |
| 15,434 | 309.92 s | none |
| 28,000 | 766.97 s | none |

Production prefill, TTFT, decode, and total request behavior are unchanged.
Further whole-request work should re-rank long-context prefill rather than add
decode complexity.

### Correctness and quality contract

The final tree returns to the previously qualified arithmetic and routing. Every
candidate was compared with the serial GPU oracle before timing, and no tolerance
was loosened. The final qualification reruns the retained test suites below.

| Contract surface | Evidence |
|---|---|
| CPU/reference and generic/optimized Metal agreement | Engine reference tests plus focused grouped-attention serial-GPU oracle checks at 1K and 2K for each candidate |
| F16 and INT8 KV schemes | Segmented/generic attention tests cover both schemes; the primary F16 route remains unchanged |
| Q4_K and Q6_K projections | Packed projection tests compare both formats with the scalar reference; mixed Q4/Q6 layer roles are exercised by the authenticated model |
| GQA, causal masking, and KV append | Grouped GQA oracle checks, attention semantic tests, and deterministic KV-cache tests |
| Teacher-forced/logit behavior | Final source is the already-qualified baseline projection/logit path; family and semantic integration suites pass |
| Sequential and multi-token lifecycle | The 1,156-token sustained generation plus engine lifecycle/integration tests |
| Benchmark reporting | Unit tests cover completion accounting, adjacent intervals, CV, and thirds; reporting is the only retained code change |

## Final Qualification

- Pure Metal: 144 unit tests passed, 2 ignored; 5 family-agnostic passed,
  1 ignored; 12 semantic passed, 1 ignored.
- Metal-hybrid: 205 unit tests passed, 2 ignored; the same integration and
  semantic counts passed.
- Focused grouped GQA GPU oracle passed at 1K and 2K for every candidate before
  performance screening; the final source is the already-qualified baseline
  arithmetic and route.
- `cargo fmt --all --check`, warning-denied workspace clippy for pure Metal,
  documentation/source-structure tests, and `git diff --check` passed.
- No temporary profiler, rejected kernel, experimental route, dependency, API,
  secret, or model artifact remained in the mission worktree. The retained
  reporting changes and report were later pushed and merged by PR #36.

The next candidate is not another decode tile. The largest whole-request cost is
now 15K--28K prefill; a future mission should rebuild its own attribution and
economic gates rather than reuse decode assumptions.
