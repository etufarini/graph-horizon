<!--
This checkpoint owns the reproducible evidence for the Apple Metal long-context
decode investigation: authenticated inputs, measurements, decisions, and the
remaining bottleneck. It does not define the runtime support contract.
-->

# Metal Long-Context Decode Optimization

Status: completed investigation on `perf/metal-long-context-decode`.

The production baseline is `87e3e4b1bb17e811caac030f69261bd0498284ec`.
No production kernel change is retained. Richer benchmark reporting is commit
`0899b74`; temporary profiling was removed by `c56aaf2`. The last production
candidate `54c1621` was removed non-destructively by `c4a49f2` after sustained
generation failed its retention gate. Final production-equivalent source before
this report is therefore `c4a49f2`.

## Environment And Inputs

- Device: MacBook Air `Mac16,13`, Apple M4 with 10 GPU cores and 24 GB unified
  memory, Metal 4, macOS 26.3 (`25D125`), connected to AC power.
- Model: `Ministral-3-14B-Instruct-2512-Q4_K_M.gguf`, 8,239,593,024 bytes,
  SHA-256 `824e0f3373e69b84f2cae46fdcb9bd1ebc6ab3bfc7acc125d818b7b8178cc613`.
- Model shape: 40 blocks, hidden width 5,120, 32 query heads, 8 KV heads,
  head dimension 128, and 16,384-wide feed-forward layers.
- KV configuration: F16, 32,768-token allocation. The K and V buffers total
  5.37 GB logically; 28,000 populated positions contain 4.59 GB of K/V data.
- Comparator: pinned llama.cpp checkout
  `2b63e0610bbc2be990ae1360d5256efcdc3f9efb`.

The synthetic prompt is the string `"a "` repeated until the tokenizer reports
the requested prompt count. Commands use a release Metal build and disable
benchmark warm-up unless a run explicitly says otherwise.

The reproducible Graph Horizon command shape is:

```sh
cargo build --locked --release --no-default-features --features metal --example bench
target/release/examples/bench <model.gguf> --context 32768 --kv f16 \
  --prompt <exact-N-token-synthetic-prompt> --max-tokens 128 --warmup 0 --reps 1
```

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

At 128 the unsplit attention route explains the non-monotonic curve. At 2K the
four-split route is already active and lowers attention despite longer history.
At 28K attention is 51.5% of reconstructed token time. The split profiler
perturbs endpoint timing most at 8K/15K, so uninstrumented intervals remain the
performance contract.

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

At 15,434 positions, K reads and V reads are 1.2645 GB each. QK and AV each
perform about 5.06 GFLOP/token across 40 layers; the online softmax processes
19.76 million query-head/history scores. Partial-result/state traffic is about
5.4 MB/token. These constituents are fused inside the measured 66.16 ms kernel
and cannot be timed independently without changing its execution, but their
exact byte and operation counts bound the roofline: 21.1 ms at a theoretical
120 GB/s for K/V alone, approximately 31.6 ms at the observed 80 GB/s hot
streaming rate, before dot products, exponentials, barriers, and reduction.

## Routing, Residency, And Capacity

| Operation | Shape per layer | Storage | Actual route |
|---|---|---|---|
| Q | 5,120 → 4,096 | Q4_K | packed two-row SIMD GEMV |
| K | 5,120 → 1,024 | Q4_K | packed two-row SIMD GEMV |
| V | 5,120 → 1,024 | 20 Q4_K + 20 Q6_K layers | matching packed SIMD GEMV |
| O | 4,096 → 5,120 | Q4_K | packed two-row SIMD GEMV |
| Gate / up | 5,120 → 16,384 | Q4_K | packed two-row SIMD GEMV |
| Down | 16,384 → 5,120 | 20 Q4_K + 20 Q6_K layers | matching packed SIMD GEMV |
| Attention ≥1,023 | dim 128, 32 Q / 8 KV heads | F16 KV | four-split grouped GQA + reduce |
| Attention fallback | other shapes, schemes, placement, or shorter KV | F16/INT8 | generic exact Metal attention |

All major decode work is GPU resident. Model weights are persistent canonical
packed buffers; K/V is request-persistent; pipelines and execution buffers are
model/runtime lifetime; no per-token allocation, map/unmap, resize, host/device
copy, or CPU fallback was observed. The split route requires pure Metal, F16 KV,
head dimension 128, exact 4:1 GQA, context at least 1,023, SIMD width 32, 256
threads, 16 KiB threadgroup memory, and sufficient scratch. Failure of any
predicate retains the generic path.

| Populated KV | F16 K+V bytes | Model + populated KV | Observed process state |
|---:|---:|---:|---|
| 2,048 | 0.336 GB | 8.58 GB | GPU resident |
| 8,192 | 1.342 GB | 9.58 GB | GPU resident |
| 15,434 | 2.529 GB | 10.77 GB | about 13.7 GB peak footprint |
| 28,000 | 4.588 GB | 12.83 GB | about 16.6 GB maximum RSS, no swap/fallback |

The full logical model plus 32,768-token K/V allocation is about 13.61 GB,
leaving capacity for execution buffers on the 24 GB device. Context growth did
not change placement or routing.

## Projection Inventory And Floors

Canonical packed bytes are read once per generated token. Q4_K blocks include
144 bytes per 256 weights; Q6_K blocks include 210. Metadata is already inside
those totals, so traffic amplification is 1x.

| Operation | Bytes/token | 15K time | Effective bandwidth |
|---|---:|---:|---:|
| Q | 0.472 GB | 5.54 ms | 85.2 GB/s |
| K | 0.118 GB | 1.87 ms | 63.2 GB/s |
| V | 0.145 GB | 2.26 ms | 64.1 GB/s |
| O | 0.472 GB | 8.31 ms | 56.8 GB/s |
| Gate | 1.887 GB | 25.99 ms | 72.6 GB/s |
| Up | 1.887 GB | 21.70 ms | 87.0 GB/s |
| Down | 2.320 GB | 25.90 ms | 89.6 GB/s |
| Logits | 0.551 GB | 6.36 ms | 86.6 GB/s |

Input/output activation traffic is below 0.01% of the 7.852 GB packed-weight
stream. At 2K the same kernels sustain about 100 GB/s in aggregate; their 76.1
ms total is the practical cool-device floor. The 65.43 ms 120-GB/s bound is
physical, not an attainable whole-projection target. The current kernels match
llama.cpp's two-row Q4_K/Q6_K arithmetic and two-SIMD-group geometry. Earlier
per-format specialization was flat, so fusion that only removes activation or
dispatch traffic cannot meet the 5% whole-token gate.

## Competitive Reference

Warmed llama.cpp results use its own `predicted_ms / predicted_n` counter.

| KV | Graph Horizon baseline | llama.cpp | Baseline position |
|---:|---:|---:|---:|
| 8,192 | 137.74 ms hot control | 144.94 ms | Graph Horizon 5.0% faster |
| 15,434 | 169.71 ms | 163.16 ms | Graph Horizon 4.0% slower |
| 28,000 | 220.96 ms | 203.83 ms | Graph Horizon 8.4% slower |

llama.cpp's vector decode uses 32 workgroups per query head and reloads shared
GQA K/V for the four query heads. Graph Horizon uses four workgroups per KV head
and stages one tile for all four heads. That explains Graph Horizon's stronger
8K attention scaling; llama.cpp retains a roughly 15--20 ms fixed-path advantage.
The rejected staging experiment briefly narrowed the 15K/28K endpoint gap
without adopting fourfold K/V traffic, but did not pass sustained generation.

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
| Aligned `half4` K/V staging | GPU-oracle checks passed at 1K and 2K; 8K attention locally -30.2% | 8K +5.7%, 15K short A/B +6.7%, 28K +6.6%, but 15K/1,156 candidate 5.82 vs baseline 5.83 tok/s | Reverted: required sustained primary workload was neutral |

Historical results also close the obvious alternatives: a grouped kernel without
history splits regressed 16% at 2K; eight splits gained 2.5% at 8K and regressed
1.3% at 28K; per-format Q4/Q6 GEMV specialization was flat; and explicit
residency hints regressed the comparator by 3.6%.

## Practical Floors And Final Decision

| Component at 15K | Baseline | Practical floor | Removable | Closure |
|---|---:|---:|---:|---|
| Attention | 66.16 ms | about 46 ms experimental envelope | no sustained removable time proven | `half4`, larger tile, more splits, matrix, and block-softmax variants rejected |
| Seven projections + logits | 97.92 ms hot | about 76 ms cool-state | no proven code-removable time | exact 1x packed traffic; same canonical kernel architecture as comparator |
| Small kernels | 4.83 ms | about 3 ms | <2 ms | fusion ceiling below 1.2% whole token |
| GPU interval residual | 9.72 ms | about 6 ms observed short-context | <4 ms | split/reduce dependency is required; unsplit route regressed 16% |
| Sampling | 1.63 ms | about 1 ms | <1 ms | below economic gate |

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

The production final decode matrix therefore equals the baseline matrix:

| KV | Baseline tok/s | Final tok/s | Baseline ms/token | Final ms/token | Delta |
|---:|---:|---:|---:|---:|---:|
| 128 | 10.12 | 10.12 | 98.81 | 98.81 | 0 |
| 2,048 | 11.33 | 11.33 | 88.26 | 88.26 | 0 |
| 8,192 | 7.62 | 7.62 | 131.23 | 131.23 | 0 |
| 15,434 | 5.94 | 5.94 | 168.35 | 168.35 | 0 |
| 28,000 | 4.56 | 4.56 | 219.30 | 219.30 | 0 |

The diagnostics-free public interval means are the baseline table near the top;
this final table uses model completion duration. Production prefill, TTFT, decode,
and total request time are unchanged. The baseline and rejected candidate both
measured about 5.83 tok/s over the observed 1,156-token generation, so no claim
of request-time improvement is made. Further whole-request work should re-rank
long-context prefill rather than add decode complexity.

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
  secret, model artifact, push, or external repository change remains.

The next candidate is not another decode tile. The largest whole-request cost is
now 15K--28K prefill; a future mission should rebuild its own attribution and
economic gates rather than reuse decode assumptions.
