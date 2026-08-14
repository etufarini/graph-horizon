<!--
This document owns the Phase 2 Vulkan long-context decode investigation: its
baseline, causal evidence, experiments, retained changes, and qualification.
-->

# Vulkan Long-Context Decode Investigation — Phase 2

## Outcome

Phase 2 starts at the qualified Phase 1 revision `6b09bdc` and improves the
same Ministral 3B Q4_K_M workload on an RTX 3060. The retained changes are:

- packed Q6_K decode loads with loop-invariant scale decoding;
- eight GQA KV splits with 256-thread workgroups, replacing four 512-thread
  workgroups.

At 28K KV, decode improves from the newly measured 24.04 to 27.94 tok/s
(+16.22%). GPU time falls from 39.488 to 33.686 ms/token (-14.69%, equivalent
to +17.22% GPU throughput). The desirable +15% target is met without a model,
quantization, KV representation, public API, or dependency change.

The final branch is `perf/vulkan-decode-phase2`. Its retained commits are
`1b12472` and `825057e`.

## Fixed Benchmark Contract

| Item | Value |
|---|---|
| Phase 2 base | `6b09bdc` |
| GPU | NVIDIA GeForce RTX 3060 12 GiB, PCI device `0x2487` |
| Driver / Vulkan | 595.84 / device API 1.4.329 |
| Reported memory clock | 7501 MHz |
| Model | `Ministral-3-3B-Instruct-2512-Q4_K_M.gguf` |
| Model size | 2,147,023,008 bytes |
| SHA-256 | `9ed150d4367e68df0ac8e1540f6ddc65b42d0ee26378329d1ecbca60f93fc5f8` |
| Context capacity | 32,768 tokens |
| KV representation | F16 |
| Prompt construction | exact string `" a"` repeated to `target - 3` |
| Decode sample | 32 tokens, one warm-up, three measured repetitions |
| Long sample | 128 tokens, one warm-up, three measured repetitions |
| Build | release, locked dependencies, `vulkan-profile` feature |

The artifact hash exactly matches the local model catalog. The model has 26
layers, hidden size 3072, Q width 4096, K/V width 1024, FFN width 9216, 32
query heads, 8 KV heads, and head dimension 128. Q, K, O, gate, and up weights
are Q4_K. V and down each contain 13 Q4_K and 13 Q6_K tensors. The tied
embedding/logits matrix is Q6_K.

The commands below express the build and benchmark shape; `<model.gguf>` is the
artifact identified above.

```sh
cargo build --locked --release --no-default-features \
  --features vulkan-profile --example bench

target/release/examples/bench --model <model.gguf> --context 32768 \
  --prompt-repeats <target-minus-3> --max-tokens 32 --warmup 1 --repetitions 3
```

## Inherited Phase 1 Result

Phase 1 made the exact 4:1 GQA decode shape reuse each K/V load for four query
heads and qualified 24.05 tok/s at 28K. Its complete record is in
[the Phase 1 investigation](vulkan-decode-investigation.md).

The old Phase 1 baseline and the inherited Phase 1 result separate as follows
at 28K:

| Quantity | Old baseline | Phase 1 / Phase 2 base | Removed |
|---|---:|---:|---:|
| Wall time, ms/token | 112.108 | 41.597 | 70.511 |
| Top-level attention, ms/token | 87.391 | 16.768 | 70.623 |
| Causal attention program, ms/token | 87.782 | 16.677 | 71.105 |
| K read + QK | 29.762 | 10.725 | 19.037 |
| Softmax | 22.618 | 2.035 | 20.583 |
| V read + AV | 33.635 | 3.765 | 29.870 |
| Skeleton | 1.767 | 0.152 | 1.615 |

The small mismatch between top-level and causal totals comes from independent
runs and profiling boundaries. Projection and MLP times remained approximately
constant, so the inherited gain is attributable to the GQA attention path.
Granular production counters from before Phase 1 are not available; the
temporary stage variants are the reproducible causal evidence.

## New Baseline

Phase 2 remeasured the inherited revision instead of reusing the earlier
headline numbers.

| KV tokens | tok/s | Std. dev. | CV | Wall ms/token | GPU ms/token | GPU accounted |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 38.77 | 0.04 | 0.10% | 25.793 | 22.907 | 99.55% |
| 2,048 | 38.31 | 0.15 | 0.39% | 26.103 | 24.014 | 99.58% |
| 8,192 | 33.45 | 0.22 | 0.64% | 29.895 | 27.732 | 99.63% |
| 16,384 | 28.82 | 0.03 | 0.11% | 34.698 | 32.614 | 99.69% |
| 28,000 | 24.04 | 0.03 | 0.14% | 41.597 | 39.488 | 99.74% |

The additional 512-token point is 38.75 tok/s, 25.806 wall ms/token, and
22.951 GPU ms/token. The 128-token long samples at 8K, 16K, and 28K are 33.60,
28.76, and 24.04 tok/s. Their CVs are 0.26%, 0.10%, and 0.15%; no within-run
decay appears.

Active-run telemetry includes prompt processing and decode because the sampling
tool cannot mark phases. It is therefore a thermal and throttling control, not
a per-kernel power measurement.

| KV | GPU util. mean (range) | VRAM MiB | Clock MHz | Power W mean (range) | Temp. C mean (range) |
|---:|---:|---:|---:|---:|---:|
| 128 | 93.3% (92–97) | 5,694 | 1,987 | 134.0 (78.6–144.4) | 51.7 (50–53) |
| 2K | 94.9% (92–97) | 5,699 | 1,989 | 160.1 (97.6–167.1) | 63.3 (59–66) |
| 8K | 95.9% (80–97) | 5,700 | 1,977 | 161.5 (84.3–169.8) | 68.7 (62–72) |
| 16K | 96.6% (88–100) | 5,700 | 1,973 | 159.1 (72.3–169.4) | 71.2 (63–73) |
| 28K | 97.2% (93–100) | 5,700 | 1,971 | 155.1 (97.0–169.6) | 71.6 (66–73) |

Clocks remain stable and temperatures plateau; throttling does not explain the
context curve.

## Scaling And Bottleneck Attribution

A least-squares fit over 128, 512, 2K, 8K, 16K, and 28K gives:

```text
wall T(N) = 25.342388 ms + 0.000576124 ms * N, R² = 0.998061
GPU  T(N) = 22.773663 ms + 0.000598283 ms * N, R² = 0.999881
```

The wall fit has 0.257 ms RMSE and predicts 41.474 ms at 28K versus 41.597 ms
measured. The GPU fit has 0.066 ms RMSE and predicts 39.526 ms versus 39.488 ms
measured. At 28K, the context-dependent term is 38.90% of wall time and 42.38%
of GPU time.

The profiler category matrix makes the growing term explicit:

| Category, ms/token | 128 | 2K | 8K | 16K | 28K |
|---|---:|---:|---:|---:|---:|
| Attention | 0.311 | 1.478 | 5.041 | 9.709 | 16.768 |
| Projection Q | 1.700 | 1.678 | 1.730 | 1.685 | 1.703 |
| Projection K | 0.797 | 0.796 | 0.804 | 0.800 | 0.801 |
| Projection V | 1.267 | 1.319 | 1.266 | 1.250 | 1.262 |
| Projection output | 1.639 | 1.661 | 1.663 | 1.659 | 1.658 |
| MLP gate | 3.251 | 3.295 | 3.277 | 3.429 | 3.308 |
| MLP up | 3.329 | 3.282 | 3.348 | 3.452 | 3.370 |
| MLP activation | 0.065 | 0.069 | 0.067 | 0.066 | 0.066 |
| MLP down | 5.966 | 5.881 | 5.973 | 5.988 | 5.970 |
| Normalization | 0.361 | 0.352 | 0.363 | 0.362 | 0.359 |
| RoPE | 0.113 | 0.119 | 0.113 | 0.113 | 0.112 |
| KV cache append | 0.063 | 0.065 | 0.065 | 0.065 | 0.065 |
| Elementwise | 0.114 | 0.113 | 0.114 | 0.112 | 0.112 |
| Embedding | 0.015 | 0.015 | 0.015 | 0.015 | 0.015 |
| Logits | 3.812 | 3.791 | 3.791 | 3.808 | 3.818 |

Temporary compile-time attention stages isolate causal program regions. Each
row is a separate executable; differences can expose overlap, cache, and
compiler scheduling and must not be interpreted as independent hardware-pipe
utilization.

| KV | Skeleton | K read | QK | Softmax | V read | AV | Full attention |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 0.150 | 0.051 | 0.041 | 0.017 | 0.008 | 0.052 | 0.318 |
| 2K | 0.173 | 0.361 | 0.448 | 0.136 | 0.184 | 0.160 | 1.462 |
| 8K | 0.152 | 1.299 | 1.836 | 0.644 | 0.700 | 0.476 | 5.106 |
| 16K | 0.151 | 2.567 | 3.796 | 1.130 | 1.316 | 0.776 | 9.735 |
| 28K | 0.152 | 4.383 | 6.342 | 2.035 | 2.596 | 1.169 | 16.677 |

All values are ms/token. The stage variants were removed after measurement.

### KV traffic and roofline

For F16 KV, each layer logically reads one K and one V vector per KV head and
history position. The Phase 1 GQA shader indexes each packed value once and
reuses it across four Q heads, so measured source-level traffic equals the
theoretical minimum: amplification 1.00.

| KV | K bytes | V bytes | Total bytes | Full-attention effective bandwidth |
|---:|---:|---:|---:|---:|
| 8K | 0.436 GB | 0.436 GB | 0.872 GB | 170.9 GB/s |
| 16K | 0.872 GB | 0.872 GB | 1.745 GB | 179.2 GB/s |
| 28K | 1.491 GB | 1.491 GB | 2.982 GB | 178.8 GB/s |

The causal K-read deltas imply 335.7, 339.9, and 340.2 GB/s at 8K, 16K, and
28K, agreeing with the 342 GB/s sustainable bandwidth measured in Phase 1.
The analogous V deltas exceed nominal bandwidth because causal subtraction
captures overlap and compiler/cache effects; they are not physical DRAM
counters.

At 28K, QK plus AV perform approximately 11.928 GFLOP and the exact KV stream
is 2.982 GB, or 4 FLOP/byte. At 342 GB/s the traffic-only floor is 8.72 ms.
The initial full attention time is 16.768 ms and the final time is 13.217 ms,
corresponding to 178.8 and 225.6 GB/s effective bandwidth. The remaining exact
attention ceiling is about 1.52x; with attention at 39.24% of final GPU time,
an optimistic end-to-end ceiling is about +15.4%. Reaching it requires reducing
QK, softmax, AV, and issue/live-state costs because K reads already reach the
measured bandwidth ceiling. Further large traffic reduction requires a narrower
KV representation, outside this phase.

### Quantized matvec inventory

Q4_K stores 128 quant bytes and 16 metadata bytes per 256 weights. Q6_K stores
192 quant bytes and 18 metadata bytes. Weight reads dominate the activation
and output traffic of these decode GEMV operations.

| Operation | Logical weight bytes/token | Baseline ms | Effective GB/s |
|---|---:|---:|---:|
| Q, Q4_K | 184.025 MB | 1.703 | 108.04 |
| K, Q4_K | 46.006 MB | 0.801 | 57.45 |
| V, mixed Q4_K/Q6_K | 56.549 MB | 1.262 | 44.82 |
| O, Q4_K | 184.025 MB | 1.658 | 110.98 |
| Gate, Q4_K | 414.056 MB | 3.308 | 125.18 |
| Up, Q4_K | 414.056 MB | 3.370 | 122.86 |
| Down, mixed Q4_K/Q6_K | 508.944 MB | 5.970 | 85.25 |
| Logits, Q6_K | 330.301 MB | 3.818 | 86.51 |

The total is 2.138 GB of logical weight reads per token. This inventory made
Q6_K the second independent target: it contributes half of V and down plus all
logits, while its baseline shader decoded individual bytes repeatedly.

### CPU and submission path

At 28K the public wall/GPU gap is 2.105 ms/token (5.88% of wall time). One
submission contains 445 dispatches and 341 barriers. CPU recording is 0.481 ms,
submission 0.032 ms, descriptor work 0.048 ms, and the 33.816 ms wait overlaps
GPU execution. The actionable CPU total is therefore about 0.562 ms/token
(1.57%), not the full wall/GPU gap. There are no steady-state allocations; 39
allocations occur during setup. CPU/dispatch restructuring did not pass the
bottleneck gate.

## Experiment Register

| Candidate | Gate / hypothesis | Result | Decision |
|---|---|---|---|
| Existing MMVQ opt-in | Replace decode GEMV path | 38.77 to 24.16 tok/s at 128 (-37.7%); Q4 kernels regress and activation Q8_1 changes arithmetic | Reject |
| Packed Q6_K loads | Q6_K V/down/logits are 18.31% of baseline GPU time; reduce byte extraction and repeated scale work | 28K 24.04 to 25.45 tok/s (+5.87%); GPU 39.488 to 37.170 ms (+6.23% throughput) | Retain |
| Exact-shape dynamic-stride attention | Remove general stride arithmetic from the fixed model shape | Attention improves 2.8% at 8K; predicted end-to-end gain about 1.3% | Reject and remove |
| 8 splits, WG256 | Improve scheduling/occupancy while preserving exact GQA reuse | Checkpoint 25.45 to 27.94 tok/s (+9.78%); GPU 37.170 to 33.686 ms (+10.34% throughput) | Retain |

Rejected production edits and all temporary instrumentation were removed.

## Retained Change 1: Packed Q6_K Decode Loads

`matmul_q6_k.comp` now reads four packed bytes at a time. Its 210-byte blocks
alternate between offsets zero and two modulo four, so the helper joins two
adjacent words only for the unaligned case. Four signed scales are decoded once
per segment instead of once per element. Quant reconstruction and FP32
accumulation order remain unchanged.

At the 28K checkpoint, V falls from 1.262 to 0.896 ms and down from 5.970 to
3.998 ms. Their effective logical-weight bandwidth rises to approximately 63.1
and 127.3 GB/s. Logits remains approximately unchanged at 3.821 ms; its larger
output dimension and execution shape do not receive the same benefit.

The affected baseline share is 18.31%, local speedup is 1.488x, and Amdahl
predicts +6.39% overall. The measured GPU-throughput gain is +6.23% and the
wall-throughput gain is +5.87% at 28K. At 128 and 8K, throughput improves by
12.28% and 9.06% at this checkpoint.

## Retained Change 2: Eight-Split GQA Decode

The GQA attention kernel changes from four 512-thread workgroups to eight
256-thread workgroups per KV head. Total threads per layer remain 16,384, but
workgroups increase from 32 to 64 and shared memory per workgroup falls from
8,320 to 4,160 bytes. On an RTX 3060, the 1,536-thread limit permits at most
three old or six new workgroups per SM before register limits; both represent
at most 48 resident warps.

The partial scratch grows by 64 KiB, while the state scratch fits the existing
allocation. Allocation count remains 39 and reported total allocated bytes do
not change at the profiler's granularity. The fixed 4:1 GQA ownership invariant
and exact one-load-per-K/V-value representation remain intact.

Nsight Compute was attempted, but profiled no Vulkan kernels in this setup, so
exact registers/thread, spills, and achieved occupancy are unavailable. Static
main-kernel state includes four Q vectors, four accumulator vectors, and four
max plus four normalization scalars (40 FP32 scalar equivalents), excluding
compiler temporaries. The performance result is empirical evidence for better
scheduling, not a claim of measured occupancy.

At its checkpoint attention falls from 16.808 to 13.217 ms (-21.36%). With a
45.22% affected share and local speedup 1.272x, Amdahl predicts +10.69%; measured
GPU throughput improves 10.34% and wall throughput 9.78%.

This differs from the Phase 1 E-256 rejection: E-256 retained four splits and
32 workgroups, whereas this candidate doubles independent split workgroups.

## Final Performance

| KV | New baseline tok/s | Final tok/s | Gain | Baseline wall ms | Final wall ms | Baseline GPU ms | Final GPU ms | Final CV |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 38.77 | 42.87 | +10.58% | 25.793 | 23.326 | 22.907 | 20.467 | 0.26% |
| 2K | 38.31 | 42.50 | +10.94% | 26.103 | 23.529 | 24.014 | 21.474 | 0.05% |
| 8K | 33.45 | 37.86 | +13.18% | 29.895 | 26.413 | 27.732 | 24.303 | 0.19% |
| 16K | 28.82 | 32.98 | +14.43% | 34.698 | 30.321 | 32.614 | 28.201 | 0.13% |
| 28K | 24.04 | 27.94 | +16.22% | 41.597 | 35.791 | 39.488 | 33.686 | 0.33% |

Against the earlier qualified Phase 1 headline measurements rather than the
new baseline, gains are +10.75%, +11.20%, +12.44%, +14.55%, and +16.17% at
128, 2K, 8K, 16K, and 28K.

The final 28K 128-token run is 27.87 tok/s with 0.13% CV. Instrumented thirds
are 27.91, 27.86, and 27.84 tok/s; the first-to-last change is -0.25%, with no
material decay. The temporary thirds instrumentation was removed.

| Final 28K GPU category | ms/token | Share |
|---|---:|---:|
| Attention | 13.217 | 39.24% |
| Q projection | 1.693 | 5.03% |
| K projection | 0.811 | 2.41% |
| V projection | 0.896 | 2.66% |
| Output projection | 1.680 | 4.99% |
| MLP gate | 3.339 | 9.91% |
| MLP up | 3.391 | 10.07% |
| MLP activation | 0.066 | 0.19% |
| MLP down | 3.998 | 11.87% |
| Normalization | 0.364 | 1.08% |
| RoPE | 0.114 | 0.34% |
| KV append | 0.065 | 0.19% |
| Elementwise | 0.112 | 0.33% |
| Embedding | 0.015 | 0.04% |
| Logits | 3.821 | 11.34% |

The profiler accounts for 99.69% of final GPU time. Sampling is a separate
approximately 0.083 ms/token.

Final active-run telemetry remains stable: utilization is 91.2%, 94.8%, 95.9%,
96.7%, and 97.3% from 128 through 28K; mean clocks are 1987, 1971, 1977, 1971,
and 1978 MHz. VRAM remains 5,694–5,700 MiB. Mean power is 140.8–162.6 W and
mean temperature 63.4–71.3 C. The long 28K run averages 97.2% utilization,
1971 MHz, 155.1 W, and 71 C. There is no thermal or clock regression.

## Prefill And Correctness Qualification

Prefill is unchanged within run noise:

| Prompt | New baseline | Final | Change |
|---:|---:|---:|---:|
| 8K | 430.07 tok/s | 429.74 tok/s | -0.08% |
| 28K | 285.54 tok/s | 286.21 tok/s | +0.23% |

The 28K long-run prefill result is 285.77 tok/s with 0.07% CV.

Correctness checks cover both retained changes:

- the Q6_K projection shader matches its CPU oracle;
- F16 Vulkan prefill attention matches sequential decode, with maximum absolute
  attention error at most `6.1035e-5` and logits error at most `4.1778e-6` in
  the focused suite;
- the INT8 path remains bit-identical in the attention parity test;
- the final build produces the exact same 128 accumulated greedy token IDs as
  a detached Phase 1 reference after a 2K prompt;
- KV append and multi-token paths remain exercised;
- formatting and lint checks pass;
- the full `vulkan-profile` library suite passes: 151 passed, 4 ignored.

Relative error near zero is not used as the qualification bound; maximum
absolute error and exact token identity are the relevant controls.

The workspace suite passes with one unrelated installer fixture skipped. That
pre-existing test gives its synthetic `PATH` no `dirname`, although
`support/install.sh` invokes `dirname` before its prerequisite loop; it exits 1
instead of the fixture's expected exit 2. Phase 2 changes neither the script nor
the test, and all other 165 binary tests pass.

## Stop Decision And Remaining Bottleneck

The requested desirable 28K improvement is achieved, all context points
improve, prefill is neutral, long-run stability holds, and correctness gates
pass. Further optimization stops here rather than expanding durable scope.

Attention remains the largest final category at 39.24%, followed by MLP down
(11.87%), logits (11.34%), MLP up (10.07%), and MLP gate (9.91%). Within exact
F16 attention, K streaming already approaches sustainable bandwidth; the
remaining work is QK/softmax/AV issue and live-state cost. A future exact-path
investigation should obtain Vulkan-compatible hardware counters and target
those regions. A larger structural bandwidth gain would require KV
quantization or another narrower representation and must be evaluated as a
separate correctness and product decision.

No dependencies, public APIs, model files, or validation tolerances changed.
