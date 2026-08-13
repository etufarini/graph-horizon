<!--
This report owns the reproducible evidence and retained conclusions for the
Phase 1 Vulkan long-context decode investigation based on commit d631461.
-->

# Vulkan Long-Context Decode Phase 1

## Outcome

The retained kernel changes F16 decode attention from one complete K/V scan per
query head to one packed K/V scan per KV head, reused by its four GQA query
heads. At 28K this raises end-to-end decode from 8.92 to 24.05 tok/s
(+169.62%) and reduces attention from 87.39 to 16.72 ms/token (-80.86%). Short
decode also improves by 1.18%, while 8K and 28K prefill remain within noise.

The phase stops because exact-representation K/V traffic has reached the 4:1
GQA minimum. Attention is still the largest single category at 28K (42.36%),
but the next broad cost is M=1 quantized matvec: MLP, projections and logits
together account for 55.69% of GPU decode time. A narrower KV representation
would be the next way to reduce K/V bytes and is intentionally outside Phase 1.

## Baseline And Method

- Baseline: `d631461`, branch point `perf/vulkan-prefill-matmul-phase4`.
- Candidate branch: `perf/vulkan-decode-kv-phase1`.
- GPU: NVIDIA GeForce RTX 3060 12 GiB, GA104 PCI device `0x2487`.
- Driver/API: NVIDIA 595.84, Vulkan 1.4.329.
- Model: `Ministral-3-3B-Instruct-2512-Q4_K_M.gguf`, 2,147,023,008 bytes,
  SHA-256 `9ed150d4367e68df0ac8e1540f6ddc65b42d0ee26378329d1ecbca60f93fc5f8`.
- Build: Cargo release, `--no-default-features --features vulkan-profile`.
- Runtime: pure Vulkan, F16 KV, context 32,768, greedy decode.
- Prompt: `" a"` repeated to produce the requested exact token count.
- Main samples: 32 generated tokens, one warm-up and three measured repetitions.
  Public throughput statistics exclude warm-up; GPU profile totals contain four
  runs and 31 decode intervals per run, hence 124 profiled decode steps.
- Stability samples: 128 generated tokens at 8K and 28K.

Model load, prefill and warm-up are excluded from decode throughput. GPU clocks
were 1,965–1,980 MHz, memory clock 7,501 MHz, board power 144.7–168.7 W,
temperature 68–72 °C and utilization 97–100% during long runs. Resident VRAM
was 5,701 MiB. No thermal or power throttle was observed. The retained logs
preserve ranges rather than a separate telemetry tuple for every KV point; no
per-point value is reconstructed after the fact.

A representative invocation is:

```sh
cargo build --locked --release --no-default-features \
  --features vulkan-profile --example bench

target/release/examples/bench <model.gguf> \
  --context 32768 --kv f16 \
  --prompt "$(perl -e 'print " a" x 27997')" \
  --max-tokens 32 --warmup 1 --reps 3
```

`GRAPH_HORIZON_DECODE_GQA=0` forces the baseline decode attention path. The new
path is selected only for F16, `head_dim=128`, and exact 4:1 GQA on a device for
which its pipelines were registered. INT8 and unsupported shapes retain their
previous kernels.

## Decode Scaling

Public `ms/token` is the reciprocal of mean tok/s. GPU and attention times are
profile totals divided by 124 steps.

| KV | baseline tok/s | stddev | CV | wall ms/token | GPU ms/token | attention ms/token | attention / KV (ms/token) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 38.26 | 0.12 | 0.0031 | 26.137 | 23.238 | 0.682 | 0.005329 |
| 512 | 36.86 | 0.03 | 0.0007 | 27.130 | 24.259 | 1.893 | 0.003697 |
| 2,048 | 32.03 | 0.05 | 0.0014 | 31.221 | 29.208 | 6.698 | 0.003270 |
| 8,192 | 19.79 | 0.02 | 0.0009 | 50.531 | 48.482 | 25.822 | 0.003152 |
| 16,384 | 13.16 | 0.00 | 0.0002 | 75.988 | 73.926 | 51.234 | 0.003127 |
| 28,000 | 8.92 | 0.00 | 0.0002 | 112.108 | 110.033 | 87.391 | 0.003121 |

The baseline converges to a nearly constant attention-time/history ratio from
2K onward: the scan is empirically O(N), not super-linear. The larger ratio at
128 and 512 is fixed launch/reduction overhead being amortized.

| KV | final tok/s | stddev | CV | wall ms/token | GPU ms/token | attention ms/token | attention / KV (ms/token) | tok/s delta |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 38.71 | 0.08 | 0.0021 | 25.833 | 22.908 | 0.318 | 0.002482 | +1.18% |
| 512 | 38.35 | 0.07 | 0.0018 | 26.076 | 23.173 | 0.559 | 0.001092 | +4.04% |
| 2,048 | 38.22 | 0.16 | 0.0041 | 26.165 | 24.129 | 1.467 | 0.000716 | +19.33% |
| 8,192 | 33.67 | 0.16 | 0.0046 | 29.700 | 27.589 | 5.025 | 0.000613 | +70.14% |
| 16,384 | 28.79 | 0.12 | 0.0043 | 34.734 | 32.592 | 9.695 | 0.000592 | +118.77% |
| 28,000 | 24.05 | 0.08 | 0.0032 | 41.580 | 39.482 | 16.724 | 0.000597 | +169.62% |

## Baseline Attribution

The normal profiler accounts for at least 99.4% of GPU time at every measured
point. To see inside the fused baseline attention kernel, one-step causal
ablations progressively enabled K read, QK, online softmax, V read and AV. Each
number below is the delta from the preceding stage. These are controlled
program deltas, not hardware-counter attribution: instruction and memory
latency can overlap, so they should not be interpreted as independent pipe
utilization.

| KV | skeleton | K read | QK | softmax | V read | AV | full attention |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2K | 0.265 | 1.782 | 0.381 | 1.641 | 2.611 | 0.136 | 6.816 |
| 8K | 0.622 | 7.214 | 1.410 | 6.651 | 9.534 | 0.780 | 26.211 |
| 16K | 1.091 | 14.146 | 3.485 | 13.646 | 18.160 | 1.131 | 51.659 |
| 28K | 1.767 | 24.167 | 5.595 | 22.618 | 31.895 | 1.740 | 87.782 |

At 28K the K- and V-read marginal deltas are 64% of attention, while online
softmax is another 26%. The evidence supports redundant K/V traffic as the
first target, but not the weaker statement that the entire kernel is only DRAM
bandwidth-bound.

The ordinary 28K baseline profile is:

| category | ms/token | GPU share |
|---|---:|---:|
| attention | 87.391 | 79.42% |
| Q projection | 1.708 | 1.55% |
| K projection | 0.806 | 0.73% |
| V projection | 1.265 | 1.15% |
| output projection | 1.649 | 1.50% |
| MLP gate/up/activation/down | 12.641 | 11.48% |
| normalization | 0.357 | 0.32% |
| RoPE | 0.111 | 0.10% |
| KV append | 0.064 | 0.06% |
| logits | 3.819 | 3.47% |
| embedding, residual and profiler residual | 0.220 | 0.20% |
| sampling, separate submission | 0.082 | — |

KV append has no resize, copy, allocation or buffer creation in steady state.
The first engine setup reported 39 Vulkan allocations and 10,479,206,400 device
bytes; repeated same-engine diagnostics reported `[0, 0, 0]` allocations during
decode.

## KV Layout And Traffic Model

The model has 26 layers, 32 query heads, 8 KV heads and head dimension 128:
the GQA ratio is exactly 4:1. K and V are separate F16 buffers with physical
indexing:

```text
((layer * context + token) * kv_heads + kv_head) * head_dim + dimension
```

Thus dimension is contiguous, then KV head, token and layer. A head is 256 B,
a token row per buffer is 2,048 B, and a full-context layer is 64 MiB. All are
aligned for the retained 8-byte `f16vec4` load. Prefill appends contiguous KV
head rows; decode walks tokens at a 2,048-byte stride for a fixed head. The
layout is a reasonable shared write/read compromise, but the old kernel looped
over query heads and rescanned the same physical KV head four times.

| KV | old K GB/token | old V GB/token | old total GB/token | GQA-min K GB/token | GQA-min V GB/token | GQA-min total GB/token |
|---:|---:|---:|---:|---:|---:|---:|
| 2K | 0.436 | 0.436 | 0.872 | 0.109 | 0.109 | 0.218 |
| 8K | 1.745 | 1.745 | 3.490 | 0.436 | 0.436 | 0.872 |
| 16K | 3.490 | 3.490 | 6.979 | 0.872 | 0.872 | 1.745 |
| 28K | 5.964 | 5.964 | 11.928 | 1.491 | 1.491 | 2.982 |

Perfect GQA reuse therefore has a 4x reuse factor and removes 75% of logical
global K/V bytes. There are no conversion or reformat passes. Both old and new
paths use streaming online softmax and never materialize a score vector or an
NxN matrix.

The per-layer physical K+V working set is 0.5, 2, 8, 32, 64 and 109.375 MiB at
128, 512, 2K, 8K, 16K and 28K respectively. It plausibly outgrows the device's
few-MiB last-level cache between 512 and 2K, but timings show no adverse cliff:
time/history improves and then converges. Exact L2 residency was not exposed by
the available Vulkan profiler, so this is a working-set correlation, not a
claim of a measured cache miss threshold.

Using logical bytes divided by attention time gives 130.3, 135.1, 136.2 and
136.5 GB/s for the old 2K, 8K, 16K and 28K scans. The final GQA-minimum traffic
gives 148.7, 173.6, 180.0 and 178.3 GB/s. A vec4 pair-reuse candidate at 28K
read 5.964 GB in 17.44 ms, about 342 GB/s, close to the RTX 3060's nominal
360 GB/s DRAM bandwidth. Full 4:1 reuse halves those bytes again but improves
attention by only about 4%. This is the decisive evidence that the retained
kernel is no longer primarily limited by DRAM traffic; online-softmax math,
instructions and live state now matter materially.

Separate K then V phases would require storing all scores or rescanning K.
Interleaving K/V would not reduce bytes and would impose a prefill repack. The
traffic model therefore selected simultaneous streamed packed K/V loads. A
manual prefetch was not pursued after the final bandwidth result, and a
persistent kernel was not justified because record, submit and descriptor work
at 28K totals about 0.55 ms/token, below 2% of wall time.

## Retained Kernel

One workgroup owns one of four disjoint history splits for one KV head. Its 16
32-lane subgroups scan an aggregate 64 history positions per loop iteration.
Each lane loads one contiguous `f16vec4` K and V fragment once, and uses it for
the four query heads in that GQA group. Each head maintains independent FP32
online-softmax state. A 32-thread reduction kernel merges the four split
partials.

- Effective KV tile/traversal: 64 positions per aggregate iteration. No K/V
  tile is copied into shared memory; the packed fragments are reused directly
  from registers.
- Workgroup: 512 threads, 16 subgroups of 32.
- Dispatch: 8 KV heads × 4 splits = 32 split workgroups per layer, followed by
  32 head reductions.
- Shared memory: 16 maxima + 16 denominators + 16 × 32 `vec4` accumulators =
  8,320 B/workgroup. Shared K tile = 0 B; shared V tile = 0 B.
- Reused external scratch: 65,536 B partial numerators plus 1,024 B state. No
  new device allocation is introduced.
- Static principal live state per thread: four Q `vec4`, four accumulator
  `vec4`, four maxima and four denominators, or 40 FP32 scalar equivalents,
  plus compiler temporaries.

The Vulkan/NVIDIA path used here did not expose NVIDIA SASS, exact
registers/thread, stack bytes, spill counts or achieved occupancy. SPIR-V and
the 8,320-byte shared allocation were inspectable, but translating source
variables into a fabricated register count would be misleading. The hardware
limit is at most three 512-thread workgroups (48 warps) per SM before register
or shared constraints; actual occupancy may be lower. This missing counter is
recorded for every experiment below. Performance and workgroup sweeps were used
as the empirical pressure gate.

## Experiment Register

All deltas compare with `d631461`. “Registers/occupancy unavailable” has the
specific instrumentation limitation described above, not an assumed zero.

| ID | architecture | reuse / traffic | tile, WG, subgroup map | shared | 8K result | 28K result | decision |
|---|---|---|---|---:|---|---|---|
| A | scalar, four Q heads, no history split | 4:1 / -75% | full history, WG1024, one WG/KV head | small reduction state | 10.74 tok/s (-45.7%), attention 2.66x slower | 3.86 tok/s (-56.7%), attention about 234 ms | reject: too few WGs and too much serial live state |
| B | scalar, four Q heads, four splits + reduce | 4:1 / -75% | 4-way, WG1024, 32 WGs/layer | partial reduction state | 16.58 tok/s (-16.2%), attention 35.73 ms | 6.86 tok/s (-23.1%), attention 121.19 ms | reject: scalar instruction/live-state cost |
| C | scalar, two Q heads, two splits + reduce | 2:1 / -50% | 2-way, WG1024, 32 WGs/layer | partial reduction state | 21.56 tok/s (+8.9%), attention 21.79 ms | 10.11 tok/s (+13.3%), attention 74.31 ms | superseded by vector loads |
| D | `f16vec4`, two Q heads, two splits | 2:1 / -50% | 2-way, WG1024, 32 WGs/layer | partial reduction state | 33.63 tok/s (+69.9%), attention 5.26 ms | 23.71 tok/s (+165.8%), attention 17.44 ms | keep as sweep reference |
| E | `f16vec4`, four Q heads, four splits | 4:1 / -75% | 4-way, WG1024, 32 WGs/layer | 16,640 B/WG | 33.62 tok/s, attention 5.19 ms | 24.01 tok/s, attention 16.79 ms | keep, then tune WG |
| E-256 | full reuse, narrower WG | 4:1 / -75% | aggregate tile 32, WG256, 8 subgroups | 4,160 B/WG | 31.56 tok/s, attention 6.96 ms | not repeated | reject: -6.5% versus WG512 at 8K |
| E-512 | retained full reuse | 4:1 / -75% | aggregate tile 64, WG512, 16 subgroups | 8,320 B/WG | 33.67 tok/s repeated, attention 5.02 ms | 24.05 tok/s repeated, attention 16.72 ms | keep |
| E-1024 | full reuse, wide WG | 4:1 / -75% | aggregate tile 128, WG1024, 32 subgroups | 16,640 B/WG | 33.62 tok/s, attention 5.19 ms | 24.01 tok/s, attention 16.79 ms | reject: no gain, stricter device requirement |

Exact registers, stack, spills and achieved occupancy are unavailable for A–E.
Static source state increased from two to four heads between D and E; the split
and WG sweeps directly tested whether that pressure outweighed traffic reuse.
A separate K-only kernel was not retained: the causal baseline ablation already
bounded K read at 25–28% of attention, while candidates A–C falsified scalar
multi-head state before vectorization attacked both larger K and V costs.

Matrix2 was also not prototyped. With M=1 there was no concrete tile that
improved utilization over the direct four-head mapping after packed reuse met
the traffic minimum. This follows the measured result rather than transferring
the prefill design to decode.

## Stability And KV Growth

| case | baseline tok/s (CV) | final tok/s (CV) | attention ms/token |
|---|---:|---:|---:|
| 8K, 32 tokens | 19.79 (0.0009) | 33.67 (0.0046) | 25.822 → 5.025 |
| 8K, 128 tokens | 19.71 (0.0023) | 33.58 (0.0012) | 25.986 → 5.052 |
| 28K, 32 tokens | 8.92 (0.0002) | 24.05 (0.0032) | 87.391 → 16.724 |
| 28K, 128 tokens | 8.91 (0.0007) | 23.99 (0.0006) | 87.603 → 16.793 |

The 128-token result differs by only -0.27% at 8K and -0.25% at 28K. Attention
increases by 0.54% and 0.41%, consistent with appending roughly 127 rows rather
than a first-token-only effect. The implementation has one command submission
per decode step. Baseline/final dispatches are 419/445 and barriers 315/341 per
step; the exact +26 of each is one split/reduction stage per layer.

## Prefill Control

No prefill shader or routing was modified.

| KV | baseline prompt tok/s | final prompt tok/s | delta | baseline TTFT | final TTFT | GPU prefill delta |
|---:|---:|---:|---:|---:|---:|---:|
| 8K | 429.43 | 431.34 | +0.44% | 19,076.53 ms | 18,992.08 ms | -0.70% |
| 28K | 286.01 | 285.44 | -0.20% | 97,900.29 ms | 98,094.06 ms | +0.01% |

Both controls are within run-to-run noise.

## Correctness

- Focused F16 attention comparisons at 2K, 8K and 28K passed against the
  sequential Vulkan path: maximum absolute error at most `1.526e-5`, mean
  absolute error about `1.0e-6` to `1.16e-6`, maximum relative error at most
  `5.11e-4`; all values finite.
- CPU-oracle attention comparisons passed: maximum absolute error at most
  `1.180e-5`, mean absolute error about `3.7e-6` to `4.0e-6`, maximum relative
  error at most `3.93e-4`.
- Logit comparisons passed with maximum absolute error below about `4e-6`.
  Relative error is intentionally not used alone around logits close to zero.
- INT8 comparisons are bit-identical (`max_abs = mean_abs = max_rel = 0`) and
  route through the unchanged INT8 shader.
- The retained real-device regression test generated 128 greedy tokens after a
  2,048-token prompt. Forced fallback and automatic GQA routing produced the
  exact same 128 token IDs, exercising accumulated KV append and logits rather
  than one isolated attention invocation.

## Final 28K Breakdown

| category | ms/token | GPU share |
|---|---:|---:|
| fused GQA attention | 16.724 | 42.36% |
| Q projection | 1.699 | 4.30% |
| K projection | 0.808 | 2.05% |
| V projection | 1.259 | 3.19% |
| output projection | 1.647 | 4.17% |
| MLP gate | 3.321 | 8.41% |
| MLP up | 3.380 | 8.56% |
| MLP activation | 0.070 | 0.18% |
| MLP down | 5.987 | 15.16% |
| normalization | 0.364 | 0.92% |
| RoPE | 0.112 | 0.28% |
| KV append | 0.068 | 0.17% |
| logits | 3.818 | 9.67% |
| embedding, residual and profiler residual | 0.224 | 0.57% |
| sampling, separate submission | 0.083 | — |

The fused final kernel does not expose separate production timestamps for K
read, QK, softmax, V read and AV without perturbing scheduling. Its exact K and
V byte counts are established by indexing and its baseline causal components
are reported above; no unmeasured final subcomponent is invented.

## Stop Condition And Next Work

The exact F16 representation now reads K and V once per KV head, the theoretical
4:1 GQA minimum. Further K/V traffic reduction requires a new representation.
Hypothetical INT8 KV would halve the 28K minimum from 2.982 to 1.491 GB/token
and halve KV memory. Even if attention time halved perfectly, Amdahl's law with
the final 42.36% attention share caps end-to-end gain at about 26.9%; actual gain
would be lower and must pass a new numerical qualification. Phase 1 therefore
does not change KV precision.

The next highest-impact exact-representation target is quantized M=1 matvec,
especially the 32.31% MLP family, then projections and logits. This is a real
bottleneck shift: those matvec families together exceed final attention, while
dispatch/control, KV append and sampling are too small to justify persistent or
fusion work first.
