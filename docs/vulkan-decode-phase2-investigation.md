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

TARGET=28000
PROMPT=$(perl -e 'print " a" x ($ARGV[0] - 3)' "$TARGET")
target/release/examples/bench <model.gguf> --context 32768 --kv f16 \
  --prompt "$PROMPT" --max-tokens 32 --warmup 1 --reps 3
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
| 512 | 38.75 | 0.17 | 0.45% | 25.806 | 22.951 | 99.55% |
| 2,048 | 38.31 | 0.15 | 0.39% | 26.103 | 24.014 | 99.58% |
| 8,192 | 33.45 | 0.22 | 0.64% | 29.895 | 27.732 | 99.63% |
| 16,384 | 28.82 | 0.03 | 0.11% | 34.698 | 32.614 | 99.69% |
| 28,000 | 24.04 | 0.03 | 0.14% | 41.597 | 39.488 | 99.74% |

The required long samples also exclude model load, prefill, and warm-up from
decode timing:

| KV | Generated | tok/s mean | Std. dev. | CV | Wall ms/token | GPU ms/token |
|---:|---:|---:|---:|---:|---:|---:|
| 8K | 128 | 33.60 | 0.09 | 0.26% | 29.762 | 27.651 |
| 16K | 128 | 28.76 | 0.03 | 0.10% | 34.771 | 32.673 |
| 28K | 128 | 24.04 | 0.04 | 0.15% | 41.597 | 39.492 |

No within-run decay appears.

The original active-run telemetry mixed prefill and decode. A temporary
benchmark-only monitor was therefore started on the first generated token on
the exact baseline commit and removed after the audit. These are decode-only
128-token samples at 100 ms; model load, prefill, and warm-up are excluded.

| KV | Samples | GPU util. mean (range) | VRAM MiB mean | Graphics / memory clock MHz | Power W mean (range) | Temp. C mean (range) |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 21 | 91.9% (91–92) | 5,772 | 1,995 / 7,501 | 152.2 (140.10–154.57) | 62.6 (62–63) |
| 512 | 21 | 92.3% (92–93) | 5,695 | 1,999 / 7,501 | 153.0 (152.31–155.18) | 59.4 (59–60) |
| 2K | 23 | 92.4% (92–93) | 5,695 | 1,980 / 7,501 | 159.8 (159.32–162.33) | 69.1 (69–70) |
| 8K | 27 | 93.1% (92–94) | 5,695 | 1,995 / 7,501 | 158.8 (156.41–159.18) | 63.6 (63–64) |
| 16K | 34 | 94.2% (94–96) | 5,646 | 1,974 / 7,501 | 164.5 (161.74–165.17) | 70.0 (70–70) |
| 28K | 42 | 96.0% (96–96) | 5,695 | 1,972 / 7,501 | 167.6 (164.47–168.15) | 70.3 (70–71) |

The 16K VRAM mean contains one process-exit sample; the steady value is 5,695
MiB. The driver's software-power-cap reason is active in some 2K samples and
throughout 8K–28K, as expected near the 170 W board limit, but graphics clocks
remain 1,972–1,995 MHz and temperature stays far below the 95 C slowdown point.
Power or thermal decay therefore does not explain the context curve.

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
| Decode timestamp residual | 0.104 | 0.102 | 0.103 | 0.101 | 0.101 |
| Sampling argmax, separate submission | 0.079 | 0.079 | 0.079 | 0.080 | 0.080 |

The decode rows account for 99.55%, 99.58%, 99.63%, 99.69%, and 99.74% of
their timestamp intervals. Sampling is serialized between generated tokens but
reported separately; its own timestamp attribution is 96.4–97.2%. The GPU
critical path is therefore the decode total plus about 0.08 ms sampling, not an
unsafe sum of overlapped host waits.

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

### Online-softmax state audit

The Phase 1 split4/WG512 kernel was rebuilt in a detached worktree with one
leave-one-out change at a time at 28K. All variants preserve the dispatch shape;
their deliberately invalid numerical output is used only for causal timing.

| Removed recurrence | Attention ms/token | Causal upper bound | Share of baseline attention | Share of baseline GPU token |
|---|---:|---:|---:|---:|
| None | 16.716 | — | — | — |
| Running max update | 15.782 | 0.934 ms | 5.59% | 2.37% |
| Two exponentials | 15.435 | 1.281 ms | 7.66% | 3.24% |
| Running-sum recurrence | 15.095 | 1.621 ms | 9.70% | 4.10% |
| AV rescaling | 16.151 | 0.565 ms | 3.38% | 1.43% |
| V-weighted AV update and compiler-removable V use | 12.170 | 4.546 ms | 27.20% | 11.51% |

These upper bounds overlap and are not additive. In particular, removing the
V-weighted update also lets the compiler remove the V use, so it is not a pure
ALU measure. The independent cumulative stage audit above assigns 1.169
ms/token to AV after the V-read stage. No isolated max/exp/sum/rescale
recurrence reaches the 5% end-to-end redesign gate; the larger AV bound cannot
be realized without changing the required value accumulation or KV traffic.

### KV traffic and roofline

For F16 KV, each layer logically reads one K and one V vector per KV head and
history position. The Phase 1 GQA shader indexes each packed value once and
reuses it across four Q heads, so measured source-level traffic equals the
theoretical minimum: amplification 1.00.

| KV | K theoretical / actual | K amp. | V theoretical / actual | V amp. | Total bytes | Full-attention effective bandwidth |
|---:|---:|---:|---:|---:|---:|---:|
| 8K | 0.436 / 0.436 GB | 1.00 | 0.436 / 0.436 GB | 1.00 | 0.872 GB | 170.9 GB/s |
| 16K | 0.872 / 0.872 GB | 1.00 | 0.872 / 0.872 GB | 1.00 | 1.745 GB | 179.2 GB/s |
| 28K | 1.491 / 1.491 GB | 1.00 | 1.491 / 1.491 GB | 1.00 | 2.982 GB | 178.8 GB/s |

The causal K-read deltas imply 335.7, 339.9, and 340.2 GB/s at 8K, 16K, and
28K, agreeing with the 342 GB/s sustainable bandwidth measured in Phase 1.
The analogous V deltas exceed nominal bandwidth because causal subtraction
captures overlap and compiler/cache effects; they are not physical DRAM
counters.

At 28K, QK plus AV perform approximately 11.928 GFLOP and the exact KV stream
is 2.982 GB, or 4 FLOP/byte. At 342 GB/s the traffic-only floor is 8.72 ms.
The initial full attention time is 16.768 ms and the final time is 13.217 ms,
corresponding to 178.8 and 225.6 GB/s effective bandwidth. Useful compute is
0.711 TFLOP/s initially and 0.903 TFLOP/s finally, far below device FP32 peak;
the kernel is not aggregate-compute-throughput bound. K-only streaming reaches
335.7–340.2 GB/s and is DRAM-bandwidth bound, while the full kernel is limited
by subgroup reductions, online-softmax dependencies, AV issue, and workgroup
scheduling rather than cache amplification or memory latency alone.

The unattainable traffic-only ratio is 1.52x attention, which would imply an
optimistic +15.4% final GPU-token ceiling. It is not a candidate prediction:
it assumes every non-streaming instruction is free. The causal state audit
shows no exact, isolated recurrence above the 5% end-to-end gate. Further large
traffic reduction requires a narrower KV representation; a larger QK/AV change
requires a different attention architecture and numerical qualification.

### Quantized matvec inventory

Q4_K stores 128 quant bytes and 16 metadata bytes per 256 weights. Q6_K stores
192 quant bytes and 18 metadata bytes. Weight reads dominate the activation
and output traffic of these decode GEMV operations.

Every decode projection takes FP16 input and emits FP16 except the FP32 logits.
The source-read column includes one FP16 activation stage per 64 output rows;
that small vector is cache-reused, while weights are streamed once. FLOPs count
the useful dot-product multiply-adds and exclude dequant instructions.

| Operation | Shape out×in | Quant | Dispatch/token | Quant / metadata MB | Total weight MB | Activation+output source MB | GFLOP | Baseline ms | Weight GB/s |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| Q | 4096×3072 | Q4_K | 26 | 163.578 / 20.447 | 184.025 | 10.437 | 0.654 | 1.703 | 108.04 |
| K | 1024×3072 | Q4_K | 26 | 40.894 / 5.112 | 46.006 | 2.609 | 0.164 | 0.801 | 57.45 |
| V | 1024×3072 | 13 Q4_K + 13 Q6_K | 26 | 51.118 / 5.431 | 56.549 | 2.609 | 0.164 | 1.262 | 44.82 |
| O | 3072×4096 | Q4_K | 26 | 163.578 / 20.447 | 184.025 | 10.384 | 0.654 | 1.658 | 110.98 |
| Gate | 9216×3072 | Q4_K | 26 | 368.050 / 46.006 | 414.056 | 23.482 | 1.472 | 3.308 | 125.18 |
| Up | 9216×3072 | Q4_K | 26 | 368.050 / 46.006 | 414.056 | 23.482 | 1.472 | 3.370 | 122.86 |
| Down | 3072×9216 | 13 Q4_K + 13 Q6_K | 26 | 460.063 / 48.882 | 508.944 | 23.163 | 1.472 | 5.970 | 85.25 |
| Logits | 131072×3072 | Q6_K | 1 | 301.990 / 28.312 | 330.301 | 13.107 | 0.805 | 3.818 | 86.51 |

The total is 2.138 GB of logical weight reads per token; weights are 93–97% of
each projection's source reads. Q4_K reaches 108–125 GB/s on the large shapes.
Q6_K contributes 665.764 MB/token across half of V, half of down, and logits,
but its baseline projection shader repeatedly extracted the same packed words
and scales. This made Q6_K instruction/dequant bound rather than weight-
bandwidth saturated. The existing MMVQ path was also measured separately at
M=1 and rejected: activation Q8_1 conversion plus its Q4 kernel reduced 128
throughput by 37.7%, confirming that prefill/Matrix2 policy does not transfer.

Both retained Q4_K and Q6_K projection kernels use four lanes per output row,
64 rows per 256-thread workgroup, shared activation staging, a four-part shared
reduction, and fused dequant-plus-dot with no intermediate weight expansion.
The activation vector is therefore not a material global-traffic target.

### CPU and submission path

At 28K the public wall/decode-GPU gap is 2.105 ms/token (5.88% of wall time),
but sampling contributes another serialized 0.080 ms GPU/token. One decode
submission contains 445 dispatches and 341 barriers; sampling adds one dispatch
and two barriers. CPU decode recording is 0.483 ms, submission 0.033 ms, and
descriptor work 0.047 ms; sampling adds 0.016, 0.009, and 0.001 ms. The 39.628
ms decode wait and 0.204 ms sampling wait overlap GPU work and cannot be added.
The actionable CPU total is about 0.588 ms/token (1.41% of wall), below the 5%
gate. The remaining wall gap is harness/driver scheduling, not proven removable
dispatch cost. There are no steady-state allocations; 39 allocations occur
during setup. CPU, command-buffer, descriptor, fusion, and allocation changes
were therefore rejected by Amdahl before production edits.

## Experiment Register

Resource fields come from `VK_KHR_pipeline_executable_properties`. Resident
workgroups/warps are theoretical limits from 65,536 registers, 102,400 B
shared, 1,536 threads, and 48 warps per GA106 SM; achieved occupancy was not
available because Nsight Compute captured no Vulkan kernels.

| ID | Target / architecture | Quant or attention traffic | Registers/thread | Shared/WG | Theoretical residency | Dispatch change |
|---|---|---|---:|---:|---:|---:|
| E0 | Existing MMVQ: Q8_1 activation plus dp4a GEMV | Q4_K, different arithmetic | n/a | n/a | n/a | adds activation quantization |
| E1 | Four packed bytes/load, scales hoisted, same four-lane reduction | Q6_K, 665.764 MB affected weights/token | 32 → 40 | 2,048 B unchanged | 6 WG / 48 warps unchanged | none |
| E2 | Compile exact 32Q/8KV/head128 strides into split4 | K/V 2.982 GB at 28K, amp. 1.00, GQA 4:1 | 64 | 34,944 B | 2 WG / 32 warps | none |
| E3 | Eight history splits, WG256, eight subgroups | K/V 2.982 GB at 28K, amp. 1.00, GQA 4:1 | 64 → 77 | 43,136 → 4,160 B | 2 WG/32 warps → 3 WG/24 warps | none |

The driver reports stack size zero for E1–E3. Its local-memory statistic is the
same impossible 64 GiB sentinel for every shader, so it is not used as a spill
counter. E2's optimized SPIR-V removes five integer divisions plus three loads
and access chains; its measured gain nevertheless fails Amdahl.

| ID | Screen / local result | GPU-token result | Wall/decode result | Prefill | Correctness | Decision |
|---|---|---|---|---|---|---|
| E0 | 128-token decode screen | not retained | 38.77 → 24.16 tok/s (-37.7%) | not needed | arithmetic changes to Q8_1 activation | Reject |
| E1 | V+down+logits categories -21.39% at 28K | 39.488 → 37.170 ms (-5.87%) | 24.04 → 25.45 tok/s (+5.87%) | 8K +0.07%; 28K -0.05% | Q6 CPU oracle and exact 128-token sequence | Retain |
| E2 | Attention -2.8% at 8K | predicted about -1.3%, below gate | no 28K run | not needed | focused attention parity passes | Reject and remove |
| E3 | Attention -19.8% at 128, -21.36% at 28K | checkpoint 37.170 → 33.686 ms (-9.38%) | checkpoint 25.45 → 27.94 tok/s (+9.78%) | 8K -0.15%; 28K +0.29% vs E1 | F16/INT8 parity and exact 128-token sequence | Retain |

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

The complete retained-patch matrix is:

| KV | Baseline / E1 tok/s | Gain | Wall baseline / E1 ms | V+down+logits baseline / E1 ms | GPU baseline / E1 ms | E1 CV |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 38.77 / 43.53 | +12.28% | 25.793 / 22.973 | 11.045 / 8.447 | 22.907 / 20.208 | 0.39% |
| 2K | 38.31 / 42.03 | +9.71% | 26.103 / 23.793 | 10.991 / 8.631 | 24.014 / 21.751 | 0.07% |
| 8K | 33.45 / 36.48 | +9.06% | 29.895 / 27.412 | 11.030 / 8.659 | 27.732 / 25.289 | 0.37% |
| 16K | 28.82 / 31.02 | +7.63% | 34.698 / 32.237 | 11.046 / 8.652 | 32.614 / 30.132 | 0.13% |
| 28K | 24.04 / 25.45 | +5.87% | 41.597 / 39.293 | 11.050 / 8.687 | 39.488 / 37.170 | 0.21% |

E1 leaves bytes, dispatches, output mapping, four-lane reduction, and FP32
accumulation order unchanged. Its register count rises from 32 to 40, but both
versions remain thread-limited at six 256-thread workgroups and 48 warps/SM;
the speedup is therefore instruction/dequant efficiency without occupancy loss.

## Retained Change 2: Eight-Split GQA Decode

The GQA attention kernel changes from four 512-thread workgroups to eight
256-thread workgroups per KV head. Total threads per layer remain 16,384, but
workgroups increase from 32 to 64 and source-declared shared memory per
workgroup falls from 8,320 to 4,160 bytes.

The partial scratch grows by 64 KiB, while the state scratch fits the existing
allocation. Allocation count remains 39 and reported total allocated bytes do
not change at the profiler's granularity. The fixed 4:1 GQA ownership invariant
and exact one-load-per-K/V-value representation remain intact.

The driver executable audit materially narrows the mechanism. Split4 compiles
to 64 registers/thread and 43,136 B shared/WG; split8 compiles to 77 registers
and 4,160 B shared. The old kernel is register/shared limited to two resident
WG (32 warps), while the new one is register limited to three WG (24 warps).
Thus theoretical resident-warps occupancy does not rise. The retained gain is
associated with 64 smaller scheduling units, one more resident workgroup, far
less compiled shared state, and an eight- rather than sixteen-subgroup local
reduction. Achieved active warps and spills remain unavailable; stack is zero
and the driver's local-memory counter is unusable. At 128 and 28K, attention
improves by similar 19.8% and 21.36%, supporting a scheduling/state mechanism
rather than a new long-history byte reduction.

At its checkpoint attention falls from 16.808 to 13.217 ms (-21.36%). With a
45.22% affected share and local speedup 1.272x, Amdahl predicts +10.69%; measured
GPU throughput improves 10.34% and wall throughput 9.78%.

This differs from the Phase 1 E-256 rejection: E-256 retained four splits and
32 workgroups, whereas this candidate doubles independent split workgroups.

Relative to E1, the complete E3 checkpoint matrix is:

| KV | E1 / E3 tok/s | Gain | Wall E1 / E3 ms | Attention E1 / E3 ms | GPU E1 / E3 ms | E3 CV |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 43.53 / 42.87 | -1.52% | 22.973 / 23.326 | 0.331 / 0.265 | 20.208 / 20.467 | 0.26% |
| 2K | 42.03 / 42.50 | +1.12% | 23.793 / 23.529 | 1.442 / 1.164 | 21.751 / 21.474 | 0.05% |
| 8K | 36.48 / 37.86 | +3.78% | 27.412 / 26.413 | 5.029 / 3.997 | 25.289 / 24.303 | 0.19% |
| 16K | 31.02 / 32.98 | +6.32% | 32.237 / 30.321 | 9.796 / 7.805 | 30.132 / 28.201 | 0.13% |
| 28K | 25.45 / 27.94 | +9.78% | 39.293 / 35.791 | 16.808 / 13.217 | 37.170 / 33.686 | 0.33% |

At 128 the target kernel still improves 19.8%; the 1.5% end-to-end checkpoint
loss comes from small shifts in the larger fixed categories. Final 128 remains
10.58% above the global Phase 2 baseline, so a length route would add policy
and tests without protecting a regression against the actual phase baseline.

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
| Decode timestamp residual | 0.104 | 0.31% |

The profiler accounts for 99.69% of final GPU time. Sampling is a separate
approximately 0.083 ms/token.

Final decode-only telemetry remains stable. At 128 it records 91.3% utilization,
5,695 MiB, 1,987 MHz, 154.3 W, and 65.5 C; at 28K it records 94.6%, 5,695 MiB,
1,955 MHz, 169.4 W, and 70.1 C. The corresponding 127/128-token screens are
44.10 and 27.90 tok/s. The full matrix's earlier active-run controls likewise
show no thermal or clock regression.

## Prefill And Correctness Qualification

Prefill is unchanged within run noise:

| Prompt | New baseline | Final | Change |
|---:|---:|---:|---:|
| 8K | 430.07 tok/s | 429.74 tok/s | -0.08% |
| 28K | 285.54 tok/s | 286.21 tok/s | +0.23% |

The 28K long-run prefill result is 285.77 tok/s with 0.07% CV.

Correctness checks cover both retained changes:

- the Q6_K projection shader matches its CPU oracle with maximum absolute error
  `7.5569e-3`, mean absolute error `2.1143e-3`, and maximum relative error
  `4.3850e-3`; this includes FP16 activation and output rounding;
- F16 Vulkan prefill attention versus sequential decode has maximum absolute
  error `6.1035e-5`, worst-case mean absolute error `3.5632e-6`, and maximum
  relative error `2.0833e-2` near zero;
- the corresponding projected logits have maximum absolute error `3.7558e-6`,
  worst-case mean absolute error `8.8033e-7`, and maximum relative error
  `5.0159e-1` on near-zero logits;
- the independent CPU F16 attention/logits controls have maximum absolute
  errors `4.0583e-5` / `4.1778e-6` and worst mean absolute errors
  `6.4359e-6` / `1.0208e-6`;
- the INT8 path remains bit-identical in the attention parity test;
- the final build produces the exact same 128 accumulated greedy token IDs as
  a detached Phase 1 reference after a 2K prompt;
- KV append and multi-token paths remain exercised;
- formatting and lint checks pass;
- the full `vulkan-profile` library suite passes: 151 passed, 4 ignored.

All INT8 attention and projected-logit max/mean/relative errors are exactly
zero in the four focused cases. Relative error near zero is not used as the
qualification bound; maximum absolute error and exact token identity are the
relevant controls.

The workspace suite passes with one unrelated installer fixture skipped. That
pre-existing test gives its synthetic `PATH` no `dirname`, although
`support/install.sh` invokes `dirname` before its prerequisite loop; it exits 1
instead of the fixture's expected exit 2. Phase 2 changes neither the script nor
the test, and all other 165 binary tests pass.

## Stop Decision And Remaining Bottleneck

The requested desirable 28K improvement is achieved, all global-baseline
context points improve, prefill is neutral, long-run stability holds, and
correctness gates pass.

Attention remains the largest final category at 39.24%, followed by MLP down
(11.87%), logits (11.34%), MLP up (10.07%), and MLP gate (9.91%). Within exact
F16 attention, K streaming already approaches sustainable bandwidth and source
traffic is at amplification 1.00. Compile-time stride specialization is below
the 5% gate, and the measured max/exp/sum/rescale recurrences each have less
than 5% whole-token upper bound. The remaining QK and AV cost is coupled to the
required dot products, subgroup reductions, and value accumulation rather than
one removable byte or dispatch.

Phase 2 therefore stops under condition 6: the next significant improvement
requires a separately qualified attention architecture or representation, not
another local exact-path patch. Candidate directions are a block/tensor-core
QK plus different softmax schedule, or a narrower KV representation. KV
compression must compare FP16, INT8, FP8-like formats, conversion cost, cache
size, logits distribution, and multi-token divergence in a new numerical
phase. The 1.52x traffic-only attention ratio is an absolute composite ceiling,
not evidence that a remaining local candidate can attain it.

No dependencies, public APIs, model files, or validation tolerances changed.
