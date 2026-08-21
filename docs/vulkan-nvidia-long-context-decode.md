<!--
This checkpoint owns the 2026-08-21 NVIDIA Vulkan long-context decode
investigation: reproducibility, attribution, routing and residency audits,
economic closures, the retained global rerank, and final qualification.
-->

# NVIDIA Vulkan Long-Context Decode

## Outcome

The reported 14B symptom (about 220 ms/decode-token at 15,434 prompt tokens)
does not reproduce at the starting revision. Fresh all-GPU hybrid measurement
is 51.334 ms/token (19.480 tok/s), 4.28 times faster than the symptom. A
1,156-token final run averages 51.721 ms/token while its KV length grows. The
current decode path is therefore the qualified result of earlier retained Q4
DP4A and vector-GQA work, not the stale slow path.

No new decode candidate clears the economic gate. Exact F16 attention already
reads K/V once, its context slope matches llama.cpp, and the remaining gap is
KV-independent. The best previously tested Q6 DP4A candidate predicts only a
3.69% 14B whole-token gain and changes arithmetic; Q4, attention, and control
candidates are smaller. Decode priority is closed under the mission stop rule.

Global reranking exposed a much larger routing defect: hybrid all-GPU prefill
kept the old 32-row batch while pure Vulkan selected the qualified 512-row
Matrix2 batch. Production commit `2228a7a` gives capable NVIDIA hybrid devices
the same route, accounts for its scratch before placement, falls back to the
32-row all-GPU plan near capacity, and activates the larger batch only when the
required pipelines exist. At 14B/15K, TTFT falls 230.888 -> 33.549 seconds
(-85.47%) while decode remains 51.334 -> 51.250 ms/token (-0.16%).

## Reproducibility

| Item | Value |
|---|---|
| Date | 2026-08-21 |
| Branch | `perf/nvidia-long-context-decode-20260821` |
| Baseline | `87e3e4b1bb17e811caac030f69261bd0498284ec` |
| Production | `2228a7a` |
| GPU | NVIDIA GeForce RTX 3060 12 GiB, Ampere GA104, PCI device `0x2487` |
| Driver / Vulkan | 595.84 / device API 1.4.329, loader 1.4.341 |
| Subgroup / caps | 32 lanes; FP16, INT8 dot, cooperative matrix, Matrix2 |
| Limits used | 256-thread GQA; 512-row Matrix2 prefill; 102,400 B shared/SM |
| Clock / power | active mean about 1,912--1,983 MHz SM, 7,501 MHz memory; board cap 170 W |
| 3B artifact | 2,147,023,008 bytes; SHA-256 `9ed150d4367e68df0ac8e1540f6ddc65b42d0ee26378329d1ecbca60f93fc5f8` |
| 14B artifact | 8,239,593,024 bytes; SHA-256 `824e0f3373e69b84f2cae46fdcb9bd1ebc6ab3bfc7acc125d818b7b8178cc613` |
| Runtime | Vulkan/Vulkan-hybrid, full GPU placement, F16 KV, greedy sampling |
| Profiler | preserved `17f77bf` Vulkan-timestamp binary; current public source separately rebuilt |
| Comparator | llama.cpp `9bebfcb4b`, Vulkan0, full offload, flash attention, F16 K/V |

The profiler binary adds 2--6% wall overhead but current public decode matches
its kernel routes. Its GPU categories account for 98.87--99.50% of timestamped
GPU time. No hardware counter tool available here reports Vulkan kernel
register spills or achieved occupancy; NVIDIA telemetry and Vulkan pipeline
properties are the available physical evidence.

Representative commands (the prompt is exact when `N-3` copies of `" a"` are
passed through the chat template):

```sh
cargo build --release --example bench --no-default-features --features vulkan-hybrid
target/release/examples/bench <model.gguf> --context 32768 --kv f16 \
  --prompt <exact-N-token-prompt> --max-tokens 32 --warmup 1 --reps 3

llama-bench -m <model.gguf> -p 0 -n 32 -d 128,2048,8192,15434,28000 \
  -b 256 -ub 256 -ctk f16 -ctv f16 -ngl 99 -fa on -dev Vulkan0 -r 3 -o jsonl
```

## Fresh Decode Matrix

The high-repetition 3B baseline uses the current public pure-Vulkan path,
context capacity 32,768, 32 generated tokens, one warm-up, and 5/5/3/3/2
repetitions. Every CV is below 0.25%. The final hybrid matrix uses the same
model and token contract; paired same-feature checks show the production edit
does not enter the decode graph.

| KV | Baseline tok/s | Final tok/s | Baseline ms/token | Final ms/token | Final CV | Delta ms |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 71.35 | 70.62 | 14.015 | 14.160 | 0.39% | +0.145 |
| 2,048 | 69.83 | 68.98 | 14.321 | 14.497 | 0.32% | +0.176 |
| 8,192 | 60.54 | 60.23 | 16.518 | 16.603 | 0.35% | +0.085 |
| 15,434 | 52.20 | 52.03 | 19.157 | 19.220 | 0.42% | +0.063 |
| 28,000 | 42.18 | 42.14 | 23.708 | 23.730 | 0.09% | +0.022 |

This cross-feature table is conservative. Same-feature hybrid A/B gives
70.94 -> 70.24 tok/s at 128 and 69.64 -> 69.18 at 2K (1.0% and 0.66%);
an 8K A/B/A gives final 60.49, baseline 60.19, final 60.20 tok/s. The single
28K hybrid comparison is 42.51 -> 42.14 tok/s (0.88%). There is no reproducible
decode regression beyond the 1% guard.

At 14B/15,434, the fresh workload match is stronger: all 40 layers and weights
remain on GPU, baseline decode is 51.334 ms/token (CV 0.79%), and the retained
candidate is 51.250 ms/token (CV 0.74%). The reported 220 ms symptom is stale.

## Competitive Reference And Scaling

| 3B KV | Ours ms/token | llama.cpp ms/token | Ratio | Gap ms/token |
|---:|---:|---:|---:|---:|
| 128 | 14.160 | 9.397 | 1.507x | 4.763 |
| 2,048 | 14.497 | 10.097 | 1.436x | 4.400 |
| 8,192 | 16.603 | 12.292 | 1.351x | 4.311 |
| 15,434 | 19.220 | 14.895 | 1.290x | 4.325 |
| 28,000 | 23.730 | 19.508 | 1.216x | 4.222 |

The final diagnostic fit is:

```text
Graph Horizon: T(N) = 13.890263 + 0.000348668461*N ms, R²=0.998537
llama.cpp:     T(N) =  9.338419 + 0.000362379974*N ms, R²=0.999963
```

The fitted KV fraction is 27.92% at 15,434 and 41.27% at 28K. The baseline fit
was `13.737277 + 0.000353751426*N` ms. The candidate changes neither term.
Graph Horizon's measured KV slope is 3.8% smaller than llama.cpp's; its roughly
4.3 ms gap is KV-independent.

The same conclusion holds for the workload-matching 14B model. Its two-point
diagnostic is `42.4019 + 0.000578727*N` ms versus llama.cpp
`26.8779 + 0.000572797*N` ms. At 15,434, ours is 51.334 ms and llama.cpp is
35.718 ms: a 15.616 ms constant-cost gap with essentially equal slopes.

## Full Decode Attribution

This same-run profiler budget reconciles each instrumented wall token. Q4 is
Q/K/O/gate/up; Q6 is V/down/logits; support contains activation quantization,
normalization, RoPE, KV append, residual/SiLU, and embedding. The last column
contains timestamp residual, host recording/submission/descriptor work, driver
scheduling, and the profiling overhead. Sampling is serialized and separate.

| 3B KV | Wall | Attention | Q4 | Q6 | Support | Sampling | Residual/control | GPU categories |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 14.883 | 0.228 | 5.065 | 5.689 | 1.071 | 0.082 | 2.748 | 98.87% |
| 2,048 | 14.919 | 0.882 | 5.019 | 5.602 | 1.074 | 0.082 | 2.260 | 98.94% |
| 8,192 | 17.106 | 2.940 | 5.134 | 5.576 | 1.094 | 0.082 | 2.280 | 99.03% |
| 15,434 | 19.681 | 5.463 | 5.091 | 5.680 | 1.095 | 0.082 | 2.270 | 99.18% |
| 28,000 | 24.207 | 9.705 | 5.163 | 5.782 | 1.113 | 0.082 | 2.363 | 99.36% |

All values are ms/token. Broad groups above are only the compact all-context
view; the workload-matching 14B/15K budget below exposes individual operations.

| 14B/15K component | ms/token | Wall % | Actual route / contents |
|---|---:|---:|---|
| Attention | 8.383 | 16.22% | vector 4:1 GQA, eight exact F16 KV splits |
| Q projection | 1.933 | 3.74% | Q4 per-8 Q8/DP4A MMVQ |
| K projection | 0.892 | 1.73% | Q4 per-8 Q8/DP4A MMVQ |
| V projection | 1.847 | 3.57% | exact packed Q6 float GEMV |
| O projection | 2.029 | 3.93% | Q4 per-8 Q8/DP4A MMVQ |
| Gate | 6.695 | 12.95% | Q4 per-8 Q8/DP4A MMVQ |
| Up | 6.717 | 13.00% | Q4 per-8 Q8/DP4A MMVQ |
| Activation quantization | 0.560 | 1.08% | persistent Q8 scratch, no allocation |
| Down | 15.217 | 29.45% | exact packed Q6 float GEMV |
| Normalization / final norm | 0.793 | 1.53% | 81 RMSNorm invocations/token |
| RoPE | 0.217 | 0.42% | Q and K only |
| KV append / management | 0.123 | 0.24% | one in-place K/V append/layer |
| Activation / residual | 0.295 | 0.57% | SiLU-mul and residual operations |
| Embedding | 0.013 | 0.03% | one token lookup |
| Logits | 3.042 | 5.89% | exact FP32-output Q6 GEMV |
| Sampling | 0.082 | 0.16% | GPU argmax, separate submission |
| GPU timestamp residual | 0.248 | 0.48% | uncategorized interval residue |
| Record/submit/descriptors | 1.138 | 2.20% | actionable CPU work; waits overlap GPU |
| Driver/harness/profile gap | 1.456 | 2.82% | reconciliation, not assumed removable |
| **Total** | **51.680** | **100%** | profiler wall boundary |

The profiler records 923 decode dispatches, 763 barriers and one command
buffer per token. Sampling adds one dispatch, two barriers and one submission.
CPU decode record/submit/descriptor work is 0.957/0.067/0.089 ms/token;
sampling adds 0.025 ms. The 49.122 ms CPU wait overlaps the GPU and is not
added. Steady decode allocates, resizes, maps and creates zero resources.
No material GPU idle interval is established: the public-minus-GPU remainder
is about 2.25 ms without profiling, of which measured actionable CPU work is
about 1.14 ms. Treating the rest as removable idle would double count driver,
harness and timestamp-boundary differences.

### Attention traffic and stages

The 14B model has 32 query heads, 8 KV heads, head dimension 128 and 40 layers.
At 15,434, K and V are each 1.264 GB/token; the source indexes each packed F16
value once per KV head and shares it across four Q heads. K and V traffic
amplification are both exactly 1.00. Split partial/state/Q/output traffic is
about 0.011 GB, so total source traffic is about 2.540 GB/token.

QK and AV each perform about 5.058 GFLOP/token. Total useful attention is about
10.116 GFLOP/token, arithmetic intensity about 3.98 FLOP/byte. At 8.383 ms,
effective bandwidth is 301.7 GB/s and useful compute is 1.21 TFLOP/s. The
measured K-only sustainable ceiling is 340 GB/s, giving a practical exact
traffic floor of 7.44 ms and only 0.94 ms removable.

K read and QK, online max/sum/exponentials, V read and AV are fused and overlap;
they cannot be assigned independent current timestamps without changing the
compiler schedule. The retained causal audit at the nearby 16K point, before
the later vector-state improvement, separated skeleton/K-read/QK/softmax/
V-read/AV as 0.151/2.567/3.796/1.130/1.316/0.776 ms. Those values are causal
architecture evidence, not additive claims about today's 8.383 ms kernel. The
current vector state removed 27% locally in its qualified A/B. This distinction
avoids inventing subcomponent time inside a fused kernel.

### Quantized M=1 inventory

Canonical Q4_K uses 128 data plus 16 metadata bytes per 256 weights; Q6_K uses
192 plus 18. Loads have traffic amplification 1.00 and dequantize directly into
FP32 accumulators. Q4 stages one per-8 Q8 activation, uses DP4A and subgroup/
shared reduction. Q6 uses four lanes/output, packed word loads, hoisted scale
decoding, shared activation and a four-part shared reduction. Weight traffic is
over 93% of source traffic; activation/output bytes are not a material fusion
target.

| 14B operation | Out x in | Quant | Weight GB/token | GPU ms | Effective GB/s | Dispatches |
|---|---:|---|---:|---:|---:|---:|
| Q | 4096 x 5120 | Q4_K | 0.472 | 1.933 | 244 | 40 |
| K | 1024 x 5120 | Q4_K | 0.118 | 0.892 | 132 | 40 |
| V | 1024 x 5120 | Q6_K | 0.172 | 1.847 | 93 | 40 |
| O | 5120 x 4096 | Q4_K | 0.472 | 2.029 | 233 | 40 |
| Gate | 16384 x 5120 | Q4_K | 1.887 | 6.695 | 282 | 40 |
| Up | 16384 x 5120 | Q4_K | 1.887 | 6.717 | 281 | 40 |
| Down | 5120 x 16384 | Q6_K | 2.753 | 15.217 | 181 | 40 |
| Logits | 131072 x 5120 | Q6_K | 0.551 | 3.042 | 181 | 1 |

Q4 large shapes are predominantly bandwidth-limited with unpack/correction
work still visible on smaller K. Q6 is instruction/dequant limited: it sustains
about 181 GB/s despite exact single-pass traffic. No spill counter is available;
the prior pipeline executable report found zero stack and stable theoretical
residency, while lane-2/4/8 variants bracketed the current geometry. Fusing QKV
or gate/up cannot remove their dominant independent weight streams.

## Routing, Residency And Memory

At every 3B context, actual decode routes are
`attention_decode_gqa_reduce_q4_vec4`, `matmul_q4k_mmvq_f16`, exact
`matmul_q6k`, and `logits_q6k`. Predicates are capability + subgroup + format +
shape: DP4A and subgroup 32 for Q4; F16 KV, head 128, exact 4:1 GQA and resource
limits for attention; canonical Q6 for V/down/logits. Context length changes
dispatch extent only. Generic Q4/Q6/attention paths remain registered fallbacks.

The primary 14B/16,384 plan is 0 CPU / 40 GPU layers:

| Device-memory category | Baseline bytes | Final bytes |
|---|---:|---:|
| Model weights | 8,230,348,800 | 8,230,348,800 |
| F16 KV capacity | 2,684,354,560 | 2,684,354,560 |
| Temporary prefill/decode scratch | 5,609,472 | 87,201,792 |
| Persistent logits/reduction/MMVQ buffers | 17,563,648 | 17,563,648 |
| Explicit reserve | 606,624,153 | 606,624,153 |
| **Placement total** | **11,544,500,633** | **11,626,092,953** |

The retained 512-row batch costs 81,592,320 bytes (77.81 MiB). CPU ownership
is 551,026,688 bytes: 550,502,400 bytes of bounded staging plus 524,288 fixed,
not resident layers. There are no CPU/GPU copies/token.

For the 3B 32,768-capacity matrix, weights are 2.139 GB and reserved F16 KV is
3.490 GB. Logical KV occupancy is:

| KV | Logical KV GiB | Observed allocation behavior |
|---:|---:|---|
| 2,048 | 0.203 | fixed capacity allocation; no resize/migration |
| 8,192 | 0.812 | same route and residency |
| 15,434 | 1.531 | same route and residency |
| 28,000 | 2.777 | same route and residency |

Peak NVIDIA-reported 3B memory is 5,986 MiB. Temporary/persistent execution,
pipelines, logits, prefill rows and driver allocations account for the remainder
after weights and reserved KV. The substantial 14B run peaks at 10,927 MiB.
Active telemetry reports 96.9% mean utilization, 166.2 W mean power, 1,912 MHz
mean SM clock and 67.3 C. Neither context growth nor the candidate causes
allocation churn, residency loss, host migration, or route changes.

A 32,768-capacity 14B plan becomes mixed (6 CPU / 34 GPU layers) and measures
about 118.5 ms/token at 2K. Its 7.9 tok/s prefill is incompatible with the
observed 72.3 tok/s symptom, so it is a capacity control, not the primary ratio.

## Practical Floors And Amdahl Ranking

The 14B/15K public baseline `T` is 51.334 ms/token. Fractions below use the
same profiled routes. Physical floors are informative bounds; practical floors
use measured sustainable behavior or completed candidates.

| Candidate domain | Current ms | Physical floor | Practical floor / local speedup | Removable ms | Predicted final | Predicted tok/s | Whole gain | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Exact F16 attention | 8.383 | 7.03 at 360 GB/s | 7.44 at 340 GB/s / 1.127x | 0.94 | 50.39 | 19.85 | 1.87% | closed; amp 1, slope competitive |
| Q6 V/down/logits DP4A | 20.106 | 10.22 at 340 GB/s | measured 1.10x local | 1.83 | 49.51 | 20.20 | 3.69% | rejected; arithmetic change, <5% |
| Q4 projections/MLP | 18.265 | 14.23 at 340 GB/s | about 1.10x local | 1.66 | 49.67 | 20.13 | 3.35% | closed; lane geometries bracketed |
| Record/submit/descriptors | 1.138 | 0 | about 0.70 ms | 0.44 | 50.89 | 19.65 | 0.87% | closed; waits overlap GPU |
| Norm/RoPE/KV/elementwise | 1.989 | 0 | about 1.50 ms | 0.49 | 50.84 | 19.67 | 0.96% | closed; no individual material share |

For Q6, `f=20.106/51.334=0.392` and the best completed M8 experiment supplies
`S_local≈1.10`; Amdahl gives `S_total=1/((1-f)+f/S)=1.0369`, 1.83 ms removable.
That experiment already included V/down and FP32 logits, passed its focused
oracle, but improved 8B whole decode only 4.91%; M4 regressed and M16 is
dominated. New 14B share evidence still predicts below the moderate 5% gate.

Attention's corresponding `f=0.163` and `S_local=1.127` give 0.94 ms removable
and 1.87% whole gain. A different KV representation could reduce bytes further,
but attention is only 16.2% of the workload-matching token, the exact path is
already at amplification 1, and F16->INT8 would add quality/read-dequant gates
for a sub-5% likely whole benefit. It is not economically justified here.

The impossible-delete ceilings prevent combinations from being mistaken for
candidate predictions. Existing Q4 L4/L16, Q6 M4/M8, split counts, scalar DP4A
prefill, QKV/gate-up fusion arguments, attention tiling, state recurrence, and
dispatch/allocation directions are closed by prior measured regressions or the
floors above. No allowed decode design space contains 5% realistic removable
time without a new numerical/representation contract.

## Retained Global Rerank

The hybrid 32-row route issued 483 full graph batches for the 15,434-token 14B
prompt. The qualified 512-row path issues 31, preserving the same Matrix2 Q4,
Q6 and attention kernels while amortizing recording, submission and weight
work across sixteen times more rows. This is routing/eligibility, not a new
kernel or numerical policy.

| Workload | Baseline hybrid TTFT | Final TTFT | Change | Baseline decode | Final decode |
|---|---:|---:|---:|---:|---:|
| 3B / 128 | 602.70 ms | 116.40 ms | -80.69% | 71.80 | 70.62 tok/s |
| 3B / 2K | 8.816 s | 1.012 s | -88.52% | 70.73 | 68.98 tok/s |
| 3B / 8K | 37.826 s | 4.846 s | -87.19% | 60.98 | 60.23 tok/s |
| 3B / 28K | 158.626 s | 27.258 s | -82.82% | 42.51 | 42.14 tok/s |
| 14B / 15,434 | 230.888 s | 33.549 s | -85.47% | 19.480 | 19.512 tok/s |

The same-feature baseline guard rows are single-run because their gap is far
larger than measured CV. Final 3B rows use repeated measurements. Pure Vulkan
14B/15K is 33.374 s, independently confirming the expected convergence.

## Real Generation And Quality

The final all-GPU run uses 14B, context capacity 16,640, a 15,434-token prompt,
F16 KV, greedy sampling and 1,156 generated deltas:

| Metric | Result |
|---|---:|
| TTFT | 33.513 s |
| Decode mean / median | 51.721 / 51.726 ms/token |
| Decode standard deviation / CV | 0.439 ms / 0.85% |
| Decode throughput | 19.335 tok/s |
| Beginning / middle / end | 51.370 / 51.766 / 52.026 ms/token |
| Total request | 93.279 s |

Against the reported symptom, decode time falls about 220 -> 51.7 ms/token,
prefill 214 -> 33.5 seconds, and total request 468 -> 93.3 seconds (-80.1%).
Against the fresh pre-change code, the retained production edit removes about
197 seconds of prefill while leaving the already-fast decode unchanged.

Qualification passes:

- focused hybrid batch selection, placement accounting, Matrix2 Q4/Q6, F16
  attention, and F16/INT8 prefill/decode parity tests;
- exact greedy output equality between hybrid and pure Vulkan at 128 and 2K;
- F16 and INT8 8K passkey recovery through the public API;
- 14B 1,156-token generation with no within-run decay or placement change;
- pure Vulkan compilation and generic fallback preservation.

Temporary probes and profiling variants are not production files. The next
candidate is **none under the current decode contract**. Reopen Q6 only with a
quality-approved persistent representation or a measured local speedup above
1.14x; reopen exact attention only if a new architecture beats the 340 GB/s
practical traffic model or changes exact byte count without quality loss.
