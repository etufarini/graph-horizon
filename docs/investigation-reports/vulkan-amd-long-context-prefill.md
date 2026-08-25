<!--
This document owns the reproducible evidence, retained route, qualification,
and stop decision for the 2026-08-21 AMD Vulkan long-context prefill work.
-->

# AMD Vulkan long-context prefill checkpoint

## Result

The retained exact F16 GQA attention path reduces 8B end-to-end TTFT by 13.22%
at 15,435 prompt tokens and 22.17% at 28,000. It leaves the 128/512 controls
within 0.10%, preserves decode, adds no allocation or dependency, and remains
behind capability, format, shape, row-count, and context-position gates.

- Baseline: `e3d23f2` on `main`.
- Retained implementation: `1942208`.
- Branch: `perf/amd-long-context-prefill`.
- Later integration: branch tip `23d8186`; merged by PR #39 (`8e2f8cc`).
- Device: AMD Radeon RX 6750 XT, Navi 22 / RDNA2 / `gfx1031`, 40 CUs.
- Driver: RADV, Mesa 26.0.3, Vulkan 1.4.335.
- Device-local memory: 12,868,124,672 bytes.
- Relevant limits: 64 KiB LDS, reported subgroup 64, required wave32 control,
  integer dot product and FP16, no Vulkan cooperative matrix.
- Placement: 34/34 blocks on Vulkan for the primary tuple.
- Primary artifact: `Ministral-3-8B-Instruct-2512-Q4_K_M.gguf`,
  5,198,911,904 bytes, SHA-256
  `33e7a72cf5e6e2cfc2f2847075acc013d68bba023e35310cef86b5cf8fdca761`.
- Primary shape: 34 blocks, hidden 4,096, Q 4,096, K/V 1,024, FFN
  14,336, 32 query heads, 8 KV heads, head dimension 128, GQA 4:1.
- Residency at context 32,768: 5,189,935,104 engine weight bytes and
  4,563,402,752 F16 KV bytes. Long-run allocation was 10.79--10.81 GB.

Temporary profiling binaries and outputs were build artifacts under `target/`;
the tables below are the persistent record and do not depend on their continued
presence.

## Reproduction tuple

Release Graph Horizon runs use Vulkan hybrid with full automatic offload, F16
KV, context capacity 32,768, greedy sampling, 32 requested tokens, one warm-up,
and three repetitions. The 128-token row uses five repetitions. A no-system
Ministral template contributes three structural tokens, so an exact depth `N`
uses `N - 3` repetitions of the one-token string `" a"`.

```bash
cargo build --release --no-default-features --features vulkan-hybrid \
  --example bench
target/release/examples/bench MODEL --context 32768 --kv f16 \
  --prompt "<N - 3 repetitions of ' a'>" --max-tokens 32 \
  --warmup 1 --reps 3
```

The pinned comparator is llama.cpp `9bebfcb4b`, built with Vulkan and invoked
with the same model, full GPU offload, F16 K/V, flash attention, batch 2,048,
micro-batch 512, one warm-up, and three measured repetitions:

```bash
target/oracle/llama.cpp-amd-9bebfcb4/build/bin/llama-bench \
  -m MODEL -p 128,512,2048,8192,15435,28000 -n 0 \
  -ctk f16 -ctv f16 -ngl 99 -fa on -dev Vulkan0 -r 3 -o jsonl
```

Graph Horizon TTFT includes tokenization and first sampling; llama.cpp prompt
evaluation does not. The comparator is therefore a directional implementation
reference, not a claim of identical timing boundaries.

## Fresh baseline and comparator

Every Graph Horizon CV is below 0.08%; the 128 row is five repetitions and the
others are three. The historical 213.24 s symptom at 15K is stale: the clean
baseline already incorporates earlier retained work and measures 134.84 s.

| Prompt | Baseline TTFT | Prompt tok/s | llama.cpp prompt | GH / llama.cpp |
|---:|---:|---:|---:|---:|
| 128 | 623.05 ms | 205.44 | 143.18 ms | 4.35x |
| 512 | 2,523.91 ms | 202.86 | 514.11 ms | 4.91x |
| 2,048 | 10,969.46 ms | 186.70 | 2,167.76 ms | 5.06x |
| 8,192 | 56,844.43 ms | 144.11 | 11,751.18 ms | 4.84x |
| 15,435 | 134,843.05 ms | 114.47 | 36,482.69 ms | 3.70x |
| 28,000 | 350,906.76 ms | 79.79 | 90,775.60 ms | 3.87x |

At 15K the baseline held 99% GPU utilization, a 2,630 MHz graphics clock,
167--170 W board power, and 82--89 C junction temperature. Record and submit
work was not on the critical path.

## Baseline attribution

A timestamp-instrumented baseline measured 135.337 s public wall time versus
134.843 s without instrumentation (+0.37%). GPU intervals totalled
135,194.023 ms: 134,662.335 ms named kernels plus a 531.688 ms GPU residual,
or 99.61% of the measured GPU interval. CPU recording was 221.582 ms, submit
10.515 ms, descriptor work 70.616 ms, and the inclusive wait 135.219 s.
There were no runtime allocations during prefill.

| Baseline 15K stage | GPU ms | Share |
|---|---:|---:|
| F16 attention | 56,537.769 | 41.82% |
| Q4_K projections, total | 37,103.5 | 27.44% |
| Q6_K V and down projections | 40,306.1 | 29.81% |
| Quantization | 175.844 | 0.13% |
| Norm | 161.785 | 0.12% |
| Elementwise | 286.825 | 0.21% |
| RoPE and KV write | 44.240 | 0.03% |
| Embedding and logits | 46.262 | 0.03% |
| GPU interval residual | 531.688 | 0.39% |

The seven individual projections were Q 3,904.663 ms, K 1,052.923 ms, V
3,704.314 ms, O 4,029.030 ms, gate 14,052.719 ms, up 14,064.187 ms, and down
36,601.775 ms. An earlier attention-only causal trace split attention into
45.72% QK dot product, 1.81% softmax, and 52.48% weighted V accumulation. On
the fresh baseline these correspond to about 25.85 s, 1.02 s, and 29.67 s.

The attention dispatch inventory before this change was one workgroup family
per query-head tile, which repeatedly streamed the same K/V data for the four
query heads in a GQA group. The matrix inventory was already specialized:
retained AMD DP4A Q4 prefill serves Q/O/gate/up, while exact Q6 serves V/down;
K is the remaining small projection. The timestamp sum and GPU residual rule
out CPU orchestration, allocation, or an unnamed transfer as the missing cost.

## Amdahl decision

Attention represented `56.537769 / 134.84305 = 41.93%` of baseline TTFT. The
pre-screen assumed a feasible 1.8x local attention speedup, predicting
109.715 s TTFT, an 18.63% full-request reduction. The retained path measures a
1.4603x causal local speedup: only attention routing changed, so the 17.821 s
whole-request reduction attributes the post-change attention cost as
38.716729 s. Amdahl then predicts `134.843 / 117.022 = 1.1523x`, exactly the
measured 13.22% within the common timing boundary.

## Retained architecture and invariants

The new split shader owns one `(8 query rows, 4 query heads, KV head, history
split)` tile. It uses a required wave32 subgroup per query/head pair. Thirty-two
queries therefore share each 64-position K/V tile instead of independently
streaming it. Eight history splits restore enough workgroup parallelism for a
long causal scan. Each split stores its FP32 accumulator and online-softmax
`(max, sum)` state; a second exact reduction shader rescales and merges all
eight partials before narrowing the output to F16.

The dispatch sequence is exactly split then reduce. The primary hybrid tuple
uses 32-row chunks: 483 prompt chunks total, with 420 at or beyond the 2,048
crossover. Across 34 layers that adds 14,280 reduction dispatches. The
standalone 64-row guard has 242 chunks, 211 specialized, and 7,174 extra
dispatches. They are GPU-resident and already included in the end-to-end result.

The path reuses existing MMVQ scratch. At the maximum qualified 64 rows, the
FP32 partials require 8,388,608 bytes and the state requires 131,072 bytes,
within the two existing 8 MiB buffers. It adds no persistent bytes, allocation,
copy, command buffer, dependency, public API, or cache-layout change.

Eligibility requires all of the following:

- AMD vendor, DP4A qualification proxy, reported subgroup 64, required wave32
  control, 1,024 invocations/workgroup, and at least 33,792 bytes LDS;
- both split and reduction pipelines built successfully;
- F16 KV, head dimension 128, exact 4:1 GQA, and nonzero rows;
- at most 64 rows and a chunk ending at or beyond position 2,048.

INT8 KV, non-AMD hardware, unsupported head/GQA shapes, larger batches, missing
capabilities, short context, or pipeline absence select the pre-existing route.
The arithmetic uses checked base-plus-row eligibility and bounded scratch
indices. The existing NVIDIA Q64 route remains first in policy order.

RADV compiler statistics for the split shader are 108 SGPRs, 64 VGPRs,
33,792 bytes LDS, zero scratch, zero spills, 16 subgroups/SIMD, and 724
instructions. The reducer uses 108 SGPRs, 32 VGPRs, 1,024 bytes LDS, zero
scratch, zero spills, 16 subgroups/SIMD, and 230 instructions.

## Final primary matrix

| Prompt | Baseline TTFT | Final TTFT | TTFT change | Baseline tok/s | Final tok/s | Throughput change |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 623.05 ms | 623.09 ms | +0.01% | 205.44 | 205.43 | -0.00% |
| 512 | 2,523.91 ms | 2,526.37 ms | +0.10% | 202.86 | 202.66 | -0.10% |
| 2,048 | 10,969.46 ms | 10,944.16 ms | -0.23% | 186.70 | 187.13 | +0.23% |
| 8,192 | 56,844.43 ms | 51,975.27 ms | -8.57% | 144.11 | 157.62 | +9.38% |
| 15,435 | 134,843.05 ms | 117,022.01 ms | -13.22% | 114.47 | 131.90 | +15.23% |
| 28,000 | 350,906.76 ms | 273,108.36 ms | -22.17% | 79.79 | 102.52 | +28.49% |

Final CVs are 0.04% at 15K and 28K and at most 0.93% in the matrix. Public
decode is unchanged at 15K (32.20 to 32.27 deltas/s); the 28K result improves
from 27.39 to 28.06 deltas/s rather than regressing.

The 28K candidate still reports 99% GPU utilization and a 2,630 MHz clock.
Allocation is unchanged, but board power and junction temperature rise from
about 173 W / 87 C on the baseline sample to about 198 W / 97 C. There was no
clock collapse, reset, spill, or correctness failure. This is the material
tradeoff: the kernel uses previously idle compute to finish sooner.

## Multi-model and decode guards

All rows below use authenticated Instruct Q4_K_M artifacts and F16 KV. The 3B
and 14B A/B controls disable only the new policy branch in the fallback binary.

| Model / live depth | Fallback TTFT | Specialized TTFT | Change | Decode control |
|---|---:|---:|---:|---:|
| 3B / 8K, context 32K | 40.223 s | 32.665 s | -18.79% | 71.11 vs 70.64 model tok/s |
| 8B / 15K, context 32K | 134.843 s | 117.022 s | -13.22% | 32.20 vs 32.27 public deltas/s |
| 14B / 8K, context 16K | 92.325 s | 83.891 s | -9.14% | 22.22 vs 22.27 model tok/s |

The three files match catalog sizes and SHA-256 digests: 3B
`9ed150d4367e68df0ac8e1540f6ddc65b42d0ee26378329d1ecbca60f93fc5f8`,
8B `33e7a72cf5e6e2cfc2f2847075acc013d68bba023e35310cef86b5cf8fdca761`,
and 14B `824e0f3373e69b84f2cae46fdcb9bd1ebc6ab3bfc7acc125d818b7b8178cc613`.

The standalone exact-prefix harness passed all required 8B depths:

| KV depth | Prime prefill | Decode tok/s | CV |
|---:|---:|---:|---:|
| 128 | 664 ms | 40.6609 | 0.2009% |
| 2,048 | 11,219 ms | 39.3895 | 0.1403% |
| 8,192 | 52,482 ms | 36.5047 | 0.1301% |
| 15,435 | 117,174 ms | 33.3751 | 0.1546% |
| 28,000 | 272,797 ms | 29.0381 | 0.0642% |

Every measured request reused the exact caller-keyed prefix with one tail token
refilled. A calibrated public request with exactly 15,435 prompt tokens then
generated the requested maximum of 512 tokens at 32.31 model tok/s and
129.784 s TTFT. A separate 8K real-model passkey check recovered `GHZ-314159`.

## Correctness qualification

- The release ignored long-context numeric test passes F16 and INT8 KV at 2K,
  8K, and 28K with 64-, 32-, and 16-row batches. The new F16 result differs
  from sequential attention by at most `1.5258789e-5`, one F16 ULP.
- The full Vulkan-hybrid workspace/all-target suite passes. It covers Q4_K and
  Q6_K projection oracles, causal GQA, KV layout/write, logits, and batched
  versus sequential prefill/decode equivalence.
- Warning-denied Vulkan-hybrid Clippy and formatting pass.
- Capability and policy tests prove the AMD/wave32/resource requirements,
  long-context crossover, pipeline fallback, oversized-row fallback, and
  preservation of the NVIDIA Q64 route.
- No rejected diagnostic code, temporary timestamp path, or fallback ablation
  remains in the final tree.

## Candidate ledger

| Candidate | Result | Decision |
|---|---:|---|
| Eight-way split-history GQA F16 attention | -13.22% at 15K; -22.17% at 28K | KEEP (`1942208`) |
| Four history splits | 72.517 s at 8,194 tokens vs about 52 s retained; decode also collapsed under sustained load | REJECT; insufficient parallelism |
| Q4 tile 16 | historical -12.9% performance | REJECT; larger tile reduced occupancy |
| Exact Q6 64 tile | historical -6.6% performance | REJECT |
| Exact Q6 packed staging | historical +2.15% whole request | REJECT; below 5% gate |
| Lossy Q6 dot path | failed parity | REJECT; numeric contract violation |
| Wave32 local attention | +7.77% local, only +2.43% whole request | REJECT; below gate |
| Grouped 4/2-head attention | regression | REJECT |
| Q16 attention | +10.45% local, +4.66% whole at 8K | REJECT; below gate |

## Post-change bottleneck and stop

Causal re-attribution at 15K gives 38.717 s attention (33.08%), 37.104 s Q4
(31.71%), 40.306 s exact Q6 (34.44%), and 0.896 s other work (0.77%). Exact
Q6 is now the largest measured component by a narrow margin. Its completed
exact candidates offer only about 2--3 seconds and fail the 5% whole-request
gate; its lossy candidate failed parity. Larger Q4 tiles regress.

Attention still has the largest theoretical headroom. The causal scan is about
66.4 TFLOP at 15K; a physical lower bound is roughly 5 s compute or 7 s K/V
traffic after 32-query reuse, while a practical online-softmax/LDS/occupancy
floor is about 30 s. That leaves roughly 8.7 s practical attention headroom,
but the remaining ownership variants either regress, fail the gate, or require
a new multi-workgroup state architecture with materially greater complexity.

The retained result clears the explicit long-context threshold on the primary
tuple, improves both other supported model sizes, and passes all guardrails.
Further available low-complexity candidates are economically closed. The final
Graph Horizon / llama.cpp ratios remain 3.21x at 15K and 3.01x at 28K; closing
that comparator gap is a new kernel-architecture program, not unfinished
cleanup in this checkpoint.
