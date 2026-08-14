<!--
This report owns the reproducible Phase 5 and Phase 6 Vulkan short-context prefill
evidence: baselines, attribution, experiments, retained changes, correctness, and regression gates.
It defines no runtime behavior and contains no benchmark-only backend constants.
-->

# Vulkan Short-Context Prefill Phase 5

## Scope and reproducibility

The investigation ran on branch `perf/vulkan-short-prefill-phase5` from clean
commit `ae49b6a5444410af32a16bbe0e6793b5c0602a95`. Required predecessors
`1b12472`, `825057e`, and `a7b997f` are ancestors. No result was inherited as
the baseline and nothing was pushed.

The measured tuple was:

- NVIDIA GeForce RTX 3060 12 GiB, PCI device `0x2487`, driver 595.84;
- Vulkan device API 1.4.329, Rust/Cargo 1.95;
- `Ministral-3-3B-Instruct-2512-Q4_K_M.gguf`, 2,147,023,008 bytes;
- SHA-256 `9ed150d4367e68df0ac8e1540f6ddc65b42d0ee26378329d1ecbca60f93fc5f8`;
- pure Vulkan release build with `vulkan-profile`, context 32,768, F16 KV;
- exact synthetic prompts, greedy sampling, and two emitted tokens.

Prompt lengths through 512 used two warm-ups and nine measured repetitions;
1K/2K used one plus five; 8K/28K used one plus three. Profiler aggregates
include warm-ups, while public TTFT statistics exclude them. Quantiles use
linear interpolation over separately logged measured samples. The temporary
sample logger and every shader ablation were removed from the final tree.

## Fresh baseline

The baseline shows low variance and no short-prompt GPU-idle regime. GPU time
is phase-only Vulkan timestamp time; TTFT also includes request setup, sampling,
readback, and the first decoded token.

| Tokens | TTFT mean ± sd | CV | Prompt tok/s | GPU prefill/request |
|---:|---:|---:|---:|---:|
| 32 | 102.25 ± 0.46 ms | 0.45% | 312.98 | 55.919 ms |
| 64 | 160.22 ± 0.62 ms | 0.38% | 399.46 | 111.342 ms |
| 128 | 277.10 ± 1.02 ms | 0.37% | 461.94 | 221.060 ms |
| 256 | 512.98 ± 2.70 ms | 0.53% | 499.05 | 446.902 ms |
| 512 | 991.87 ± 2.33 ms | 0.23% | 516.20 | 903.146 ms |
| 1K | 1,975.99 ± 5.55 ms | 0.28% | 518.22 | 1,841.087 ms |
| 2K | 4,049.22 ± 4.08 ms | 0.10% | 505.78 | 3,823.669 ms |
| 8K | 19,079.88 ± 50.38 ms | 0.26% | 429.35 | 18,282.647 ms |
| 28K | 98,180.15 ± 77.95 ms | 0.08% | 285.19 | 95,562.269 ms |

Nine independent warm-start processes established the noisier baseline tails
at the primary short points:

| Tokens | p10 | median | p90 |
|---:|---:|---:|---:|
| 32 | 103.122 ms | 107.800 ms | 109.100 ms |
| 64 | 160.846 ms | 161.820 ms | 163.494 ms |
| 128 | 278.168 ms | 278.380 ms | 279.820 ms |
| 256 | 514.488 ms | 515.050 ms | 516.962 ms |
| 512 | 995.368 ms | 996.250 ms | 1,000.086 ms |

Telemetry sampled at 100 ms while GPU utilization was at least 80%:

| Tokens | GPU util mean | VRAM | Power mean (range) | Temperature mean (range) |
|---:|---:|---:|---:|---:|
| 32 | 92.4% | 5,212 MiB | 113.3 W (76.6--117.8) | 58.3 C (56--60) |
| 64 | 92.5% | 5,204 MiB | 127.9 W (93.3--131.6) | 62.1 C (59--64) |
| 128 | 94.1% | 5,076 MiB | 138.9 W (81.3--146.8) | 65.2 C (60--67) |
| 256 | 95.1% | 5,116 MiB | 146.2 W (69.9--157.8) | 67.7 C (63--70) |
| 512 | 95.6% | 5,282 MiB | 157.5 W (96.1--163.2) | 69.7 C (65--71) |
| 1K | 94.0% | 5,593 MiB | 154.4 W (58.7--169.1) | 66.2 C (60--68) |
| 2K | 95.7% | 5,331 MiB | 161.0 W (91.9--169.5) | 68.2 C (63--70) |
| 8K | 95.7% | 5,454 MiB | 158.8 W (71.2--168.7) | 69.9 C (64--71) |
| 28K | 97.4% | 5,693 MiB | 153.2 W (63.4--165.9) | 70.2 C (62--72) |

Graphics clocks stayed about 1.97--2.00 GHz and memory at 7.501 GHz. The path
was neither idle, downclocked, memory-pressure limited, nor thermally throttled.

The global fit was `T(N)=39.568+1.835720N+5.96180e-5N²` ms (`R²=0.999999991`).
The fixed term is 14.36% of predicted TTFT at 128 and 3.98% at 512. A fit to
GPU prefill alone has effectively zero fixed term: the old 32-row request
chunking turns per-submit setup into the linear coefficient.

## Baseline attribution and Matrix2 utilization

At 128 tokens, quantized matmuls consume about 82% of GPU prefill. RoPE and
logits consume 6.63% and 6.83%; fused attention is only 0.84%. At 512 the same
matmuls remain dominant. Attention grows to 6.9% at 2K, 22.1% at 8K, and
48.9% at 28K.

Every required prompt length is divisible by the Matrix2 row tile 32. The
3B output dimensions are divisible by N tile 32 and K dimensions
3072/4096/9216 are divisible by K tile 128. Useful Matrix2 MMA work therefore
equals executed MMA work at all nine official points: no row, output, or K
padding is executed. The retained 256-row command batch contains up to eight
complete Matrix2 row tiles; it does not change the numerical tile.

Two existing controlled alternatives close the routing question:

| Route | 128 | 512 | 2K | Decision |
|---|---:|---:|---:|---|
| Matrix2 baseline | 277.10 ms | 991.87 ms | 4,049.22 ms | reference |
| KHR cooperative fallback | 407.64 ms | 1,509.50 ms | 6,113.57 ms | reject at every point |
| vector fallback | 1,552.86 ms | 6,098.06 ms | — | reject, about 5.6--6.1x slower |

There is no short-length crossover away from Matrix2. For the final full-tile
points, a rough logical recurring-weight model falls from one seven-matmul plus
logits load per 32-row chunk to one seven-matmul load per 256-row batch and one
logits load per request. Estimated logical weight bytes fall by about 75% at
128 and 89% from 512 through 28K. Cache makes this a work model, not a DRAM
counter.

The unchanged attention Matrix2 pipeline has the qualified driver values 125
registers/thread, 38,272 bytes driver shared memory, two resident workgroups,
16 resident warps, and 33.3% modeled occupancy. The quantized matmul shaders
declare 12,288 bytes shared plus 8,192 bytes driver reservation. Nsight Compute
2024.1 attached to the Vulkan process but reported `No kernels were profiled`;
fresh matmul register counters are therefore unavailable and occupancy is not
fabricated. Timestamp, dispatch, barrier, and CPU recording measurements remain
direct.

The important dispatch timings and geometry are:

| Tokens | Attention avg | Q avg | Gate avg | Down avg | largest groups X,Y |
|---:|---:|---:|---:|---:|---:|
| 128 | 0.0443 ms | 0.6324 ms | 1.3734 ms | 1.4231 ms | attention 32,4; matmul 4,288 |
| 512 | 0.2174 ms | 1.2244 ms | 2.7372 ms | 2.8209 ms | attention 32,8; matmul 8,288 |
| 2K | 0.7826 ms | 1.2338 ms | 2.7559 ms | 2.8456 ms | attention 32,8; matmul 8,288 |
| 8K | 3.0273 ms | 1.2369 ms | 2.7651 ms | 2.8560 ms | attention 32,8; matmul 8,288 |

Both Matrix2 kernels launch 256 threads, eight subgroups per workgroup. The
synchronous backend exposes no CPU/GPU command overlap: fence wait tracks GPU
time, while recording, submission, and descriptor pushes are the separately
measured host costs.

At 28K the seven matmuls perform 169.467 useful TFLOP in 33.300 s, or
5.089 TFLOP/s. The 256-row traffic model contains about 199 GB repeated
quantized weights, 10,592 GB activation tile reads, and 44.7 GB output, for
about 325 GB/s logical traffic and 15.6 FLOP/byte. Against nominal 360 GB/s the
logical bandwidth roof is about 5.63 TFLOP/s; the configured dense tensor roof
is about 61.1 TFLOP/s. Cache reuse means this is not a DRAM counter, but it
places the retained kernel near its logical movement roof and far below dense
tensor peak: unpack/dequant, tensor loads, shared synchronization, and issue
remain part of the limit.

## Fresh fused-attention sub-attribution

A one-run cold 8K control measured 2,492.057 ms in attention. Temporary
single-change shaders retained downstream liveness:

| Bucket removed | Attention | Direct delta | Share | Interpretation |
|---|---:|---:|---:|---|
| none | 2,492.057 ms | — | 100% | control |
| Q/K tensor loads + QK MMA | 1,659.988 ms | 832.069 ms | 33.39% | direct |
| V tensor load + AV MMA/store | 1,495.159 ms | 996.898 ms | 40.00% | direct |
| max/exp/reduce replaced by a live clamp | 2,586.188 ms | +94.131 ms | no positive saving | replacement changed scheduling; not additive |
| setup, mask/store, online merge, synchronization, and softmax residual | — | at most 663.090 ms | at most 26.61% | conservative remainder |

The softmax replacement is an explicit negative result, not a claim that
softmax is free. It proves that this simple bypass is not an isolating faster
substitute. The direct QK and AV deltas are additive to 73.39%; the remainder
is the safe ceiling for all setup/softmax/control work combined.

Applying only those measured 8K shares to the separately timestamped fused
category gives an attribution model, clearly not separate counters:

| Tokens | Fused attention | QK model | AV model | residual ceiling |
|---:|---:|---:|---:|---:|
| 128 | 1.152 ms | 0.385 ms | 0.461 ms | 0.307 ms |
| 512 | 11.306 ms | 3.775 ms | 4.522 ms | 3.009 ms |
| 2K | 162.773 ms | 54.350 ms | 65.110 ms | 43.313 ms |
| 8K | 2,518.710 ms | 840.989 ms | 1,007.484 ms | 670.237 ms |

## Experiment registry

| ID | Isolated change | 128 / 512 TTFT | Decision |
|---|---|---:|---|
| TAIL | output norm/logits only on final request batch | 265.90 / 931.36 ms | keep; overwritten intermediate logits removed |
| Q4-SHARE | subgroup-sharing packed Q4 nibbles | 300.72 / 1,070.78 ms | reject; shuffle cost regressed 13--15% |
| Q6-PACK4 | four-byte packed Q6 low/high loads | 254.50 / 898.92 ms | keep; Q6 kernels about 21.7% faster |
| ROPE-BATCH | one Q and one K dispatch per scale bucket | 238.08 / 822.01 ms | keep; 128 recording time fell about 74% |
| ROW-64 | Vulkan request batch 64 | 213.42 / 723.65 ms | useful, superseded |
| ROW-128 | batch 128 | 208.09 / 697.24 ms | useful, superseded |
| ROW-256 | batch 256 | 207.73 / 681.94 ms; 2K 2,724.51 ms | keep |
| ROW-512 | batch 512 | 207.54 / 685.96 ms; 2K 2,736.11 ms | reject; 512/2K regress and capacity doubles |

The pre-edit Amdahl checks and observed local effects were:

| Experiment | Affected baseline share / local effect | Amdahl expectation | Observed |
|---|---|---:|---:|
| TAIL | logits 6.83% GPU at 128; all but last invocation removable | at most -5.1% GPU for four chunks | -4.0% TTFT, -4.9% GPU |
| Q6-PACK4 | Q6 V+down about 15.4% GPU at 128; about 1.28x local | about -3.4% GPU | -4.3% TTFT on top of TAIL |
| ROPE-BATCH | RoPE 6.63% GPU plus 6,656 row dispatches/request | at most -6.4% GPU, larger host ceiling | -6.5% TTFT; recording -74% |
| ROW-256 | repeated graph work and weight traffic across eight 32-row chunks | upper bound set by remaining linear coefficient | -5.8% TTFT at 512 vs ROW-64 |
| Q4-SHARE | Q4 matmuls dominate, but shuffles are added to every tile | positive only if saved loads exceed shuffle cost | +13--15% TTFT, falsified |

The row boundary is explicit: 255/256/257-token TTFT is
362.95/361.03/379.79 ms. The first two use one command batch; 257 uses two.
All complete successfully. Only pure Vulkan selects 256; Metal and both hybrid
placement budgets retain the established 32-row policy.

## Retained implementation

Four focused changes remain:

1. Intermediate prefill chunks extend KV only; final norm/logits execute once.
2. Q6 Matrix2 loads four adjacent packed bytes and reconstructs the exact Q6
   low/high fields without subgroup shuffles.
3. RoPE rotates packed Q/K rows in two dispatches per original-context bucket;
   the portable backend default preserves row-wise behavior.
4. Pure Vulkan owns a measured 256-row request capacity. It is eight complete
   Matrix2 row tiles and does not alter Metal or hybrid sizing.

The invariants are unchanged causal mask, GQA mapping, F16/FP32 precision,
original-context YaRN query scaling, KV layout, final logits, fallback routes,
and decode kernels. No public API, dependency, weight format, or KV format
changed.

## Final stable matrix

These observed distributions are the final retained tree. The GPU column comes
from a separate same-protocol profiler pass.

| Tokens | mean | median | p10 | p90 | sd | CV | tok/s | delta TTFT | GPU prefill |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | 96.49 | 96.507 | 95.880 | 97.066 | 0.64 | 0.67% | 331.65 | -5.63% | 50.179 ms |
| 64 | 131.29 | 131.574 | 129.983 | 132.171 | 0.92 | 0.70% | 487.50 | -18.06% | 85.369 ms |
| 128 | 206.00 | 205.960 | 205.431 | 206.538 | 0.67 | 0.32% | 621.36 | -25.66% | 160.718 ms |
| 256 | 358.23 | 358.507 | 357.296 | 359.030 | 0.89 | 0.25% | 714.63 | -30.17% | 314.812 ms |
| 512 | 678.61 | 678.285 | 676.493 | 680.631 | 1.97 | 0.29% | 754.49 | -31.58% | 630.996 ms |
| 1K | 1,333.80 | 1,333.896 | 1,330.013 | 1,337.409 | 3.64 | 0.27% | 767.73 | -32.50% | 1,285.854 ms |
| 2K | 2,705.47 | 2,704.993 | 2,697.856 | 2,713.180 | 7.54 | 0.28% | 756.99 | -33.19% | 2,652.847 ms |
| 8K | 12,644.82 | 12,648.441 | 12,616.775 | 12,671.404 | 34.29 | 0.27% | 647.86 | -33.73% | 12,494.209 ms |
| 28K | 63,923.65 | 63,882.810 | 63,874.696 | 63,988.940 | 79.68 | 0.12% | 438.02 | -34.89% | 63,755.048 ms |

The final fit is `T(N)=44.088+1.229654N+3.75634e-5N²` ms
(`R²=0.999999823`). The linear coefficient falls 33.0% and the quadratic
coefficient 37.0% relative to baseline: larger dispatches improve both repeated
matmul work and attention scheduling across query tiles. The primary goal is
exceeded at 128 by 10.66 percentage points.

At 128, commands/dispatches/barriers per request fall from approximately
4/8,248/4,608 to 1/546/442; CPU recording falls from 7.775 to 0.662 ms. At
512, command batches fall 16 to 2 and dispatches 32,992 to 1,346. At 8K they
fall 256 to 32 and 527,872 to 21,506. Vulkan profiler accounting is at least
99.92% at 128 and 99.95% from 512 through 2K; 8K/28K are 99.96/99.97%.

Pipeline construction and model loading occur before the timed public request.
A cold 128-token request measured 222.90 ms versus 209.20 ms after one warm-up:
13.70 ms or 6.5% first-execution overhead. No production warm-up or persistent
allocation policy was added.

## Correctness and regression gates

- Q4 Matrix2 CPU oracle: max/mean relative error
  `4.934e-4 / 1.943e-4` on the real wide shape.
- Q6 Matrix2 CPU oracle: max/mean absolute `0.007330 / 0.001676`, max/mean
  relative `0.002245 / 0.000412`.
- F16 fused attention versus sequential decode through 28K: max attention
  absolute `1.526e-5`; projected-logit max absolute `3.601e-6`.
- F16 fused attention versus CPU through 28K: max attention absolute
  `1.180e-5`; projected-logit max absolute `4.001e-6`.
- INT8 attention and projected logits match the sequential Vulkan reference
  exactly at rapid and 2K/8K/28K qualification points.
- Batched YaRN Q/K matches row-wise CPU across the 127/128 scale boundary.
- Batched final logits match sequential execution for F16 and INT8 KV; the
  synthetic dense/quantized profiles report max error zero.
- Full-model outputs are finite at every required prompt length.

Decode controls remain within 2%: 128 is 44.75 versus 44.96 tok/s baseline,
2K is 41.95 versus 42.28, and 28K is 27.81 versus 28.04. The alternative
observed-distribution pass measured 27.73 tok/s at 28K with 0.21% CV. The
8K/28K TTFT gains are 33.7/34.9%, so long-context behavior is improved rather
than traded away.

Standalone Vulkan tests pass 145 with four intentional ignores. The
Vulkan-hybrid library suite passes 213 with three ignores; the focused ignored
long-context numeric qualification also passes. The family-agnostic
`docs_contract` retains its known baseline failure because the intentionally
gitignored local `VALIDATION.md` is absent. Formatting and `git diff --check`
pass.

## Final state and remaining bottleneck

All rejected shader probes and benchmark sample logging are removed. The
retained tree has no benchmark numbers in backend code, no new dependency, and
no decode-path modification. Complexity grows only at the backend contract
needed to batch RoPE; its default is direct and portable, and the Vulkan
override contains scale-bucket splitting locally.

At 128/512 the remaining GPU bottleneck is the seven Matrix2 matmuls, about
94.27/94.94% of prefill. At 28K attention is 46.30%, while the seven matmuls
sum to 52.23%; attention's direct AV/V bucket is nevertheless the largest
measured internal sub-operation. Further short-context optimization should
first reduce quantized matmul dequant/tensor-load issue without increasing row
capacity, because the
512-row sweep already crossed the measured optimum.

# Vulkan Short-Context Prefill Phase 6

## Baseline and protocol

Phase 6 ran locally on `perf/vulkan-short-prefill-phase6`; nothing was pushed.
The executable baseline is clean commit `c4898e34ff0d6873b14f5865d05e59a51317f93b`,
the current tip of `perf/vulkan-short-prefill-phase5`. It contains retained
implementation `d541d6e`, evidence report `25366dd`, and one behavior-preserving
Q6 source simplification. The retained Phase 6 implementation is `af87c80`.

The hardware, driver, artifact, context, KV format, greedy policy, and exact
synthetic prompts are unchanged from Phase 5. The artifact was reauthenticated
at size 2,147,023,008 and SHA-256
`9ed150d4367e68df0ac8e1540f6ddc65b42d0ee26378329d1ecbca60f93fc5f8`.
Each prompt is `" a"` repeated `tokens - 3`, producing the exact reported token
count. Runs through 512 use two warm-ups and fifteen samples; 1K/2K use one and
seven; 8K/28K use one and three. TTFT distributions exclude warm-ups; GPU
profiler totals include them. Medians come from temporary per-sample logging
removed from the retained tree.

The inherited Phase 5 reference remains 206.00 ms at 128. Its gains were
31.58% at 512, 33.19% at 2K, 33.73% at 8K, and 34.89% at 28K. The fresh Phase 6
baseline is tuple-compatible and 0.22% faster at 128, well within normal run
variation.

| Tokens | TTFT mean | Median | SD | CV | GPU prefill | CPU wall prefill | Prompt tok/s |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | 130.95 ms | 130.828 | 0.69 | 0.52% | 85.446 ms | 88.877 ms | 488.76 |
| 128 | 205.55 ms | 205.405 | 0.69 | 0.33% | 160.136 ms | 163.734 ms | 622.73 |
| 256 | 359.62 ms | 359.584 | 1.18 | 0.33% | 313.298 ms | 317.527 ms | 711.87 |
| 512 | 677.46 ms | 677.425 | 2.19 | 0.32% | 629.908 ms | 635.376 ms | 755.77 |
| 1K | 1,325.47 ms | 1,325.719 | 2.42 | 0.18% | 1,275.800 ms | 1,283.573 ms | 772.56 |
| 2K | 2,700.93 ms | 2,698.943 | 7.53 | 0.28% | 2,642.828 ms | 2,658.746 ms | 758.26 |
| 8K | 12,562.09 ms | 12,570.392 | 35.20 | 0.28% | 12,440.955 ms | 12,519.148 ms | 652.12 |
| 28K | 64,013.51 ms | 64,012.096 | 48.92 | 0.08% | 63,642.122 ms | 63,969.462 ms | 437.41 |

Telemetry includes only 100 ms samples with GPU utilization at least 80%:

| Tokens | GPU util | VRAM | Graphics clock | Memory clock | Power | Temperature |
|---:|---:|---:|---:|---:|---:|---:|
| 64 | 98.4% | 5,176 MiB | 1,997 MHz (1,987--2,010) | 7,501 MHz | 113.3 W (87.4--126.7) | 53.4 C (53--54) |
| 128 | 98.0% | 4,304 MiB | 1,995 MHz (1,980--2,010) | 7,501 MHz | 135.3 W (117.5--140.5) | 54.8 C (49--57) |
| 256 | 96.6% | 4,842 MiB | 1,982 MHz (1,972--2,010) | 7,501 MHz | 139.2 W (63.4--156.7) | 57.4 C (51--60) |
| 512 | 99.4% | 5,567 MiB | 1,970 MHz (1,957--1,987) | 7,501 MHz | 157.6 W (104.4--164.7) | 61.9 C (59--64) |
| 1K | 98.6% | 5,612 MiB | 1,971 MHz (1,942--1,987) | 7,501 MHz | 158.9 W (83.8--170.5) | 60.4 C (57--63) |
| 2K | 99.5% | 5,499 MiB | 1,954 MHz (1,897--2,002) | 7,501 MHz | 164.5 W (90.2--170.1) | 65.4 C (57--68) |
| 8K | 99.4% | 5,696 MiB | 1,947 MHz (1,807--1,980) | 7,501 MHz | 167.5 W (68.6--170.2) | 68.6 C (59--72) |
| 28K | 99.7% | 5,716 MiB | 1,938 MHz (1,912--1,980) | 7,501 MHz | 169.0 W (70.4--170.1) | 73.2 C (62--75) |

There is no short-prompt downclock, thermal throttle, or memory-pressure regime.

## Complete 128 ms budget and critical path

Environment-gated timing separated template rendering, session construction,
prefill wall time, device sampling, and first public delta. Feature-gated Vulkan
timestamps then subdivided GPU prefill. The temporary request logger is absent
from `af87c80`.

| Critical-path component | Baseline ms | Execution and dependency |
|---|---:|---|
| CPU template/token preparation | 0.014 | CPU, serialized before session creation |
| request KV session construction | 40.957 | CPU/driver, serialized; two device allocations |
| embedding | 1.620 | GPU, first command operation |
| attention RMSNorm | 0.226 | GPU, 26 serial layer instances |
| Q projection | 16.392 | GPU Matrix2, serial by layer |
| K projection | 4.474 | GPU Matrix2; Q/K/V are mutually independent after norm |
| V projection | 4.592 | GPU Matrix2; overlaps neither in the synchronous command |
| RoPE Q + K | 0.374 | GPU, two batched dispatches per layer |
| attention QK model | 0.384 | GPU; Phase 5 measured-share model, not a new counter |
| attention AV model | 0.460 | GPU; Phase 5 measured-share model |
| attention softmax/setup ceiling | 0.307 | GPU conservative residual of the fused kernel |
| attention output projection | 17.595 | GPU Matrix2 |
| attention residual | 0.302 | GPU |
| MLP RMSNorm | 0.226 | GPU |
| gate projection | 35.603 | GPU Matrix2 |
| up projection | 35.721 | GPU Matrix2 |
| SiLU multiply | 0.981 | GPU |
| down projection | 36.510 | GPU Matrix2 |
| MLP residual | 0.302 | GPU |
| KV copy/write | 0.138 | GPU; no host upload or map/unmap |
| final norm | 0.009 | GPU, final row only |
| logits | 3.783 | GPU, final batch only |
| timestamp residual | 0.138 | GPU command span not assigned to a kernel |
| prefill host-only remainder | 3.598 | scratch allocation/free, recording, submission and wait residual |
| argmax/readback/first-token preparation | 0.843 | GPU reduction + copy + CPU wait/decode |
| **TTFT** | **205.548** | **100% attributed** |

The profiler directly measured 160.136 ms GPU and 163.734 ms wall prefill.
Recording was 0.667 ms, queue submission 0.049 ms, and descriptor work
0.058 ms (descriptor time is a subset of recording). One command contained
546 dispatches and 442 barriers. No inter-kernel gap exceeded 50 microseconds;
there was one queue submission and no CPU/GPU command overlap. The GPU sequence
is therefore one continuously occupied command:

```text
CPU render -> allocate KV -> allocate batch scratch -> record
-> embedding -> 26 serial transformer layers -> final norm -> logits
-> submit/fence completion -> argmax command -> four-byte readback -> first delta
```

The post-prefill tail, defined from the end of layer 25 to the first token, is
final norm 0.009 + logits 3.783 + sampling/readback 0.843 = **4.635 ms**, or
2.25% of baseline TTFT. The sampling command itself averages 0.083 ms GPU,
0.015 ms recording, 0.009 ms submission, and 0.209 ms fence wait; host mapping,
token decode, and callback delivery form the remaining wall time. This tail is
too small to be the first target.

## Per-layer budget and variance

Fresh 128 TTFT is 7.906 ms per layer; final TTFT is 6.303 ms per layer. Direct
GPU layer timestamps average 5.992 ms with 0.156 ms standard deviation and
2.60% cross-layer CV:

| Per-layer class | Mean ms |
|---|---:|
| attention block (Q/K/V, RoPE, KV, attention, O) | 1.718 |
| MLP block (gate, up, activation, down) | 4.194 |
| norms and residuals | 0.042 |
| timestamp/control residual | about 0.038 |

The measured layer range is 5.785--6.354 ms. Layers 14 and 20 are 6.0% and
5.6% above the mean; layer 21 is 3.5% below it. The first and last layers are
only +1.1% and +2.7%. There is no first/last/batch-boundary hotspot and no layer
large enough to justify routing.

## Sub-256 and Matrix2 utilization audit

Phase 5's 256-row capacity was used as acquired. No capacity sweep was repeated.
The Matrix2 geometry is `M=32, N=32, K=128`, one output tile per 256-thread
workgroup, eight 32-lane subgroups. All model K/N dimensions and all official
short rows are exact multiples of their tiles.

| Prompt | Logical rows | Command batches | Physical rows | M tiles/batch | Partial tiles | Active lanes | Useful MMA |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 128 | 1 | 128 | 4 | 0 | 256/256 | 100% |
| 256 | 256 | 1 | 256 | 8 | 0 | 256/256 | 100% |
| 512 | 512 | 2 | 512 | 8 + 8 | 0 | 256/256 | 100% |
| 2K | 2,048 | 8 | 2,048 | 8 each | 0 | 256/256 | 100% |

At 128, Q launches `4 x 128` workgroups and gate/up `4 x 288`; at 256 the
row dimension doubles to eight. Each Q/gate workgroup executes 24 complete K
tiles and each down workgroup 72. There are no wasted MMA instructions, padded
rows, partial Matrix2 tiles, or idle software lanes. A tiny/short Matrix2 route
therefore has no utilization loss to remove. The inherited KHR cooperative and
vector alternatives remain 47% and roughly 6x slower at 128 and were not rerun.

Whole-kernel timestamp averages confirm that 128 is useful half-batch work,
not a launch-bound or padded special case:

| Prompt | Q projection | Gate projection | Down projection | Interpretation |
|---:|---:|---:|---:|---|
| 128 | 0.630 ms | 1.372 ms | 1.406 ms | four useful M tiles |
| 256 | 1.221 ms | 2.709 ms | 2.801 ms | eight useful M tiles |
| 512 | 1.226 ms | 2.729 ms | 2.820 ms | two eight-tile batches |
| 2K | 1.229 ms | 2.746 ms | 2.837 ms | eight eight-tile batches |

The 128 kernels take 50--52% of a full 256-row dispatch, while full-batch
times remain stable through 2K. This rules out a fixed launch or inactive-lane
cost large enough to motivate a sub-256 shader.

The kernels declare 12,288 bytes shared plus the driver's 8,192-byte reservation.
Fresh register, spill, resident-workgroup, and occupancy counters remain
unavailable: Nsight Compute attaches to the Vulkan process but reports no
kernels. No occupancy value is fabricated. Nsight Systems captured a QDSTRM,
but this installation lacks its importer; timestamp gap accounting is direct
and covers 99.92% of GPU time.

## Quantized matmul traffic and residual limit

Q4_K stores 256 values in 144 bytes: 128 quant bytes and 16 bytes metadata.
Q6_K stores them in 210 bytes: 192 quant bytes and 18 bytes metadata. Stored
weights have no shape padding. Four 32-row M tiles independently consume each
weight at 128, so shader-visible unique weight bytes are four times the stored
minimum; cache can satisfy the repeated copies and no DRAM counter is claimed.

| Representative matmul | M/N/K | Stored weight (quant + metadata) | Executed weight bytes | Tensor activation loads | Output | GPU ms | FLOP/s | Logical BW |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Q4 Q | 128/4096/3072 | 7.078 MB (6.291 + 0.786) | 28.312 MB | 100.663 MB | 1.049 MB | 0.630 | 5.11 TF/s | 206 GB/s |
| Q4 gate/up | 128/9216/3072 | 15.925 MB (14.156 + 1.769) | 63.701 MB | 226.492 MB | 2.359 MB | 1.372--1.378 | 5.26--5.28 TF/s | 212--213 GB/s |
| Q6 down | 128/3072/9216 | 23.224 MB (21.234 + 1.991) | 92.897 MB | 226.492 MB | 0.786 MB | 1.409 | 5.14 TF/s | 227 GB/s |

Minimum/visible effective weight bandwidth is 11.2/44.9 GB/s for Q4 Q,
11.6/46.4 GB/s for Q4 gate, and 16.5/65.9 GB/s for Q6 down. Total logical
traffic, rather than raw stored weights alone, gives the 187--227 GB/s rows
above. Across all 26 layers the recurrent matmuls perform 774.705 GFLOP in
about 151.02 ms, or 5.13 TFLOP/s.

For each 128x32 K tile, Q4 issues 4,096 quant-byte extractions plus metadata
loads, about 4,864 32-bit load operands before cache coalescing; each weight is
converted to float and receives scale/multiply/subtract. Packed Q6 issues 11
32-bit loads per lane for four-byte-aligned blocks and 19 for two-byte-aligned
blocks, averaging 15 x 256 = 3,840 load operands per tile. It reconstructs each
of 4,096 values from four low and two high bits, converts it, and multiplies by
the scale. This is the already-retained packed Q6 route; it was not rewritten.

Vulkan timestamps delimit the complete shader, not its load, unpack/dequant,
MMA, and epilogue instruction regions. Nsight Compute exposed no Vulkan kernel
or instruction counters on this installation, so independent elapsed times for
those regions are unavailable and are not fabricated. The byte counts,
instruction inventory, whole-kernel time, effective bandwidth, and FLOP rate
above are the measurable attribution. A liveness-changing shader ablation would
not provide additive stage times; it was unnecessary after the measured 41 ms
CPU allocation target exceeded the excellent TTFT goal.

The stored-weight minimum over all seven layer matmuls is 1.808 GB/request;
final logits add 0.330 GB. At a realistic 300 GB/s, raw minimum weight transfer
is only 7.13 ms. Recurrent shader-visible traffic is instead 7.231 GB weights,
24.210 GB tensor activation loads, and 0.204 GB output, 31.645 GB total. The
observed limit is thus not raw DRAM weight bandwidth or Matrix2 padding. It is
the combined tensor-load, unpack/dequant, synchronization, cache, and issue
roof already identified in Phase 5.

## Intermediate traffic, shared inputs, and fusion screen

Logical buffer traffic below counts whole-tensor writes and consumer reads; it
does not pretend cache hits are DRAM transactions.

| Tensor | Shape/type at 128 | Size | Writes/layer | Reads/layer | Approx. traffic/layer |
|---|---|---:|---:|---:|---:|
| residual `x` | 128x3072 F32 | 1.50 MiB | 2 | norms + 2 residuals | 9.00 MiB |
| normalized | 128x3072 F16 | 0.75 MiB | 2 | Q/K/V + gate/up | 5.25 MiB |
| Q | 128x4096 F16 | 1.00 MiB | projection + RoPE | RoPE + attention | 4.00 MiB |
| K | 128x1024 F16 | 0.25 MiB | projection + RoPE | RoPE + KV write | 1.00 MiB |
| V | 128x1024 F16 | 0.25 MiB | 1 | KV write | 0.50 MiB |
| attention | 128x4096 F16 | 1.00 MiB | 1 | O projection | 2.00 MiB |
| O projection | 128x3072 F16 | 0.75 MiB | 1 | residual | 1.50 MiB |
| gate / up / activation | each 128x9216 F16 | 2.25 MiB | 1 each | 1 each | 4.50 MiB each |
| down output | 128x3072 F16 | 0.75 MiB | 1 | residual | 1.50 MiB |

The inventory is about 38.25 MiB/layer or 994.5 MiB/request. Q/K/V read the
0.75 MiB normalized activation three times. Perfect cross-projection reuse
could remove at most 39 MiB/request (about 0.13 ms at 300 GB/s), while distinct
weights remain. Gate/up could remove at most 19.5 MiB (about 0.07 ms). Both
inputs fit cache and neither bound supports a fused multi-weight matmul.

| Producer -> consumer | Intermediate round trip | Dispatch/barrier affected | Optimistic removable time | Decision |
|---|---:|---:|---:|---|
| RMSNorm -> Q/K/V | normalized, at most 39 MiB duplicate reads/request | 26 / 26 | under 0.4 ms | reject by bound |
| Q/K/V -> RoPE | Q+K, 1.25 MiB/layer | 52 / 52 | RoPE total 0.374 ms | reject by bound |
| RoPE -> attention | Q+K, 1.25 MiB/layer | 26 / 26 | under 1.53 ms including attention | reject by bound |
| attention -> O | 1.00 MiB/layer | 26 / 26 | about 0.1 ms traffic | reject by bound |
| residual -> RMSNorm | F32 `x` | 52 / 52 | residual+all norms 1.07 ms | reject by bound |
| gate/up -> activation | 4.50 MiB written/layer | 26 / 26 | under 1.8 ms | reject; distinct weights remain |
| activation -> down | 2.25 MiB/layer | 26 / 26 | under 0.5 ms | reject by bound |
| final norm -> logits | 6 KiB final row | 1 / 1 | 0.009 ms plus negligible traffic | reject by bound |

No local fusion reaches the mandatory 4% / 6.55 ms redesign threshold. A
monolithic layer would add register/shared pressure without an economic bound
and was not attempted.

## Retained experiment

The resource audit found 13 device allocations per hot request: two 1.745 GB
KV buffers and eleven 26 MiB aggregate batch buffers. After the profiler's first
request latch, baseline allocation was 3,516,923,904 bytes and 42.46 ms/request.
The KV pair alone is exactly 3,489,660,928 bytes. Phase 5's memory preflight
already charges one full KV pair, but the ordinary request path freed and
reallocated it each time.

`P6-KV-PERSIST` reuses the existing single Vulkan cache slot for both ordinary
and explicitly prefix-keyed requests. Ordinary requests always prefill from
position zero and retain no prefix metadata. Stale bytes beyond the current
position are unreachable under causal attention. Keyed calls preserve the old
exact-key/exact-token prefix rule. Cancellation or any error invalidates and
frees the slot; model destruction frees a retained slot. Holding the existing
mutex across generation also matches the server's one-generation-at-a-time
invariant. Exactly one KV pair remains resident, so preflight and peak VRAM do
not change.

After the change, hot allocation is only the eleven batch buffers:
27,262,976 bytes and 1.26 ms/request. No shader, dispatch, barrier, precision,
weight/KV format, public API, or dependency changed.

Pre-edit Amdahl: the affected fraction was 40.957/205.548 = 19.93%; complete
local removal predicted 164.59 ms. Observed 163.87 ms agrees within run noise.

| ID | Target / architecture | 128 affected or upper bound | 128 result | 512 result | Long/decode | Decision |
|---|---|---:|---:|---:|---|---|
| P6-KV-PERSIST | retain one already-budgeted KV allocation | 40.96 ms | -20.28% | -6.01% | protected | keep |
| P6-SUB256 | smaller Matrix2 geometry | 0 padded MMA | not built | not built | prior alternatives slower | reject by utilization |
| P6-ROPE | further batched RoPE work | 0.374 ms | not built | not built | n/a | close: 0.23% TTFT |
| P6-CONTROL | persistence/dispatch amortization | under 0.8 ms CPU; zero 50 us gaps | not built | not built | n/a | close: under 1% |
| P6-FUSION | local producer-consumer fusion set | each under 1.8 ms | not built | not built | complexity disproportionate | reject by Amdahl |

No Phase 5 falsified shader probe was repeated.

## Final performance and regression gates

| Tokens | Baseline TTFT | Final TTFT | Delta | Delta | Final median | Final SD/CV | Baseline/final GPU | Final wall | Final tok/s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | 130.95 | 89.03 | -41.92 ms | -32.01% | 89.027 | 0.59 / 0.67% | 85.446 / 85.396 | 88.140 | 718.91 |
| 128 | 205.55 | **163.87** | **-41.68 ms** | **-20.28%** | 164.003 | 0.70 / 0.43% | 160.136 / 160.140 | 163.007 | 781.13 |
| 256 | 359.62 | 317.67 | -41.95 ms | -11.67% | 317.277 | 1.51 / 0.47% | 313.298 / 313.487 | 316.749 | 805.90 |
| 512 | 677.46 | 636.73 | -40.73 ms | -6.01% | 635.906 | 2.68 / 0.42% | 629.908 / 631.307 | 635.864 | 804.12 |
| 1K | 1,325.47 | 1,285.90 | -39.57 ms | -2.99% | 1,286.059 | 2.30 / 0.18% | 1,275.800 / 1,277.140 | 1,284.962 | 796.33 |
| 2K | 2,700.93 | 2,662.77 | -38.16 ms | -1.41% | 2,663.182 | 3.35 / 0.13% | 2,642.828 / 2,646.890 | 2,661.799 | 769.12 |
| 8K | 12,562.09 | 12,553.27 | -8.82 ms | -0.07% | 12,561.790 | 38.27 / 0.30% | 12,440.955 / 12,472.604 | 12,551.943 | 652.58 |
| 28K | 64,013.51 | 64,063.93 | +50.42 ms | +0.08% | 64,043.507 | 94.80 / 0.15% | 63,642.122 / 63,700.380 | 64,061.528 | 437.06 |

The 8K/28K GPU differences are +0.25%/+0.09%, explaining why the constant
41 ms host removal is hidden by run drift at long lengths. The 28K TTFT change
is below both distributions' noise scale and is not a material regression.

Decode controls from the same protocol are 44.33 -> 44.29 tok/s at 128
(-0.09%), 41.86 -> 41.60 at 2K (-0.62%), and 27.79 -> 27.75 at 28K
(-0.14%). All remain below 1% and within noise.

Correctness and build evidence:

- repeated fresh/reused F16 and INT8 full-model runs emitted the identical
  four-token sequence `2757,10637,2479,1636`;
- INT8 attention and projected logits remain exact against sequential Vulkan
  through 28K;
- F16 attention max absolute error through 28K is `1.526e-5`; projected-logit
  maximum is `3.601e-6`;
- Q4 Matrix2 CPU oracle max/mean relative is
  `4.934e-4 / 1.943e-4`; Q6 max/mean absolute is
  `0.007330 / 0.001676`, max/mean relative
  `0.002245 / 0.000412`;
- batched final logits match sequential execution exactly for F16 and INT8;
- Vulkan-profile library tests pass 152 with four intentional ignores; the
  focused ignored 28K numeric qualification passes;
- release Clippy with all targets and warnings denied, formatting, and
  `git diff --check` pass;
- `docs_contract` retains its known baseline failure because gitignored local
  `VALIDATION.md` is absent.

## Final critical path and physical floor

At 128, retained session lookup replaces the 40.957 ms allocation stage. The
final measured timeline is:

| Final critical-path stage | ms | TTFT share |
|---|---:|---:|
| render + cache-slot retrieval | about 0.013 | 0.01% |
| scratch allocation, record, submit, GPU prefill, cleanup | 163.007 | 99.47% |
| first sampling/readback/delta | 0.850 | 0.52% |
| **TTFT** | **163.87** | **100%** |

The seven recurrent Matrix2 matmuls remain about 151.02 ms, 94.3% of GPU
prefill and 92.2% of final TTFT. RoPE is 0.376 ms (0.23% TTFT), attention
1.189 ms (0.73%), recording 0.611 ms, submission 0.041 ms, and hot scratch
allocation about 1.26 ms. No remaining non-matmul cost has a 4% TTFT bound.

The 128-specific physical floor is a constraint maximum, not a sum of
overlapping traffic:

- raw minimum weights: 2.138 GB, about 7.13 ms at 300 GB/s;
- Matrix2 activation-tile loads: 24.210 GB, about 74.5 ms at 325 GB/s;
- all recurrent logical movement: 31.645 GB, about 97.4 ms at 325 GB/s;
- measured dequant/tensor-load issue roof: 5.63 TFLOP/s from Phase 5, placing
  774.705 GFLOP at a 137.60 ms recurrent-matmul floor;
- current serialized non-recurrent GPU work: about 9.12 ms;
- current control/allocation/first-token wall outside GPU: about 3.73 ms.

The optimistic TTFT floor is therefore about **150.5 ms**. Current/floor is
1.089, leaving about 13.4 ms or 8.2% theoretical headroom. Reaching it requires
a new Matrix2 architecture, not a tail, dispatch, RoPE, attention, or fusion
cleanup.

The next architectural candidate would let one workgroup own multiple
independent M tiles so it can reuse a dequantized weight tile and amortize its
barriers. It must prove that the larger accumulator live set does not reduce
resident workgroups or introduce spills. Because the excellent <=170 ms target
is already exceeded, current TTFT is within 9% of the derived floor, and all
smaller directions are below the 4% gate, Phase 6 stops here rather than adding
a high-risk kernel for the last single-digit theoretical margin.
