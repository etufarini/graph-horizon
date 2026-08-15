<!--
This report owns the reproducible Phase 5 through Phase 8 Vulkan short-context prefill
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

The same per-request trace gives the remaining baseline distributions; means
are the values in the table above:

| Tokens | CPU wall median / SD / CV | Prompt tok/s median / SD / CV |
|---:|---:|---:|
| 64 | 89.062 ms / 0.688 / 0.77% | 489.192 / 2.557 / 0.52% |
| 128 | 163.542 ms / 0.649 / 0.40% | 623.161 / 2.070 / 0.33% |
| 256 | 317.442 ms / 1.005 / 0.32% | 711.934 / 2.332 / 0.33% |
| 512 | 635.938 ms / 2.175 / 0.34% | 755.803 / 2.439 / 0.32% |
| 1K | 1,284.010 ms / 2.248 / 0.18% | 772.411 / 1.411 / 0.18% |
| 2K | 2,656.098 ms / 7.650 / 0.29% | 758.816 / 2.109 / 0.28% |
| 8K | 12,527.842 ms / 34.809 / 0.28% | 651.690 / 1.829 / 0.28% |
| 28K | 63,967.903 ms / 48.780 / 0.08% | 437.417 / 0.334 / 0.08% |

The retained patch changes no GPU command. Baseline/final aggregate GPU means
above and below differ by at most 0.25%. A removed per-command logger therefore
measured the shared GPU path once more, summing all 256-row command batches into
each request and excluding warm-ups:

| Tokens | GPU mean | Median | SD | CV |
|---:|---:|---:|---:|---:|
| 64 | 84.240 ms | 84.384 | 0.474 | 0.56% |
| 128 | 159.485 ms | 159.553 | 0.464 | 0.29% |
| 256 | 312.606 ms | 312.669 | 1.054 | 0.34% |
| 512 | 629.272 ms | 629.976 | 1.663 | 0.26% |
| 1K | 1,282.696 ms | 1,281.220 | 4.411 | 0.34% |
| 2K | 2,649.860 ms | 2,651.550 | 3.542 | 0.13% |
| 8K | 12,538.333 ms | 12,548.000 | 17.441 | 0.14% |
| 28K | 63,675.233 ms | 63,687.000 | 24.563 | 0.04% |

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
| prompt ID input/upload | 0.000 standalone | no device input buffer; 512 ID bytes ride embedding push constants |
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

The command structure is also fully counted. A 128 request records 128
embedding dispatches, 16 dispatches for each of 26 layers, and final norm plus
logits: 546 dispatches, 546 pipeline binds, and 546 push-descriptor updates.
Q/K/V, gate/up, and the two RoPE launches elide only dependency-safe barriers,
leaving 12 barriers/layer plus 128 embedding and two tail barriers, 442 total.
There is one command-buffer begin/end pair, one prefill submission, and one
prefill fence wait. The 128 token IDs are push constants; there is no prompt
staging allocation, host-to-device copy, or prompt map/unmap.

On the final tree, recording is 0.611 ms/request, descriptor preparation is
0.050 ms within it, submission is 0.041 ms, and recording amortizes to about
0.0235 ms/layer. Shape arithmetic and tensor-view bookkeeping are inside that
recording bound. The eleven hot scratch allocations cost about 1.26 ms; the KV
allocation was removed from the hot path. Pipelines, shaders, pipeline cache,
weights, and descriptor mechanism are created at model load, not per request.
Command persistence has a sub-1% upper bound and cannot cross transformer-layer
dependencies, so it is closed.

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

Small-shape work per workgroup is likewise fully useful:

| Representative WG | Useful outputs | Active / idle lanes | Logical bytes/WG | FLOP/WG |
|---|---:|---:|---:|---:|
| Q4 Q/gate/up, K=3072 | 1,024 | 256 / 0 | 253,952 | 6.291 MFLOP |
| Q4 O, K=4096 | 1,024 | 256 / 0 | 337,920 | 8.389 MFLOP |
| Q6 down, K=9216 | 1,024 | 256 / 0 | 833,792 | 18.874 MFLOP |

Bytes include one weight tile, one 32-row activation tile, and 2,048 output
bytes. Owning multiple independent M tiles per WG could reuse weights, but it
would not activate idle lanes—there are none—and would enlarge the accumulator
live set.

Temporary `VK_KHR_pipeline_executable_properties` capture on the unmodified
compiled shaders reports 64 registers/thread for Q4 and 63 for Q6, 20,992 bytes
shared/WG for both (12,288 shader-declared plus 8,704 driver), and zero stack.
The GA106 exposes 65,536 registers, 102,400 shared bytes, 1,536 threads, and 48
warps/SM. Registers and shared memory both limit either kernel to four resident
256-thread workgroups: 32 resident warps and **66.7% modeled occupancy**. The
driver exposes no dedicated spill counter; zero stack and no source local arrays
are recorded, but are not overstated as proof of zero spills. Its reported local
memory value is an invalid 64 GiB sentinel and is discarded.

Nsight Compute still attaches without reporting Vulkan kernels, and Nsight
Systems captured a QDSTRM that this installation cannot import. The Vulkan
pipeline statistics close the register/shared/residency fields; direct
timestamp gap accounting covers 99.92% of GPU time.

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

At the GLSL operation level, either format reconstructs 4,096 weights/tile:
4,096 integer-to-F32 conversions and 4,096 F32-to-F16 shared stores, plus 256
Q6 scale conversions or 256 Q4 scale/min conversions. Q4 extracts one nibble
per value after byte selection; Q6 combines four low and two high bits and
subtracts its zero point. These are logical source operations, not fabricated
post-compiler instruction counters.

A removed `VK_KHR_shader_clock` probe measured the workgroup critical stages
ten times. It used a complete 32-row tile, the representative Q4 gate
`K=3072,N=9216` shape, and Q6 down `K=9216,N=3072` shape. These are workgroup
stage times, not values to multiply by concurrent workgroups; the whole-dispatch
GPU times remain the direct timestamps above.

| Kernel/WG stage | Weight load + unpack/dequant + barrier | Activation/shared load + MMA + barrier | Epilogue | Stage share |
|---|---:|---:|---:|---:|
| Q4 gate, 24 K tiles | 59.699 us | 37.478 us | 1.126 us | 60.73% / 38.12% / 1.15% |
| Q6 down, 72 K tiles | 200.192 us | 98.611 us | 0.819 us | 66.81% / 32.91% / 0.27% |

Raw weight loads and unpack/dequant are instruction-interleaved before the
same synchronization point, so separating their elapsed time would change
liveness and scheduling; they are reported as the one physically measurable
stage. The probe establishes that epilogue is negligible and the residual is
primarily quantized load/unpack/dequant, then tensor/shared load plus MMA—not
fixed setup, padding, or low occupancy. The timing and executable-statistics
instrumentation was removed before qualification.

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

The complete Phase 6 registry makes non-applicable shader fields explicit:

| ID | Hypothesis / architecture / routing | WG, subgroup, tile, Matrix2 | Useful / padding | Registers, shared, occupancy | Traffic / effective BW |
|---|---|---|---|---|---|
| P6-KV-PERSIST | retain the one preflight-budgeted KV state; all serialized pure-Vulkan requests; no length threshold | unchanged; n/a to resource ownership | unchanged, 100% / 0% | unchanged; Q4 64/20,992 B/66.7%, Q6 63/20,992 B/66.7% | removes 3,489,660,928 allocation bytes/request; shader BW unchanged |
| P6-SUB256 | smaller short-row Matrix2 path; screened, not built | current 256-thread, 8-subgroup, 32x32x128 Matrix2 | 100% / 0% already | 64 or 63 regs, 20,992 B, 66.7% | no redundant row/weight traffic to remove |
| P6-ROPE | further RoPE specialization; screened, not built | current batched route unchanged | all requested rows / 0% | n/a: no candidate shader | 0.376 ms total ceiling |
| P6-CONTROL | persistent command structure; screened, not built | no shader/routing change | n/a | n/a | 0.611 ms recording, 0.041 ms submit, zero >50 us gaps |
| P6-FUSION | local producer-consumer fusion; screened, not built | no candidate after bounds | n/a | candidate pressure unknown; redesign gate not met | largest local bound 1.8 ms; QKV shared input only 39 MiB |

| ID | 128 affected / upper | 64 delta | 128 delta | 256 delta | 512 delta | 2K delta | 8K delta | 28K delta | Decode 128/2K/28K | Correctness | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| P6-KV-PERSIST | 40.957 ms | -32.01% | -20.28% | -11.67% | -6.01% | -1.41% | -0.07% | +0.08% | -0.09% / -0.62% / -0.14% | full F16/INT8/Q4/Q6/GQA qualification passes | **keep** |
| P6-SUB256 | 0 padded-MMA ms | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | existing route retained | reject: no underutilization |
| P6-ROPE | 0.376 ms | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | baseline retained | close: 0.23% TTFT |
| P6-CONTROL | under 0.8 ms | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | baseline retained | close: under 1% |
| P6-FUSION | at most 1.8 ms each | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | baseline retained | reject: below 4% gate |

Best candidate: `P6-KV-PERSIST` is resource persistence, not a shader. It has
no WG/subgroup/tile/Matrix2/register/shared/occupancy threshold of its own and
routes every ordinary pure-Vulkan request through the existing serialized
single-slot session cache. Prefix-keyed routing remains exact-key/exact-token;
ordinary routing always starts at position zero. It removes the measured
3.490 GB allocation event without changing model traffic or bandwidth.

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

Final non-TTFT distributions from the same samples are:

| Tokens | CPU wall median / SD / CV | Prompt tok/s median / SD / CV |
|---:|---:|---:|
| 64 | 88.199 ms / 0.608 / 0.69% | 718.884 / 4.803 / 0.67% |
| 128 | 163.143 ms / 0.725 / 0.45% | 780.473 / 3.350 / 0.43% |
| 256 | 316.381 ms / 1.519 / 0.48% | 806.867 / 3.820 / 0.47% |
| 512 | 635.115 ms / 2.704 / 0.43% | 805.150 / 3.378 / 0.42% |
| 1K | 1,285.124 ms / 2.296 / 0.18% | 796.231 / 1.422 / 0.18% |
| 2K | 2,662.289 ms / 3.333 / 0.13% | 769.005 / 0.966 / 0.13% |
| 8K | 12,560.502 ms / 38.168 / 0.30% | 652.136 / 1.991 / 0.31% |
| 28K | 64,041.058 ms / 94.763 / 0.15% | 437.203 / 0.647 / 0.15% |

After every temporary profiler and shader probe was removed, a clean release
rebuild repeated the primary point at 163.89 ms TTFT with 0.27% CV, confirming
the qualified binary contains only the retained production change.

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

The remaining removable-cost ranking is therefore:

| Rank | Component | Current ms | Realistic removable ms | Predicted 128 gain | Gate |
|---:|---|---:|---:|---:|---|
| 1 | recurrent Matrix2 issue gap to measured roof | 151.02 | about 13.42 | 8.2% | architectural multi-tile/WG candidate |
| 2 | largest local fusion screen | at most 1.80 | at most 1.80 | at most 1.1% | below redesign threshold |
| 3 | hot scratch allocation | 1.26 | at most 1.26 | at most 0.8% | cleanup only |
| 4 | recording + submission | 0.65 | less than 0.65 | less than 0.4% | close persistence direction |
| 5 | RoPE | 0.376 | at most 0.376 | at most 0.23% | close baseline direction |

The 128-specific roofline constraints are not summed when they describe the
same recurrent matmuls:

- raw minimum weights: 2.138 GB, about 7.13 ms at 300 GB/s;
- Matrix2 activation-tile loads: 24.210 GB, about 74.5 ms at 325 GB/s;
- all recurrent logical movement: 31.645 GB, about 97.4 ms at 325 GB/s;
- measured dequant/tensor-load issue roof: 5.63 TFLOP/s from Phase 5, placing
  774.705 GFLOP at a 137.60 ms recurrent-matmul floor;
- 128 attention operation-only compute/traffic lower bound: under 0.01 ms, but
  the measured practical fused-attention stage is 1.189 ms;
- remaining serialized non-recurrent GPU work excluding attention: 7.93 ms;
- control, scratch allocation, and first-token wall outside GPU: 3.73 ms.

Using the measured practical attention and fixed graph/tail costs gives a
current-architecture attainable floor of about **150.5 ms**: 137.60 recurrent
Matrix2 + 1.19 attention + 7.93 other GPU + 3.73 control/tail. Current/floor is
1.089, leaving about 13.4 ms or 8.2% theoretical headroom. Using the
operation-only attention bound would lower it by only about 1.18 ms and does
not change the decision. Reaching the useful margin requires a new Matrix2
architecture, not a tail, dispatch, RoPE, attention, or fusion cleanup.

The next architectural candidate would let one workgroup own multiple
independent M tiles so it can reuse a dequantized weight tile and amortize its
barriers. It must prove that the larger accumulator live set does not reduce
resident workgroups or introduce spills. Because the excellent <=170 ms target
is already exceeded, current TTFT is within 9% of the derived floor, and all
smaller directions are below the 4% gate, Phase 6 stops here rather than adding
a high-risk kernel for the last single-digit theoretical margin.

# Vulkan Short-Context Prefill Phase 7

## Baseline and protocol

Phase 7 ran locally on `perf/vulkan-short-prefill-phase7` from clean commit
`b8cc630`, the final evidence tip of `perf/vulkan-short-prefill-phase6`.
Inherited Phase 6 retained implementation is `af87c80`; its diagnostics-free
128-token reference is 163.89 ms. Nothing was pushed.

The hardware and exact tuple remained unchanged: RTX 3060 12 GiB device
`0x2487`, driver 595.84, Vulkan device API 1.4.329, Rust/Cargo 1.95, pure
Vulkan `vulkan-profile`, context 32,768, F16 KV, greedy sampling, and two
emitted tokens. The model remained the 2,147,023,008-byte
`Ministral-3-3B-Instruct-2512-Q4_K_M.gguf` with SHA-256
`9ed150d4367e68df0ac8e1540f6ddc65b42d0ee26378329d1ecbca60f93fc5f8`.
Each prompt is `" a"` repeated `tokens - 3`.

Runs through 512 used two warm-ups and fifteen measured repetitions; 1K/2K
used one plus seven; 8K/28K used one plus three. TTFT statistics exclude
warm-ups while profiler aggregates include them. Temporary per-sample,
allocation, command-timeline, shader-clock, and pipeline-executable probes
were removed before the final clean rebuild.

The fresh baseline is distinct from the inherited 163.89 ms reference: the
new 128 run is 0.18% slower and therefore tuple-compatible within run drift.

| Tokens | TTFT mean | Median | SD | CV | GPU prefill | CPU wall prefill | Prompt tok/s |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | 89.010 ms | 88.973 | 0.418 | 0.47% | 85.543 ms | 88.152 ms | 719.03 |
| 128 | 164.188 ms | 164.275 | 0.702 | 0.43% | 160.738 ms | 163.323 ms | 779.61 |
| 256 | 317.973 ms | 318.086 | 1.200 | 0.38% | 314.235 ms | 317.077 ms | 805.11 |
| 512 | 637.223 ms | 636.512 | 2.164 | 0.34% | 631.758 ms | 636.335 ms | 803.49 |
| 1K | 1,294.284 ms | 1,294.087 | 1.497 | 0.12% | 1,286.303 ms | 1,293.378 ms | 791.17 |
| 2K | 2,675.951 ms | 2,679.906 | 11.008 | 0.41% | 2,658.111 ms | 2,674.988 ms | 765.35 |
| 8K | 12,588.244 ms | 12,588.570 | 2.942 | 0.02% | 12,532.946 ms | 12,586.849 ms | 650.77 |
| 28K | 64,083.867 ms | 64,080.112 | 87.393 | 0.14% | 63,829.773 ms | 64,081.403 ms | 436.93 |

## Complete 128-token critical path

The baseline profiler accounts for 160.609 ms of named GPU work plus
0.129 ms timestamp residual, or all 160.738 ms of GPU prefill. Adding the
2.585 ms CPU-only portion of the prefill wall and the 0.865 ms request/tail
outside that wall accounts for the complete 164.188 ms TTFT. There are no
GPU gaps above 25 us in the dedicated timeline audit.

| Critical-path component | Baseline ms | TTFT share |
|---|---:|---:|
| CPU hot request setup | 0.013 | 0.01% |
| resource lifecycle, retained-KV request scratch allocation | 1.270 | 0.77% |
| command recording | 0.630 | 0.38% |
| queue submission | 0.044 | 0.03% |
| other prefill CPU/control outside GPU | 0.641 | 0.39% |
| embedding | 1.540 | 0.94% |
| attention RMSNorm, 26 layers | 0.234 | 0.14% |
| Q projection | 16.408 | 9.99% |
| K projection | 4.522 | 2.75% |
| V projection | 4.601 | 2.80% |
| RoPE | 0.377 | 0.23% |
| QK | 0.394 | 0.24% |
| attention setup/softmax | 0.314 | 0.19% |
| AV | 0.471 | 0.29% |
| O projection | 17.569 | 10.70% |
| MLP RMSNorm, 26 layers | 0.234 | 0.14% |
| gate projection | 35.802 | 21.81% |
| up projection | 35.886 | 21.86% |
| activation | 0.966 | 0.59% |
| down projection | 36.789 | 22.41% |
| KV writes and residual copies | 0.737 | 0.45% |
| final RMSNorm | 0.009 | 0.01% |
| logits | 3.757 | 2.29% |
| GPU timestamp residual | 0.129 | 0.08% |
| first readback, sampling, and delta tail | 0.852 | 0.52% |
| **TTFT** | **164.188** | **100.00%** |

The QK/AV split uses the acquired Phase 5 causal-attention shares; it is not a
new fabricated sub-timestamp. GPU useful work is 160.738 ms, measured GPU idle
is zero above 25 us, CPU-only prefill critical path is 2.585 ms, and the hot
setup plus post-prefill tail is 0.865 ms. Recording cannot overlap because all
546 dispatches are recorded before the one queue submission.

The final-output tail starts at final RMSNorm: 0.009 ms final norm, 3.757 ms
logits, and about 0.852 ms for synchronization/readback, argmax sampling, and
first delta. Decode preparation is below timer resolution. The serialized
total is about 4.618 ms, 2.81% of TTFT; only 0.852 ms is outside GPU prefill.

## Resource lifecycle and queue/control audit

One request owns eleven batch buffers totaling 27,262,976 bytes. Their measured
allocation is about 1.27 ms and destruction about 0.89 ms. Retaining all of
them would add 26 MiB persistent memory for at most 2.16 ms, only 1.3% TTFT,
so the resource-lifetime direction is closed. The 3,489,660,928-byte KV pair
remains retained by Phase 6 and has no hot allocation.

| Resource | Lifetime before/after Phase 7 | Setup ms | Memory cost | Result |
|---|---|---:|---:|---|
| KV K/V pair | retained model session / unchanged | approximately 0 | 3,489,660,928 B already budgeted | preserve Phase 6 |
| eleven batch buffers | one request / unchanged | 1.27 alloc + 0.89 free | 26 MiB if retained | reject: 1.3% ceiling |
| pipelines and pipeline cache | model / unchanged | zero hot creation | already resident | closed |
| decode/sampling/logits scratch | model / unchanged | zero hot creation | already resident | closed |
| command buffer and fence | one command / unchanged | about 0.003 alloc+begin | negligible | closed |
| descriptors | push descriptors / unchanged | 0.050 ms | no pool allocation | closed |
| prompt and sampling staging | push constants/persistent mirror / unchanged | negligible | no request staging buffer | closed |

The request has one prefill command buffer, one submission, one fence wait,
546 dispatches/pipeline binds/push-descriptor updates, and 442 barriers.
Recording is 0.630 ms, submission 0.044 ms, descriptor updates are a 0.050 ms
subset of recording, and the wait is 160.889 ms because it contains the GPU
work. There are no gaps above 25 us and no separate staging or transfer wait.

The per-layer baseline GPU distribution is mean 5.973 ms, median 5.977 ms,
minimum 5.884 ms, maximum 6.097 ms, standard deviation 0.048 ms, and 0.81% CV.
No first-layer, last-layer, or isolated fixed-cost anomaly exists. The exact
timeline is embedding, then 26 serial repetitions of norm, Q/K/V, Q/K RoPE,
KV write, attention, O, residual, MLP norm, gate/up, activation, down, residual,
then final norm and logits.

## Ranking, Amdahl bound, and experiment

The seven recurrent Matrix2 projections total 151.576 ms/request, 92.3% of
TTFT and 94.3% of GPU prefill. The Phase 6 stage probe showed quantized weight
load/unpack/dequant plus its barrier at 60--67% of representative workgroup
time. Giving a workgroup 64 rather than 32 prompt rows reuses each dequantized
128x32 weight tile across two old row tiles. Only a 10% local reduction was
needed to remove about 15 ms and reach the desired target.

| Rank | Component | Current ms | Realistic removable ms before experiment | Predicted TTFT gain |
|---:|---|---:|---:|---:|
| 1 | recurrent Matrix2 dequant/tile issue | 151.576 | at least 14.0 if reuse holds | at least 8.5% |
| 2 | batch-buffer allocation and free | 2.16 | at most 2.16 | at most 1.3% |
| 3 | largest local producer/consumer fusion | at most 1.80 | at most 1.80 | at most 1.1% |
| 4 | recording and submission | 0.674 | less than 0.674 | less than 0.4% |
| 5 | RoPE | 0.377 | at most 0.377 | at most 0.3% |

`P7-M64` changes only the established Q4_K and Q6_K Matrix2 prefill kernels:
one 256-thread workgroup, subgroup 32, tile 64x32x128, workgroup-scope Matrix2,
with `ceil(n/64)` dispatch. It uses the same all-length routing and guarded
tail. Fallbacks, formats, precision, KV, decode, barriers, public API, and
dependencies are unchanged.

Removed `VK_KHR_pipeline_executable_properties` instrumentation reported 98
registers/thread for Q4 and 108 for Q6, 33,792 bytes driver-reported shared
memory (16,384 bytes declared by the shader plus driver reservation), zero
stack, subgroup 32, and 256 threads. GA106 limits permit two resident
workgroups by registers, versus three by shared memory and six by threads:
16 resident warps, or 33.3% nominal occupancy. The former M32 kernel allowed
four workgroups, so both geometries keep 128 resident output rows; M64 halves
weight-tile dequantization without reducing resident row capacity.

At 128 there is no row padding. At 28K only the last 96-row batch pads 32 rows,
0.11% of the prompt. Recurrent shader-visible logical traffic changes from
7.231 to 3.616 GB of quantized-weight tile loads while activation-tile loads
remain 24.210 GB and output 0.204 GB: 28.030 GB total. This is a shader-visible
reuse model, not a DRAM performance-counter claim. The candidate executes the
774.705 GFLOP recurrent work in 106.491 ms, 7.275 TFLOP/s and about 263 GB/s
logical traffic. It introduces no GPU gaps above 25 us and changes no request
memory allocation.

| ID | Target | Architecture | 128 delta | 512 delta | Long delta | Decode delta | Memory delta | Decision |
|---|---|---|---:|---:|---|---|---:|---|
| P7-M64 | recurrent Matrix2 dequant reuse | 64x32x128 Matrix2/WG | -27.16% | -29.38% | 2K -29.67%, 8K -25.12%, 28K -17.06% | fresh 128 A/B +1.28%; path unchanged | 0 B | **keep** |
| P7-BATCH-PERSIST | request scratch lifetime | retain eleven buffers | bounded, not built | bounded, not built | n/a | n/a | +26 MiB | reject: 1.3% ceiling |
| P7-M128 | further row ownership | 128x32x128 Matrix2/WG | not built | not built | n/a | n/a | 0 B | stop: strong target already exceeded |

No Phase 5 or Phase 6 falsified experiment was repeated.

## Final performance and regression gates

| Tokens | Baseline TTFT | Final TTFT | Delta ms | Delta | Final median | Final SD/CV | Baseline/final GPU | Final wall | Final tok/s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | 89.010 | 68.872 | -20.138 | -22.63% | 68.555 | 0.964 / 1.40% | 85.543 / 65.289 | 68.009 | 929.42 |
| 128 | 164.188 | **119.586** | **-44.602** | **-27.16%** | 119.532 | 0.854 / 0.71% | 160.738 / 115.794 | 118.703 | 1,070.41 |
| 256 | 317.973 | 221.929 | -96.044 | -30.20% | 221.805 | 0.792 / 0.36% | 314.235 / 218.032 | 221.076 | 1,153.53 |
| 512 | 637.223 | 450.034 | -187.189 | -29.38% | 450.034 | 1.265 / 0.28% | 631.758 / 444.171 | 449.132 | 1,137.70 |
| 1K | 1,294.284 | 901.838 | -392.446 | -30.32% | 901.821 | 1.577 / 0.17% | 1,286.303 / 893.217 | 900.900 | 1,135.46 |
| 2K | 2,675.951 | 1,881.907 | -794.044 | -29.67% | 1,884.159 | 3.748 / 0.20% | 2,658.111 / 1,866.880 | 1,880.922 | 1,088.26 |
| 8K | 12,588.244 | 9,425.960 | -3,162.284 | -25.12% | 9,411.047 | 29.641 / 0.31% | 12,532.946 / 9,364.059 | 9,424.644 | 869.09 |
| 28K | 64,083.867 | 53,147.995 | -10,935.872 | -17.06% | 53,141.033 | 32.681 / 0.06% | 63,829.773 / 52,946.123 | 53,145.540 | 526.83 |

A diagnostics-free release rebuild repeated the primary point at 118.35 ms
with 0.44% CV over fifteen samples. Boundary runs at 63/64/65 and 255/256/257
all complete; the step after 64 or 256 is the expected additional 64-row
workgroup, not a correctness boundary.

Decode controls with 32 emitted tokens are 43.52 tok/s at 128, 43.07 at 2K,
and 27.77 at 28K. The stale inherited Phase 6 values are 44.33, 41.86, and
27.79; their 128 comparison is confounded by lower clocks in this run. A fresh
same-environment M32/M64 A/B is 42.97 -> 43.52 tok/s (+1.28%) at 128. The
source does not route decode through Matrix2; 2K improves and 28K changes
-0.07%. There is no evidenced decode regression.

Correctness and build evidence:

- focused Q4/Q6 batched Matrix2 CPU-oracle tests pass five of five, including
  the three-row M64 tail; Q4 max/mean relative error is
  `4.934e-4 / 1.943e-4`, and Q6 max/mean absolute is
  `0.007330 / 0.001676`, max/mean relative `0.002245 / 0.000412`;
- the focused full-model F16 greedy gate emits the inherited exact four-token
  sequence `2757,10637,2479,1636`; repeated four-token runs also complete under
  INT8 KV, exercising Q4, Q6, Matrix2, GQA, causal attention, final logits,
  retained request state, and multi-token decode;
- the focused ignored numeric qualification passes F16 and bit-exact INT8 at
  2K, 8K, and 28K; F16 attention maximum absolute error is `1.526e-5` and
  projected-logit maximum is `3.601e-6`;
- CPU/Vulkan dense greedy, intermediate tensor, RoPE, format coverage,
  prefill-versus-sequential decode, failure lifecycle, and resource-release
  tests pass in the Vulkan-profile library suite;
- formatting, release Clippy for all targets with warnings denied, and
  `git diff --check` pass;
- Vulkan-profile library tests pass 152 with four intentional ignores; the
  focused ignored 28K test passes separately;
- `docs_contract` retains its known baseline failure because the gitignored
  local `VALIDATION.md` is absent.

## Final critical path and physical floor

At 128 the M64 candidate changes only the dominant Matrix2 stage:

| Final critical-path stage | ms | Final TTFT share |
|---|---:|---:|
| request setup, resource/control, and first-token tail outside GPU | 3.792 | 3.17% |
| recurrent Matrix2 projections | 106.491 | 89.05% |
| attention | 1.220 | 1.02% |
| other serialized GPU work | 8.083 | 6.76% |
| **TTFT** | **119.586** | **100.00%** |

Matrix2 falls from 151.576 to 106.491 ms, a 45.085 ms or 29.74% local
reduction. Candidate GPU prefill is 115.794 ms, CPU wall prefill 118.703 ms,
and TTFT adds 0.883 ms. Its per-layer mean/median/minimum/maximum/standard
deviation are 4.242/4.260/4.082/4.366/0.092 ms, 2.17% CV. No >25 us GPU gap
appears at 128 or 512, and command, descriptor, submission, and barrier counts
are unchanged.

The floor bounds describe overlapping limits and must not be added blindly:

- minimum stored-weight transfer is 2.138 GB/request, 7.13 ms at 300 GB/s;
- recurrent logical movement after reuse is 28.030 GB, about 86.2 ms at
  325 GB/s, dominated by activation Matrix2 tensor loads;
- measured useful recurrent compute at the completed 256-row point is about
  7.64 TFLOP/s, placing 774.705 GFLOP at a 101.4 ms practical Matrix2 floor;
- practical attention is about 1.22 ms; its operation-only lower bound remains
  below 0.01 ms and is not attainable without changing the graph;
- remaining serialized non-Matrix2 GPU work is about 8.08 ms;
- minimum current control, resource, synchronization, and first-token tail is
  about 3.79 ms.

Using the practical compute roof gives a revised exact-representation floor of
about **114.5 ms**: 101.4 Matrix2 + 9.3 other GPU + 3.8 control/tail. The final
119.586 ms is 1.044x this floor, leaving about 5.1 ms or 4.3% practical
headroom. The earlier 150.5 ms Phase 6 floor was explicitly a floor for the
M32 architecture; M64 changes its weight-tile reuse and therefore legitimately
crosses it.

The largest remaining bottleneck is the 106.491 ms recurrent Matrix2 stage.
Its remaining gap to the measured practical roof is only about 5.1 ms for the
entire TTFT, while every resource, control, fusion, RoPE, attention, and tail
direction has an individual removable bound below 3%. A further M128 design
would raise accumulator liveness and register/spill risk after the strong
target has already been exceeded, and could harm long prefill.

Phase 7 therefore stops with M64 retained: 128 TTFT is 119.586 ms, all short
and long prefill points improve, decode is protected, exact representation is
unchanged, and the result is within 4.3% of the revised practical floor. The
next architectural step, only if a later target requires it, is an isolated
M128 or multi-output-tile Matrix2 experiment that must first prove register,
spill, occupancy, and 28K behavior; it is not justified for this phase.

# Vulkan Prefill Phase 8

## Baseline and measurement protocol

Phase 8 ran on local branch `perf/vulkan-short-prefill-phase8`; nothing was
pushed. The requested Phase 7 implementation/report commits `0e0ae71` and
`c98d7b1` are ancestors. The actual clean branch tip was `a731459`, a narrow
follow-up that corrects the M64 capability gate from 12,288 to 16,384 declared
shader bytes and adds its unit assertion. It changes neither the selected RTX
3060 kernel nor runtime math, and was retained as the safe executable baseline.

The tuple is unchanged: RTX 3060 12 GiB device `0x2487`, driver 595.84,
Vulkan 1.4.329, Rust/Cargo 1.95, pure Vulkan, context 32,768, F16 KV, greedy
sampling, and two emitted tokens. The model is the authenticated 2,147,023,008
byte `Ministral-3-3B-Instruct-2512-Q4_K_M.gguf`, SHA-256
`9ed150d4367e68df0ac8e1540f6ddc65b42d0ee26378329d1ecbca60f93fc5f8`.
Every prompt is `" a"` repeated `tokens - 3`.

The official TTFT and prompt-throughput baseline is a release build without
`vulkan-profile`. Runs through 512 use two warm-ups; 1K through 28K use one.
There are 20 measured repetitions at 128 and 512, nine at 64/256, seven at
1K/2K, and three at 8K/28K. A temporary post-measurement logger emitted values
already held by the harness only after every timed repetition. It changed no
GPU command or measured interval and is absent from the final tree.

GPU and prefill-wall distributions come from a separate `vulkan-profile`
build because Vulkan timestamps cannot exist in a diagnostics-free binary.
They are attribution evidence, not the official TTFT reference. Profiler
commands were grouped back into requests and warm-ups discarded. The
instrumented wall is correspondingly higher and is never mixed into an A/B
retention decision.

### Diagnostics-free TTFT

| Tokens | Mean | Median | p10 | p90 | SD | CV | Prompt tok/s mean |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | 65.840 | 65.149 | 64.532 | 67.619 | 2.016 | 3.06% | 972.81 |
| 128 | **114.615** | 114.551 | 114.115 | 115.289 | 0.496 | **0.43%** | 1,116.80 |
| 256 | 219.310 | 219.103 | 218.225 | 220.639 | 0.971 | 0.44% | 1,167.32 |
| 512 | 439.856 | 439.860 | 438.583 | 441.706 | 1.450 | **0.33%** | 1,164.03 |
| 1K | 895.831 | 896.219 | 892.642 | 898.797 | 3.003 | 0.34% | 1,143.08 |
| 2K | 1,867.537 | 1,867.345 | 1,865.972 | 1,869.510 | 1.673 | 0.09% | 1,096.63 |
| 8K | 9,335.657 | 9,330.545 | 9,318.505 | 9,354.853 | 23.145 | 0.25% | 877.50 |
| 28K | 52,636.119 | 52,605.824 | 52,595.618 | 52,688.738 | 63.840 | 0.12% | 531.95 |

The 64-token first measured sample retained a process-level tail and makes
that secondary row 3.06% CV. The decisive 128/512 rows are below 0.5%, every
other medium/long row is below 0.5%, and no favorable sample was selectively
rerun.

### Profiled GPU prefill and CPU wall

| Tokens | GPU mean / median | GPU p10 / p90 | GPU SD / CV | Wall mean / median | Wall p10 / p90 | Wall SD / CV |
|---:|---:|---:|---:|---:|---:|---:|
| 64 | 64.465 / 64.382 | 64.110 / 64.888 | 0.405 / 0.63% | 67.556 / 68 | 67 / 68 | 0.527 / 0.78% |
| 128 | 114.162 / 114.315 | 113.249 / 114.997 | 0.769 / 0.67% | 117.200 / 117 | 116 / 118 | 0.834 / 0.71% |
| 256 | 219.026 / 218.746 | 217.999 / 220.404 | 1.003 / 0.46% | 222.222 / 222 | 221 / 223.2 | 0.972 / 0.44% |
| 512 | 440.329 / 440.094 | 437.979 / 442.880 | 1.852 / 0.42% | 445.150 / 445 | 443 / 448 | 1.755 / 0.39% |
| 1K | 900.559 / 900.796 | 898.592 / 902.729 | 1.842 / 0.20% | 908.714 / 909 | 906.6 / 911 | 1.890 / 0.21% |
| 2K | 1,883.073 / 1,884.515 | 1,878.354 / 1,886.919 | 4.395 / 0.23% | 1,897.571 / 1,899 | 1,893 / 1,901.4 | 4.198 / 0.22% |
| 8K | 9,400.619 / 9,400.537 | 9,400.106 / 9,401.165 | 0.666 / 0.01% | 9,453.333 / 9,453 | 9,453 / 9,453.8 | 0.577 / 0.01% |
| 28K | 52,914.100 / 52,909.328 | 52,907.166 / 52,922.944 | 10.693 / 0.02% | 53,092.667 / 53,088 | 53,085.6 / 53,101.6 | 10.786 / 0.02% |

Prompt-throughput statistics use each sample's `tokens / TTFT`, rather than
inverting the aggregate TTFT mean:

| Tokens | Mean tok/s | Median | p10 | p90 | SD | CV |
|---:|---:|---:|---:|---:|---:|---:|
| 64 | 972.814 | 982.366 | 947.018 | 991.757 | 28.211 | 2.90% |
| 128 | 1,116.800 | 1,117.404 | 1,110.251 | 1,121.678 | 4.825 | 0.43% |
| 256 | 1,167.316 | 1,168.399 | 1,160.265 | 1,173.100 | 5.160 | 0.44% |
| 512 | 1,164.029 | 1,164.007 | 1,159.143 | 1,167.395 | 3.841 | 0.33% |
| 1K | 1,143.083 | 1,142.577 | 1,139.301 | 1,147.158 | 3.834 | 0.34% |
| 2K | 1,096.632 | 1,096.744 | 1,095.474 | 1,097.551 | 0.982 | 0.09% |
| 8K | 877.499 | 877.977 | 875.697 | 879.111 | 2.174 | 0.25% |
| 28K | 531.955 | 532.260 | 531.423 | 532.364 | 0.645 | 0.12% |

## Stage A: floor reconstruction

The fresh 128 result is 3.0% faster than Phase 7's diagnostics-free 118.35 ms
and effectively equals its old 114.5 ms estimate. The floor was therefore
rebuilt from the current session rather than copied.

The profiled category pass assigns 113.642 ms GPU at 128. Its seven recurrent
Matrix2 projections total 104.472 ms; attention is 1.179, embedding 1.534,
logits 3.795, normalization/RoPE/activation/KV/residual copies 2.537, and the
GPU timestamp residual 0.125 ms. Reconciliation to the separate official
114.615 ms TTFT leaves 0.973 ms for non-GPU request setup, resource/control,
sampling/readback, and cross-pass drift. No hidden `other` bucket exceeds
0.5 ms: embedding, logits, activation, elementwise copies, and non-GPU work
are named explicitly.

| Component | Measured ms | Physical minimum | Practical minimum | Removable ms | Confidence |
|---|---:|---:|---:|---:|---|
| Q4/Q6 recurrent Matrix2 | 104.472 | max(12.68 compute, 77.86 logical movement) | 101.47 | **3.00** | high: current 256-row rate |
| fused attention | 1.179 | below 0.01 operation-only | 1.179 | about 0 | high at this scale |
| embedding | 1.534 | raw lookup only | 1.534 | about 0 | high |
| final logits | 3.795 | required final vocabulary reduction | 3.795 | about 0 | high |
| norms, RoPE, activation, KV, residuals | 2.537 | non-zero required graph work | 2.537 | about 0 | medium-high |
| GPU residual/control | 0.125 | zero | 0.125 | about 0 | high: 99.89% accounted |
| host/resource/sampling reconciliation | 0.973 | non-zero serialized tail | 0.973 | about 0 | medium: separate pass |
| **Total** | **114.615** | **about 22 optimistic whole-graph** | **111.615** | **3.000** | **high for practical floor** |

The physical minimum is deliberately not used as a target. The recurrent
compute bound is 774.705 GFLOP / 61.1 TFLOP/s = 12.68 ms, while the 28.030 GB
logical movement bound at nominal 360 GB/s is 77.86 ms. Those overlap and the
logical bytes are not DRAM counters. Adding irreducible graph/tail work gives
an optimistic whole-request physical floor near 22 ms, useful only to reject
claims that the practical floor is a law of physics.

The practical floor uses the same final M64 kernel at the fully occupied
256-row point. Its per-128-row recurrent rate is 101.47 ms. Keeping every other
measured stage gives **111.615 ms**. A speculative M128 traffic-only model
would approach roughly 108 ms, but it ignores its occupancy cliff and is not
a realistic exact-path floor.

### Matrix2 M64 resource and traffic audit

At 128, each weight value is unpacked/dequantized twice: one M64 workgroup per
64 prompt rows. Each dequantized 128x32 weight tile feeds 64 rows, a reuse
factor of two over M32. Shapes are exact multiples of M64/N32/K128 and tile
utilization is 100%; there is no padded Matrix2 instruction.

| Representative projection | Stored quant / metadata | Executed quant / metadata | GPU role |
|---|---:|---:|---|
| Q4 Q | 6.291 / 0.786 MB | 12.582 / 1.572 MB | 64-row weight reuse |
| Q4 gate/up | 14.156 / 1.769 MB | 28.312 / 3.538 MB | dominant Q4 shape |
| Q6 down | 21.234 / 1.991 MB | 42.468 / 3.982 MB | dominant Q6 shape |

Across all seven projections, stored weights are 1.808 GB, shader-visible
weight loads 3.616 GB, activation Matrix2 loads 24.210 GB, and outputs
0.204 GB. About 1.478 million workgroup Matrix2 calls model the 774.705 GFLOP
at 100% tile utilization. Current effective compute is 7.42 TFLOP/s;
minimum/visible effective weight bandwidth is 17.3/34.6 GB/s and total logical
bandwidth is 268 GB/s. Raw weight bandwidth is not the limit; tensor loads,
unpack/dequant, synchronization, cache, issue, and occupancy overlap.

Removed executable statistics from Phase 7 remain applicable to unchanged
SPIR-V: Q4/Q6 use 98/108 registers per thread, 16,384 shader-declared and
33,792 driver-reported shared bytes, zero stack, two resident workgroups,
16 resident warps, and 33.3% modeled occupancy. The `a731459` baseline fix
ensures the capability gate accounts for the full 16,384 shader bytes.

M128 would reduce visible weight traffic by 1.808 GB, only 6.45% of total
logical movement. A traffic-only model predicts about 6.9% local and 5.9%
TTFT improvement. Extrapolating the measured M32→M64 register growth gives
roughly 166 Q4 and 198 Q6 registers/thread, one resident workgroup, eight
warps, and 16.7% occupancy. The fully occupied M64 measurement already limits
realistic remaining TTFT saving to 3.00 ms or 2.62%. The traffic-only gain is
therefore not realistic enough to pass the experiment gate; reuse=2 is the
practical optimum for 128.

### Stage A decision

```text
current diagnostics-free TTFT = 114.615 ms
practical floor              = 111.615 ms
remaining                    =   3.000 ms
remaining                    =   2.62%
measured / practical floor   =   1.0269
```

Decision: **STOP short-prefill optimization**. Both mandatory limits are met:
headroom is below 3% and below 3.5 ms. No residual component independently
offers 3% TTFT, and further M128 reuse fails the realistic occupancy-adjusted
gate. No Phase 8 short-prefill production experiment was built.

## Updated matmul/attention balance

| Tokens | Matmul ms | Attention ms | Other ms | Attention / GPU | Attention / TTFT |
|---:|---:|---:|---:|---:|---:|
| 128 | 104.472 | 1.179 | 7.991 | 1.04% | 1.03% |
| 512 | 405.494 | 11.204 | 20.405 | 2.56% | 2.55% |
| 2K | 1,632.006 | 162.159 | 71.061 | 8.69% | 8.68% |
| 8K | 6,606.621 | 2,525.602 | 274.376 | 26.85% | 27.05% |
| 16K | 13,136.332 | 10,005.572 | 592.097 | 42.16% | 41.96% |
| 28K | 22,583.784 | **29,370.118** | 927.576 | **55.54%** | **55.80%** |

The measured crossover is about 20K tokens: matmul still leads by 3.13 s at
16K, while attention leads by 6.79 s at 28K. The Phase 7 M64 improvement has
therefore moved the largest absolute 28K cost back to attention.

The detailed profile below reconciles every measured GPU millisecond. Q, K,
V, O, gate, up, and down are the seven recurrent Matrix2 projections. `Copies`
contains KV writes and elementwise residual copies. QK and AV are direct
downstream-live ablation deltas; the fused remainder contains softmax setup,
shared score/probability/AV traffic, barriers, addressing, and online merge.
Those operations are not separately timestampable without splitting the
kernel, so the residual is named rather than misreported as softmax alone.

| Component (ms) | 2K | 8K | 28K |
|---|---:|---:|---:|
| Embedding | 24.851 | 99.543 | 339.473 |
| Norms | 8.757 | 35.017 | 120.275 |
| Q projection | 174.430 | 705.860 | 2,414.197 |
| K projection | 48.154 | 194.679 | 662.766 |
| V projection | 51.269 | 207.216 | 709.362 |
| RoPE | 5.203 | 20.812 | 71.945 |
| Attention QK | 52.773 | 803.806 | 9,585.723 |
| Attention setup/softmax/barrier/merge | 50.528 | 842.369 | 9,655.183 |
| Attention AV | 58.858 | 879.427 | 10,129.212 |
| O projection | 180.880 | 734.273 | 2,495.243 |
| Gate projection | 382.766 | 1,552.749 | 5,320.431 |
| Up projection | 382.947 | 1,548.993 | 5,297.707 |
| Activation | 15.441 | 62.352 | 211.881 |
| Down projection | 411.561 | 1,662.851 | 5,684.078 |
| Copies | 11.755 | 47.508 | 162.084 |
| Logits | 3.777 | 3.850 | 3.808 |
| Dispatch/timestamp residual | 1.278 | 5.294 | 18.111 |
| Unnamed other | 0.000 | 0.000 | 0.000 |
| **Attributed GPU total** | **1,865.226** | **9,406.599** | **52,881.479** |
| **Coverage** | **100%** | **100%** | **100%** |

## Current attention architecture and physical model

The selected F16 Matrix2 path is unchanged: one 256-thread workgroup, eight
32-lane subgroups, one query head, Q tile 32, K/V tile 64, head dimension 128,
and online causal softmax. Workgroup-scope Matrix2 performs QK 32x64x128 and
AV 32x128x64. Q/K/V use direct tensor loads. Score is stored as FP32 shared,
probability as FP16 shared, AV as FP32 shared, and 16 FP32 output values/thread
carry the online merge. The shader declares 20,864 bytes shared; the driver
reports 38,272 bytes, 125 registers/thread, two resident workgroups, 16 warps,
and 33.3% modeled occupancy. There is no global score or probability tensor.

At 28K there are 159,614,208 layer/head/query workgroup-KV-tile visits:

| Quantity | 28K model |
|---|---:|
| repeated Q reads | 1.308 TB |
| K reads | 2.615 TB |
| V reads | 2.615 TB |
| QK FLOP | 83.684 TFLOP |
| AV FLOP | 83.684 TFLOP |
| score/probability/AV shared round trips | about 10.460 TB on-chip |
| effective Q/K/V logical bandwidth | 225 GB/s |
| effective QK+AV compute | 5.76 TFLOP/s |

The dense compute minimum is about 2.74 s. Q/K/V logical bytes at nominal
360 GB/s give 18.16 s, while K+V alone at a realistic 300 GB/s give 17.43 s.
Current attention is 29.05--29.37 s. The path is therefore primarily a
tensor-load/cache/shared/synchronization/occupancy pipeline, not softmax SFU
or dense Matrix compute alone. The traffic model is not a DRAM counter.

### Fresh QK, AV, and residual attribution

Matching one-run cold controls removed one downstream-live bucket at a time.
The QK ablation keeps softmax and AV live on zero scores. The AV ablation keeps
QK and softmax live and merges a zero AV result. Deltas are causal bounds, not
independent hardware counters.

| Tokens | Attention | QK direct delta | AV direct delta | Setup/softmax/barrier/merge residual |
|---:|---:|---:|---:|---:|
| 128 | 1.261 | 0.158 | 0.470 | 0.633 |
| 512 | 11.250 | 3.550 | 4.132 | 3.568 |
| 2K | 160.686 | 52.773 | 58.858 | 49.055 |
| 8K | 2,474.021 | 803.806 | 879.427 | 790.788 |
| 28K | 29,049.931 | **9,585.723** | **10,129.212** | **9,334.996** |

The direct QK and AV deltas are 33.0% and 34.9% of 28K attention. The residual
contains scale/mask, score store/read, max, two exponentials, sum reduction,
probability store/read, four barriers per KV tile, AV scratch round trip,
online merge, addressing, and epilogue. A downstream-live softmax replacement
made attention 2.57% slower at 28K and 3.63% slower at 8K. It therefore exposes
no positive removable softmax time; softmax is not reported as the entire
9.335 s residual. Its source was removed.

## Stage B ranking and experiments

Ranking uses absolute milliseconds per request. Decode request time is 31
measured public-token intervals from the diagnostics-free 32-token controls.

| Rank | Workload | Dominant component | Component ms/request | Realistic local speedup | Removable ms | Predicted total gain |
|---:|---|---|---:|---:|---:|---:|
| 1 | 28K prefill | attention | 29,370 | 10--15% architectural | 2,937--4,406 | 5.6--8.4% |
| 2 | 16K prefill | attention | 10,006 | 10--15% architectural | 1,001--1,501 | 4.2--6.3% |
| 3 | 28K prefill | recurrent Matrix2 | 22,584 | at most about 3% | at most 678 | at most 1.3% |
| 4 | 8K prefill | attention | 2,526 | 10--15% architectural | 253--379 | 2.7--4.1% |
| 5 | 8K prefill | recurrent Matrix2 | 6,607 | at most about 3% | at most 198 | at most 2.1% |
| 6 | 2K prefill | recurrent Matrix2 | 1,632 | at most about 3% | at most 49 | at most 2.6% |
| 7 | decode 28K | attention | about 410/31-token request | below 5% after Decode Phase 2 | below 21 | below 2% |
| 8 | decode 2K | recurrent matmul | about 487 | below 3% | below 15 | about 2% |
| 9 | decode 128 | recurrent matmul | about 482 | below 3% | below 15 | about 2% |
| 10 | 512 prefill | recurrent Matrix2 | 405 | at most about 3% | at most 12 | at most 2.8% |

The chosen Stage B target was 28K attention. It has the largest removable
millisecond range and also applies from the 8K crossover region upward.

| ID | Stage | Architecture / diagnostic | Local delta | End-to-end delta | 128 control | Decision |
|---|---|---|---:|---:|---:|---|
| P8-Q-STAGE | B | stage Q32x128 once/WG; +8 KiB shared | attention +10.62% 8K, +11.27% 28K | +2.01% 8K, +5.86% 28K | unchanged source after reject | reject |
| P8-Q64 | B | Q64/KV64 Matrix2 ownership | Matrix2 pipeline unavailable; Q16 KHR fallback | +103% at 128, +113% at 512 | failed | reject before long rows |
| P8-QK-OFF | attribution | downstream-live zero score | -33.0% attention 28K | not a candidate | n/a | measurement only |
| P8-AV-OFF | attribution | downstream-live zero AV | -34.9% attention 28K | not a candidate | n/a | measurement only |
| P8-SOFTMAX | attribution | live uniform probability/merge | +2.57% attention 28K | regression | n/a | reject |

`P8-Q-STAGE` removed about 1.307 TB of shader-visible Q reads at 28K, but
shared cooperative loads plus the extra staging dependency were slower than
the direct tensor/cache path. It used an estimated 46,464 driver shared bytes
and retained the two-workgroup register limit; traffic reduction did not reach
the critical path.

`P8-Q64` targeted half the K/V tensor traffic with one 64-query workgroup.
Shader compilation succeeded, but pipeline construction rejected that flexible
Matrix2 shape and the registry safely selected
`attention_prefill_tiled_fullmma_q16_kv64`. The fallback regression is the
resource/capability result; no 8K/28K time was wasted after the kernel identity
proved the experiment invalid.

No experiment changed a public API, dependency, format, precision, KV layout,
decode shader, or final routing. All candidate and diagnostic source is removed.
The best retained candidate is therefore **none**: architecture, tile shape,
resource use, routing, precision, and traffic remain the Phase 7 baseline.
The next plausible attention step must combine ownership across more than 32
queries with staged/reused K/V while preserving a supported Q32 Matrix2 shape,
or connect score/softmax/AV without the current shared round trips. That is a
new multi-head/multi-query kernel architecture with separate resource and
numeric qualification, not another Phase 8 local patch.

## Final performance, decode, and correctness

There is no retained production candidate, so final prefill is the Phase 8
baseline by source identity:

| Tokens | Baseline TTFT | Final TTFT | Delta |
|---:|---:|---:|---:|
| 64 | 65.840 | 65.840 | 0% |
| 128 | 114.615 | 114.615 | 0% |
| 256 | 219.310 | 219.310 | 0% |
| 512 | 439.856 | 439.856 | 0% |
| 1K | 895.831 | 895.831 | 0% |
| 2K | 1,867.537 | 1,867.537 | 0% |
| 8K | 9,335.657 | 9,335.657 | 0% |
| 28K | 52,636.119 | 52,636.119 | 0% |

Diagnostics-free 32-token decode controls are 46.63 tok/s at 128 (0.32% CV),
46.22 at 2K (0.10% CV), and 29.11 at 28K (0.34% CV). All are faster than the
Phase 7 session's 43.52/43.07/27.77 tok/s; no decode regression is evidenced.
The same control repeated 128 TTFT at 115.87 ms with 0.79% CV, within normal
session drift and with unchanged source. After every diagnostic was removed,
a clean diagnostics-free release rebuild measured 114.66 ms with 0.55% CV
over fifteen samples, reproducing the official 114.615 ms baseline.

Final restored-tree evidence:

- Vulkan-profile library suite: 152 passed, four intentionally ignored;
- focused Q4/Q6 Matrix2 CPU-oracle coverage passes in that suite, including
  three-row M64 tails;
- focused long qualification through 28K passes in 421.05 s;
- F16 attention maximum absolute error remains `1.526e-5` and projected-logit
  maximum `3.601e-6`; CPU-oracle maxima remain `1.180e-5` / `4.001e-6`;
- INT8 attention and logits remain bit-exact at 2K, 8K, and 28K;
- the 128-token full-model greedy oracle emits exactly
  `2757,10637,2479,1636`;
- F16/INT8, Q4_K/Q6_K, Matrix2, GQA, causal attention, logits, retained KV,
  and multi-token decode are exercised without changing tolerances;
- release Clippy with all targets and warnings denied, formatting, and
  `git diff --check` pass;
- `docs_contract` retains the known pre-existing local failure when the
  gitignored `VALIDATION.md` is absent.

## Final floor and stop rationale

```text
128 measured                  114.615 ms
128 practical floor           111.615 ms
gap                             3.000 ms
gap                              2.62%

final largest bottleneck       28K attention, 29.37 s/request
next realistic headroom        2.94--4.41 s at 28K
retained Phase 8 code           none
```

Stage A stops because 128 is within 2.62% of its rebuilt practical floor and
no experiment can meet both 3.5 ms and 3%. Stage B successfully pivots to the
largest absolute cost and falsifies two materially different reuse designs:
direct Q staging regresses, while Q64 Matrix2 ownership is unsupported. The
remaining 28K opportunity is still architecturally significant, but the next
step is a new supported multi-query attention kernel rather than incremental
tuning. Phase 8 therefore stops under the architectural-mission boundary with
the clean Phase 7 production path preserved.
