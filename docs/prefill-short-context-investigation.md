<!--
This report owns the reproducible Phase 5 Vulkan short-context prefill evidence:
baseline, attribution, experiments, retained changes, correctness, and regression gates.
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

Telemetry sampled at 100 ms while GPU utilization was at least 80%. Across
32/64/128/256/512/1K/2K/8K/28K, mean utilization was respectively
92.4/92.5/94.1/95.1/95.6/94.0/95.7/95.7/97.4%. Graphics clocks stayed about
1.97--2.00 GHz, memory at 7.501 GHz, VRAM at 5.1--5.7 GiB, and temperature at
58--70 C. The path was neither idle, downclocked, memory-pressure limited, nor
thermally throttled.

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
