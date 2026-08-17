<!--
This document owns the Phase 28 Vulkan checkpoint: rejected prefill execution
representations, retained GQA decode vectorization, qualification, attribution,
competitive bounds, and the next measured candidate order.
-->

# Vulkan GQA Decode Vectorization — Phase 28

## Outcome

Phase 28 retains one exact long-context decode optimization. The split-GQA
attention kernel now evaluates the four query heads sharing one KV head as
`vec4` score and online-softmax state. It preserves the same FP32 operations per
head, F16 KV representation, eight history splits, dispatch topology, buffers,
and final reduction.

The change improves decode throughput by 12.9% on 3B/28K and 9.1% on 8B/28K.
It is neutral within about 2% at 128 and 2K, where attention is not dominant,
and neutral on both capacity-qualified 14B variants. All six authenticated
models preserve their acquired 128-token greedy sequence exactly.

The global target is not yet met. 8B/28K is now 19.52 tok/s versus llama.cpp
27.69 tok/s, a 1.42x latency ratio just outside the 1.3–1.4x maturity band.
Short-context decode and all prefill rows remain outside target. This is a
**CONTINUE checkpoint**, not a project stop.

## Reproducibility

| Item | Value |
|---|---|
| Date | 2026-08-17 |
| Branch | `perf/systematic-llamacpp-gap` |
| Starting checkpoint | `21b7348` |
| Production commit | `e21fc12` |
| GPU / driver | NVIDIA GeForce RTX 3060 12 GiB / 595.84 |
| Runtime | pure Vulkan, full offload, F16 KV, greedy, 32 generated tokens |
| 3B/8B capacity | 32,768 tokens |
| 14B capacity guard | 512 tokens |
| Candidate public SHA-256 | `01701aee41d5c74f490f26c21184b581c45e0ee30e5c39bccd8d8c61534747d4` |
| Candidate profiler SHA-256 | `a155a4af710afb790d430ec8ba79f712011e4ea259f7db30ba06d4ba0ba13f4b` |

The immutable Phase 27 public binary is the control. Measurements use paired
same-session runs where a causal comparison matters. Nothing was pushed.

## Experiment register

| ID | Candidate | Result | Quality | Decision |
|---|---|---|---|---|
| E1 | FP16 Q4 Matrix2 accumulator | four of five Q4 oracles fail | unchanged tolerances | REJECT |
| E2 | Q4 Matrix2 N64 tile | 8B/128 TTFT 259.26 -> 321.51 ms | five oracles pass | REJECT |
| E3 | transient exact Q4-to-FP16 tensor | 8B/128 +89%; 8B/512 +33% | orientation oracles pass | REJECT |
| E4 | pre-scaled attention Q | 8B/8K TTFT +0.53% | boundary oracle passes | REJECT |
| E5 | Q4 decode subgroup-XOR reduction | 8B/128 decode -5.2% | Q4 oracle passes | REJECT |
| E6 | vectorized four-head GQA state | attention -27.4%; 8B/28K +9.1% | six exact sequences | KEEP |

E3 allocated an approximately 112 MiB scratch view and performed one exact
conversion per projection/chunk before direct FP16 Matrix2. Its unchanged bits
prove correctness, but conversion and extra traffic dominate even at 512 rows.
The prototype files and wiring were removed completely. E4 tested a distinct
rounding boundary by scaling F16 Q before QK; it was neutral at 128 and negative
at 8K. E5 removed the Q4 reduction barrier but exposed insufficient useful work
per subgroup. All rejected production edits were restored before `e21fc12`.

## Causal attribution

At 8B/8K, paired profiler runs use the same 36,022 dispatches and 27,590
barriers over 62 decode steps:

| Metric | Phase 27 | Vector state | Change |
|---|---:|---:|---:|
| Attention GPU | 337.770 ms | 243.647 ms | -27.87% |
| Attention per token | 5.448 ms | 3.930 ms | -1.518 ms |
| Total decode GPU | 2,628.532 ms | 2,554.203 ms | -2.83% |
| Public decode | 23.13 tok/s | 23.86 tok/s | +3.16% |

The post-KEEP 8B/28K profile accounts for 1,557.582 of 1,562.587 GPU ms
(99.68%) across 31 decode steps. Command topology remains 18,011 dispatches
and 13,795 barriers.

| Category | Phase 26 ms/token | Phase 28 ms/token | Current share |
|---|---:|---:|---:|
| Attention | 17.971 | 13.045 | 25.88% |
| Q projection | 2.466 | 2.448 | 4.86% |
| K projection | 0.907 | 0.894 | 1.77% |
| V projection | 1.430 | 1.418 | 2.81% |
| O projection | 2.498 | 2.528 | 5.02% |
| Gate | 7.333 | 7.530 | 14.94% |
| Up | 7.573 | 7.564 | 15.01% |
| Down | 10.740 | 10.907 | 21.64% |
| Logits | 2.764 | 2.779 | 5.51% |

Untouched projection categories remain near their prior values. Attention
falls 557.106 -> 404.394 ms per request, while total GPU decode falls
1,703.620 -> 1,562.587 ms. The local and whole-request changes agree with
Amdahl's prediction and isolate the retained shader as the cause.

## Public matrix

| Model / context | Phase 27 tok/s | Vector state tok/s | Change |
|---|---:|---:|---:|
| 3B / 128 | 56.13 | 56.74 | +1.09% |
| 3B / 2K | 54.77 | 55.92 | +2.10% |
| 3B / 28K | 32.34 | 36.52 | **+12.93%** |
| 8B / 128 | 26.34 | 26.64 | +1.14% |
| 8B / 2K | 25.50 | 25.91 | +1.61% |
| 8B / 8K | 23.13 | 23.86 | +3.16% |
| 8B / 28K | 17.89 | 19.52 | **+9.11%** |
| 14B Instruct / 128 | 17.22 | 17.23 | +0.06% |
| 14B Reasoning / 251 | 17.20 | 17.25 | +0.29% |

The 3B/28K paired CV is at most 0.06%; the 8B/28K paired CV is at most 0.26%.
TTFT is causally unchanged: the committed shader is decode-only. The 8B/28K
paired values are 75,099.00 and 75,091.04 ms; the post-KEEP profile measures
75,199.21 ms and confirms no prefill topology change.

## Correctness and structural guard

- The F16 and INT8 prefill/sequential-decode attention parity cases pass at
  bases and lengths 0/3, 0/64, 33/32, and 65/9.
- Both instruct and reasoning artifacts at 3B, 8B, and 14B exactly match their
  acquired 128-step greedy token IDs.
- The 3B/8B gates use context 32,768 and 2K-class prompts; the 14B gates use
  context 512 and their qualified smaller prompts.
- The retained file is a 105-line category-K kernel. It adds no allocation,
  dependency, public API, dispatch, barrier, tolerance, or model-specific rule.

## Quantitative continuation

At 8B/28K, latency is now 51.23 ms/token. The 1.3x target permits 46.94 ms,
so another 4.29 ms/token must be removed. Attention's approximately 10.8 ms
F16 traffic floor leaves only about 2.25 ms of theoretical attention headroom;
attention alone therefore cannot close the target. Down projection is the next
largest single category at 10.91 ms/token, but its exact Q6 lane and direct-load
variants have already regressed.

The next decode cycle must combine projection work with any remaining attention
headroom. The measured high-value unresolved path is a selective numerical
sensitivity map for the existing fast cooperative DP4A Q4 route: the all-Q4
policy improved 8B/128 by 44.5% but diverged at greedy token 2. A shape-family
isolation can determine whether a materially useful subset preserves the full
quality contract. This is distinct evidence, not a repeat of the rejected
all-Q4 policy.

Prefill remains the larger global failure: the post-KEEP 8B/28K profile is
75.20 s versus a 37.26 s target. Matrix2 projection families consume 52.31 s
and attention 21.38 s, so neither family alone can close the 37.94 s gap. The
next prefill representation must therefore alter work ownership materially;
FP16 accumulation, N64, transient full-tensor dequantization, pre-scaled Q,
and prior direct/cached variants are quantitatively closed.
