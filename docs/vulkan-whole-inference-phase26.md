<!--
This document owns the Phase 26 whole-inference Vulkan checkpoint: immutable
baselines, accepted decode changes, rejected experiments, full feasible model
matrix, correctness evidence, competitive gaps, and the next-cycle decision.
-->

# Vulkan Whole-Inference Performance — Phase 26

## Outcome

Phase 26 retains three production optimizations and one test-only control:

1. Q4_K decode reads the cache-resident activation directly, removing two
   workgroup barriers per super-block.
2. Q6_K logits use packed four-byte weight loads and hoisted scales.
3. Q4_K decode assigns 16 lanes to each output row and 16 rows to each
   256-thread workgroup.
4. The ignored real-model sequence test accepts an optional context capacity,
   allowing the capacity-bound 14B artifacts to use the same exact-ID gate.

The accepted stack improves decode throughput at every measured context:
9.6–21.1% on 3B, 12.1–21.6% on 8B, and 11.1–12.1% on the capacity-qualified
14B variants. It preserves the exact 128-token greedy sequence of the acquired
baseline on all six authenticated artifacts. Prefill code is unchanged except
for the single final Q6 logits dispatch; no causal prefill improvement is
claimed.

The global target is **not met**. Final decode remains 1.55–1.92x slower than
llama.cpp on the comparable 3B/8B rows, and 14B/128 remains 2.10x slower.
Prefill remains 2.26–3.75x slower on 3B/8B. This is therefore a persistent
**CONTINUE checkpoint**, not a project stop: the next cycle returns to the
8B prefill Matrix2/attention gap.

## Reproducibility

| Item | Value |
|---|---|
| Date | 2026-08-17 |
| Branch | `perf/systematic-llamacpp-gap` |
| Starting source | `8e636af` |
| Retained commits | `bb41cc3`, `cf02f8a`, `9f383e4`, `a6fa97a`, `53871d1` |
| GPU / driver | NVIDIA GeForce RTX 3060 12 GiB / 595.84 |
| Vulkan API | 1.4.329 |
| Memory clock / power ceiling | 7,501 MHz / 170 W |
| Runtime | pure Vulkan, full GPU offload, F16 KV, greedy, 32 generated tokens |
| 3B/8B capacity | 32,768 tokens |
| 14B capacity | 512 tokens; 32,768-token initialization does not fit 12 GiB |
| Public repetitions | warmup 1; 5/5/3/3/2/2 at 128/512/2K/8K/16K/28K |
| Final public executable SHA-256 | `57441c7a0a87baf485ba3c5ffdaf43316a2dc59028605ea3305ac468951818fc` |
| Final profiler SHA-256 | `27b9880f581d037b107943b7143ca9a74bf78fae9a862aa32a85212ac52afca4` |
| llama.cpp | commit `9bebfcb4`, build 9826, Vulkan0, F16 KV, flash attention |

The six model artifacts and hashes remain those authenticated in Phase 22.
Instruct variants are the performance representatives because each reasoning
variant has the same tensor shapes and formats. Reasoning variants receive
separate initialization, performance, and exact-sequence guards below.

Ignored raw evidence is under `target/phase26`. The initial worktree was clean,
the performance branch predated the first production edit, and nothing was
pushed.

## Initial global attribution and candidate order

The acquired 8B/128 decode step was 47.50 ms GPU. Q4 Q/K/O/gate/up consumed
26.66 ms (56.1%); Q6 V/down and logits about 19.3 ms; attention 0.33 ms. At
28K, attention grew to 17.07 ms/token while Q4 remained 27.29 ms/token.

The initial Amdahl ranking was:

| Rank | Candidate | Affected baseline | Predicted whole gain | Main risk |
|---:|---|---:|---:|---|
| 1 | Cooperative DP4A Q4_K decode | 26.66 ms/token | 17–19% | Q8 activation changes arithmetic |
| 2 | Packed/execution-tuned Q6 decode | about 9–10 ms/token | 5–9% | 210-byte layout and cache behavior |
| 3 | Selective FP16 execution views | dominant matmuls | up to 36% | 12 GiB capacity and rounding |
| 4 | Pre-scaled attention Q | long-context attention | about 10% | exact-token quality |
| 5 | Narrower long-KV representation | 17.07 ms at 28K | about 6% | F16 baseline semantics |

DP4A was selected first because it was reusable across model sizes and all
contexts. Its quality failure caused the program to move to exact-FP16
execution changes rather than retain a fast but invalid result.

## Experiment register

| ID | Candidate | Causal result | Whole/public result | Quality | Decision |
|---|---|---|---|---|---|
| E1 | Q8_1 + cooperative DP4A Q4 | Q4 categories -54.3% | 8B/128 +44.5%; 3B/128 +32.3% | diverges at greedy token 2 | REJECT |
| E2 | Q4 direct cached activation | Q4 categories -10.6% | 8B +2.6%; 3B +3.0% | exact 128 IDs | KEEP |
| E3 | Q6 direct cached activation | V/down slower | negative profile | oracle pass | REJECT |
| E4 | packed Q6 logits loads | logits 5.75 -> 2.67 ms | incremental 8B +7.8%; 3B +11.7% | exact 128 IDs | KEEP |
| E5 | Q4 eight-lane row split | total GPU -8.3% | incremental 8B +6.9%; 3B +2.5% | exact 128 IDs | KEEP |
| E6 | Q6 eight-lane row split | down +5.2% | total GPU +1.4% | oracle pass | REJECT |
| E7 | Q4 sixteen-lane split | total GPU -2.7% | incremental 8B +2.9%; 3B +2.6% | exact 128 IDs | KEEP |
| E8 | Q4 thirty-two-lane split | total GPU +14% | profile rejection | oracle pass | REJECT |
| E9 | GQA sixteen history splits | attention +3.8% at 8K | profile rejection | not required | REJECT |

E1 is materially different from the old serial one-invocation MMVQ candidate:
eight outputs share a 256-thread workgroup and each row uses a 32-lane
reduction. It proves that integer dot product can close much of the performance
gap, but also proves that Q8 activation quantization violates the accepted
inference semantics. That route is not eligible for retention.

The final lane-count evidence brackets the source-visible optima on this GPU:
Q4 improves from four to eight to sixteen lanes and regresses at thirty-two;
Q6 regresses when widened from four to eight; GQA attention was previously
improved from four to eight splits and now regresses at sixteen.

## Public 3B matrix

Baseline rows are the fresh phase-26 128/2K measurements or the acquired
phase-21 bookends for unchanged contexts.

| Context | Baseline TTFT ms / tok/s | Final TTFT ms / tok/s | Decode gain | llama.cpp tok/s | Final llama/GH |
|---:|---:|---:|---:|---:|---:|
| 128 | 98.19 / 46.28 | 96.99 / 56.03 | +21.1% | 102.78 | 1.83x |
| 512 | 364.55 / 46.44 | 371.10 / 55.13 | +18.7% | 101.97 | 1.85x |
| 2,048 | 1,521.77 / 45.86 | 1,530.74 / 54.73 | +19.3% | 95.06 | 1.74x |
| 8,192 | 7,033.01 / 40.68 | 7,116.24 / 46.86 | +15.2% | 78.21 | 1.67x |
| 16,384 | 16,801.53 / 35.06 | 16,942.69 / 39.48 | +12.6% | 63.01 | 1.60x |
| 28,000 | 35,381.53 / 29.39 | 35,810.88 / 32.21 | +9.6% | 50.02 | 1.55x |

Final prefill ratios versus llama.cpp are 2.26x, 2.82x, 2.83x, 2.73x,
2.63x, and 2.55x from 128 through 28K. Cross-session TTFT movement is within
1.8%; the retained recurrent shaders are decode-only, and the one changed
prefill logits dispatch is faster and below 3 ms.

## Public 8B matrix

| Context | Baseline TTFT ms / tok/s | Final TTFT ms / tok/s | Decode gain | llama.cpp tok/s | Final llama/GH |
|---:|---:|---:|---:|---:|---:|
| 128 | 281.00 / 21.74 | 281.12 / 26.44 | +21.6% | 50.74 | 1.92x |
| 512 | 1,063.53 / 21.94 | 1,079.49 / 26.31 | +19.9% | 49.37 | 1.88x |
| 2,048 | 4,389.16 / 21.11 | 4,389.76 / 25.63 | +21.4% | 47.15 | 1.84x |
| 8,192 | 18,601.47 / 19.74 | 19,070.71 / 23.03 | +16.7% | 39.32 | 1.71x |
| 16,384 | 40,751.46 / 18.02 | 41,705.33 / 20.54 | +14.0% | 35.17 | 1.71x |
| 28,000 | 78,518.39 / 15.90 | 80,010.92 / 17.83 | +12.1% | 27.69 | 1.55x |

Fresh llama.cpp pure-prefill times are 287.81 ms at 512, 1,185.37 ms at 2K,
5,300.13 ms at 8K, 12,283.39 ms at 16K, and 24,842.22 ms at 28K. Together
with the acquired 109.44 ms at 128, the final GH/llama prefill ratios are
2.57x, 3.75x, 3.70x, 3.60x, 3.40x, and 3.22x. Prefill remains the largest
global maturity failure.

## 14B capacity-qualified coverage

Both 14B artifacts fully offload at context capacity 512 but do not initialize
at the 32,768-token F16-KV tuple on this 12 GiB device. Long-context 14B rows
are therefore capacity-blocked, not silently converted to partial offload.

| Artifact | Rendered prompt | Baseline tok/s | Final tok/s | Gain | llama.cpp tok/s | Ratio |
|---|---:|---:|---:|---:|---:|---:|
| 14B Instruct | 128 | 15.43 | 17.30 | +12.1% | 36.25 | 2.10x |
| 14B Reasoning | 251 | 15.42 | 17.13 | +11.1% | shape reference 36.25 | about 2.12x |

TTFT is unchanged: 4,491.86 -> 4,482.33 ms for instruct and 8,802.51 ->
8,816.82 ms for reasoning. llama.cpp 14B pure prefill at 128 is 205.95 ms;
Graph Horizon's capacity-qualified 14B prefill route is far outside target.

## Final 8B/28K GPU attribution

The final timestamp run accounts for 1,699.445 of 1,703.620 ms across 31
decode steps (99.75%). It issues 18,011 dispatches and 13,795 barriers, exactly
the established command topology.

| Category | GPU ms / request | ms/token | GPU share |
|---|---:|---:|---:|
| Attention | 557.106 | 17.971 | 32.70% |
| Q projection | 76.456 | 2.466 | 4.49% |
| K projection | 28.119 | 0.907 | 1.65% |
| V projection | 44.346 | 1.430 | 2.60% |
| O projection | 77.452 | 2.498 | 4.55% |
| Gate | 227.310 | 7.333 | 13.34% |
| Up | 234.773 | 7.573 | 13.78% |
| Down | 332.929 | 10.740 | 19.54% |
| Logits | 85.694 | 2.764 | 5.03% |
| All other kernels | 35.260 | 1.137 | 2.07% |

The corresponding prefill run accounts for 79,313.973 of 79,335.971 ms
(99.97%). Attention is 21.262 s (26.80%); Q/K/V/O/gate/up/down matmuls total
55.975 s (70.57%); everything else is 2.077 s (2.62%). This agrees with the
Phase 25 attribution and shows that decode work did not move the prefill limit.

## Correctness and validation

- The focused Q4_K and Q6_K CPU/Vulkan projection oracles pass at their
  unchanged tolerances.
- Final Vulkan output exactly matches the acquired baseline for 128 accumulated
  greedy tokens on all six authenticated 3B/8B/14B instruct/reasoning models.
- The 3B/8B sequence gates use context 32,768 and a 2K-class prompt. The 14B
  gates use context 512, 100 repeated prompt units, and 128 generated steps.
- The rejected DP4A candidate matches the first token but diverges on token 2;
  it was restored before any retained checkpoint.
- `cargo fmt --all -- --check` passes.
- The full release Vulkan-hybrid workspace suite passes: engine 231 passed / 4
  ignored, integration 6 passed / 1 ignored, semantic 12 passed / 1 ignored.
- Workspace clippy passes for all targets with warnings denied.
- No tolerance, reference ID, model artifact, dependency, or public API changed.

## Quantitative continuation decision

The decode target requires no more than 1.3x llama.cpp. At 8B/128, final
latency is 37.82 ms/token versus llama.cpp 19.71 ms; the target ceiling is
25.62 ms, leaving 12.20 ms to remove. At 8B/28K, final latency is 56.09 ms
versus 36.11 ms; the target ceiling is 46.94 ms, leaving 9.15 ms.

The 28K attention path alone cannot close that gap: it reads approximately
3.90 GB of F16 K/V per token, a 10.8 ms ideal floor at 360 GB/s versus 17.97 ms
measured, so deleting every non-traffic cost saves at most about 7.2 ms. The
remaining gain must combine attention with projection work. Exact split-count
sweeps are bracketed; narrower KV changes the fixed F16 tuple.

At short context, the fast Q8/DP4A Q4 route demonstrates enough speed but fails
the exact-token contract. Persistent FP16 execution views for the dominant
8B gate/up pair require roughly 8 GiB in addition to canonical weights and do
not fit the 12 GiB full-offload tuple. The exact Q4 lane sweep is bracketed at
16 lanes, Q6 at four lanes, and attention at eight splits. These facts stop
further decode micro-sweeps, but they do not prove the global program complete.

The next cycle targets the larger failure: 8B prefill. Meeting 1.5x at 28K
requires reducing 80.01 s to at most 37.26 s, a 42.75 s improvement. Attention
is only 21.26 s, so the candidate must materially restructure the 55.98 s
Matrix2 matmul family or combine a new matmul architecture with attention.
Previously rejected M128, direct canonical callbacks, metadata-only caching,
fusion, and lossy native formats must not be repeated without a materially new
work-ownership or representation argument.

