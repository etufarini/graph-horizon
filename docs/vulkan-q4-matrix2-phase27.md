<!--
This document owns the Phase 27 Q4 Matrix2 packed-load checkpoint: hypothesis,
causal attribution, public measurements, model/quality guards, and continuation.
-->

# Vulkan Q4 Matrix2 Packed Loads — Phase 27

## Outcome

Phase 27 retains one exact Q4_K prefill optimization. Each Q4 Matrix2 lane
previously extracted 16 consecutive payload bytes through `gb()`, issuing four
source-visible storage reads for every packed word. The final kernel loads four
aligned words once and extracts their bytes locally. Dequantized FP16 weights,
Matrix2 shapes, FP32 accumulation, output rounding, dispatch counts, barriers,
buffers, and public routing are unchanged.

The change improves 8B TTFT by 8.0% at 128, 7.7% at 2K, and 5.95% at 28K.
It improves 3B TTFT by 3.9% at 128 and 1.85% at 28K. The 14B shapes do not
select this Matrix2 route and remain within 0.5%. Decode throughput is unchanged.
All six authenticated models preserve the acquired 128-token greedy sequence.

The global performance target remains unmet. 8B/28K prefill falls from 80.01
to 75.25 seconds, improving the llama.cpp ratio from 3.22x to 3.03x, still well
outside 1.5x. This is another **CONTINUE checkpoint**, not a project stop.

## Reproducibility

| Item | Value |
|---|---|
| Date | 2026-08-17 |
| Branch | `perf/systematic-llamacpp-gap` |
| Starting checkpoint | `bf3f756` (Phase 26 report) |
| Production commit | `9c56cd8` |
| GPU / driver | NVIDIA GeForce RTX 3060 12 GiB / 595.84 |
| Runtime | pure Vulkan, full offload, F16 KV, greedy, 32 generated tokens |
| 3B/8B capacity | 32,768 tokens |
| 14B capacity guard | 512 tokens |
| Candidate public SHA-256 | `4d4758c3a356e023fbaf1711c97e7e5e5764eaccd9956dd08a65a58df7fdf6f9` |
| Candidate profiler SHA-256 | `4a4698168637c65dc2ab6648f35012bc77f29d49395b9a12a8b38d9a549ba9db` |

Ignored raw measurements are under `target/phase27-*`. Phase 26's immutable
public and profiler executables provide the baseline bookends. Nothing was
pushed.

## Hypothesis and invariant

At the Phase 26 8B/28K checkpoint, Q4 Matrix2 Q/K/O/gate/up plus the Q4 halves
of V/down consumed about 47.9 seconds, 60% of GPU prefill. A 10% local gain
therefore predicted roughly 6% whole-request improvement.

Every lane owns an aligned 16-byte payload span. The old inner loop called
`gb()` 16 times; four adjacent calls addressed the same `uint`. The candidate
loads that `uint` once, extracts four bytes, and retains the original slot order.
The critical invariant is that every shared-memory FP16 weight bit remains
identical before `coopMatLoad`.

The five focused Q4 prefill oracles pass without tolerance changes, including
3B Matrix2, 8B gate/up, large-K down, metadata fallback, and small-output tail
coverage.

## 8B/128 causal attribution

The first candidate profile was cold and invalid: untouched categories slowed
broadly. The required warmed rerun follows a baseline warm-up and isolates the
Q4 movement. Both runs issue 674 prefill dispatches and 538 barriers.

| Category | Baseline ms | Packed ms | Change |
|---|---:|---:|---:|
| Q Q4 | 24.818 | 21.566 | -13.1% |
| K Q4 | 7.785 | 7.745 | -0.5% |
| V Q4 half | 3.875 | 3.297 | -14.9% |
| O Q4 | 24.525 | 21.321 | -13.1% |
| Gate Q4 | 76.559 | 65.443 | -14.5% |
| Up Q4 | 76.756 | 65.808 | -14.3% |
| Down Q4 half | 42.869 | 37.780 | -11.9% |
| Full GPU prefill | 316.469 | 279.851 | **-11.6%** |

K has only 1,024 output channels and is already underfilled. Untouched
attention and Q6 categories move by only a few percent with clocks. The seven
large Q4 categories all move in the predicted direction, proving that packed
payload extraction—not global scheduling—causes the gain.

## Public measurements

| Model / context | Baseline TTFT ms | Packed TTFT ms | Change | Decode guard |
|---|---:|---:|---:|---:|
| 3B / 128 | 96.78 | 92.98 | -3.93% | 55.98 -> 56.27 tok/s |
| 3B / 28K | 35,810.88 | 35,148.33 | -1.85% | 32.21 -> 32.28 tok/s |
| 8B / 128 | 282.73 | 260.05 | **-8.02%** | 26.27 -> 26.27 tok/s |
| 8B / 2K | 4,415.73 | 4,073.85 | **-7.74%** | 25.43 -> 25.43 tok/s |
| 8B / 28K | 80,010.92 | 75,248.31 | **-5.95%** | 17.83 -> 17.85 tok/s |
| 14B Instruct / 128 | 4,482.33 | 4,491.16 | +0.20% | 17.30 -> 17.21 tok/s |
| 14B Reasoning / 251 | 8,816.82 | 8,777.20 | -0.45% | 17.13 -> 17.20 tok/s |

Candidate CV is 0.61% or lower at short points and 0.35% or lower at long
points. The 14B movements are noise and remain inside the 1% guard.

Against the pinned llama.cpp references, final prefill ratios are now 2.16x
for 3B/128, 2.50x for 3B/28K, 2.38x for 8B/128, 3.44x for 8B/2K, and 3.03x
for 8B/28K. The change is material but not sufficient for maturity.

## Correctness and validation

- Five focused Q4_K batched CPU/Vulkan oracles pass.
- All six 3B/8B/14B instruct/reasoning artifacts exactly match their acquired
  128-step greedy token sequence.
- 3B/8B use 32,768-token capacity; 14B uses its qualified 512-token capacity.
- Full release Vulkan-hybrid tests pass with the same ignored authenticated
  cases as Phase 26.
- Workspace clippy passes for all targets with warnings denied.
- Formatting and documentation-contract tests pass.
- No allocation, dependency, public API, tolerance, or model artifact changed.

## Continuation decision

After the retained change, the 8B/28K target is at most 37.26 seconds and the
measured request is 75.25 seconds: another 37.99 seconds must be removed.
The unchanged attention category is about 21.3 seconds, so attention cannot
close the target alone even at zero cost. Q6 Matrix2 already uses packed payload
loads and contributes only about 8.9 seconds in the Phase 26 attribution.

The remaining Q4 Matrix2 family is still the largest coherent prefill target,
but another byte-load cleanup cannot provide the required factor. Meeting the
target through Q4 alone would require roughly a 2.5x local speedup after this
checkpoint. The next cycle therefore needs a materially new exact work-
ownership or execution representation with acceptable 12 GiB capacity—not
another M128 sweep, direct canonical callback, metadata-only cache, fusion, or
lossy native format already rejected in Phases 21–25.

