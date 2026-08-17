<!--
This document owns the Phase 29 global Vulkan stop: final decode experiments,
restored-tree proof, cross-domain quantitative exhaustion, target status, and
the conditions that would reopen the investigation.
-->

# Vulkan Global Performance Exhaustion — Phase 29

## Outcome

Phase 29 retains no production change. It brackets the remaining exact Q6_K
decode ownership, reconstructs and optimizes the fast Q8/DP4A Q4_K route, and
maps its numerical sensitivity across broad projection-shape families. The
fastest candidate improves 8B/128 decode by 40.0%, but every useful numerical
subset changes the acquired greedy sequence. All candidates were removed.

The final diagnostics-free executable is byte-identical to the qualified Phase
28 binary. Consequently, Phase 28's GQA vectorization remains the last retained
change and its six-model exact-sequence matrix remains the final quality state.

The competitive target is not met. This phase declares **GLOBAL STOP by
quantitative exhaustion**, not success by target attainment. Across 29 phases,
the investigation has measured or analytically bounded exact projection
ownership, numerical projection routes, Matrix2 geometry and representation,
long-attention ownership, fusion, dispatch, allocation, synchronization, and
capacity. The remaining target-sized step requires changing the numerical
execution contract that the unchanged quality gates reject.

## Reproducibility

| Item | Value |
|---|---|
| Date | 2026-08-17 |
| Branch | `perf/systematic-llamacpp-gap` |
| Final production commit | `e21fc12` |
| Phase 28 report commit | `0a564df` |
| GPU / driver | NVIDIA GeForce RTX 3060 12 GiB / 595.84 |
| Runtime | pure Vulkan, full offload, F16 KV, greedy |
| Final public SHA-256 | `01701aee41d5c74f490f26c21184b581c45e0ee30e5c39bccd8d8c61534747d4` |
| Phase 28 public SHA-256 | `01701aee41d5c74f490f26c21184b581c45e0ee30e5c39bccd8d8c61534747d4` |

The identical hashes were produced by fresh release builds after every Phase
29 candidate was restored. The source worktree was clean before this report;
nothing was pushed.

## Phase 29 experiment register

| ID | Candidate | Local/public result | Quality | Decision |
|---|---|---|---|---|
| E1 | cooperative DP4A, byte-extracted Q4 | 8B/128 26.42 -> 20.72 tok/s | local oracle passes | REJECT |
| E2 | cooperative DP4A, packed Q4 words | 26.53 -> 25.81 tok/s | local oracle passes | REJECT |
| E3 | eight-lane packed DP4A | 26.25 -> 36.76 tok/s, +40.0% | local oracle passes | numerical screen only |
| E4 | DP4A expanding shapes, gate/up | first five IDs match; token 6 differs | 128-ID gate fails | REJECT |
| E5 | DP4A contracting shapes | first ID matches; sequence collapses | 128-ID gate fails | REJECT |
| E6 | DP4A balanced shapes | immediate sequence divergence | 128-ID gate fails | REJECT |
| E7 | Q6_K two lanes per row | 26.27 -> 17.61 tok/s, -33.0% | Q6 oracle passes | REJECT |

E1–E3 establish the mechanism rather than relying on the old Phase 26 binary.
Aligned packed-word loads recover almost all of E1's loss. Replacing the
per-superblock subgroup reductions with eight independent Q4 sub-block owners
then exposes the integer-dot speedup. E4–E6 show that the failure is not caused
only by combining unrelated projections: gate/up alone diverges, as do the two
other broad shape families. Role- or layer-specific selection would add durable
model-graph policy without evidence that any target-sized subset survives.

E7 closes the remaining exact Q6 lane count. The two-lane dependency chain is
33% slower; the prior eight-lane candidate made down 5.2% slower. Four lanes is
therefore bracketed by measured lower and upper ownership points.

## Final target matrix

| Workload | Graph Horizon | llama.cpp | Final ratio | Contract limit | Status |
|---|---:|---:|---:|---:|---|
| 3B prefill / 128 | 92.98 ms | about 43.05 ms | 2.16x | 1.5x | miss |
| 3B prefill / 28K | 35.15 s | about 14.06 s | 2.50x | 1.5–1.7x | miss |
| 8B prefill / 128 | 260.05 ms | 109.44 ms | 2.38x | 1.5x | miss |
| 8B prefill / 2K | 4.07 s | 1.19 s | 3.44x | 1.5–1.7x | miss |
| 8B prefill / 28K | 75.10 s | 24.84 s | 3.02x | 1.5–1.7x | miss |
| 3B decode / 128 | 56.74 tok/s | 102.78 tok/s | 1.81x latency | 1.3–1.4x | miss |
| 3B decode / 28K | 36.52 tok/s | 50.02 tok/s | 1.37x latency | 1.3–1.4x | inside broad band |
| 8B decode / 128 | 26.64 tok/s | 50.74 tok/s | 1.90x latency | 1.3–1.4x | miss |
| 8B decode / 28K | 19.52 tok/s | 27.69 tok/s | 1.42x latency | 1.3–1.4x | near miss |
| 14B decode / capacity guard | 17.23 tok/s | 36.25 tok/s | 2.10x latency | 1.3–1.4x | miss |

All ratios use the pinned llama.cpp commit and same quantized model artifacts.
The 14B row remains limited to context capacity 512 on this 12 GiB GPU.

## Prefill exhaustion bound

The final 8B/28K profile accounts for 99.97% of 74.90 GPU seconds and measures
75.20 seconds wall time. A 1.5x target permits 37.26 seconds, so 37.94 seconds
must be removed.

The current category totals and optimistic practical removals are:

| Domain | Current s | Optimistic practical removal | Optimistic floor s | Evidence |
|---|---:|---:|---:|---|
| attention | 21.38 | 5.39 | 15.99 | QK/AV attribution and measured llama-rate floor |
| gate + up | 26.69 | 1.78 | 24.91 | M64 -> M128 reuse/resource model |
| down | 15.15 | 0.55 | 14.60 | acquired M128 and mixed-Q4/Q6 bound |
| Q/K/V/O | 10.30 | 0.50 | 9.80 | larger-BN and direct-callback screens |
| all other work | 1.37 | 1.37 | 0.00 | impossible-delete bound |
| **total** | **74.89** | **9.59** | **65.30** | deliberately optimistic |

Even deleting every residual/normalization/embedding/logits/control cost leaves
an optimistic same-numeric floor of 65.30 seconds, 28.04 seconds above the
37.26-second target. This is a 1.75x ratio to the target and 2.63x to llama.cpp.
The bound is intentionally more generous than the practical implementation
model; it is not presented as a physical law.

Closing the remaining 28.04 seconds requires the execution-contract changes
that distinguish the competing path: larger M/N ownership with FP16
accumulation or a persistent lower-precision execution representation. The
former fails four of five unchanged Q4 numerical oracles in the direct Phase 28
screen. Persistent FP16 gate/up views exceed the 12 GiB full-offload capacity;
an exact transient view is 89% slower at 128 and 33% slower at 512. N64, K256,
M128, direct canonical callbacks, metadata caches, fusion, Q staging, Q64
attention, and persistent attention accumulators have all measured negative or
fallen below their Amdahl gates. Sparse/windowed and narrower representations
do not preserve the accepted dense F16 inference contract.

## Decode exhaustion bound

At 8B/128, reaching 1.3x requires at least 39.03 tok/s, a 46.5% gain over the
final 26.64 tok/s. Attention is negligible there. Exact Q4 ownership is
bracketed at 4/8/16/32 lanes, exact Q6 at 2/4/8 lanes, direct activation reads
were retained only where positive, packed logits loads are retained, and
subgroup-only reductions regress. The only measured candidate of target scale
is E3 at 36.76 tok/s, and it changes the model sequence in every useful broad
projection family.

At 8B/28K, 1.4x requires 19.78 tok/s and 1.3x requires 21.30 tok/s. The final
19.52 tok/s is close to the broad threshold, but attention's 10.8 ms/token F16
traffic floor leaves about 2.25 ms/token theoretical local headroom and no
untried exact ownership: history split 4/8/16, Q reuse, Q staging, pre-scaling,
subgroup state, and four-head vector state are measured. The remaining strict
1.3x gap also requires projection work, returning to the exhausted exact or
quality-failing numerical routes above.

## Hypothesis-class closure

| Class | Measured closure |
|---|---|
| Q4 decode ownership | direct-cache KEEP; lanes 4/8/16/32 bracketed; subgroup reduction negative |
| Q6 decode ownership | packed loads KEEP; lanes 2/4/8 bracketed; direct-cache negative |
| numerical decode | serial/cooperative/packed DP4A measured; broad sensitivity map fails IDs |
| Matrix2 rows/columns/K | M32/M64 retained; M128, N64, K256 negative |
| Matrix2 representation | packed Q4 KEEP; FP16 accumulator fails; exact transient/persistent views closed |
| prefill attention | Q staging, Q64, split ownership, persistent state, softmax, pre-scale measured |
| approximation | sparse/windowed, INT8/native low precision fail quality or dense-F16 contract |
| orchestration | allocations zero steady-state; CPU/dispatch/fusion ceilings below target |
| capacity | 3B/8B qualified to 32K; 14B long context blocked by 12 GiB, not hidden fallback |

## Final correctness and reopening condition

The final state is exactly Phase 28: six authenticated artifacts retain their
128-step greedy IDs, focused Q4/Q6 and attention parity gates pass without
tolerance changes, and no model, dependency, public API, allocation policy, or
KV representation changed.

Final release validation passes: root 166/166, engine 231 passed with four
intentional authenticated/capability ignores, integration 6 passed with one
authenticated ignore, and semantic 12 passed with one authenticated ignore.
Workspace all-target clippy passes with warnings denied; formatting,
`git diff --check`, and the documentation contract pass.

Reopen this investigation only with evidence that changes one exhausted bound:

- a quality contract explicitly permitting different accumulated token IDs;
- hardware capacity sufficient for persistent exact execution views;
- a supported cooperative-matrix primitive with larger ownership and FP32
  accumulation that avoids the measured occupancy cliff; or
- a new algorithm with a quantified whole-request bound below 65.30 seconds,
  not another local tile, staging, barrier, cache, or precision variant already
  covered above.

Absent one of those changes, further experiments repeat a closed class or have
an Amdahl ceiling far below the remaining target gap.
