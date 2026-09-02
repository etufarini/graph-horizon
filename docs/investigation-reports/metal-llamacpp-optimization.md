<!--
This report is the persistent checkpoint for Metal competition with llama.cpp.
It owns the comparable endpoint matrix, gap attribution, Amdahl ranking,
experiment decisions, qualification evidence, and reproduction commands; it
defines no runtime API or backend support contract.
-->

# Metal / llama.cpp competitive optimization

## Mission start

| Field | Value |
|---|---|
| Date | 2026-08-20/21 Europe/Rome |
| Metal competitive baseline | `8deb73464d34d6da0bea9dce7a0c3fb6dcfaeaa3` |
| Starting branch | `perf/metal-vulkan-parity` |
| Experiment branch | `perf/metal-llamacpp-competitive` |
| Final tested retained state | `aa6f176` (M10 reverted; M07 behavior retained) |
| Initial worktree | clean |
| llama.cpp reference | `2b63e0610bbc2be990ae1360d5256efcdc3f9efb` (build 10237) |
| Repository publication | mission closed locally; branch tip `5375cf3` was later pushed and merged into `main` by PR #34 (`87e3e4b`) |

At mission start, repository history contained no qualified Metal revision newer
than `8deb734`. The Vulkan parity mission was acquired history and was not a
performance target. Its report is `metal-vulkan-optimization-parity.md`; the
earlier hardware-specialization evidence remains in Git history. Rejected
occupancy, FP16 staging, query-reuse, and wider-split variants are not repeated
without new evidence.

## Qualified tuple

- Apple silicon validation host with 10 GPU cores, 24 GB unified memory, Metal 4, macOS 26.3.
- Both diagnostics-free release binaries were rebuilt from the SHAs above.
- llama.cpp was built with `GGML_METAL=ON`, its embedded Metal library, full
  GPU offload, F16 K/V, flash attention enabled, batch 64, and microbatch 64.
- Graph Horizon used pure Metal, F16 K/V, context allocation 32,768, its
  qualified 64-row request ownership, greedy sampling, and 32 requested tokens.
- Inputs contain the exact same token IDs: `[1, 3, 1097]`, token `1261`
  repeated `N - 5` times, then `[1032, 4]`. llama.cpp reports `cache_n=0` and
  `prompt_n=N` for every retained request.
- Each workload uses one warm-up and three measured requests. Long workloads
  run serially because the validation host is fanless. CV above 1% is retained
  and identified rather than filtered.
- Graph Horizon TTFT includes tokenization, first sampling, and the first public
  text delta. llama.cpp `prompt_ms` is prompt evaluation only. Graph Horizon
  decode counts public text deltas; llama.cpp counts model tokens. These event
  boundaries prevent an exact tail comparison and slightly disadvantage the
  Graph Horizon TTFT number.
- The repository's current representative matrix is a complete 3B context
  curve plus 128/2K size checks for 8B and 14B. Reasoning variants have the
  same performance-relevant tensor shapes and remain correctness artifacts.

An initial control forced llama.cpp flash attention off to match the historical
Metal study. Source inspection showed this disabled a current production fast
path, so those measurements are not the primary competitive reference. They
are retained below only to isolate the structural value of llama.cpp's fused
attention route. No Graph Horizon endpoint was rerun between the controls; its
diagnostics-free binary and source revision are unchanged.

All six catalogued GGUF files were opened read-only and their SHA-256 values
matched `support/models.tsv` before measurement.

## Fresh competitive matrix

Latency ratio is Graph Horizon time divided by llama.cpp time. A decode ratio
below 1.0 means Graph Horizon is faster. There is no separate numeric
performance contract, so the competitive gap is measured against parity.

| Model | Workload | Ours | llama.cpp | Ratio | Absolute gap |
|---|---|---:|---:|---:|---:|
| 3B | prefill 128 | 486.77 ms | 290.42 ms | 1.68x | 196.35 ms |
| 3B | prefill 512 | 1,980.78 ms | 1,196.05 ms | 1.66x | 784.73 ms |
| 3B | prefill 2K | 8,777.22 ms | 5,190.83 ms | 1.69x | 3,586.39 ms |
| 3B | prefill 8K | 48,851.51 ms | 28,862.63 ms | 1.69x | 19,988.88 ms |
| 3B | prefill 28K | 335,020.18 ms | 173,509.39 ms | 1.93x | 161,510.79 ms |
| 8B | prefill 128 | 1,215.15 ms | 684.07 ms | 1.78x | 531.08 ms |
| 8B | prefill 2K | 20,896.70 ms | 12,144.89 ms | 1.72x | 8,751.81 ms |
| 14B | prefill 128 | 1,973.43 ms | 1,234.37 ms | 1.60x | 739.07 ms |
| 14B | prefill 2K | 35,542.08 ms | 22,634.22 ms | 1.57x | 12,907.86 ms |
| 3B | decode KV128 | 33.64 ms/token | 23.67 ms/token | 1.42x | 9.97 ms/token |
| 3B | decode KV512 | 40.00 ms/token | 25.34 ms/token | 1.58x | 14.66 ms/token |
| 3B | decode KV2K | 28.07 ms/token | 26.44 ms/token | 1.06x | 1.63 ms/token |
| 3B | decode KV8K | 39.35 ms/token | 31.51 ms/token | 1.25x | 7.85 ms/token |
| 3B | decode KV28K | 77.04 ms/token | 53.58 ms/token | 1.44x | 23.46 ms/token |
| 8B | decode KV128 | 63.90 ms/token | 48.76 ms/token | 1.31x | 15.14 ms/token |
| 8B | decode KV2K | 58.17 ms/token | 51.67 ms/token | 1.13x | 6.51 ms/token |
| 14B | decode KV128 | 96.06 ms/token | 76.66 ms/token | 1.25x | 19.40 ms/token |
| 14B | decode KV2K | 92.17 ms/token | 81.24 ms/token | 1.13x | 10.93 ms/token |

The largest relative and absolute prefill gap is 3B/28K at 1.93x and 161.51
seconds/request. No decode row is ahead against llama.cpp's production route.
The 3B/2K decode row is close at 1.06x and remains a regression guard; the
largest decode gap is 23.46 ms/token at 3B/28K.

## Prefill statistics

| Model/workload | Ours mean / median / SD / CV (ms) | llama.cpp mean / median / SD / CV (ms) |
|---|---:|---:|
| 3B/128 | 486.77 / 487.73 / 1.90 / 0.39% | 290.42 / 290.12 / 1.06 / 0.36% |
| 3B/512 | 1,980.78 / 1,980.65 / 3.90 / 0.20% | 1,196.05 / 1,201.31 / 12.58 / 1.05% |
| 3B/2K | 8,777.22 / 8,779.93 / 6.20 / 0.07% | 5,190.83 / 5,179.71 / 22.48 / 0.43% |
| 3B/8K | 48,851.51 / 48,866.60 / 73.17 / 0.15% | 28,862.63 / 29,184.77 / 2,338.59 / 8.10% |
| 3B/28K | 335,020.18 / 336,649.63 / 2,862.23 / 0.85% | 173,509.39 / 173,937.72 / 918.57 / 0.53% |
| 8B/128 | 1,215.15 / 1,214.62 / 1.19 / 0.10% | 684.07 / 686.22 / 7.81 / 1.14% |
| 8B/2K | 20,896.70 / 20,787.49 / 195.85 / 0.94% | 12,144.89 / 12,191.52 / 164.89 / 1.36% |
| 14B/128 | 1,973.43 / 1,969.48 / 21.72 / 1.10% | 1,234.37 / 1,227.33 / 29.84 / 2.42% |
| 14B/2K | 35,542.08 / 36,417.68 / 1,724.47 / 4.85% | 22,634.22 / 22,762.58 / 1,518.30 / 6.71% |

## Decode statistics and closure

| Model/history | Ours tok/s (mean / median / SD / CV) | llama ms/token (mean / median / SD / CV) | Status |
|---|---:|---:|---|
| 3B/128 | 29.73 / 29.81 / 0.15 / 0.51% | 23.67 / 23.66 / 0.06 / 0.26% | GAP |
| 3B/512 | 25.00 / 25.00 / 0.03 / 0.10% | 25.34 / 24.25 / 1.98 / 7.82% | GAP |
| 3B/2K | 35.62 / 35.62 / 0.15 / 0.42% | 26.44 / 27.24 / 1.43 / 5.39% | NEAR |
| 3B/8K | 25.41 / 25.50 / 0.21 / 0.82% | 31.51 / 31.45 / 0.14 / 0.46% | GAP |
| 3B/28K | 12.98 / 13.08 / 0.28 / 2.19% | 53.58 / 54.34 / 1.40 / 2.60% | GAP |
| 8B/128 | 15.65 / 15.65 / 0.00 / 0.01% | 48.76 / 48.73 / 0.14 / 0.28% | GAP |
| 8B/2K | 17.19 / 17.23 / 0.09 / 0.51% | 51.67 / 51.69 / 0.21 / 0.40% | GAP |
| 14B/128 | 10.41 / 10.41 / 0.01 / 0.12% | 76.66 / 76.66 / 0.01 / 0.02% | GAP |
| 14B/2K | 10.85 / 10.83 / 0.15 / 1.35% | 81.24 / 80.50 / 2.17 / 2.67% | GAP |

## Memory checkpoint

With context capacity 32,768, 3B Graph Horizon reached 4,370,087,936 bytes
maximum RSS and 5,756,454,952 bytes peak footprint. The live llama.cpp 3B
server RSS after the 28K matrix was 5,641,984 KiB. At 14B the corresponding
RSS values were 13,493,846,016 bytes for Graph Horizon and 13,362,096 KiB for
llama.cpp. These are process-level unified-memory observations, not isolated
GPU allocation counters.

## Fresh Graph Horizon attribution

The isolated `metal-profile` feature samples adjacent operation categories at
Metal encoder boundaries. Ordinary `metal` builds do not contain this path.
The diagnostic endpoint uses one request and four generated tokens; its timing
is never used as diagnostics-free throughput evidence.

| 3B prefill | GPU command | Accounted | Attention | Seven matrices | Other kernels | Residual |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 573.78 ms | 90.08% | 7.50 ms | 490.23 ms | 19.11 ms | 56.94 ms |
| 512 | 2,077.27 ms | 96.22% | 82.17 ms | 1,857.97 ms | 58.59 ms | 78.54 ms |
| 2K | 8,951.06 ms | 97.99% | 1,176.78 ms | 7,358.11 ms | 235.92 ms | 180.25 ms |
| 28K | 319,691.04 ms | 99.44% | 213,416.25 ms | 101,105.91 ms | 3,366.52 ms | 1,802.37 ms |

The matrix column is Q, K, V, O, gate, up, and down. At 2K it is 83.9% of
diagnostics-free TTFT; at 28K attention crosses over to 63.7% of TTFT. The 28K
CPU record and submit totals are 646.05 ms and 4.07 ms against 320,016.57 ms of
wait time. CPU orchestration is not on the critical-path floor.

The 128 point accounts for only 90.1%; it retains the explicit 56.94 ms
residual and is not used for a redesign bound. The prioritized 512/2K/28K
points exceed the 95% attribution gate.

## Comparator component inference and structural audit

The production llama.cpp endpoints fit a fixed + linear + quadratic model:

```text
T_llama(N) = -0.129946 + 0.00242700 N + 1.34806e-7 N^2 seconds
```

This is an inference from five end-to-end points, not direct llama.cpp GPU
counter attribution. It predicts every 2K--28K point within 0.22 seconds and
assigns about 67.96 seconds of the 28K endpoint to linear work and 105.69
seconds to quadratic work. The flash-off control has a similar linear term but
a 161.80-second quadratic term at 28K. Flash attention therefore removes about
56.1 seconds from llama.cpp's long-context path while leaving short prefill
nearly unchanged.

Routing was checked before selecting a new kernel:

| Operation | Representative shape | Expected path | Actual path / reason |
|---|---|---|---|
| Q projection | M=64, N=4096, K=3072, Q4_K | wide cooperative matrix | `metal_matmul_batched_wide`; pure Metal, aligned shape |
| K/V projection | M=64, N=1024, K=3072, Q4_K | wide cooperative matrix | `metal_matmul_batched_wide`; pure Metal, aligned shape |
| O projection | M=64, N=3072, K=4096, Q4_K | wide cooperative matrix | `metal_matmul_batched_wide`; pure Metal, aligned shape |
| gate/up | M=64, N=9216, K=3072, Q4_K | wide cooperative matrix | `metal_matmul_batched_wide`; pure Metal, aligned shape |
| down | M=64, N=3072, K=9216, Q6_K | wide cooperative matrix | `metal_matmul_batched_wide`; pure Metal, aligned shape |
| GQA prefill attention | rows=64, heads=32/8, dim=128, F16 KV | SIMD-matrix tile | `metal_attention_prefill_matrix`; exact qualified route |
| GQA decode | row=1, heads=32/8, dim=128, F16 KV | split or native GQA | qualified native/split routes by history |

No dominant operation is falling through a generic route. Graph Horizon's
matrix tile is 64 outputs by 64 prompt rows with eight SIMD groups; it stages
simple row-major K32 weights/activations and a 16 KiB FP32 result tile. Current
llama.cpp on pre-M5 Apple hardware uses a 64-output by 32-row tile with four
SIMD groups, 6 KiB of swizzled weight/activation threadgroup storage, and
direct FP32 result stores. Graph Horizon's previously rejected occupancy-only
variants did not test this swizzled data mapping.

Graph Horizon attention assigns one 1,024-thread workgroup to each KV head and
loops serially over K/V in eight-token tiles. For this 64-row, 32-head shape,
llama.cpp flash attention dispatches one workgroup per eight query rows and
query head, uses 64-column cache tiles and four or eight SIMD groups per
workgroup, and prepares causal-mask blocks separately. That exposes 256
workgroups per layer instead of Graph Horizon's eight, at the cost of less GQA
cross-head K/V sharing. The structural gap is work ownership, history
parallelism, and tile width, not an unused Graph Horizon route.

## Amdahl bounds and global ranking

For the first matrix candidate at 3B/2K:

```text
T = 8.777 s
component = 7.358 s
f = 0.838
practical floor ~= 4.970 s (llama linear fit)
S_local ~= 1.48x
realistically removable = 2.388 s
predicted final = 6.389 s
predicted whole-workload improvement = 27.2%
```

At 512 the same comparator-local speedup removes about 0.615 seconds and
predicts a 31.0% whole-request improvement. At 28K it removes about 33.15
seconds, just under 10% of that attention-dominated endpoint, before counting
reuse across 8B and 14B.

For a multi-workgroup attention redesign at 3B/28K:

```text
T = 335.020 s
component = 213.416 s
f = 0.637
practical floor ~= 105.688 s (llama quadratic fit)
S_local ~= 2.02x
realistically removable = 107.728 s
predicted final = 227.292 s
predicted whole-workload improvement = 32.2%
```

The attention bound clears the complex-redesign gate, but has higher numeric,
state-reduction, and maintenance risk. The first candidate is the matrix
threadgroup mapping because it affects every prompt and model, has a 27--31%
predicted gain at stable prioritized points, preserves representation and
arithmetic, and is a bounded category-K experiment. The attention redesign is
ranked second and remains mandatory if the matrix direction closes or is kept.
Short decode is third: its largest measured gap is 23.46 ms/token, but the
matrix experiment also changes the shared quantized decode source only if
explicitly routed, which this first candidate will not do.

## Experiment registry

### M01 — swizzled wide-projection staging: KEEP

M01 changes only `metal_matmul_batched_wide` in the existing category-K
projection file. It preserves the 64-output by 64-row work ownership, canonical
Q4_K/Q6_K bytes, FP32 accumulators, F16 output, and production route. Weight
and activation K8 tiles are swizzled so each SIMD matrix load reads contiguous
threadgroup addresses. There is no new buffer, dependency, API, or dispatch.

The first local form incorrectly used a four-block activation stride even
though 64 rows require eight blocks. The focused Q4/Q6 64-row oracle rejected
it before any benchmark. Correcting the stride to eight made the same oracle
pass; no result from the rejected form was measured or retained.

| Diagnostics-free A/B/A tuple | Baseline A | M01 | Baseline bookend | M01 change | M01 CV |
|---|---:|---:|---:|---:|---:|
| 3B/128 TTFT | 482.73 ms | 451.58 ms | 483.35 ms | -6.52% | 0.06% |
| 3B/512 TTFT | 1,971.90 ms | 1,845.45 ms | 1,972.05 ms | -6.42% | 0.31% |
| 3B/2K TTFT | 8,737.15 ms | 8,225.25 ms | 8,736.24 ms | -5.86% | 0.08% |
| 8B/128 TTFT | 1,187.96 ms | 1,108.35 ms | 1,186.63 ms | -6.64% | 0.09% |
| 8B/512 TTFT | 4,809.49 ms | 4,490.01 ms | 4,804.85 ms | -6.60% | 0.14% |

Decode throughput is unchanged within noise on all five tuples. An earlier
8B/2K sequence had 10.8% drift between baseline bookends and was classified
inconclusive rather than used for selection; the cooled 8B/512 row above is
the bounded larger-model replacement.

Required reprofiling confirms the intended component rather than an endpoint
artifact:

| Profile tuple | Baseline seven matrices | M01 seven matrices | Local change | Baseline GPU | M01 GPU | GPU change |
|---|---:|---:|---:|---:|---:|---:|
| 3B/512 | 1,857.97 ms | 1,725.80 ms | -7.11% | 2,077.27 ms | 1,946.79 ms | -6.28% |
| 3B/2K | 7,358.11 ms | 6,828.50 ms | -7.20% | 8,951.06 ms | 8,416.51 ms | -5.97% |

Attention was 82.31 ms at 512 and 1,176.05 ms at 2K after M01, effectively
unchanged from 82.17 ms and 1,176.78 ms. The temporary profiler was reverted
again after measurement. M01 is retained at `2cd6043` because it clears the 5%
whole-workload threshold at every thermally valid point, generalizes across 3B
and 8B, and reduces rather than expands the kernel's data-movement model.

The focused Q4/Q6 oracle, complete pure-Metal tests, complete Metal-hybrid
tests, `git diff --check`, and pure-Metal clippy with warnings denied pass.
Metal-hybrid clippy with warnings denied is blocked by two pre-existing unused
`simd` arguments in the CPU attention module; M01 does not touch that module
and Metal-hybrid's complete tests pass.

### M02 — query-head prefill-attention ownership: REJECT

M02 replaced the 1,024-thread, eight-KV-head dispatch with one 128-thread
workgroup per eight query rows and query head. Four SIMD groups retained matrix
QK/PV, read disjoint strided history tiles directly from device memory, kept
independent FP32 online-softmax states, and merged those states in 3.8 KiB of
threadgroup memory. This tested the principal ownership difference identified
in llama.cpp without adding a buffer, kernel family, or numerical-policy
change. The serial causal-attention oracle passed at base 0 and base 512.

The 32.2% long-context Amdahl prediction did not materialize at medium context:
relative to retained M01, 3B/512 improved only 1.44% (1,845.45 to 1,818.81 ms)
and 3B/2K improved 3.08% (8,225.25 to 7,972.12 ms). Both candidate CVs were
below 0.1%, so these sub-gate results are not noise. A cool single 8K probe did
show the expected crossover, improving 47,681.63 to 43,900.42 ms (7.93%).

The required sustained 8K A/B/A could not validate that probe on the fanless
device. Baseline A was 48,277.26 ms (1.59% CV), M02 was 50,117.17 ms (1.81%
CV), and the baseline bookend was 50,233.84 ms (4.21% CV). The 4.05% baseline
drift makes the sequence inconclusive, while the stable 512/2K evidence remains
below the economic gate for this moderate, higher-traffic redesign. M02 is
therefore rejected: extra per-query-head K/V traffic and merge complexity are
not justified by a qualified whole-workload gain. Commits `ac6edc9` and
`d458cd6` retain the experiment and its non-destructive removal; production is
back to M01.

### M03 — remove swizzled-load SIMD barriers: REJECT

M03 removed the three `mem_none` SIMD-group barriers around each K8 matrix-load
step in M01. The K32 threadgroup tiles are immutable during this phase, so this
was a trivial, broadly reusable test with unchanged arithmetic and traffic; its
predicted whole-request range was 1.7--4.0% at 2K.

The Q4/Q6 oracle passed, but the 3B/512 A/B/A was 1,899.01 ms for M01,
1,910.15 ms for M03, and 1,896.78 ms for the M01 bookend. Candidate CV was
0.06% and both controls were below 0.23%. The barriers are therefore retained
as useful compiler scheduling hints: removing them produces no gain and a
small repeatable regression. Exact candidate `2a3a55f` was removed by
`f124a37`.

### M04 — cooperative C64 query-head attention: KEEP

M04 keeps the query-head workgroup granularity tested by M02 but removes its
history split and final state merge. Four SIMD groups cooperatively form one
64-column score tile; each group owns two query rows for online softmax and 32
output dimensions for PV. The kernel retains matrix QK/PV, FP32 accumulation,
FP16 KV/probabilities/output, the exact causal predicate, and one final
normalization. Static threadgroup state is about 9.3 KiB and replaces the
production kernel's 1,024-thread/large-register ownership without a new buffer,
pipeline, dependency, or API.

The unchanged serial attention oracle passes at base 0 and base 512. At 2K,
temporary component profiling measured attention at 910.64 ms versus M01's
1,176.05 ms, a 22.57% local reduction. Projections remained effectively
unchanged and total profiled GPU time fell from 8,416.51 to 8,148.40 ms
(3.19%), matching the medium-context Amdahl limit. Diagnostics were reverted
afterward.

| Diagnostics-free tuple | M01 A | M04 | M01 bookend | M04 vs control mean | M04 CV |
|---|---:|---:|---:|---:|---:|
| 3B/4K TTFT | 18,708.68 ms | 17,681.80 ms | 18,689.79 ms | -5.44% | 0.06% |

The two controls differ by 0.10%. Decode is unchanged within noise because M04
is prefill-only. The already-recorded stable screens were 1,824.79 ms at 512
and 7,961.28 ms at 2K (CV below 0.1%); their roughly 3--4% gains are expected
below the attention crossover. M04 is retained as exact production commit
`01a0c45`. It passes 139 unit tests (2 ignored), 5 family-agnostic tests (1
ignored), and 12 semantic tests (1 ignored) under pure Metal; Metal-hybrid has
203 unit tests with the same integration counts and ignores. Pure-Metal clippy
with warnings denied and `git diff --check` also pass.

### M05 — K64 wide-projection staging: REJECT

M05 doubled the swizzled weight/activation tile from K32 to K64 while retaining
the 64-by-64 output tile, dequantization, arithmetic, and traffic. It halved
full-threadgroup synchronization frequency but raised static threadgroup memory
from 24 to exactly 32 KiB. The Q4/Q6 oracle passed.

The 3B/512 A/B/A was 1,826.51 ms for M04, 2,149.93 ms for M05, and 1,827.80 ms
for the M04 bookend. M05 regressed 17.66%; candidate CV was 0.07% and both
controls were below 0.3%. The exact-capacity tile causes a resource/scheduling
cliff that overwhelms the saved barriers and would also tighten device
admission. Exact candidate `3a2d34d` was removed by `b02ae2a`; K32 and the 24
KiB footprint remain production.

### M06 — FP32-backed prefill projection scratch: REJECT

M06 tested the largest remaining projection-occupancy hypothesis after M05.
The wide kernel stored its FP32 SIMD-matrix accumulators directly to device
memory, removing the 16 KiB threadgroup result tile and its final barrier. Pure
Metal prefill projection scratch widened from two to four bytes per element;
every RoPE, KV-write, attention, activation, and residual consumer explicitly
cast the stored value through `half`, preserving the established graph boundary.
Decode, hybrid placement, KV representation, weights, and attention output did
not change.

At 2K the current seven-matrix component is 6,828.21 ms. The comparator-linear
practical floor is about 4,970 ms, so the full projection design space has an
upper recoverable envelope of 1,858 ms, or 23.3% of the 7,961.28 ms workload.
M06 addressed only result staging and the occupancy it consumed, not the
dominant dequantization/matrix-feed work; it was tested as a prerequisite for
lower-footprint projection mappings rather than assuming that full envelope.

The focused oracle exercised the actual 64-row FP32 wide route for Q4_K and
Q6_K and compared its values after half-rounding with sequential F16 output.
It and all ten Metal kernel tests passed. The diagnostics-free screens were:

| Tuple | M04 A | M06 | M04 bookend | M06 vs control mean | M06 CV |
|---|---:|---:|---:|---:|---:|
| 3B/512 TTFT | 1,826.47 ms | 1,798.38 ms | 1,824.48 ms | -1.48% | 0.17% |
| 3B/2K TTFT | 7,933.81 ms | 7,838.91 ms | 7,965.07 ms | -1.39% | 0.07% |

The gain is real but far below the 5% moderate-redesign gate and costs doubled
projection scratch plus format-aware logic across 18 files. It is not an
economically justified production change while a large gap remains. Exact
candidate `743e4a9` was removed by `726201a`; the retained M04 binary and F16
scratch contract are restored.

Final matrices and reproduction commands follow after global reranking.

### M07 — Metal 4 cooperative-tensor projection: KEEP

The current llama.cpp source contains a `GGML_METAL_HAS_TENSOR` matrix path,
but deliberately disables it by default on pre-M5/pre-A19 hardware because its
FP32-destination implementation is not faster there. The structural reference
was still decisive: dequantize only one 64-by-K32 weight tile into about 4 KiB
of threadgroup memory, read activations directly from device memory, and let
Metal Performance Primitives own the cooperative destination. Graph Horizon's
F16 projection boundary makes a direct half destination profitable on this M4.

For 3B/2K, the pre-experiment bound was `T=7.961 s`, affected matrix fraction
`f=6.828/7.961=0.858`, and practical comparator floor 4.970 s. Therefore
`S_local=1.374`, realistically removable time was 1.858 s, predicted final time
was 6.103 s, and predicted whole-workload improvement was 23.3%.

M07 keeps canonical Q4_K/Q6_K weights and the request-local F16 scratch layout.
One 128-thread group dequantizes a complete 64-by-K32 weight tile, then a Metal
4 cooperative matmul reads token activations directly and stores its F16
destination. The qualified route is exact rows=64, aligned Q4/Q6 shapes, pure
Metal placement, and runtime Metal 4 family support. The retained legacy SIMD
wide kernel is the fallback; unsupported devices never create the tensor
pipeline. No file, dependency, persistent representation, or public API is
added.

| Diagnostics-free tuple | M04 A | M07 | M04 bookend | M07 vs control mean | M07 CV |
|---|---:|---:|---:|---:|---:|
| 3B/512 TTFT | 1,827.32 ms | 1,366.73 ms | 1,827.41 ms | -25.21% | 0.07% |
| 3B/2K TTFT | 7,947.47 ms | 6,111.52 ms | 7,954.20 ms | -23.14% | 0.02% |

The 2K endpoint is within 8.5 ms of its Amdahl prediction. Required reprofiling
measures the seven matrices at 4,960.32 ms versus M04's 6,828.21 ms, a 27.36%
local reduction and effectively the 4,970 ms practical floor. Attention is
unchanged at 910.27 ms versus 910.64 ms. Profiled total GPU time falls from
8,148.40 to 6,295.15 ms. At 512, the seven matrices total 1,261.38 ms and total
GPU time is 1,477.21 ms; the diagnostic path remains excluded from endpoints.

The cooperative destination changes accumulation ordering/precision, so it was
gated as a numerical candidate. Against the sequential FP32-accumulation/F16
oracle, maximum absolute projection differences are 0.00586 for Q4_K and
0.00195 for Q6_K, inside the unchanged 0.05 limit. Authenticated 3B teacher rows
with both F16 and INT8 KV produce all 16 exact oracle token IDs and pass the
teacher-forced top-two/lifecycle/sequential-request gates.

Complete qualification passes: pure Metal has 139 unit tests with 2 ignored,
5 family tests with 1 ignored, and 12 semantic tests with 1 ignored;
Metal-hybrid has 203 unit tests with the same integration counts and ignores.
Pure-Metal clippy passes with warnings denied. Hybrid retains two pre-existing
unused-`simd` warnings in CPU attention. The production implementation is
`bcefc59`; diagnostics were removed by `6322e33`.

### M08 — grouped-attention long-context crossover: REJECT

M08 tested whether the pre-M04 grouped kernel should return above 8K. The old
kernel loads each K/V tile once per KV head and shares it across four query
heads; M04 instead spends fourfold logical K/V traffic to expose one workgroup
per eight query rows and query head. Both kernels were already present, so the
candidate was one dispatch guard plus an extension of the serial attention
oracle to base 8,192. The oracle passed.

A back-to-back 3B/12K component trace rejected the crossover. Keeping M04 for
the whole prompt used 38,423.44 ms of attention and 76,409.14 ms total GPU
time. Switching only chunks whose base was at least 8,192 to grouped attention
used 52,544.54 ms of attention and 85,686.05 ms total GPU time: +36.75% in the
target component and +12.14% on the GPU interval. The control ran second, when
non-attention kernels were 12--15% slower from heat, yet its attention was
still 14.12 seconds faster. Increased occupancy is therefore worth more than
four-head K/V sharing on this M4. Exact candidate `fa1d6af` was removed by
`be0bbbc`; temporary profiling was removed by `a8b0f5f`.

### M09 — eight-group C64 attention: REJECT

M09 tested the last comparator-derived occupancy variant on the retained M04
kernel. Eight SIMD groups formed the same C64 score tile; each group owned one
softmax row and 16 output dimensions instead of two rows and 32 dimensions.
Traffic, arithmetic, FP32 online state, FP16 boundaries, dispatch count, and
9.3 KiB threadgroup allocation were unchanged. The serial oracle passed.

The diagnostics-free 3B/2K A/B/A TTFT was 6,261.90 ms for M07, 6,309.03 ms for
M09, and 6,292.25 ms for the M07 bookend. M09 regressed 0.51% against the
control mean; candidate CV was 0.07%. Decode was unchanged. Because the same
256-thread resource geometry applies to every history tile, there is no
evidence-based long-context crossover. Exact candidate `f0ed961` was removed
by `918b208`.

## Final diagnostics-free matrix

The final binary is SHA-256
`a166863382875d596f5cc71406c2d6782b8b15b95c58427350bfb951f7122c1c`.
It contains retained M01, M04, and M07 and no diagnostics or rejected
production code. Ratios and gaps use the same llama.cpp measurements from the
fresh baseline session.

| Model | Workload | Final ours | llama.cpp | Ratio | Parity contract | Status | Absolute gap | Ours vs initial |
|---|---|---:|---:|---:|---:|---|---:|---:|
| 3B | prefill 128 | 342.46 ms | 290.42 ms | 1.18x | <=1.00x | GAP | 52.04 ms | -29.65% |
| 3B | prefill 512 | 1,399.19 ms | 1,196.05 ms | 1.17x | <=1.00x | GAP | 203.14 ms | -29.36% |
| 3B | prefill 2K | 6,262.51 ms | 5,190.83 ms | 1.21x | <=1.00x | GAP | 1,071.68 ms | -28.65% |
| 3B | prefill 8K | 38,749.78 ms | 28,862.63 ms | 1.34x | <=1.00x | GAP | 9,887.15 ms | -20.68% |
| 3B | prefill 28K | 372,482.44 ms | 173,509.39 ms | 2.15x | <=1.00x | THERMAL GAP* | 198,973.05 ms | +11.18%* |
| 8B | prefill 128 | 907.79 ms | 684.07 ms | 1.33x | <=1.00x | GAP | 223.72 ms | -25.29% |
| 8B | prefill 2K | 16,067.86 ms | 12,144.89 ms | 1.32x | <=1.00x | GAP | 3,922.97 ms | -23.11% |
| 14B | prefill 128 | 1,672.76 ms | 1,234.37 ms | 1.36x | <=1.00x | GAP | 438.40 ms | -15.24% |
| 14B | prefill 2K | 27,085.42 ms | 22,634.22 ms | 1.20x | <=1.00x | GAP | 4,451.20 ms | -23.79% |
| 3B | decode KV128 | 34.28 ms/token | 23.67 ms/token | 1.45x | <=1.00x | GAP | 10.62 ms/token | +1.92% |
| 3B | decode KV512 | 40.92 ms/token | 25.34 ms/token | 1.61x | <=1.00x | GAP | 15.57 ms/token | +2.29% |
| 3B | decode KV2K | 28.84 ms/token | 26.44 ms/token | 1.09x | <=1.00x | NEAR | 2.39 ms/token | +2.71% |
| 3B | decode KV8K | 40.95 ms/token | 31.51 ms/token | 1.30x | <=1.00x | GAP | 9.44 ms/token | +4.05% |
| 3B | decode KV28K | 104.71 ms/token | 53.58 ms/token | 1.95x | <=1.00x | THERMAL GAP* | 51.13 ms/token | +35.92%* |
| 8B | decode KV128 | 65.45 ms/token | 48.76 ms/token | 1.34x | <=1.00x | GAP | 16.69 ms/token | +2.42% |
| 8B | decode KV2K | 59.49 ms/token | 51.67 ms/token | 1.15x | <=1.00x | GAP | 7.82 ms/token | +2.26% |
| 14B | decode KV128 | 98.33 ms/token | 76.66 ms/token | 1.28x | <=1.00x | GAP | 21.67 ms/token | +2.36% |
| 14B | decode KV2K | 95.24 ms/token | 81.24 ms/token | 1.17x | <=1.00x | GAP | 14.00 ms/token | +3.33% |

Final prefill dispersion was:

| Model/workload | Mean / median / SD / CV (ms) |
|---|---:|
| 3B/128 | 342.46 / 342.35 / 0.26 / 0.07% |
| 3B/512 | 1,399.19 / 1,399.40 / 1.84 / 0.13% |
| 3B/2K | 6,262.51 / 6,261.23 / 3.96 / 0.06% |
| 3B/8K | 38,749.78 / 38,018.14 / 3,261.56 / 8.42% |
| 3B/28K | 372,482.44 / 372,568.54 / 1,709.53 / 0.46%* |
| 8B/128 | 907.79 / 919.96 / 24.55 / 2.70% |
| 8B/2K | 16,067.86 / 16,133.18 / 595.52 / 3.71% |
| 14B/128 | 1,672.76 / 1,673.35 / 9.27 / 0.55% |
| 14B/2K | 27,085.42 / 27,256.16 / 1,049.36 / 3.87% |

`*` The 28K final run started after the complete model/context matrix and took
more than 22 minutes for warm-up plus three repetitions. Its low within-run CV
describes a stable throttled state, not cross-run comparability: it is 11.2%
slower than the initial binary even though M07 changes only projection prefill,
reduces the seven measured matrices by 27.36% at 2K, and leaves attention and
decode source unchanged. It is retained exactly as observed, but is not used
to infer a code regression or retained speedup. The 8K point is also thermally
noisy; a nominally cooled repeat was 42,181.82 ms with 13.35% CV. Both 8K runs
agree on direction versus the initial 48,851.51 ms but not on a precise gain.

At stable 3B points through 2K, the competitive prefill gap fell from
1.66--1.69x to 1.17--1.21x. The retained projection changes are prefill-only;
the roughly 2--4% decode movement through 8K is session drift rather than a
claimed regression. The raw thermally stressed 28K decode row has the same
qualification warning as its TTFT.

## Post-M07 attribution and attention local stop

M07 reaches the comparator-derived projection floor at 2K: its seven matrices
take 4,960.32 ms, versus the 4,970 ms practical linear floor inferred from
llama.cpp. Attention is then 910.27 ms and 14.46% of the 6,295.15 ms GPU
interval. Even halving that component could remove only 7.2% of the request,
below the 10% complex-redesign gate.

Long prefill remains the largest bottleneck. In the post-M07 12K all-M04 trace,
attention is 38,423.44 ms, 50.97% of accounted kernel time. The llama.cpp fit
assigns about 20.36 seconds to quadratic work at the same length, leaving a
theoretical 18.07-second attention envelope. This is not treated as an
implementation floor. The concrete mechanisms that could plausibly reach it
were tested:

- split query-head history and merge (M02): at most 3.08% at stable 2K and no
  qualified 8K win;
- four-group query-head C64 SIMD-matrix attention (M04): retained, -22.57%
  locally at 2K and -5.44% at the qualified 4K endpoint;
- grouped four-head K/V reuse above 8K (M08): +36.75% local attention time at
  12K, rejected;
- eight-group C64 occupancy (M09): +0.51% end-to-end at 2K, rejected.

Larger C tiles cannot keep score/probability, scaling, K/V, and FP32 online
output state within the 32 KiB M4 threadgroup contract. Current llama.cpp flash
attention also uses SIMD-group matrices rather than the Metal tensor path.
MPP cooperative matmul cannot retain its destination across the intervening
online-softmax nonlinearity; using it would require extra tile/state
materialization and has no comparator or measured evidence for a 10% request
gain. This is an explicit mechanism boundary, not a claim that quadratic
causal attention is inherently optimal.

Attention reaches a local stop: every identified ownership/occupancy mechanism
was retained or falsified, and another local tile/split variant has no evidence
above its economic gate. This does not close the global mission.

## M07 checkpoint reproduction and state

Build the diagnostics-free Graph Horizon endpoint with:

```sh
cargo build --locked --release --no-default-features --features metal \
  --example bench
```

For a requested prompt length `N`, the synthetic input is `"a "` repeated
`N - 4` times. For example, the exact 2K command is:

```sh
prompt=$(printf 'a %.0s' {1..2044})
target/release/examples/bench \
  "<model.gguf>" \
  --context 32768 --kv f16 --prompt "$prompt" \
  --max-tokens 32 --warmup 1 --reps 3
```

The llama.cpp reference was built at
`2b63e0610bbc2be990ae1360d5256efcdc3f9efb` with Metal enabled and run as a
fresh server for each model with full GPU offload, context 32,768, F16 K/V,
flash attention, batch/microbatch 64, greedy sampling, no prompt-cache reuse,
and the same generated text. The exact token IDs are recorded under
"Qualified tuple" so a tokenizer-level reproduction can verify equivalence.

The final source tree contains only retained M01/M04/M07 production changes,
tests, and this report. M02/M03/M05/M06/M08/M09 production candidates and all
temporary profilers are removed by ordinary revert commits. Complete pure
Metal and Metal-hybrid suites, the focused Q4/Q6 and attention oracles,
authenticated F16/INT8 llama.cpp parity, pure-Metal clippy with warnings
denied, formatting, and `git diff --check` pass. The mission closed before
publication; the retained state was later pushed and merged by PR #34.

## Completion audit: decode gap reopened

The full mission audit rejected the earlier global-stop claim. A fresh
route-equivalent `metal-profile` trace on final M07 source measured 14B/KV128
decode over 31 timed token transitions:

| Decode component | GPU total | Per token | Share of GPU interval |
|---|---:|---:|---:|
| Q/K/V/O projections | 411.75 ms | 13.28 ms | 12.92% |
| gate/up/down projections | 1,886.94 ms | 60.87 ms | 59.23% |
| all seven quantized projections | 2,298.69 ms | 74.15 ms | 72.15% |
| attention | 400.79 ms | 12.93 ms | 12.58% |
| logits | 167.55 ms | 5.40 ms | 5.26% |
| norm/RoPE/KV/elementwise/embedding | 76.10 ms | 2.45 ms | 2.39% |
| measured command residual | 242.78 ms | 7.83 ms | 7.62% |
| complete GPU interval | 3,185.90 ms | 102.77 ms | 100.00% |

CPU record and submit total only 35.74 ms across all 31 transitions; wait is
3,199.00 ms. Decode is GPU-critical. The diagnostics-free endpoint is 98.33
ms/token versus llama.cpp's 76.66 ms/token, a 21.67 ms/token competitive gap.
The profiler is attribution evidence only and was removed after capture.

### M10 structure checkpoint: per-format decode GEMV specialization

Current Q4_K/Q6_K arithmetic and two-SIMD/four-output-row geometry are nearly
identical to current llama.cpp. The structural difference is compilation:
llama.cpp has separate Q4_K and Q6_K entry points, while Graph Horizon's single
`metal_matmul` retains runtime Q4/Q6/F16/Q5 and output-precision branches. M10
tests only whether removing unrelated format bodies reduces instruction and
register pressure. Canonical weights, half activation/output boundaries, FP32
accumulation, geometry, traffic, and mixed-placement fallback remain unchanged.

```text
crates/graph_horizon_engine/src/backend/metal/
├── shaders/matmul.metal       (category K, no line limit)
├── pipeline.rs                (~200 productive lines, excluding tests)
└── kernels/matmul.rs          (~170 productive lines, excluding tests)
```

No new file, dependency, representation, buffer, or public API is warranted.
The two specialized pipeline entries are one operation and stay in the current
registry. The invariant is that format-specific dispatch accepts exactly its
canonical block layout and writes the same FP16/FP32 boundary. Main risks are
pipeline-index ordering, fallback drift, and numerical reordering; existing
Q4/Q6 scalar oracles gate the performance screen.

For 14B/KV128 diagnostics-free decode:

```text
T = 98.328 ms/token
component = 74.151 ms/token
f = 0.754
realistic S_local = 1.08x
theoretical parity budget for the component = 52.484 ms/token
practical candidate floor = 68.658 ms/token
realistically removable = 5.493 ms/token
recoverable competitive gap = min(5.493, 21.667) = 5.493 ms/token
predicted final = 92.835 ms/token
predicted whole-token improvement = 5.59%
```

The practical candidate floor reflects only an 8% compiler-specialization
gain, not the entire parity budget. M10 is a small, Apple-family-wide route and
clears the 5% moderate gate at that realistic bound.

### M10 result — per-format decode GEMV specialization: REJECT

The Q4_K/Q6_K scalar oracle and source-structure tests passed. The candidate
was then measured diagnostics-free at 14B/KV128 with one warm-up and three
requests in an A/B/A sequence:

| Position | TTFT | Decode | Decode CV |
|---|---:|---:|---:|
| M07 control A1 | 1,225.69 ms | 10.51 tok/s | 0.17% |
| M10 specialized entries | 1,227.66 ms | 10.51 tok/s | 0.07% |
| M07 control A2 | 1,227.95 ms | 10.52 tok/s | 0.10% |

M10's 10.51 tok/s is equal to or slightly below the 10.515 tok/s control
mean. Removing the runtime format bodies therefore does not improve register
allocation or instruction scheduling on this compiler. Candidate `81f0737`
was removed by `aa6f176`; production again has M07 behavior.

The trace also closes an ambiguity in the projection bound. The seven layer
projections read 7,301,529,600 weight bytes per token and sustain about 98.5
GB/s at 74.15 ms/token. Including the dedicated 550,502,400-byte logits matrix,
the decode weight stream is 7,852,032,000 bytes and sustains about 98.7 GB/s.
The nominal 120 GB/s device limit gives a non-achievable all-weight-only bound
of 65.43 ms/token; it excludes attention, graph state, cache traffic, and
sampling. Current llama.cpp's complete 76.66 ms/token endpoint leaves only
2.90 ms/token between Graph Horizon's measured projection-plus-logits time and
the comparator's entire token. Current source inspection finds the same
canonical Q4_K/Q6_K blocks, two SIMD groups, and four output rows in both
decode GEMV kernels. A new local arithmetic variant therefore has less than a
3% realistically recoverable endpoint envelope unless it changes the durable
weight representation.

### M11 — Metal residency policy: REJECT

Current llama.cpp requests Metal residency for its model buffers and exposes
`GGML_METAL_NO_RESIDENCY` as an exact policy control. Graph Horizon does not
create residency sets. Before adding a new resource owner, the policy was
isolated inside the pinned comparator at 14B/KV126, F16 KV, flash attention,
full GPU offload, and 32 generated tokens. Each position loaded a fresh server
and used one warm-up plus three measured requests:

| Position | Prompt mean | Decode mean | Decode CV |
|---|---:|---:|---:|
| residency enabled A1 | 1,129.73 ms | 82.58 ms/token | 0.09% |
| residency disabled | 1,140.13 ms | 79.68 ms/token | 0.75% |
| residency enabled A2 | 1,146.22 ms | 82.65 ms/token | 0.26% |

Disabling residency improved decode by 3.55% against the 82.62 ms/token
bookend mean; prompt evaluation changed by only 0.19% against its control mean.
Residency sets are therefore not a missing llama.cpp advantage on this M4 and
would add lifetime machinery with the wrong measured sign. No Graph Horizon
production edit was made.

## Final gap attribution

Times below are diagnostics-free workload times where available; individual
components come from the adjacent-boundary diagnostic traces and are marked as
such. A practical floor is the best measured member of the qualified design
space, not the hardware-only ideal.

| Workload | Component | Current time | llama.cpp equivalent | Practical floor | Realistically removable | Reason to close |
|---|---|---:|---:|---:|---:|---|
| 3B prefill 2K | seven projections | 4,960.32 ms diagnostic | about 4,970 ms linear fit | 4,960 ms | about 0 ms | M07 reached the comparator-derived floor |
| 3B prefill 2K | attention | 910.27 ms diagnostic | about 565 ms quadratic fit | 910 ms | about 0 ms | M04 is the measured winner; M02/M08/M09 falsified the available ownership variants |
| 3B prefill 2K | other kernels and residual | 424.56 ms diagnostic | unknown | not isolated | below 1,071.68 ms competitive gap | no dominant unaccounted stage; record/submit is not critical |
| 3B prefill 12K | attention | 38,423.44 ms diagnostic | about 20,360 ms quadratic fit | 38,423 ms | 0 ms with a viable mechanism | theoretical 18.06 s envelope remains, but every feasible ownership/occupancy mechanism is retained or slower |
| 14B decode KV128 | seven projections | 74.15 ms/token diagnostic | unknown | 74.15 ms/token | about 0 ms | same canonical geometry as llama.cpp; M10 is flat at the endpoint |
| 14B decode KV128 | logits | 5.40 ms/token diagnostic | unknown | about 5.40 ms/token | below 1 ms/token | already part of the same 98.7 GB/s weight stream |
| 14B decode KV128 | attention | 12.93 ms/token diagnostic | unknown | about 12.93 ms/token | about 0 ms | short grouped-GQA candidate P05a was flat |
| 14B decode KV128 | other kernels plus measured residual | 10.28 ms/token diagnostic | unknown | at least 9.14 ms/token | at most 1.14 ms/token | CPU record is only 1.14 ms/token; residency has the wrong sign |
| 3B decode KV28K | retained split attention | 51.78 ms/token diagnostic | unknown | about 51.78 ms/token | about 0 ms | doubling history splits regressed 1.29% at 28K; final endpoint is thermally non-comparable |

The 12K prefill row is the largest known theoretical envelope. It is retained
as an explicit gap, not converted into a claim of parity. Global closure rests
on the absence of a viable Metal-native mechanism with realistically
recoverable value, not on pretending that this theoretical envelope vanished.

## Remaining candidates and global ranking

| Candidate / design space | Predicted whole-workload gain | Quality risk | Memory cost | Engineering cost | Decision |
|---|---:|---|---|---|---|
| alternate Q4/Q6 decode entry points | 5.59% before test; 0.00% measured | none | none | low | REJECT M10; compiler already specializes effectively |
| new packed decode weight representation | less than 3% realistic short-decode envelope | medium: durable format conversion | second representation or load-time repack | high | NOT PURSUED; canonical llama.cpp geometry is already matched and M10 is flat |
| Metal tensor decode at M=1 | at most 0% on this M4 | medium accumulation change | pipeline/state only | high | NOT PURSUED; current llama.cpp disables its tensor route on pre-M5 and M07's gain depends on 64-row reuse |
| short grouped-GQA attention | about 0% measured at KV128 | low | scratch/state | medium | REJECT P05a; 29.04 versus 29.01 tok/s |
| more long-decode history splits | -1.29% at KV28K; best screen +2.51% at KV8K | low | doubled reduction scratch | medium | REJECT P05e; below gate and context-inconsistent |
| wider/split long-prefill attention | -12.14% GPU for M08; -0.51% endpoint for M09 | medium softmax/reduction | threadgroup state at 32 KiB boundary | high | REJECT; C128 cannot fit and MPP would materialize nonlinear state |
| command fusion / indirect command recording | at most 1.16% from CPU record bound | low | command metadata | high | NOT PURSUED; GPU wait dominates and one buffer/encoder already spans a token |
| Metal residency sets | -3.55% comparator decode effect | none | residency objects and renewal thread | medium | REJECT M11; measured sign is wrong |
| private weight buffers | about 0% | none | model-load blit / duplicate lifetime policy | medium | NOT PURSUED; llama.cpp also selects shared buffers on unified M4 |
| INT8 KV as F16 replacement | negative: 2.64x slower prefill at 2K, 4.80x at 28K | high numerical-policy change | lower KV bytes, extra quantization state | high | REJECT as competitive route; existing correct fallback remains |
| approximate softmax or model adaptation | unbounded without a new quality contract | high | model/validation artifacts | very high | NOT PURSUED; changes the agreed output-quality boundary |

No remaining candidate clears the 5% moderate-change gate with positive
measured evidence. The only untested theoretical envelope above that gate is a
new long-prefill attention architecture, but the concrete Metal-native forms
are either resource-infeasible or already falsified by M02, M04, M08, and M09.
The matrix, attention, decode, KV, representation, dataflow, resource lifetime,
orchestration, numerical-policy, and Apple-family-specialization spaces have
therefore all been explicitly ranked.

## Global stop and completion answer

The parity contract is not universally met: the stable final rows range from
1.09x to 1.61x, while thermally stressed 28K results are worse. Nevertheless,
the realistically recoverable Metal performance gap has been substantially
exhausted on the important qualified workloads. Retained M01/M04/M07 reduce
stable prefill by 23.11--29.65% on the principal matrix and bring stable
prefill ratios to 1.17--1.33x except the 14B/128 1.36x row. M07 lands within
0.2% of its 3B/2K Amdahl prediction and at the inferred projection floor.

The remaining short-decode gap is not an unprofiled projection hypothesis:
79.56 ms/token of projection-plus-logits work already streams 7.85 GB at about
98.7 GB/s, M10 measures no endpoint gain, short GQA reuse is flat, CPU record
can recover at most 1.16%, and residency is counterproductive. Long attention
still has a theoretical comparator gap, but every feasible current ownership,
split, occupancy, and tile mechanism has been retained or rejected with direct
measurements. This satisfies the second GLOBAL STOP condition: no remaining
identified Metal-native candidate contains sufficient *realistically*
recoverable end-to-end value, even though literal llama.cpp parity has not
been achieved.

Final retained production is M01 `2cd6043`, M04 `01a0c45`, and M07 `bcefc59`.
The tested retained state is `aa6f176`, whose production behavior equals M07
after the ordinary M10 revert (apart from a formatting-only source change).
The diagnostics-free binary remains
`a166863382875d596f5cc71406c2d6782b8b15b95c58427350bfb951f7122c1c`.
No profiler, rejected shader, alternate routing, dependency, or public API
survives. PR #34 later integrated the retained state into `main`.
