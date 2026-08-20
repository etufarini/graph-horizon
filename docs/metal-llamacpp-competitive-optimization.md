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
| Initial worktree | clean |
| llama.cpp reference | `2b63e0610bbc2be990ae1360d5256efcdc3f9efb` (build 10237) |
| Push state | nothing pushed |

Repository history contains no qualified Metal revision newer than `8deb734`.
The Vulkan parity mission is acquired history and is not a performance target.
The acquired reports are `metal-vulkan-optimization-parity.md` and
`metal-hardware-specialization.md`; their rejected occupancy, FP16 staging,
query-reuse, and wider-split variants are not repeated without new evidence.

## Qualified tuple

- Apple M4 with 10 GPU cores, 24 GB unified memory, Metal 4, macOS 26.3.
- Both diagnostics-free release binaries were rebuilt from the SHAs above.
- llama.cpp was built with `GGML_METAL=ON`, its embedded Metal library, full
  GPU offload, F16 K/V, flash attention enabled, batch 64, and microbatch 64.
- Graph Horizon used pure Metal, F16 K/V, context allocation 32,768, its
  qualified 64-row request ownership, greedy sampling, and 32 requested tokens.
- Inputs contain the exact same token IDs: `[1, 3, 1097]`, token `1261`
  repeated `N - 5` times, then `[1032, 4]`. llama.cpp reports `cache_n=0` and
  `prompt_n=N` for every retained request.
- Each workload uses one warm-up and three measured requests. Long workloads
  run serially because this MacBook Air is fanless. CV above 1% is retained and
  identified rather than filtered.
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

Final matrices and reproduction commands will be appended as the investigation
proceeds.
