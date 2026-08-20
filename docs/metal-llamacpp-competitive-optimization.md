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
| Date | 2026-08-20 Europe/Rome |
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
  GPU offload, F16 K/V, flash attention disabled, batch 64, and microbatch 64.
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

All six catalogued GGUF files were opened read-only and their SHA-256 values
matched `support/models.tsv` before measurement.

## Fresh competitive matrix

Latency ratio is Graph Horizon time divided by llama.cpp time. A decode ratio
below 1.0 means Graph Horizon is faster. There is no separate numeric
performance contract, so the competitive gap is measured against parity.

| Model | Workload | Ours | llama.cpp | Ratio | Absolute gap |
|---|---|---:|---:|---:|---:|
| 3B | prefill 128 | 486.77 ms | 304.84 ms | 1.60x | 181.93 ms |
| 3B | prefill 512 | 1,980.78 ms | 1,200.23 ms | 1.65x | 780.55 ms |
| 3B | prefill 2K | 8,777.22 ms | 5,275.39 ms | 1.66x | 3,501.83 ms |
| 3B | prefill 8K | 48,851.51 ms | 33,367.07 ms | 1.46x | 15,484.44 ms |
| 3B | prefill 28K | 335,020.18 ms | 228,558.03 ms | 1.47x | 106,462.15 ms |
| 8B | prefill 128 | 1,215.15 ms | 853.80 ms | 1.42x | 361.35 ms |
| 8B | prefill 2K | 20,896.70 ms | 13,972.92 ms | 1.50x | 6,923.78 ms |
| 14B | prefill 128 | 1,973.43 ms | 1,558.32 ms | 1.27x | 415.11 ms |
| 14B | prefill 2K | 35,542.08 ms | 24,141.65 ms | 1.47x | 11,400.44 ms |
| 3B | decode KV128 | 33.64 ms/token | 23.63 ms/token | 1.42x | 10.01 ms/token |
| 3B | decode KV512 | 40.00 ms/token | 24.85 ms/token | 1.61x | 15.15 ms/token |
| 3B | decode KV2K | 28.07 ms/token | 29.87 ms/token | 0.94x | -1.79 ms/token |
| 3B | decode KV8K | 39.35 ms/token | 47.77 ms/token | 0.82x | -8.41 ms/token |
| 3B | decode KV28K | 77.04 ms/token | 118.86 ms/token | 0.65x | -41.81 ms/token |
| 8B | decode KV128 | 63.90 ms/token | 52.52 ms/token | 1.22x | 11.38 ms/token |
| 8B | decode KV2K | 58.17 ms/token | 59.98 ms/token | 0.97x | -1.81 ms/token |
| 14B | decode KV128 | 96.06 ms/token | 81.90 ms/token | 1.17x | 14.16 ms/token |
| 14B | decode KV2K | 92.17 ms/token | 94.98 ms/token | 0.97x | -2.81 ms/token |

The largest relative prefill gap is 3B/2K at 1.66x. The largest absolute gap
is 3B/28K at 106.46 seconds/request. Short decode remains behind, but decode at
2K and longer is `AHEAD` and is closed to competitive tuning unless later work
regresses it.

## Prefill statistics

| Model/workload | Ours mean / median / SD / CV (ms) | llama.cpp mean / median / SD / CV (ms) |
|---|---:|---:|
| 3B/128 | 486.77 / 487.73 / 1.90 / 0.39% | 304.84 / 298.09 / 13.50 / 4.43% |
| 3B/512 | 1,980.78 / 1,980.65 / 3.90 / 0.20% | 1,200.23 / 1,200.78 / 4.37 / 0.36% |
| 3B/2K | 8,777.22 / 8,779.93 / 6.20 / 0.07% | 5,275.39 / 5,292.11 / 48.94 / 0.93% |
| 3B/8K | 48,851.51 / 48,866.60 / 73.17 / 0.15% | 33,367.07 / 33,580.69 / 809.13 / 2.42% |
| 3B/28K | 335,020.18 / 336,649.63 / 2,862.23 / 0.85% | 228,558.03 / 227,309.95 / 2,558.77 / 1.12% |
| 8B/128 | 1,215.15 / 1,214.62 / 1.19 / 0.10% | 853.80 / 845.89 / 30.32 / 3.55% |
| 8B/2K | 20,896.70 / 20,787.49 / 195.85 / 0.94% | 13,972.92 / 13,938.54 / 496.67 / 3.55% |
| 14B/128 | 1,973.43 / 1,969.48 / 21.72 / 1.10% | 1,558.32 / 1,554.99 / 10.02 / 0.64% |
| 14B/2K | 35,542.08 / 36,417.68 / 1,724.47 / 4.85% | 24,141.65 / 24,312.08 / 888.05 / 3.68% |

## Decode statistics and closure

| Model/history | Ours tok/s (mean / median / SD / CV) | llama ms/token (mean / median / SD / CV) | Status |
|---|---:|---:|---|
| 3B/128 | 29.73 / 29.81 / 0.15 / 0.51% | 23.63 / 23.62 / 0.14 / 0.59% | GAP |
| 3B/512 | 25.00 / 25.00 / 0.03 / 0.10% | 24.85 / 24.85 / 0.07 / 0.29% | GAP |
| 3B/2K | 35.62 / 35.62 / 0.15 / 0.42% | 29.87 / 28.99 / 1.66 / 5.57% | AHEAD |
| 3B/8K | 25.41 / 25.50 / 0.21 / 0.82% | 47.77 / 47.41 / 0.70 / 1.47% | AHEAD |
| 3B/28K | 12.98 / 13.08 / 0.28 / 2.19% | 118.86 / 116.78 / 3.75 / 3.15% | AHEAD |
| 8B/128 | 15.65 / 15.65 / 0.00 / 0.01% | 52.52 / 52.43 / 0.18 / 0.34% | GAP |
| 8B/2K | 17.19 / 17.23 / 0.09 / 0.51% | 59.98 / 59.83 / 0.40 / 0.66% | AHEAD |
| 14B/128 | 10.41 / 10.41 / 0.01 / 0.12% | 81.90 / 81.77 / 0.36 / 0.44% | GAP |
| 14B/2K | 10.85 / 10.83 / 0.15 / 1.35% | 94.98 / 94.97 / 0.76 / 0.80% | AHEAD |

## Memory checkpoint

With context capacity 32,768, 3B Graph Horizon reached 4,370,087,936 bytes
maximum RSS and 5,756,454,952 bytes peak footprint. The live llama.cpp 3B
server RSS after the 28K matrix was 5,641,984 KiB. At 14B the corresponding
RSS values were 13,493,846,016 bytes for Graph Horizon and 13,362,096 KiB for
llama.cpp. These are process-level unified-memory observations, not isolated
GPU allocation counters.

## Current ranking and next evidence

1. Long prefill has the largest absolute competitive gap: 106.46 seconds at
   3B/28K. Fresh operation attribution and a structural comparison with the
   current llama.cpp Metal attention path are required before any redesign.
2. Quantized projection/MLP work is expected to dominate shorter prefill and
   must be freshly attributed at 128/512/2K before selecting a matrix candidate.
3. Short decode has 10--15 ms/token gaps, but its absolute recoverable time per
   request is below the prefill candidates. Existing routing is checked before
   any new decode kernel.
4. Decode at 2K and longer is a regression guard, not an optimization target.

Fresh GPU/CPU attribution, practical floors, Amdahl calculations, retained and
rejected experiments, qualification, and final reproduction commands will be
appended as the investigation proceeds.
