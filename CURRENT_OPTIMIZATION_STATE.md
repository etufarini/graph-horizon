<!--
This document is the compact, reviewed record of the retained Vulkan
optimizations. Detailed experiment logs remain available in Git history.
-->

# Current Optimization State

## Scope

The retained implementation targets Vulkan inference for the supported
Ministral 3B, 8B, and capacity-guarded 14B Q4_K_M models on NVIDIA and AMD.
It adds no dependency, public API, model-name route, or device-name route.
Every specialized path is capability-, vendor-family-, format-, and shape-gated
and preserves the generic fallback.

The authenticated 3B artifact used throughout the investigation was
2,147,023,008 bytes with SHA-256
`9ed150d4367e68df0ac8e1540f6ddc65b42d0ee26378329d1ecbca60f93fc5f8`.

## Retained Improvements

| Path | Eligibility | Fallback | Measured effect |
|---|---|---|---|
| Direct Q4 Matrix2 prefill with FP16 accumulation | NVIDIA Matrix2, Q4_K, qualified shapes | staged cooperative/scalar Q4 | 8B/128 TTFT 228.12 → 150.66 ms; 8B/28K 69.246 → 51.937 s |
| 14B Matrix2 routing | legal 5,120/16,384/4,096/1,024 dimensions | existing Q4/Q6 batch kernels | 14B/128 prefill 4.390 s → 249.32 ms across the full program |
| Per-8 Q8/DP4A Q4 decode | DP4A, Q4_K, scratch and shape contract | exact float Q4 decode | 8B/128 decode 26.64 → 35.06 tok/s across the full program |
| Vectorized GQA decode | exact GQA relation and device limits | generic attention decode | retained after focused parity and six-model qualification |
| Tiled and Matrix2 attention | qualified subgroup, shared-memory, and shape caps | wide/generic attention | improves short and long prefill without changing the public contract |
| AMD Q4 MMQ prefill | AMD, accelerated integer dot, Q4_K, aligned shape and bounded scratch | portable Q4 batch kernel | RX 6750 XT 3B/128 TTFT 624.35 → 223.07 ms |
| AMD bounded prefill submissions | AMD vendor family | existing 256-row Vulkan capacity elsewhere | RX 6750 XT 3B/28K reset → 191.055 s completion |
| Request KV reuse | serialized Vulkan requests; keyed calls may also reuse an identical token prefix | fresh state on cache failure | avoids reallocating request-local cache while preserving fresh tail logits |

The benchmark harness also reports medians alongside means, dispersion, and CV,
so performance decisions are less sensitive to outliers.

## Fresh Comparison With `origin/main`

On 2026-08-18, commit `bc0145d` and this branch were rebuilt in release mode
and measured on the same RTX 3060 with the authenticated 3B model, Vulkan,
F16 KV, context 4096, a 128-token prompt, one warm-up, and three repetitions.

| Metric | `origin/main` | Clean branch | Change |
|---|---:|---:|---:|
| TTFT mean | 1,356.07 ms | 78.70 ms | -94.2% |
| Prompt throughput mean | 94.39 tok/s | 1,628.55 tok/s | 17.3x |
| Decode throughput mean | 43.27 tok/s | 69.81 tok/s | +61.3% |

The current TTFT CV was 4.36% versus 0.25% on main, but the measured gap is
far larger than either run's variation.

## Historical Representative Result

These values track the optimization program's later qualified entry point, not
`main`. They use pure Vulkan, full offload, F16 KV, greedy sampling, the same
model artifact per comparison, and the pinned llama.cpp Vulkan build recorded
in Git history.

| Workload | Program entry | Final | llama.cpp | Final ratio |
|---|---:|---:|---:|---:|
| 3B prefill / 128 | 92.98 ms | 74.08 ms | 42.00 ms | 1.76x |
| 3B prefill / 28K | 35.15 s | 28.995 s | 13.603 s | 2.13x |
| 8B prefill / 128 | 260.05 ms | 149.68 ms | 97.51 ms | 1.54x |
| 8B prefill / 28K | 75.10 s | 51.937 s | 24.523 s | 2.12x |
| 14B prefill / 128 | 4.390 s | 249.32 ms | 174.94 ms | 1.43x |
| 3B decode / 128 | 56.74 tok/s | 71.24 tok/s | 104.81 tok/s | 1.47x |
| 8B decode / 128 | 26.64 tok/s | 35.06 tok/s | 49.84 tok/s | 1.42x |
| 8B decode / 28K | 19.52 tok/s | 23.93 tok/s | 27.69 tok/s | 1.16x |
| 14B decode / 128 | 17.23 tok/s | 23.61 tok/s | 36.42 tok/s | 1.54x |

Graph Horizon includes tokenization and first sampling in TTFT; llama.cpp pure
prompt processing does not. This boundary was measured at about 11 ms for 128
tokens and is not subtracted from the table.

## Correctness And Portability

- Focused CPU/Vulkan numeric oracles cover Q4, Q6, attention, GQA, and retained
  Matrix2 shapes.
- Teacher-forced top-two checks cover the six supported 3B/8B/14B Instruct and
  Reasoning artifacts for numeric paths that do not promise bit identity.
- The final local matrix passes CPU, Vulkan, and Vulkan-hybrid tests and
  warning-denied Clippy. Metal remains external verification on macOS.
- Specialized kernels are selected only after feature, resource, format, and
  shape checks. Unsupported tuples remain on the pre-existing generic paths.

The AMD program's complete capability inventory, llama.cpp comparison,
experiment registry, rejected candidates, and raw-evidence locations are in
[`docs/vulkan-amd-specialization.md`](docs/vulkan-amd-specialization.md).

## Deliberately Excluded

The final source does not contain the rejected sparse-attention, low-precision
weight, native-FP16 predecode, M256/N256, direct-Q6, K128 direct-Q4,
experimental tracing, or one-shot diagnostic paths. They regressed
performance, failed the established quality gate, or did not clear the minimum
whole-request gain. Their detailed evidence is retained in Git history rather
than in the working documentation.

## Remaining Limit

The largest remaining bottleneck is 8B/28K prefill. The final profile attributes
about 42% to direct Q4 Matrix2, 39% to attention, 16% to staged Q6 Matrix2, and
2% to other work. Tested runtime-only variants cannot close the remaining gap
to the 1.7x target; doing so would require a separately approved model-adaptation
and quality-validation effort.

On the RX 6750 XT, the largest remaining 3B/8K shares are tiled attention
(40.18%), retained Q4 MMQ projections (31.87%), and portable exact Q6_K
(26.71%). Wave32, grouped-GQA, wider attention tiles, and exact/lossy Q6
candidates did not pass the established whole-request or quality thresholds.
