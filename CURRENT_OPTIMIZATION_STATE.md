<!--
This document is the compact, reviewed record of retained backend
optimizations and their measured boundaries. Detailed experiment logs remain
available in Git history.
-->

# Current Optimization State

## Scope

The retained implementation specializes CPU, Vulkan, and Metal inference for
Ministral 3B, 8B, and capacity-guarded 14B Q4_K_M models. It adds no dependency,
public API, model-name route, or device-name route. Specialized device paths are
capability-, vendor-family-, format-, and shape-gated and preserve a generic
operation fallback.

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
| AMD bounded prefill submissions | AMD vendor plus graph depth/context | existing 256-row Vulkan capacity elsewhere | 3B/28K reset → 191.055 s at 128 rows; 8B/28K reset → 352.347 s at 64 rows |
| AMD required-wave32 Q6 decode/logits | AMD plus Vulkan subgroup-size control | default-subgroup pipeline | adds 6.3--6.5% at 3B KV128/2K on top of GQA |
| AMD required-wave32 4:1 GQA decode | AMD, required subgroup 32, F16 KV, head 128, exact 4:1 GQA | exact wave64 or generic decode | 3B/28K 16.144 → 49.398 tok/s; 8B/28K 18.868 fallback → 28.374 tok/s |
| Exact wave64 4:1 GQA decode | default subgroup 64, F16 KV, head 128, exact 4:1 GQA | generic decode | 3B/28K 16.144 → 29.762 tok/s without LDS or changed dot order |
| Request KV reuse | serialized Vulkan requests; keyed calls may also reuse an identical token prefix | fresh state on cache failure | avoids reallocating request-local cache while preserving fresh tail logits |
| CPU 32-row prefill and single-token SIMD routing | AVX2/F16C capability and supported Q4_K/Q6_K shapes | scalar and smaller-batch kernels | 3B/128 TTFT 7,030.85 → 5,024.78 ms; decode 5.71 → 6.79 tok/s |
| CPU four-query GQA attention | exact 4:1 GQA and supported SIMD width | pair/serial attention | 8K attention improved 1.542x locally |
| Metal early tiled GQA prefill | unified-memory Apple9, F16 KV, head 128, width 32, exact 4:1 GQA | serial/segmented Metal attention | 3B/512 TTFT 6,188.66 → 2,554.70 ms |

The benchmark harness also reports medians alongside means, dispersion, and CV,
so performance decisions are less sensitive to outliers.

## Historical NVIDIA A/B Comparison

On 2026-08-18, commit `bc0145d` and the retained Vulkan candidate were rebuilt
in release mode and measured on the same RTX 3060 with the authenticated 3B
model, Vulkan, F16 KV, context 4096, a 128-token prompt, one warm-up, and three
repetitions.

| Metric | `origin/main` | Retained candidate | Change |
|---|---:|---:|---:|
| TTFT mean | 1,356.07 ms | 78.70 ms | -94.2% |
| Prompt throughput mean | 94.39 tok/s | 1,628.55 tok/s | 17.3x |
| Decode throughput mean | 43.27 tok/s | 69.81 tok/s | +61.3% |

The candidate TTFT CV was 4.36% versus 0.25% on main, but the measured gap is
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

### AMD Final Comparison

These rows use the RX 6750 XT, RADV 26.0.3, full Vulkan placement, F16 KV, and
the pinned llama.cpp Vulkan build `9bebfcb4b`.

| Workload | AMD final | llama.cpp | Final ratio |
|---|---:|---:|---:|
| 3B prefill / 128 | 223.07 ms | 158.46 ms | 1.41x |
| 3B prefill / 28K | 191.055 s | 56.226 s | 3.40x |
| 3B decode / 2K | 74.907 tok/s | 40.215 tok/s | 0.54x latency |
| 3B decode / 28K | 49.398 tok/s | 59.826 tok/s | 1.21x latency |
| 8B prefill / 28K | 352.347 s | 87.818 s | 4.01x |
| 8B decode / 28K | 28.374 tok/s | 37.766 tok/s | 1.33x latency |
| 14B prefill / 8K | 75.383 s | 23.831 s | 3.16x |
| 14B decode / 8K | 21.840 tok/s | 36.715 tok/s | 1.68x latency |

### CPU Reference Comparison

These rows use six pinned i5-9600K cores, F16 KV, the authenticated 3B artifact,
and the pinned llama.cpp CPU comparator. CPU remains a correctness reference,
not a performance-qualified backend.

| Workload | CPU final | llama.cpp | Final ratio |
|---|---:|---:|---:|
| 3B prefill / 128 | 5.025 s | 0.317 s | 15.83x latency |
| 3B prefill / 2K | 96.000 s | 2.385 s | 40.26x latency |
| 3B decode / 128 | 6.79 tok/s | 10.312 tok/s | 1.52x latency |
| 3B decode / 2K | 5.73 tok/s | 8.957 tok/s | 1.56x latency |

### Metal Final Comparison

These rows use an Apple M4 10-core GPU on macOS 26.3, full Metal placement,
F16 KV, and pinned llama.cpp `13f2b28b0`. Graph Horizon TTFT includes a broader
public-event boundary than llama.cpp prompt evaluation.

| Workload | Metal final | llama.cpp | Final ratio |
|---|---:|---:|---:|
| 3B prefill / 128 | 677.26 ms | 323.55 ms | 2.09x |
| 3B prefill / 2K | 11.449 s | 5.768 s | 1.99x |
| 8B prefill / 2K | 26.685 s | 15.427 s | 1.73x |
| 14B prefill / 2K | 45.329 s | 24.351 s | 1.86x |
| 3B decode / 2K | 25.40 tok/s | 27.56 tok/s | 1.09x latency |
| 14B decode / 2K | 9.01 tok/s | 10.81 tok/s | 1.20x latency |

## Correctness And Portability

- Focused CPU/Vulkan numeric oracles cover Q4, Q6, attention, GQA, and retained
  Matrix2 shapes.
- Teacher-forced top-two checks cover the six catalogued 3B/8B/14B Instruct and
  Reasoning artifacts for numeric paths that do not promise bit identity. The
  final AMD route additionally passes the pinned external 16-token teacher on
  3B/8B/14B with exact prompt IDs and zero crossings.
- The final local matrix passes CPU, Vulkan, and Vulkan-hybrid tests and
  warning-denied Clippy. The Metal checkpoint passes pure/hybrid suites,
  focused numeric tests, and the authenticated 3B teacher row on the M4 host.
- Specialized kernels are selected only after feature, resource, format, and
  shape checks. Unsupported tuples remain on the pre-existing generic paths.

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

On the RX 6750 XT, the largest remaining prefill shares are tiled attention,
retained Q4 MMQ projections, and portable exact Q6_K. Grouped-prefill GQA
regressed, Q16 attention gained only 4.66% total, larger Q4 tiles regressed, exact
Q6 gained 2.15%, and lossy Q6 failed quality. Long decode now retains wave32
GQA/Q6 and meets the historical envelope on 3B/8B; larger-model short decode is
weight/Q6 dominated and further large gains require a separately approved
numeric or persistent-representation contract.

CPU prefill remains dominated by attention and quantized matrix multiplication;
the next material designs require new packed-GEMM or query-tiled-attention
subsystems. Metal's largest remaining gap is tiled F16 long-context attention,
with INT8 long prefill on a separate non-tiled route. These are new performance
programs, not unfinished cleanup tasks.
