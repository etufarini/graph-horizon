<!--
This document is the compact, reviewed record of retained backend
optimizations and their measured boundaries. Detailed experiment logs remain
available in Git history.
-->

# Current Performance Status

## Scope

The retained implementation specializes CPU, Vulkan, and Metal inference for
Ministral 3B, 8B, and capacity-guarded 14B Q4_K_M models, and includes a
standalone CUDA backend plus a CUDA hybrid composition profile qualified on one
frozen 3B tuple. It adds no public API, model-name route, or device-name route.
Specialized device paths are capability-, vendor-family-, format-, and
shape-gated and preserve a generic operation fallback where the backend defines
one.

The authenticated 3B artifact used throughout the investigation was
2,147,023,008 bytes with SHA-256
`9ed150d4367e68df0ac8e1540f6ddc65b42d0ee26378329d1ecbca60f93fc5f8`.

## CUDA Qualification Measurement

CUDA is **qualified**, not production, only for the recorded Linux `x86_64`
validation environment with driver 595.84, CUDA Toolkit 12.4.131,
authenticated 3B Instruct Q4_K_M, context-4096, and f16/int8 KV tuple in
[validation evidence](validation-evidence.md#cuda-implementation-gate--1-september-2026).
The qualification run's separate context-2048 f16 measurement recorded 1.28
prompt tok/s and 0.59 end-to-end decode tok/s.

A later CUDA/Vulkan benchmark completed its short and medium CUDA workloads but
timed out on the long CUDA workload, so it reports no comparative winner. Its
generated report is untracked local performance evidence only and does not
broaden the qualified tuple.

CUDA hybrid was qualified separately at implementation commit `61c8b57` for
context 4096 and both f16 and int8 KV. Its 100% all-GPU, 25% mixed, and 0%
CPU-only rows passed the frozen numeric, endpoint, crossing, and lifecycle
gates. This campaign recorded no performance comparison and makes no production
claim; neighboring devices, software, artifacts, model sizes, contexts, and KV
schemes remain unmeasured.

## Integration Identity

This record was audited against `main`/`origin/main` `6b46331` on 22 August
2026. The retained performance lineages are not local-only work: they were
pushed and integrated as follows.

| Work | Main integration |
|---|---|
| Metal parity and llama.cpp competition | PR #34, `87e3e4b` |
| NVIDIA long-context decode | PR #35, `e955b26` |
| Metal long-context decode | PR #36, `ae69781` |
| AMD long-context decode | PR #37, `e3d23f2` |
| NVIDIA long-context prefill | PR #38, `8884465` |
| AMD long-context prefill | PR #39, `8e2f8cc` |
| Metal long-context prefill | PR #40, `6e514c1` |
| Final cleanup and AMD synchronization repair | PR #41, `24eac82` |

The package version is `0.1.5`. The immutable annotated `v0.1.0` tag retains
the numeric qualification evidence; `v0.1.1` remains the packaging correction,
`v0.1.2` adds explicit Web and News search plus the revised Web workspace, and
`v0.1.3` corrects release identity, `v0.1.4` prepares generic release model
selection and a closed family dispatcher, and `v0.1.5` prepares the qualified
CUDA profiles and installer paths.
[Validation evidence](validation-evidence.md) owns all tag-derived identities
and their distinct evidence boundaries.

## Retained Improvements

| Path | Eligibility | Fallback | Measured effect |
|---|---|---|---|
| Direct Q4 Matrix2 prefill with FP16 accumulation | NVIDIA Matrix2, Q4_K, qualified shapes | staged cooperative/scalar Q4 | 8B/128 TTFT 228.12 → 150.66 ms; 8B/28K 69.246 → 51.937 s |
| Hybrid NVIDIA Matrix2 prefill batches | NVIDIA family plus Matrix2 Q4/Q64 pipelines and all-GPU capacity | established 32-row all-GPU plan, then generic hybrid placement | 14B/15K TTFT 230.888 → 33.549 s; decode unchanged at about 51.3 ms/token |
| 14B Matrix2 routing | legal 5,120/16,384/4,096/1,024 dimensions | existing Q4/Q6 batch kernels | 14B/128 prefill 4.390 s → 249.32 ms across the full program |
| Per-8 Q8/DP4A Q4 decode | DP4A, Q4_K, scratch and shape contract | exact float Q4 decode | 8B/128 decode 26.64 → 35.06 tok/s across the full program |
| Vectorized GQA decode | exact GQA relation and device limits | generic attention decode | retained after focused parity and six-model qualification |
| Tiled and Matrix2 attention | qualified subgroup, shared-memory, and shape caps | wide/generic attention | improves short and long prefill without changing the public contract |
| AMD Q4 MMQ prefill | AMD, accelerated integer dot, Q4_K, aligned shape and bounded scratch | portable Q4 batch kernel | AMD Vulkan validation GPU 3B/128 TTFT 624.35 → 223.07 ms |
| AMD bounded prefill submissions | AMD vendor plus graph depth/context | existing 256-row Vulkan capacity elsewhere | 3B/28K reset → 191.055 s at 128 rows; 8B/28K reset → 352.347 s at 64 rows |
| AMD split-history GQA prefill attention | AMD DP4A/wave32 resources, F16 KV, head 128, exact 4:1 GQA, rows <=64, ending position >=2K | existing tiled/Matrix2/portable attention | 8B/15K TTFT 134.843 → 117.022 s; 8B/28K 350.907 → 273.108 s |
| AMD required-wave32 Q6 decode/logits | AMD plus Vulkan subgroup-size control | default-subgroup pipeline | adds 6.3--6.5% at 3B KV128/2K on top of GQA |
| AMD required-wave32 4:1 GQA decode | AMD, required subgroup 32, F16 KV, head 128, exact 4:1 GQA | exact wave64 or generic decode | 3B/28K 16.144 → 49.398 tok/s; 8B/28K 18.868 fallback → 28.374 tok/s |
| Exact wave64 4:1 GQA decode | default subgroup 64, F16 KV, head 128, exact 4:1 GQA | generic decode | 3B/28K 16.144 → 29.762 tok/s without LDS or changed dot order |
| Request KV reuse | serialized standalone Vulkan or Metal requests; keyed calls may also reuse an identical token prefix | ordinary generation on CPU/hybrid; prefix zero on a key/token mismatch and cache invalidation on failure | avoids reallocating request-local cache while preserving fresh tail logits |
| CPU 32-row prefill and single-token SIMD routing | AVX2/F16C capability and supported Q4_K/Q6_K shapes | scalar and smaller-batch kernels | 3B/128 TTFT 7,030.85 → 5,024.78 ms; decode 5.71 → 6.79 tok/s |
| CPU cache-sized prefill token tiling | x86_64 architectural L2 report and supported quantized batched shapes | historical 256 KiB cache premise and existing 16--64 tile bounds | validation environment short prompt 30.68 → 33.20 tok/s; long prompt 27.32 → 31.09 tok/s |
| CPU four-query GQA attention | exact 4:1 GQA and supported SIMD width | pair/serial attention | 8K attention improved 1.542x locally |
| Metal early tiled GQA prefill | unified-memory Apple9, F16 KV, head 128, width 32, exact 4:1 GQA | serial/segmented Metal attention | 3B/512 TTFT 6,188.66 → 2,554.70 ms |
| Metal pipelined C64 attention QK | Metal matrix path, rows 64, F16 KV, head 128, exact 4:1 GQA | established tiled/segmented/serial Metal attention | 3B/15K TTFT 103.184 → 84.298 s; 28K 280.058 → 214.704 s |

The benchmark harness also reports medians alongside means, dispersion, and CV,
so performance decisions are less sensitive to outliers.

## Historical NVIDIA A/B Comparison

On 2026-08-18, commit `bc0145d` and the retained Vulkan candidate were rebuilt
in release mode and measured on the same NVIDIA Vulkan validation GPU with the
authenticated 3B model, Vulkan, F16 KV, context 4096, a 128-token prompt, one
warm-up, and three repetitions.

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

These rows use the AMD Vulkan validation GPU, RADV 26.0.3, full Vulkan placement, F16 KV, and
the pinned llama.cpp Vulkan build `9bebfcb4b`.

| Workload | AMD final | llama.cpp | Final ratio |
|---|---:|---:|---:|
| 3B prefill / 128 | 223.07 ms | 158.46 ms | 1.41x |
| 3B prefill / 28K | 191.055 s | 56.226 s | 3.40x |
| 3B decode / 2K | 74.907 tok/s | 40.215 tok/s | 0.54x latency |
| 3B decode / 28K | 49.398 tok/s | 59.826 tok/s | 1.21x latency |
| 8B prefill / 15K | 117.022 s | 36.483 s | 3.21x |
| 8B prefill / 28K | 273.108 s | 90.776 s | 3.01x |
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

These rows use a 10-core Apple silicon validation GPU on macOS 26.3, full Metal
placement, F16 KV, and pinned llama.cpp `13f2b28b0`. Graph Horizon TTFT includes
a broader public-event boundary than llama.cpp prompt evaluation.

| Workload | Metal final | llama.cpp | Final ratio |
|---|---:|---:|---:|
| 3B prefill / 128 | 331.65 ms | 299.18 ms | 1.11x |
| 3B prefill / 2K | 5.694 s | 5.649 s | 1.01x |
| 3B prefill / 15K | 84.298 s | 72.853 s | 1.16x |
| 3B prefill / 28K | 214.704 s | 176.111 s | 1.22x |
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
- Corrected runtime `e7edc83` passes the available CPU, Vulkan, and
  Vulkan-hybrid matrix (`40 pass`, `34 external verification`, zero failures),
  both warning-denied profiles, and all six semantic rows. The Metal checkpoint
  passes pure/hybrid suites, focused numeric tests, and the authenticated 3B
  teacher row on the M4 host.
- The CUDA checkpoint passes the release engine suite, physical-device
  operation oracles, lifecycle gates, and authenticated 3B f16/int8 teacher
  rows on the frozen qualified tuple.
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

Fresh NVIDIA Vulkan validation GPU long-context profiling is recorded in
[the NVIDIA prefill checkpoint](../investigation-reports/vulkan-nvidia-long-context-prefill.md) and
[the NVIDIA decode checkpoint](../investigation-reports/vulkan-nvidia-long-context-decode.md).
The motivating 15,435-token prefill symptom no longer reproduces: fresh 8B
all-GPU hybrid TTFT is 22.062 s at 699.63 tok/s, and the matching 14B/16K
capacity row is 33.945 s. Current 512-row timestamps attribute 8B/28K prefill
as 40.5% exact attention, 56.8% projections/MLP, and 2.7% support/control.
The remaining optimistic exact-attention removal is 3.70 s, or 7.64% of whole
prefill, below the complex-redesign gate; acquired Q4/Q6 alternatives remain
closed by performance or unchanged numerical qualification.

At 14B/15K, decode is 51.334 ms/token rather than the stale 220 ms symptom;
exact attention has only 0.94 ms of practical traffic headroom and the best
completed Q6 candidate predicts 1.83 ms (3.69% whole-token gain), below its
quality-sensitive gate. The remaining llama.cpp gap is KV-independent. Hybrid
prefill was the larger removable request cost and now uses the qualified
capacity-accounted Matrix2 batch without changing decode or fallback behavior.

The largest remaining bottleneck is 8B/28K prefill. The final profile attributes
about 42% to direct Q4 Matrix2, 39% to attention, 16% to staged Q6 Matrix2, and
2% to other work. Tested runtime-only variants cannot close the remaining gap
to the 1.7x target; doing so would require a separately approved model-adaptation
and quality-validation effort.

On the AMD Vulkan validation GPU, exact split-history F16 GQA attention is now retained after
reducing 8B TTFT by 13.22% at 15K and 22.17% at 28K. At 15K the causally
re-attributed shares are exact Q6_K 34.44%, attention 33.08%, Q4 31.71%, and
other work 0.77%. Four history splits regressed about 39%, larger Q4/Q6 tiles
regressed, exact Q6 staging gained only 2.15% historically, and lossy Q6 failed
quality. The detailed route, Amdahl model, thermal tradeoff, model controls, and
stop decision are in
[the AMD prefill checkpoint](../investigation-reports/vulkan-amd-long-context-prefill.md). Long
decode retains wave32 GQA/Q6 and remains guarded through KV depth 28K.

CUDA remains far below the mature Vulkan path on its completed benchmark rows,
and the long workload did not finish before the runtime adapter timeout. No
CUDA optimization or production claim follows from that incomplete comparison.

CPU prefill now sizes its quantized-matmul activation tile from architectural
L2 capacity; the retained A/B improves short prompt throughput by 8.21% and
long by 13.80% without a decode regression. Quantized batched matmul remains
the largest measured CPU limit; the next material designs require new
packed-GEMM or query-tiled-attention subsystems. Metal's retained pipelined C64
QK schedule removes 44.4% of 15K
attention time; the complete evidence is in
[the Metal prefill checkpoint](../investigation-reports/metal-long-context-prefill.md).
The remaining Metal gap is exact attention's quadratic term, but grouped GQA,
wider/split ownership, PV pipelining, and head-major KV have the wrong measured
sign or fail to reproduce. INT8 remains on its correct separate non-tiled route.
