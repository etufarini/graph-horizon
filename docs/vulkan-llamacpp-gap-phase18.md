<!--
This document owns the Phase 18 Vulkan competitive-gap investigation: the
same-device benchmark, component attribution, source/dataflow comparison, and
the evidence-gated Phase 19 recommendation.
-->

# Vulkan llama.cpp Competitive Gap — Phase 18

## Outcome

Graph Horizon is 3.332x slower than llama.cpp on the 28K prefill comparison:
46,777.105 versus 14,038.202 ms, a 32,738.903 ms wall-time gap. Fresh Vulkan
timestamps reproduce 32,860.507 ms of component gap and attribute all of it:

| 28K macro component | Graph Horizon ms | llama.cpp ms | Gap ms | Gap share |
|---|---:|---:|---:|---:|
| Fused dense attention | 26,920.199 | 7,041.940 | 19,878.259 | 60.49% |
| Q/K/V/O/gate/up/down matmuls | 19,037.603 | 6,238.093 | 12,799.510 | 38.95% |
| Activation and other work | 954.505 | 771.767 | 182.738 | 0.56% |
| **Total** | **46,912.307** | **14,051.800** | **32,860.507** | **100.00%** |

The primary cause is an attention algorithm/dataflow difference, not less
causal work. Both backends evaluate the same 326.156 billion useful causal
query/key pairs, and their padded Matrix2 QK+AV work differs by only 0.225%.
llama.cpp instead keeps online-softmax state in workgroup-scope cooperative
matrices, loads a 64-row Q tile once, traverses K/V with twice Graph Horizon's
query reuse, and avoids explicit score/probability shared-memory round trips
and four static barriers. It performs the same QK+AV work at 23.77 versus 6.22
effective TFLOP/s.

The secondary gap is large-M matmul architecture, but it contains an important
numeric difference. llama.cpp directly consumes canonical Q4_K/Q6_K weights
with a 128x256x64 cooperative-matrix tile and FP16 accumulation. Graph Horizon
uses 64x32x128 tiles with FP32 accumulation; gate/up are persistently expanded
to FP16. A temporary llama.cpp selector probe forcing its existing FP32
accumulator path reduced pp512 from 3,887 to 48.7 token/s. The default matmul
lead is therefore real competitive behavior, but it is not evidence for a
parity-preserving copy of that numeric policy.

Phase 18 retains no production performance change. The top Phase 19 candidate
is an RTX 3060/Ampere-gated workgroup-cooperative Matrix2 attention path with
the current exact dense semantics and FP32 QK/softmax/AV contract. Closing a
conservative 50% of the measured attention gap predicts 9.939 seconds removed,
or 21.19% of the whole profiled request. The current kernel remains the generic
Vulkan fallback.

## Reproducible baseline

| Item | Value |
|---|---|
| Date | 2026-08-16 |
| Graph Horizon branch | `perf/vulkan-llamacpp-gap-phase18` |
| Graph Horizon Phase 18 base | `c4657ed` |
| Effective production source | `6cb170f`; Phase 16/17 production probes are reverted |
| llama.cpp checkout | `/home/emanuele/Documenti/llama.cpp` |
| llama.cpp commit / build | `9bebfcb4bc8b12a316e96ae03f33671eac1e72fd` / 9826 |
| llama.cpp build | Release, shared; CPU, CUDA, and Vulkan enabled; native CPU and OpenMP enabled |
| GPU | NVIDIA GeForce RTX 3060 12 GiB, PCI device `0x2487` |
| Driver / Vulkan | 595.84 / device API 1.4.329, loader 1.4.341 |
| CPU / RAM | Intel i5-9600K, 6 physical cores / 14 GiB |
| Model | `Ministral-3-3B-Instruct-2512-Q4_K_M.gguf` |
| File size / SHA-256 | 2,147,023,008 bytes / `9ed150d4367e68df0ac8e1540f6ddc65b42d0ee26378329d1ecbca60f93fc5f8` |
| Model shape | 26 layers, hidden 3072, FFN 9216, 32 Q heads, 8 KV heads, head dimension 128 |
| Tensor inventory | F32 53, Q4_K 156, Q6_K 27 |

The model's original context is 16,384 and its advertised maximum is 262,144;
both programs were run with a 32,768-token capacity. The idle control was P8,
210/405 MHz, 19 W, and 46 C. Alternated public runs operated at 1,965–1,987 MHz
graphics, 7,501 MHz memory, 150–159 W, and 62–71 C. There is no clock or thermal
decay matching the context-dependent gap.

## Fair-comparison configuration and timing boundary

| Parameter | Graph Horizon | llama.cpp |
|---|---|---|
| Device | Vulkan RTX 3060 | `-dev Vulkan0`, split none, main GPU 0 |
| GPU layers | all model work | `-ngl 99`, reported 27/27 layers |
| KV cache | F16, capacity 32,768 | `-ctk f16 -ctv f16`, offload enabled |
| Attention | exact causal dense | `-fa on` |
| Batch | fixed row/microbatch 256 | batch 2,048, microbatch 512 |
| Threads | harness default | 6 |
| Model mapping | local GGUF | mmap enabled, tensor split 0 |
| Sampling | greedy | model benchmark generation path |
| Graph Horizon MLP policy | `GRAPH_HORIZON_PREFILL_PREDECODE_MLP=1` | canonical compressed weights |

`llama-bench` generates its benchmark token IDs, while Graph Horizon uses the
exact chat-templated `" a"` repetition that produces the requested public token
count. Token values differ, but tensor shapes, causal work, model, quantization,
and KV capacity do not.

The public Graph Horizon TTFT begins before chat-template tokenization and ends
at the first public text delta; it includes tokenization, prefill, logits,
greedy sampling, and UTF-8 text emission. llama.cpp's pure `pp` measurement
excludes tokenization and sampling. A second `-pg N,1` series measures the
model-side first-token tail. That tail is about 10.8 ms at 128 and 21.6–26.8 ms
at long contexts. It is too small to alter the conclusion, but the headline
comparison is not claimed to have an identical host boundary.

Graph Horizon startup is also excluded from TTFT repetitions. With a hot page
cache it took about 11.35 seconds outside inference versus about 1.43 seconds
for llama.cpp. The retained gate/up preprocessing measured in Phase 12 accounts
for 9.013 seconds of Graph Horizon initialization. llama.cpp source and runtime
logs show no host repack, GPU conversion, or one-time shader-side weight
preparation: its matmul shader decodes canonical compressed blocks directly.

## Public performance

Runs were ordered Graph Horizon A, llama.cpp A, Graph Horizon B, llama.cpp B.
Graph Horizon used 15/10/5/3/3/3 repetitions from 128 through 28K and llama.cpp
used three repetitions per prefill point. The table averages the two bookends;
SD and CV are the averages of each run's sample statistics, not pooled values.

### Prefill

| Context | GH TTFT mean / median ms | GH SD / CV | llama pp mean / median ms | llama SD / CV | Ratio | Absolute gap ms |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 99.125 / 99.100 | 0.330 / 0.33% | 42.983 / 43.048 | 0.314 / 0.73% | 2.306x | 56.142 |
| 512 | 376.600 / 376.835 | 1.485 / 0.40% | 131.722 / 131.817 | 0.419 / 0.32% | 2.859x | 244.878 |
| 2,048 | 1,595.380 / 1,596.745 | 6.095 / 0.38% | 541.391 / 540.891 | 2.289 / 0.42% | 2.947x | 1,053.989 |
| 8,192 | 8,087.790 / 8,096.185 | 19.755 / 0.25% | 2,611.573 / 2,612.132 | 6.143 / 0.24% | 3.097x | 5,476.217 |
| 16,384 | 20,781.565 / 20,780.905 | 49.905 / 0.24% | 6,435.127 / 6,432.129 | 10.596 / 0.16% | 3.229x | 14,346.438 |
| 28,000 | 46,777.105 / 46,807.650 | 63.690 / 0.14% | 14,038.202 / 14,040.651 | 12.563 / 0.09% | 3.332x | 32,738.903 |

Graph Horizon prompt throughput falls from 1,291.35 token/s at 128 to 598.59
at 28K. llama.cpp falls from 2,978.03 to 1,994.56 token/s. The absolute gap
grows by 583x while context grows by 219x, ruling out fixed setup as the main
cause.

The llama.cpp `-pg N,1` model-side means were 53.802, 141.029, 549.933,
2,621.433, 6,453.402, and 14,059.779 ms. Their difference from pure prefill is
small beside the 28K competitive gap.

### Decode

Graph Horizon values again average the A/B bookends. The first llama.cpp 28K
series had a 7.7% CV and was rejected in full; the table is its five-repetition
replacement.

| Context | GH mean / median token/s | GH SD / CV | llama mean / median token/s | llama SD / CV | llama/GH |
|---:|---:|---:|---:|---:|---:|
| 128 | 45.950 / 45.925 | 0.195 / 0.42% | 102.780 / 103.038 | 0.551 / 0.54% | 2.237x |
| 2,048 | 45.280 / 45.300 | 0.125 / 0.28% | 95.064 / 95.234 | 0.589 / 0.62% | 2.099x |
| 8,192 | 39.785 / 39.790 | 0.150 / 0.38% | 78.208 / 78.277 | 0.215 / 0.27% | 1.966x |
| 16,384 | 34.315 / 34.300 | 0.115 / 0.33% | 63.010 / 63.033 | 0.100 / 0.16% | 1.836x |
| 28,000 | 28.580 / 28.560 | 0.065 / 0.22% | 50.016 / 49.861 | 0.430 / 0.86% | 1.750x |

Fresh timestamp categories explain why decode is closer. Graph Horizon values
are current-source GPU timestamps. llama.cpp values subtract the prompt-only
M=1 tail from paired `-pg N,1` timestamps; `other` is the residual to the public
decode wall and therefore includes logits and host/control time. The boundary
difference makes this diagnostic, not an additive competitive-gap budget.

| Context | Backend | Attention ms/token | Q/K/V/O ms/token | MLP ms/token | Other ms/token |
|---:|---|---:|---:|---:|---:|
| 128 | Graph Horizon | 0.262 | 5.451 | 11.202 | 4.484 |
| 128 | llama.cpp | 0.408 | 2.131 | 5.084 | 2.107 |
| 2,048 | Graph Horizon | 1.147 | 5.240 | 11.198 | 4.669 |
| 2,048 | llama.cpp | 1.042 | 2.131 | 5.482 | 1.864 |
| 28,000 | Graph Horizon | 13.340 | 5.168 | 11.602 | 4.865 |
| 28,000 | llama.cpp | 10.690 | 2.033 | 5.180 | 2.090 |

The same quantized weights are much closer at M=1 because neither backend uses
its large-M prefill mapping. llama.cpp explicitly selects scalar attention for
one query row and uses fused quantized GEMV. At 28K only 2.650 ms/token of the
decode gap is attention, while Graph Horizon's nearly fixed projections and MLP
remain about twice llama.cpp. Raw quant format alone therefore cannot explain
the prefill gap; large-M ownership/reuse and the prefill attention dataflow are
the differentiators.

## Fresh prefill timestamp reconstruction

Graph Horizon used the feature-gated Vulkan timestamp profiler. Each point was
an independent cold run plus a warm-up and three measured repetitions; the
stable value is `(paired aggregate - cold aggregate) / 3`. llama.cpp used
`GGML_VK_PERF_LOGGER=1`, one measured pass after its internal warm-up. The 128
point is directionally useful but has unavoidable first-kernel clock ramp.

| Context | Backend | Attention ms | Seven matmuls ms | Other ms | Total ms |
|---:|---|---:|---:|---:|---:|
| 128 | Graph Horizon | 1.452 | 115.938 | 7.694 | 125.084 |
| 128 | llama.cpp | 0.951 | 45.717 | 4.651 | 51.319 |
| 512 | Graph Horizon | 10.834 | 348.576 | 21.704 | 381.114 |
| 512 | llama.cpp | 4.656 | 119.600 | 11.019 | 135.275 |
| 2,048 | Graph Horizon | 148.974 | 1,359.972 | 71.774 | 1,580.720 |
| 2,048 | llama.cpp | 44.163 | 459.927 | 55.067 | 559.157 |
| 8,192 | Graph Horizon | 2,296.769 | 5,506.034 | 282.280 | 8,085.083 |
| 8,192 | llama.cpp | 616.135 | 1,819.414 | 224.341 | 2,659.890 |
| 28,000 | Graph Horizon | 26,920.199 | 19,037.603 | 954.505 | 46,912.307 |
| 28,000 | llama.cpp | 7,041.940 | 6,238.093 | 771.767 | 14,051.800 |

| Context | Attention GH/llama | Matmul GH/llama |
|---:|---:|---:|
| 128 | 1.527x | 2.536x |
| 512 | 2.327x | 2.915x |
| 2,048 | 3.373x | 2.957x |
| 8,192 | 3.728x | 3.026x |
| 28,000 | 3.823x | 3.052x |

Attention's rising ratio identifies a scaling/dataflow gap. Matmul converges
near a constant 3.05x once large-M kernels dominate, identifying a different
per-tile architecture/precision gap rather than hidden one-time dequantization.

### Detailed 28K budget

| Component | Graph Horizon ms | llama.cpp ms | Gap ms | Competitive-gap share |
|---|---:|---:|---:|---:|
| Q projection | 2,498.668 | 756.501 | 1,742.167 | 5.30% |
| K projection | 692.250 | 212.931 | 479.319 | 1.46% |
| V projection | 731.912 | 237.989 | 493.923 | 1.50% |
| Fused QK + softmax + AV | 26,920.199 | 7,041.940 | 19,878.259 | 60.49% |
| O projection | 2,560.639 | 666.644 | 1,893.995 | 5.76% |
| MLP gate | 3,330.681 | 1,432.227 | 1,898.454 | 5.78% |
| MLP up | 3,342.566 | 1,432.227 | 1,910.339 | 5.81% |
| Activation | 222.805 | 235.963 | -13.158 | -0.04% |
| MLP down | 5,880.887 | 1,499.574 | 4,381.313 | 13.33% |
| Norm/RoPE/KV/logits/embedding/residual | 731.700 | 535.804 | 195.896 | 0.60% |
| **Total** | **46,912.307** | **14,051.800** | **32,860.507** | **100.00%** |

Graph Horizon's Phase 15 causal stage probes allocate its fresh attention time
as 9,968.550 ms QK/load, 3,607.307 ms softmax, 9,422.070 ms V/AV, and 3,922.273
ms state/skeleton. llama.cpp exposes only one fused timestamp: QK, row maximum,
exponentiation/rescaling, row sum, and AV remain in cooperative-matrix state.
Assigning parts of its 7,041.940 ms to those labels would fabricate additive
measurements, so the table preserves the measured fused boundary. Exact dynamic
work and source-level traffic below reconstruct the internal stages without
pretending that overlapped instructions have independent wall times.

## Attention architecture and physical model

| Feature | Graph Horizon | llama.cpp |
|---|---|---|
| Algorithm | exact causal tiled online softmax | exact causal tiled online softmax, mask-tile skip |
| Ownership | one Q-head tile; explicit shared score/probability state | workgroup-scope cooperative matrices retain all online state |
| Q tile / KV tile | 32 / 64 | 64 / 64 |
| Workgroup / subgroups | 256 / 8 | 128 / 4 |
| Q/K/V/O precision | F16/F16/F16/F16 | F32/F16/F16/F32 |
| QK/softmax/AV accumulation | FP32 | FP32 (`GGML_PREC_F32`) |
| Registers/thread | 124 | 255 |
| Shared/workgroup | 38,272 B | 26,112 B |
| Theoretical resident WG / threads | 2 / 512, 33.3% thread occupancy | 2 / 256, 16.7% thread occupancy |
| Q logical traffic at 28K | 1.308 TB; Q reloaded per KV visit | 11.927 GB; Q loaded once per 64-row WG |
| K logical traffic | 2.615 TB | 1.308 TB |
| V logical traffic | 2.615 TB | 1.308 TB |
| Score/probability shared traffic | 5.254 TB | no explicit array round trip |
| Static control barriers | 4 | 0 |
| Matrix2 use | 2 MMA + cooperative accumulator, manual subgroup reductions | 3 MMA including `P*ones` row sum; cooperative reductions/per-element online state |
| Causal work | 326.156B useful pairs; 326.890B padded | same |

The driver reports stack size zero for both. Its local-memory value is the same
impossible 64 GiB sentinel observed in prior phases and is not treated as a
spill counter. Achieved hardware occupancy, DRAM, and cache counters were not
available for Vulkan on this setup. Driver executable properties, static
SPIR-V, exact work counts, logical traffic, and timestamps are used instead.

Static optimized SPIR-V reinforces the dataflow distinction. Graph Horizon's
attention module is 12,684 bytes and 770 instructions, with 134 loads, 70
stores, four control barriers, two cooperative MMA operations, and two
per-element operations. llama.cpp's specialized module is 39,324 bytes and
2,440 instructions including specialization branches, with 173 loads, 385
stores, no control barrier, three MMA operations, six cooperative reductions,
and fourteen cooperative per-element operations. These are static module
counts, not executed instruction counts.

The occupancy hypothesis is falsified: llama.cpp is 3.823x faster despite half
the modeled thread occupancy and more registers. The transferable principle is
larger workgroup-owned reuse and matrix-native online state, not higher
occupancy.

The source audit followed Graph Horizon's prefill attention shader and Vulkan
dispatch directly. On llama.cpp it followed `src/llama-graph.cpp` precision
selection, `ggml/src/ggml-vulkan/ggml-vulkan.cpp` capability/tile routing, and
`vulkan-shaders/{flash_attn_cm2,mul_mm_cm2}.comp` dataflow. The selected runtime
pipeline names and driver executable properties agree with those branches;
this is not a comparison against dormant source.

### Attention roofline

| Backend | Logical bytes | Useful/padded QK+AV FLOPs | Arithmetic intensity | Effective logical BW | Effective compute |
|---|---:|---:|---:|---:|---:|
| Graph Horizon | 12.792 TB including modeled shared traffic | 167.368 TF | 13.08 FLOP/B | 475 GB/s | 6.22 TFLOP/s |
| llama.cpp | 2.628 TB global tensor traffic; compiler-internal shared excluded | 167.368 TF | 63.69 FLOP/B | 373 GB/s | 23.77 TFLOP/s |

Graph Horizon uses about 36.16 logical bytes per useful pair when explicit
shared score/probability traffic is included; llama.cpp uses about 8.06 global
bytes per pair. llama.cpp is not faster because it reports a higher aggregate
logical bandwidth. It removes redundant traversal and exposes the unchanged
QK+AV work to Matrix2 more effectively.

## Matmul architecture and numeric boundary

| Feature | Graph Horizon | llama.cpp |
|---|---|---|
| Weight representation | Q4_K/Q6_K direct except persistent exact FP16 gate/up | canonical Q4_K/Q6_K direct |
| Large tile | token 64 x output 32 x K 128 | output 128 x token 256 x K 64 |
| Workgroup | 256 | 256 |
| Accumulator | FP32 | FP16 on the measured default path |
| Weight reuse | one weight traversal per 64-token tile | one compressed traversal per 256-token tile |
| Metadata/dequant | shader unpack/shared staging; absent for native gate/up | tensor-load decode callbacks consume block metadata inline |
| Cooperative mapping | Matrix2 input loads, one MMA, cooperative output | workgroup-scope Matrix2 tensor loads, decode callback, one MMA, cooperative output |
| Q4 resources | 98 registers, 33,792 B shared, 2 WG | 247 registers, 63,616 B shared, 1 WG on large path |
| Q6 resources | 108 registers, 33,792 B shared, 2 WG | 255 registers, 55,296 B shared, 1 WG on large path |
| Gate/up logical weight bytes | 1,289.648 GB total | 91.092 GB total |
| Gate/up effective logical BW | 193.3 GB/s | 31.8 GB/s |
| Gate/up useful compute | 12.35 TFLOP/s | 28.78 TFLOP/s |

For gate/up, useful work is 82.443 TFLOP. Graph Horizon's 1.472 GB FP16 tensor
is revisited across 438 64-row tiles per operation; llama.cpp's 414.056 MB
canonical compressed tensor is revisited across 110 256-row tiles. The logical
arithmetic intensities are about 63.9 and 905.1 FLOP/B respectively. These are
source-level logical bytes, not physical DRAM counters, but the 14.2x reduction
is large enough to explain why higher llama.cpp register pressure and lower
residency do not prevent a 2.33x gate/up speedup.

No preprocessing advantage is hidden inside the benchmark: llama.cpp neither
repacks nor transcodes the weights. Conversely, its FP16 accumulator is not
semantically interchangeable with Graph Horizon's retained FP32 accumulator.
The temporary F32 selector diagnostic used a separate Vulkan-only build under
`/tmp`; pp512 changed from the 131.722 ms bookend mean to 10,505.160 ms, or
48.738 token/s. The llama.cpp checkout was restored clean. This falsifies
"select llama's existing FP32 pipeline" as a candidate on the RTX 3060; it does
not falsify a new FP32 mapping designed around this hardware.

## Command, memory, and startup economics

At 28K Graph Horizon records about 73,762 dispatches and 62,322 barriers into
110 command buffers. Stable CPU costs are approximately 79.5 ms recording,
6.1 ms submission, and 6.6 ms descriptor work; GPU gaps above 50 microseconds
are negligible. llama.cpp's timestamp logger sees about 24,065 timed dispatches
and its graph queues at most 100 nodes per submission. The entire measured
`other` competitive gap is only 182.738 ms (0.56%). Command/control work is
therefore closed under the 3% threshold. The zero barriers in llama.cpp's
attention shader remain important as kernel dataflow, not host overhead.

Changing llama.cpp microbatch from 512 to 256 raised 28K prefill from
14,038.202 to 14,613.496 ms, a 575.293 ms or 4.10% llama.cpp penalty. It explains
only 1.76% of the Graph Horizon/llama.cpp wall gap and cannot be added again to
the component attribution.

Peak active VRAM is 8,260 MiB for Graph Horizon versus 5,191 MiB for llama.cpp,
a 3,069 MiB (37.2%) Graph Horizon premium. llama.cpp reports a 2,039.54 MiB
Vulkan model buffer and roughly 315 MiB CPU-mapped storage; F16 KV capacity
accounts for most of the remainder. Graph Horizon's persistent native gate/up
copy increased its prior peak by about 2.808 GiB. This is an explicit
throughput/startup/memory trade-off, not an unexplained runtime advantage.

## Confirmed and falsified causes

| Hypothesis | Evidence | Result |
|---|---|---|
| Less causal attention work | Both execute 326.156B useful pairs and about 326.890B padded pairs | Falsified |
| Better attention ownership/dataflow | 4.9x fewer modeled logical bytes/pair, no explicit score/probability round trip, 3.823x measured speedup | Confirmed |
| Higher occupancy | llama.cpp models 16.7% versus Graph Horizon 33.3% | Falsified |
| Fixed host/command overhead | Gap grows to 32.739 s; 28K residual is 0.56% | Falsified |
| Hidden weight preparation | Source and runtime show direct canonical Q4_K/Q6_K execution | Falsified |
| Large-M matmul mapping/reuse | Ratio converges near 3.05x; gate/up logical rereads differ 14.2x | Confirmed, numeric confound |
| Quant format alone | Same weights are much closer on M=1 decode | Falsified |
| Precision difference | llama default large-M path is `_f16acc`; forced existing F32 path is 79.7x slower at pp512 | Confirmed and non-transferable as-is |
| Microbatch 512 | llama ubatch-256 loses 575 ms at 28K | Confirmed but minor |
| Power/thermal decay | Stable clocks and low CV; no matching temperature curve | Falsified |

## Transferability and candidate gate

Classifications are: A directly transferable; B transferable with Graph Horizon
redesign; C dependent on llama-specific representation; D hardware-specific;
E irrelevant to the measured gap.

| Architectural difference | Class | Transfer decision |
|---|---|---|
| Matrix-native online max/sum/output state and no explicit score/probability shared round trip | B+D | Transfer as an Ampere fast path with the current kernel as fallback |
| Q64 ownership and one Q load per workgroup | B+D | Transfer together with the online-state redesign, not as another local tile edit |
| Canonical compressed weight decode inside Matrix2 loads | A for Q/K/V/O/down; B for gate/up | Graph Horizon already does the direct form except its retained native gate/up trade-off |
| FP16 large-M accumulation | Numeric incompatibility | Do not transfer without separate quality approval; it is outside the retained FP32 contract |
| 256-token matmul reuse | B+D | Reconsider only with a viable FP32 accumulator mapping |
| 512-row microbatch | B | Quantified but below the implementation gate |
| Smaller host graph / fewer dispatches | E | Residual is below 1%; close |

| Difference | Class | Current GH s | llama equivalent s | Maximum gap s | Realistically closable | Predicted removed s | Whole request |
|---|---|---:|---:|---:|---:|---:|---:|
| Workgroup Matrix2 online-attention state, Q64/KV64 | B+D | 26.920 | 7.042 | 19.878 | 50% | 9.939 | 21.19% |
| Large-BN compressed matmul with FP32 contract | B+D | 19.038 | 6.238 | 12.800 | 10% pending a viable FP32 mapping | 1.280 | 2.73% |
| 512-row microbatch economics | B | 46.912 total | measured llama delta | 0.575 | 100% upper bound | 0.575 | 1.23% |
| Host graph/command reduction | E | 0.955 other | 0.772 other | 0.183 | 100% upper bound | 0.183 | 0.39% |

The attention candidate is materially different from the Phase 15 local,
split-K, and Q-reuse variants. Competitive source evidence therefore reopens
attention only for this different matrix-native ownership/decomposition. Sparse
attention, native INT8, additional exact weight expansion, and command tuning
remain closed.

### Ranked Phase 19 candidates

| Rank | Candidate principle | Evidence / affected component | Competitive gap available | Predicted recoverable | Risk | Next action |
|---:|---|---|---:|---:|---|---|
| 1 | Ampere-gated workgroup Matrix2 online attention, Q64/KV64/WG128 | 19.878 s attention gap; same work, 4.9x traffic-model reduction | 19.878 s | 9.939 s, 21.19% TTFT | High: new K kernel, compiler/resource sensitivity | Prototype first in Phase 19 with generic fallback |
| 2 | New FP32 large-BN direct-quant Matrix2 mapping | 12.800 s matmul gap, but llama advantage includes FP16 accumulation | 12.800 s | 1.280 s, 2.73% TTFT until a viable FP32 kernel exists | Very high numeric/performance risk | Source-level design and local kernel screen only after rank 1 |
| 3 | Larger Graph Horizon prefill microbatch/ownership | llama 512→256 penalty is only 0.575 s | 0.575 s | 0.575 s, 1.23% TTFT ceiling | Medium memory/graph risk | Document; do not implement under gate |
| 4 | Fewer command buffers/dispatches | residual competitive gap 0.183 s | 0.183 s | 0.183 s, 0.39% TTFT ceiling | Low benefit | Close |

Rank 1 is expected to require roughly 180–260 productive category-K shader
lines plus no more than about 40 orchestration/routing lines and focused tests.
The maintenance cost is high but proportional to a 57.4%-affected component
and a predicted gain above 10%. Routing must require NVIDIA/Ampere-compatible
workgroup cooperative Matrix2 capabilities; every other device keeps the
current exact portable kernel. Public API and tolerances remain unchanged.

No Phase 18 prototype was executed. The mission permits a no-patch conclusion,
and its stop condition was already reached: 100% of the 28K timestamp gap is
attributed, more than three causes are quantitatively confirmed or falsified,
and rank 1 exceeds the preferred 10% whole-request prediction. Implementing a
large kernel after that point would mix competitive diagnosis with Phase 19
qualification without adding evidence required to close this phase.

## Final conclusion

The primary reason Graph Horizon loses prefill is its dense-attention
ownership/dataflow: repeated Q/K/V traffic, explicit shared score/probability
state, manual reductions, and barriers make the same causal QK+AV work 3.823x
slower at 28K. The secondary reason is a roughly 3.05x large-M matmul gap from
tile reuse and direct compressed execution, inseparable in llama.cpp's current
fast path from FP16 rather than FP32 accumulation. Decode is closer because M=1
selects scalar attention and quantized GEMV instead of either large-M prefill
architecture; its attention gap is only 2.650 ms/token at 28K.

Phase 19 should begin with the exact dense, Ampere-gated workgroup Matrix2
attention path described below. Its conservative predicted gain is 9.939
seconds, or 21.19% at 28K, and it is the only evidence-backed candidate above
the preferred 10% whole-request gate.

## Phase 19 acceptance contract

The prototype should preserve F16 input/KV/output and FP32 QK, softmax, and AV;
evaluate exact dense causal attention; load Q once per workgroup; keep online
max/sum/output in cooperative state; use matrix-native reductions and a
`P*ones` row sum; directly tensor-load K/V; and avoid global scores. It should
screen local attention, GPU prefill, and public TTFT at 2K, 8K, and 28K before
full qualification.

Keep it only if actual whole-request gain is at least 5%, or if a precisely
identified incomplete portion gives a convincing measured path to at least
10%. A retained implementation must pass the existing CPU/Vulkan F16, INT8,
Q4, Q6, attention, logits, greedy-sequence, and decode-guard checks without
tolerance changes. It must also preserve the current generic Vulkan fallback.

The 28K competitive target is at most 2.5x llama.cpp, or at most 35.096 seconds.
Graph Horizon must remove about 11.681 seconds from its public bookend mean.
The conservative attention prediction supplies 9.939 seconds, so rank 1 alone
is likely necessary but may not be sufficient for the final target. The
largest remaining bottleneck after a successful prototype should determine
whether Phase 19 continues within attention or returns to a numerically
equivalent large-M matmul design.
