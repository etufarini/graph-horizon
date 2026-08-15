<!--
This report owns Phase 11 evidence for the global Vulkan prefill bottleneck:
fresh whole-request attribution, quantized-matmul traffic/resource models,
the M128 reuse experiment, exact-path stop decision, and next-phase ranking.
It defines no runtime behavior or support contract.
-->

# Vulkan Global Prefill Phase 11

## Result

Phase 11 stops without a production change. Fresh 28K profiling attributes
99.96% of GPU prefill: exact attention is 26.344 s, the seven recurrent
quantized Matrix2 projections are 22.831 s, and every remaining GPU category
combined is 0.937 s. Attention remains the largest component, but Phase 10
already closed its local exact-representation candidates. The largest new
removable-cost hypothesis was therefore Q4_K weight/metadata reuse.

A shape-specific Q4 M128 prototype halves the number of Q4 workgroups and the
shader-visible quantized-weight/metadata traversal. It regresses the Q4 group
by about 4.4% and whole-request TTFT by 1.1% at 28K. The lower workgroup
residency and latency hiding cost more than the removed dequantization work.
The candidate is removed. Acquired Matrix2 geometry, fusion, intermediate,
control, and attention experiments then leave no exact-representation
component with a realistic 5% whole-request opportunity.

The final production tree is byte-identical to the Phase 10 baseline commit
`340e9af2127b8e0d73e42c76921f4793b6300791`. The investigation branch is
`perf/vulkan-global-prefill-phase11`; nothing is pushed.

## Phase 10 inherited state

The source-identical Phase 10 28K attention attribution is:

| Region | Time |
|---|---:|
| QK | 9.894 s |
| Softmax/online state | 3.582 s |
| AV | 9.352 s |
| Loop/address/conversion/synchronization | 3.891 s |
| **Attention total** | **26.719 s** |

Registers and shared memory independently hold the attention kernel at two
resident workgroups: 124 registers/thread, 38,272 B driver shared/WG, 16
resident warps, and 33.3% modeled occupancy. QK and AV Matrix2 utilization are
above 99.7%. K-hot has only a 1.13% whole-prefill ceiling, V-hot regresses
33.8%, and simple multi-Q reuse has about a 2.9% whole-request ceiling before
its likely occupancy loss. Phase 10 therefore satisfied stop conditions 5, 6,
and 7 and retained no production change. Local exact-attention tuning is not
reopened in Phase 11.

Phase 10's 117.49 ms final short measurement was an environmental miss against
the 114.61 ms qualified reference with source-identical production. Phase 11
therefore uses candidate/same-session-baseline ratios for the 128 guard.

## Baseline and invariants

The required starting branch was `perf/vulkan-long-prefill-phase10` at clean
commit `340e9af`. The complete Phase 10 report was read before measurement or
source changes. Phase 5--10 negative results are treated as acquired and are
not repeated.

The model is `Ministral-3-3B-Instruct-2512-Q4_K_M.gguf`, 2,147,023,008 bytes,
SHA-256 `9ed150d4367e68df0ac8e1540f6ddc65b42d0ee26378329d1ecbca60f93fc5f8`.
The device is an RTX 3060 (`0x2487`), driver 595.84, Vulkan device API 1.4.329,
12 GiB, 7,501 MHz memory clock, and a 170 W board limit. The profile is pure
Vulkan release, context 32,768, F16 KV, greedy sampling, and exact synthetic
prompts made from `" a"` repeated `tokens - 3` times.

The invariant is unchanged Q4_K/Q6_K storage, FP16 activation and output,
FP32 Matrix2 accumulation, GQA, exact causal attention, public APIs, routing,
decode behavior, and numeric tolerances. No precision or quantization change
is authorized.

Diagnostics-free TTFT uses independent cold processes. Long points use three
samples, enough to resolve the requested 3% screening delta. The profiler uses
one cold request and a separate warm-up plus three measured requests; stable
per-request categories are `(four-request total - cold) / 3`. Absolute wall
time shifted again relative to the historic 114.61 ms short reference, so only
same-session A/B is causal.

## Fresh Phase 11 baseline

| Context | TTFT mean | Median | SD | CV | Prompt tok/s | GPU prefill | Profiler fence wait |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 163.062 ms | 161.850 ms | 4.663 ms | 2.860% | 784.98 | 118.077 ms | 118.243 ms |
| 512 | 496.872 ms | 497.230 ms | 1.639 ms | 0.330% | 1,030.45 | 435.808 ms | 436.112 ms |
| 2K | 1,905.010 ms | 1,904.650 ms | 3.038 ms | 0.159% | 1,075.06 | 1,862.508 ms | 1,863.582 ms |
| 8K | 9,130.083 ms | 9,135.590 ms | 18.302 ms | 0.200% | 897.25 | 9,176.103 ms | 9,180.496 ms |
| 16K | 22,688.777 ms | 22,688.190 ms | 16.728 ms | 0.074% | 722.12 | 22,873.969 ms | 22,882.158 ms |
| 28K | 49,746.173 ms | 49,780.050 ms | 62.278 ms | 0.125% | 562.86 | 50,111.556 ms | 50,126.372 ms |

The diagnostics-free and profiled builds are separate evidence streams and are
not subtracted from one another. Profiler fence wait is reported instead of
mislabeling it as isolated GPU time; it contains the GPU critical path.

Active telemetry excludes model upload/unload samples and requires at least
80% GPU utilization with the resident model allocation present:

| Context | GPU utilization | SM clock | Memory clock | Power mean (range) | Temperature mean (range) | VRAM |
|---:|---:|---:|---:|---:|---:|---:|
| 8K | 99.69% | 1,932 MHz | 7,501 MHz | 162.70 W (70.66--170.10) | 69.55 C (64--71) | 5,871 MiB |
| 16K | 99.77% | 1,931 MHz | 7,501 MHz | 167.15 W (80.96--170.16) | 71.79 C (68--73) | 5,871 MiB |
| 28K | 99.79% | 1,925 MHz | 7,501 MHz | 168.37 W (81.42--170.11) | 73.09 C (71--74) | 5,868 MiB mean |

No row is capacity, downclock, thermal-throttle, or GPU-idle limited.

## Full 28K attribution

The normal profiler directly measures every non-attention row below. Repeating
the Phase 10 attention ablations would violate the acquired-negative-results
rule, so the source-identical Phase 10 bucket shares are normalized to the
fresh Phase 11 attention timestamp. Those four rows are explicitly a causal
model, not new simultaneous counters. The two recurrent RMSNorm classes have
identical shapes and counts; their tiny combined timestamp is split by dispatch
count, with one final-norm dispatch separated.

| Component | GPU time | GPU prefill | Evidence |
|---|---:|---:|---|
| Embedding/input | 344.437 ms | 0.687% | direct timestamp |
| Attention RMSNorm | 60.854 ms | 0.121% | dispatch-count split |
| Q projection | 2,442.041 ms | 4.873% | direct timestamp |
| K projection | 670.946 ms | 1.339% | direct timestamp |
| V projection | 714.930 ms | 1.427% | direct timestamp |
| RoPE | 71.967 ms | 0.144% | direct timestamp |
| QK | 9,755.132 ms | 19.467% | Phase 10 share normalized to fresh attention |
| Softmax/online state | 3,531.512 ms | 7.047% | Phase 10 share normalized to fresh attention |
| AV | 9,221.205 ms | 18.401% | Phase 10 share normalized to fresh attention |
| Attention loop/address/conversion/synchronization | 3,836.343 ms | 7.656% | Phase 10 share normalized to fresh attention |
| Attention output projection | 2,525.494 ms | 5.040% | direct timestamp |
| Residual kernels | 135.210 ms | 0.270% | direct timestamp |
| MLP RMSNorm | 60.854 ms | 0.121% | dispatch-count split |
| Gate projection | 5,374.538 ms | 10.725% | direct timestamp |
| Up projection | 5,352.310 ms | 10.681% | direct timestamp |
| SiLU/gate multiply | 215.040 ms | 0.429% | direct timestamp |
| Down projection | 5,750.082 ms | 11.474% | direct timestamp |
| KV write/copy | 26.519 ms | 0.053% | direct timestamp |
| Final norm | 0.021 ms | less than 0.001% | dispatch-count split |
| Logits | 3.796 ms | 0.008% | direct timestamp |
| GPU timestamp/dispatch residual | 18.327 ms | 0.037% | direct residual |
| **Total** | **50,111.556 ms** | **100%** | **99.96% kernel coverage** |

There is no standalone conversion kernel; narrowing and unpack conversions are
inside their owning kernels. Required barriers likewise remain inside kernel
timestamps rather than being double-counted as a separate duration.

Macro attribution is:

| Macro category | Time | GPU prefill |
|---|---:|---:|
| Exact attention | 26.344 s | 52.57% |
| Seven quantized Matrix2 projections | 22.831 s | 45.56% |
| Norm/RoPE/activation/residual elementwise | 0.544 s | 1.09% |
| Embedding | 0.344 s | 0.69% |
| KV copy/write | 0.027 s | 0.05% |
| Logits | 0.004 s | 0.01% |
| Timestamp/control residual | 0.018 s | 0.04% |

## CPU/control and GPU gaps

At 28K the profiler records 110 command buffers/submissions, 73,762
dispatches, 62,322 graph-level barriers, 76.885 ms CPU recording, 5.681 ms
submission, and 6.184 ms descriptor work (a subset of recording). Fence wait
is 50.126 s. The sum outside the GPU critical path is about 0.1 s, roughly
0.2% of the request.

GPU gaps above 50 us total 0.404 ms in the stable cold-subtracted result, or
0.0008% of GPU prefill. The largest cold-run gap was 0.216 ms. Control is
closed far below the 3% audit threshold and the workload is GPU-saturated.

## Quantized matmul inventory

All recurrent operations receive at most 256 rows per command. At 28K this is
109 full commands plus one 96-row tail, or 110 dispatches per layer and 2,860
per operation across 26 layers. The retained workgroup tile is M64/N32/K128,
256 threads and eight subgroups. Input/output are FP16 and accumulation is
FP32. Q4 is used in all layers except that V and down use Q6 in 13 of 26
layers.

| Op | Logical MxNxK | Format by layer | GPU s | Per layer | Registers/thread | Driver shared | Resident WG / occupancy | Matrix2 utilization | Useful compute | Logical BW |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Q | 28000x4096x3072 | 26 Q4 | 2.442 | 93.925 ms | 98 | 33,792 B | 2 / 33.3% | 99.886% | 7.502 TF/s | 270.2 GB/s |
| K | 28000x1024x3072 | 26 Q4 | 0.671 | 25.806 ms | 98 | 33,792 B | 2 / 33.3% | 99.886% | 6.826 TF/s | 245.8 GB/s |
| V | 28000x1024x3072 | 13 Q4 + 13 Q6 | 0.715 | 27.497 ms | 98 / 108 | 33,792 B | 2 / 33.3% | 99.886% | 6.406 TF/s | 237.2 GB/s |
| O | 28000x3072x4096 | 26 Q4 | 2.525 | 97.134 ms | 98 | 33,792 B | 2 / 33.3% | 99.886% | 7.254 TF/s | 260.6 GB/s |
| Gate | 28000x9216x3072 | 26 Q4 | 5.375 | 206.713 ms | 98 | 33,792 B | 2 / 33.3% | 99.886% | 7.670 TF/s | 276.2 GB/s |
| Up | 28000x9216x3072 | 26 Q4 | 5.352 | 205.858 ms | 98 | 33,792 B | 2 / 33.3% | 99.886% | 7.702 TF/s | 277.3 GB/s |
| Down | 28000x3072x9216 | 13 Q4 + 13 Q6 | 5.750 | 221.157 ms | 98 / 108 | 33,792 B | 2 / 33.3% | 99.886% | 7.169 TF/s | 263.8 GB/s |

The resource values are the acquired Phase 7 executable-property results for
unchanged SPIR-V: Q4/Q6 use 98/108 registers/thread, 16,384 B shader-declared
and 33,792 B driver-reported shared memory, zero stack, two resident
workgroups, 16 resident warps, and 33.3% modeled thread occupancy. Fresh SASS,
spill, and hardware-counter access remains unavailable; no physical DRAM or
spill claim is fabricated.

Current SPIR-V assembly contains the work expected from source. Q4/Q6 have
106/115 static loads, 70 stores, 5/4 conversions, 12/16 shifts, 10/18 integer
multiplications, 3/4 integer divisions, and 18/21 access chains. Each contains
two cooperative load sites, one cooperative Matrix2 multiply-add, and one
cooperative store. The compiler has not already removed the unpack/address
work targeted by reuse.

## Weight, metadata, activation, and output traffic

The source-exact model counts one unique stored matrix as the theoretical
minimum and one traversal per M64 workgroup as actual shader-visible work.
There are 438 row workgroups across the 28K command batches, hence 438x
quantized-weight and metadata amplification. Cache means this is not physical
DRAM traffic. Hardware physical-byte counters are unavailable, so effective
rates below are explicitly logical rates.

| Op | Minimum quant | Minimum metadata | Actual quant | Actual metadata | Activation | Output | Logical weight BW |
|---|---:|---:|---:|---:|---:|---:|---:|
| Q | 0.164 GB | 0.020 GB | 71.647 GB | 8.956 GB | 573.177 GB | 5.964 GB | 33.0 GB/s |
| K | 0.041 GB | 0.005 GB | 17.912 GB | 2.239 GB | 143.294 GB | 1.491 GB | 30.0 GB/s |
| V | 0.051 GB | 0.005 GB | 22.390 GB | 2.379 GB | 143.294 GB | 1.491 GB | 34.6 GB/s |
| O | 0.164 GB | 0.020 GB | 71.647 GB | 8.956 GB | 573.177 GB | 4.473 GB | 31.9 GB/s |
| Gate | 0.368 GB | 0.046 GB | 161.206 GB | 20.151 GB | 1,289.648 GB | 13.418 GB | 33.7 GB/s |
| Up | 0.368 GB | 0.046 GB | 161.206 GB | 20.151 GB | 1,289.648 GB | 13.418 GB | 33.9 GB/s |
| Down | 0.460 GB | 0.049 GB | 201.507 GB | 21.410 GB | 1,289.648 GB | 4.473 GB | 38.8 GB/s |
| **Total** | **1.615 GB** | **0.192 GB** | **707.515 GB** | **84.242 GB** | **5,301.886 GB** | **44.728 GB** | **34.7 GB/s** |

Total logical movement is about 6.138 TB in 22.831 s, or 269 GB/s, while the
seven operations execute 169.467 useful TFLOP at 7.42 TFLOP/s. The nominal
360 GB/s logical-traffic floor is 17.05 s and the 61.1 TFLOP/s dense-compute
floor is 2.77 s; neither is a directly attainable DRAM or application roof.
Measured behavior and prior ablations classify the path as tensor-load,
unpack/dequant, shared-synchronization, issue, and occupancy bound—not raw
weight-bandwidth bound and not dense Matrix2 compute bound.

Matrix2 N and K dimensions are exact for every operation. M utilization is:

| Context | Useful work | Executed work | Utilization |
|---:|---:|---:|---:|
| 128 | 0.775 TFLOP | 0.775 TFLOP | 100% |
| 512 | 3.098 TFLOP | 3.098 TFLOP | 100% |
| 2K | 12.395 TFLOP | 12.395 TFLOP | 100% |
| 8K | 49.581 TFLOP | 49.581 TFLOP | 100% |
| 28K | 169.467 TFLOP | 169.661 TFLOP | 99.886% |

Tile underutilization is closed. The only 28K waste is 32 padded rows in the
last 96-row command.

The next matmul residency threshold is three workgroups. Q4/Q6 must fall from
98/108 to at most 85.33 registers/thread, reductions of 12.67/22.67. Driver
shared is 33,792 B/WG; three workgroups use 101,376 of 102,400 B, leaving only
1,024 B/SM. A fourth workgroup would require at most 64 registers/thread and
25,600 B shared/WG, so it is not a local cleanup target. M128 instead moves in
the wrong direction: its modeled 166 Q4 registers require a 38-register
reduction merely to restore two workgroups.

## Intermediate tensor audit

The table records the minimum materialized producer-to-consumer round-trip,
not repeated reads internal to attention or Matrix2. All tensors are FP16.
Sizes use logical 28K rows; scratch buffers are reused between layers.

| Tensor | Shape | Bytes/layer | Producer -> consumer | Writes / reads per layer | Minimum request traffic |
|---|---|---:|---|---:|---:|
| Gate output | 28000x9216 | 516.096 MB | gate -> SiLU/multiply | 1 / 1 | 26.837 GB |
| Up output | 28000x9216 | 516.096 MB | up -> SiLU/multiply | 1 / 1 | 26.837 GB |
| Activated gate product | 28000x9216 | 516.096 MB | SiLU/multiply -> down | 1 / 1 | 26.837 GB |
| Q projection output | 28000x4096 | 229.376 MB | Q -> RoPE/attention | 1 / 1 minimum | 11.928 GB |
| Attention output | 28000x4096 | 229.376 MB | attention -> O | 1 / 1 | 11.928 GB |
| Hidden/norm/O/residual value | 28000x3072 | 172.032 MB | adjacent block operations | 1 / 1 per boundary | 8.946 GB per boundary |
| K projection output | 28000x1024 | 57.344 MB | K -> RoPE/KV write | 1 / 1 minimum | 2.982 GB |
| V projection output | 28000x1024 | 57.344 MB | V -> KV write | 1 / 1 minimum | 2.982 GB |

The three 9216-wide MLP round-trips lead by bytes, but their combined 80.511
GB costs only 0.224 s at 360 GB/s. That physical byte ceiling is much smaller
than the compute/dequant kernels surrounding it. Q/K/V and attention/O
materializations are smaller still, which is why fusion gates use measured
time rather than byte rank alone.

## Projection and intermediate fusion gates

Q/K/V total 3.828 s. Their weight streams remain distinct. Sharing their
input removes at most two unique 4.473 GB activation reads plus roughly 5.7 ms
of CPU dispatch recording. At 300 GB/s the byte ceiling is 29.8 ms, under
0.1% of the request; a complex QKV kernel cannot reach the 5% gate.

Gate/up total 10.727 s, but their two distinct weight streams dominate the
useful work. Acquired Phase 3 shared-A prototypes already removed half of the
modeled activation staging and produced only a 0.43% GPU improvement at best;
another variant regressed. The current direct Matrix2 tensor load does not
provide a new global round-trip large enough to overturn that result.

The 28K gate/up/activation/down path is 16.692 s. The activation kernel itself
is only 0.215 s. Eliminating all 26.8 GB of the final activation-to-down
round-trip at 300 GB/s is another 0.089 s ceiling; even impossible combined
deletion is under 0.7% of the request. A mega-kernel would also combine
incompatible output and down-projection mappings. Norm-to-projection,
Q/K/V-to-RoPE, attention-to-O, and residual-to-next-norm have still smaller
direct timestamps or mandatory materialization boundaries.

## Removable-cost ranking and revised Amdahl table

Ranking uses realistically removable milliseconds, not component size:

| Rank | Component | Current | Practical floor | Removable ceiling | Predicted whole gain | Decision |
|---:|---|---:|---:|---:|---:|---|
| 1 | Seven quantized matmuls | 22.831 s | about 22.146 s | about 0.685 s | about 1.37% | local geometries and reuse closed |
| 2 | Exact attention local residual | 26.344 s | about 25.839 s | about 0.505 s | about 1.01% | Phase 10 closed |
| 3 | Gate/up shared input | 10.727 s affected | at least 10.512 s | at most 0.215 s | at most 0.43% | acquired prototype below gate |
| 4 | Activation/intermediate traffic | 0.304 s optimistic | nonzero | less than 0.304 s | less than 0.61% | below gate |
| 5 | Elementwise outside fusion | 0.544 s | about 0.47 s | about 0.07 s | about 0.14% | below gate |
| 6 | CPU/control/GPU idle | about 0.101 s | about 0.05 s | about 0.05 s | about 0.10% | below gate |

The matmul practical floor applies the acquired at-most 3% remaining local
headroom after M64. It is supported by the fully occupied 256-row rate, failed
M128 experiment, exact Matrix2 utilization, and acquired N64/K256 geometry and
fusion results. The attention ceiling combines the favorable half K-hot and
half invalid barrier upper bounds from Phase 10; it is not an implementable
candidate.

| Component | Current s | Request share | Realistic local speedup | New s | Removable s | Predicted TTFT gain |
|---|---:|---:|---:|---:|---:|---:|
| Quantized matmul | 22.831 | 45.56% | at most 1.03x | 22.146 | 0.685 | 1.37% |
| Exact local attention | 26.344 | 52.57% | about 1.02x upper | 25.839 | 0.505 | 1.01% |
| Elementwise | 0.544 | 1.09% | about 1.15x | 0.473 | 0.071 | 0.14% |
| Embedding/KV/logits | 0.375 | 0.75% | about 1.05x | 0.357 | 0.018 | 0.04% |
| Control/residual | 0.018 GPU plus 0.083 CPU | 0.20% | at most 2x | about 0.051 | about 0.050 | 0.10% |

These ceilings overlap and must not be summed as an implementation plan. No
single remaining exact component reaches 5%, and no compatible set has a
resource-feasible architecture that does.

## Experiment P11-M128-Q4

### Pre-code gate

Target: Q4-backed Q/K/V/O/gate/up/down work, about 19.4 s of the fresh 50.1 s
GPU request. The candidate changes only Q4 Matrix2 from M64/N32/K128 to
M128/N32/K128; Q6 remains M64. It conceptually raises row reuse from two to
four former M32 tiles and halves Q4 quant/metadata workgroups.

The acquired M32-to-M64 delta implied that about 48% of current local Q4 time
could still be associated with the remaining dequant traversal. Halving that
gave an optimistic 24% local and 9% whole-request ceiling, above the 5% coding
gate. The resource model predicted about 166 registers/thread from Phase 7
growth, 24,576 B shader shared plus 8,192 B driver reservation, one resident
workgroup, eight resident warps, 16.7% modeled occupancy, and no credible room
for another resident workgroup. Resident output rows remain 128, but active
warps halve. Registers and spills are modeled because executable counters are
not available in this session.

### Same-session A/B

The experiment was built and run on the dedicated Phase 11 branch, then fully
removed. Baseline values are the mean of the surrounding M64 measurements.
The profiler's retained kernel name still says M64 during the temporary probe;
grid X=2 rather than 4 and the source diff prove M128 selection.

| Context | Metric | Baseline | M128 Q4 | Delta |
|---:|---|---:|---:|---:|
| 8K | TTFT | 9,154.235 ms | 9,254.380 ms | +1.09% |
| 8K | GPU prefill | 9,056.943 ms | 9,156.653 ms | +1.10% |
| 8K | all seven matmuls | 6,562.668 ms | 6,721.220 ms | +2.42% |
| 8K | Q4 Matrix2 group | 5,577.446 ms | 5,758.593 ms | +3.25% |
| 28K | TTFT | 49,533.165 ms | 50,081.310 ms | +1.11% |
| 28K | GPU prefill | 49,311.845 ms | 49,858.388 ms | +1.11% |
| 28K | all seven matmuls | 22,447.114 ms | 23,260.840 ms | +3.63% |
| 28K | Q4 Matrix2 group | 19,079.187 ms | 19,925.034 ms | +4.43% |

At 28K candidate attention happens to be 1.0% faster than the surrounding
average, masking part of the matmul regression. The targeted component itself
is unambiguous. Weight and metadata traffic fall as designed, but lower
residency makes compute/dequant issue slower. The candidate is **REJECTED**.
No candidate source, capability change, diagnostic name, or instrumentation
remains.

Acquired materially different alternatives already cover Matrix2 N64 output
ownership (regression), K256 (regression), KHR subgroup output partitioning
(slower fallback), Q4/Q6 metadata hoisting (retained and present), shared-A
gate/up (below 1%), activation fusion (below 1%), and Phase 10 multi-Q
attention reuse (below 3% whole before overhead). Repeating them without a new
architecture would violate the mission gate.

## Final performance and guards

No production candidate is retained, so baseline and final are source-identical:

| Context | Baseline TTFT | Final TTFT | Delta |
|---:|---:|---:|---:|
| 128 | 163.062 ms mean | identical source | 0% by construction |
| 512 | 496.872 ms mean | identical source | 0% by construction |
| 2K | 1,905.010 ms mean | identical source | 0% by construction |
| 8K | 9,130.083 ms mean | identical source | 0% by construction |
| 16K | 22,688.777 ms mean | identical source | 0% by construction |
| 28K | 49,746.173 ms mean | identical source | 0% by construction |

The 128 absolute baseline is 42% above the qualified 114.61 ms historical
guard while the long rows are not similarly displaced. There is no retained
software delta: same-session candidate comparisons reject M128, and final
runtime/shader source matches `340e9af`. This is another environmental short
guard miss, not a production regression claim.

Phase 10 decode controls remain 45.52 tok/s at 128, 45.02 at 2K, and 28.54 at
28K. Decode never routes through the temporary prefill-only M128 shader, and
the final source is identical, so the final decode delta is 0% by
construction. No attention source or shared resource is changed; the 28K
attention guard is likewise 0% by construction.

Final source-restored verification:

- Vulkan-profile library suite: 152 passed, four intentionally ignored;
- authenticated 128-token greedy sequence:
  `2757,10637,2479,1636`;
- Q4_K, Q6_K, Matrix2, GQA, F16/INT8 attention, sequential decode parity,
  logits, failure lifecycle, and resource release are exercised by the suite;
- Phase 10's source-identical long numeric qualification remains the applicable
  result: F16 and bit-exact INT8 through 28K without changed tolerances;
- release `vulkan-profile` benchmark and all shaders build after restoration;
- no production diff exists against `340e9af`.

## Exact-path stop rationale

Phase 11 satisfies stop conditions 1, 2, 3, 4, and 6:

1. No remaining exact-representation component has a realistic 5% predicted
   whole-request gain.
2. Quantized Matrix2 uses exact tiles, runs near its logical movement rate,
   retains all acquired metadata/K128/M64 wins, and loses when reuse rises.
3. QKV, gate/up, activation/down, norm, residual, and RoPE fusion remove too few
   critical-path bytes or have acquired negative prototypes.
4. CPU control is about 0.2% and GPU idle is 0.0008%.
5. The next credible jumps change weight/activation representation or the
   attention primitive/algorithm.

The present exact-representation optimization cycle is therefore closed. This
does not mean attention or matmul are physically optimal; it means no measured,
supported, resource-feasible architecture isolates at least 5% more whole
request time under the current representation and graph.

## Next architectural phase

The ranking below is theoretical and authorizes no implementation:

| Rank | Architectural change | Current affected time | Bytes/FLOPs potentially removed | Predicted whole gain | Memory impact | Numeric/correctness risk | Complexity |
|---:|---|---:|---|---:|---|---|---|
| 1 | Predecoded or native tensor weight representation | 22.831 s matmul | much of 0.792 TB logical quant/metadata traversal and unpack issue | about 9--18% if 20--40% local | FP16 expansion could add about 5 GB and approach 12 GiB capacity | representation change; must preserve current dequant rounding | high |
| 2 | Different exact attention primitive/runtime mapping | 26.344 s attention | raise 33.3% occupancy and remove shared/state handoffs; 167.4 TFLOP remains exact | about 5--10% for 10--20% local | similar KV, kernel scratch may change | exact semantics possible but reduction-order parity must be requalified | very high |
| 3 | Lower/native activation representation | 22.831 s matmul | up to half of 5.302 TB logical activation traversal | about 5--10% practical target | lower transient traffic; conversion buffers possible | activation precision/order change | high |
| 4 | Lower-precision KV/attention state | 26.344 s attention | nominally halve K/V bytes, but Phase 10 K/V probes show weak byte sensitivity | likely below 5%, upper about 8% | KV can fall by about half | attention/logit drift and long-context quality | high |
| 5 | Sparse/windowed attention | 26.344 s attention | an 8K window removes about 51% of 28K causal pairs | about 27% if time tracks pairs | KV may remain or shrink with eviction | changes exact causal semantics | high |
| 6 | Radical cross-operation/cross-layer fusion | most of the graph | avoid selected intermediates and submissions | unproven; ordinary local fusion is below 1% | larger live state and code | exactness possible, lifecycle risk | very high |

Persistent FP16 weights are the most direct next measured bottleneck pivot:
they can remove dequant issue without the M128 occupancy loss, but their memory
cost and representation contract require a separate mission. A new attention
primitive ranks next because it could attack occupancy rather than repeat the
closed local attention edits. Sparse/windowed attention offers the largest
asymptotic jump but is explicitly non-exact.
