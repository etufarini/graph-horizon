<!--
This checkpoint owns the 2026-08-21 NVIDIA Vulkan long-context prefill
investigation: fresh measurements, attribution, acquired frontier closures,
routing and residency audits, and the final quantitative stop decision.
-->

# NVIDIA Vulkan Long-Context Prefill

## Outcome

The reported `15,435`-token symptom (`213.24 s`, `72.4 tok/s`) does not
reproduce at the current production revision. Fresh 8B all-GPU Vulkan-hybrid
measurement is `22.062 s` (`699.63 tok/s`), 9.67 times faster than the symptom.
The workload-matching 14B capacity tuple is `33.945 s` (`454.71 tok/s`) with
all 40 layers resident. A full 8B request with 954 generated tokens completes
the prefill/TTFT boundary in `21.702 s`, decodes at `27.62 tok/s`, and takes
about `56.24 s` end to end.

The historical cause was a routing defect, not a missing kernel: capable
all-GPU hybrid execution retained the old 32-row prefill batch instead of the
qualified NVIDIA Matrix2 batch. The retained capability-gated fix in `2228a7a`
routes eligible NVIDIA all-GPU plans through 512 rows while preserving the
32-row capacity fallback. Its acquired 14B/15K result was
`230.888 -> 33.549 s` (-85.47%), with unchanged decode. Current source also
contains the direct canonical-Q4 Matrix2 path, staged exact-Q6 Matrix2 path,
and Q64 matrix-native online attention that made that wider route useful.

No new production change is retained in this investigation. Fresh current-
source profiling accounts for 99.96--99.97% of GPU prefill and leaves no
untried exact candidate above the economic gates. At 28K, the optimistic
remaining exact-attention removal is 3.70 seconds, only 7.64% of whole prefill
and below the 10% complex-redesign gate. Exact Q4/Q6 ownership,
representations, fusion, orchestration, and attention alternatives are already
bracketed by acquired experiments. The result is therefore a quantitative
global stop, not a claim that the hardware is at a physical limit.

## Reproducibility

| Item | Value |
|---|---|
| Date | 2026-08-21 Europe/Rome |
| Branch | `perf/nvidia-long-context-prefill-20260821` |
| Baseline / production SHA | `e3d23f2102661c5cb95310a11873ad974ab2c831` (no production delta) |
| Checkpoint / integration | branch tip `011952b`; merged by PR #38 (`8884465`) |
| GPU | NVIDIA GeForce RTX 3060 12 GiB, GA104/Ampere, PCI device `0x2487` |
| Driver / Vulkan | NVIDIA 595.84 / device API 1.4.329, loader 1.4.341 |
| Subgroup / capabilities | 32 lanes; FP16; integer dot; KHR cooperative matrix; NV Matrix2 |
| Physical limits | 28 SMs; 360.048 GB/s nominal memory; 2.25 MiB L2; 65,536 registers/SM; 102,400 B shared/SM |
| 8B artifact | `Ministral-3-8B-Instruct-2512-Q4_K_M.gguf`, 5,198,911,904 bytes |
| 14B artifact | `Ministral-3-14B-Instruct-2512-Q4_K_M.gguf`, 8,239,593,024 bytes |
| Runtime | locked release; Vulkan-hybrid all-GPU; F16 KV; greedy; context capacity 32,768 (8B) or 16,384 (14B) |
| Public protocol | exact `N` prompt tokens, one warm-up, three measured repetitions, 32 requested tokens |
| Profiler | isolated rebuild of `fdb58c6`; same current 512-row prefill shaders/routes, Vulkan timestamps |
| Comparator | llama.cpp `9bebfcb4b`, Vulkan0, full offload, flash attention, F16 K/V, batch/ubatch 512 |

The synthetic Graph Horizon prompt is `N - 3` copies of `" a"`; the chat
template contributes the other three tokens. Compilation, model loading,
pipeline creation, and the public warm-up are outside measured TTFT. The
profiler revision predates later decode-only and hybrid-planning edits, but its
pure-Vulkan prefill shaders and 512-row NVIDIA routing are unchanged relative
to production; this was verified by source diff before use. No retired
diagnostics were restored to the production worktree.

Representative commands:

```sh
cargo build --locked --release --example bench \
  --no-default-features --features vulkan-hybrid

target/release/examples/bench <8b.gguf> --context 32768 --kv f16 \
  --prompt <exact-N-token-prompt> --max-tokens 32 --warmup 1 --reps 3

llama-bench -m <8b.gguf> -p 128,512,2048,8192,15435,28000 -n 0 \
  -b 512 -ub 512 -ctk f16 -ctv f16 -ngl 99 -fa on -dev Vulkan0 \
  -r 3 -o jsonl
```

## Fresh And Final Prefill / Competitive Matrix

Graph Horizon reports public TTFT, which is the measured CPU wall boundary and
includes tokenization and first sampling/emission. llama.cpp reports pure
prompt processing. The difference is not subtracted. All Graph Horizon CVs are
below 0.75%.

| Context | TTFT mean | Median | Stddev | CV | Our tok/s | GPU prefill | llama.cpp | llama tok/s | Ratio | Gap |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 205.06 ms | 205.77 ms | 1.34 ms | 0.66% | 624.23 | 166.55 ms | 97.71 ms | 1,310.06 | 2.10x | 107.35 ms |
| 512 | 582.77 ms | 581.70 ms | 2.25 ms | 0.39% | 878.57 | 535.74 ms | 285.80 ms | 1,791.49 | 2.04x | 296.97 ms |
| 2,048 | 2,238.81 ms | 2,239.32 ms | 2.87 ms | 0.13% | 914.77 | 2,195.13 ms | 1,175.59 ms | 1,742.10 | 1.90x | 1,063.22 ms |
| 8,192 | 10,033.02 ms | 10,025.37 ms | 24.99 ms | 0.25% | 816.51 | 10,112.00 ms | 5,279.64 ms | 1,551.62 | 1.90x | 4,753.38 ms |
| 15,435 | 22,061.52 ms | 22,066.17 ms | 15.29 ms | 0.07% | 699.63 | 22,050.36 ms | 11,419.29 ms | 1,351.66 | 1.93x | 10,642.23 ms |
| 28,000 | 48,436.05 ms | 48,464.98 ms | 54.62 ms | 0.11% | 578.08 | 48,622.31 ms | 24,727.57 ms | 1,132.34 | 1.96x | 23,708.48 ms |

GPU values come from a separate timestamped pure-Vulkan run. At 128--512 the
public all-GPU hybrid boundary has about 40--55 ms more fixed overhead than the
pure profile; at 8K and above the separate-run difference changes sign or is
below 0.4%. It is not treated as additive CPU time. A same-session 8B/128
hybrid/pure/hybrid check measured `204.98 / 149.66 / 204.21 ms`; this is an
existing frontend/profile difference, not a regression from this source-
identical investigation. Decode remained `35.1--35.3 public deltas/s` in that
check.

The curve has distinct regions. Efficiency improves through 2K as fixed work
and the 512-row architecture amortize. Attention then raises milliseconds per
token from 1.093 at 2K to 1.225 at 8K, 1.429 at 15K, and 1.730 at 28K. Its GPU
share grows from 4.80% at 2K to 16.41%, 27.42%, and 40.50%. No residency,
allocation, or matmul route crossover accompanies this change; dense causal
attention is the growing term.

## Fresh 8K Attribution

A clean single-request timestamp rerun after the completion audit measures a
10.084-second profiler wall boundary. Kernel timestamps cover 99.96% of GPU
time and the complete wall boundary reconciles to rounding:

| Component | Seconds | Wall % | Dispatch invocations |
|---|---:|---:|---:|
| Exact attention | 1.655 | 16.41% | 544 |
| Q / K / V projections | 0.571 / 0.185 / 0.268 | 5.66% / 1.84% / 2.66% | 544 each |
| O projection | 0.570 | 5.66% | 544 |
| Gate / up / activation | 1.553 / 1.554 / 0.192 | 15.40% / 15.41% / 1.90% | 544 / 544 / 1,632 |
| Down projection | 3.256 | 32.28% | 544 |
| Normalization / RoPE / KV write | 0.066 / 0.024 / 0.009 | 0.65% / 0.24% / 0.09% | 1,089 / 1,088 / 544 |
| Embedding / logits | 0.079 / 0.003 | 0.79% / 0.03% | 8,192 / 1 |
| GPU timestamp residual | 0.004 | 0.04% | n/a |
| CPU recording / submission | 0.019 / 0.001 | 0.19% / 0.01% | 16 command buffers |
| Frontend, driver, and first-token boundary | 0.074 | 0.73% | wall reconciliation |
| **Total** | **10.084** | **100%** | **16,898 dispatches; 14,722 barriers** |

The fused attention dispatch cannot expose live sub-stages independently.
Applying the acquired same-shader Q64 causal proportions gives approximately
1.102 seconds Q/K load plus QK, 0.104 seconds online softmax/state, 0.423
seconds probability/V plus AV, and 0.026 seconds other attention. These are
normalized estimates, not independently timestamped counters.

## Fresh 15K Attribution

The table uses the 22.179-second profiler wall boundary so it reconciles
exactly. Direct timestamp categories are fresh. The four internal attention
rows apply the acquired additive Q64 causal-stage proportions to the fresh
6.045-second attention total; they are labeled estimates because Vulkan cannot
timestamp live QK, online softmax, and AV regions independently inside one
fused dispatch.

| Component | Seconds | Wall % | Evidence / actual route |
|---|---:|---:|---|
| Q/K load + QK Matrix2 | ~4.024 | ~18.14% | acquired Q64 causal split, normalized fresh |
| online max/exp/sum/rescale | ~0.380 | ~1.71% | acquired Q64 causal split, normalized fresh |
| probability/V load + AV Matrix2 | ~1.545 | ~6.97% | acquired Q64 causal split, normalized fresh |
| attention loop/output residual | ~0.097 | ~0.44% | acquired Q64 causal split, normalized fresh |
| Q projection | 1.099 | 4.96% | direct Q4 Matrix2 |
| K projection | 0.378 | 1.71% | direct Q4 Matrix2 |
| V projection | 0.522 | 2.35% | 17 Q4 / 17 staged exact-Q6 layers |
| O projection | 1.101 | 4.96% | direct Q4 Matrix2 |
| MLP gate | 2.997 | 13.51% | direct Q4 Matrix2 |
| MLP up | 2.992 | 13.49% | direct Q4 Matrix2 |
| activation / residual | 0.363 | 1.64% | fused SiLU-mul and residual kernels |
| MLP down | 6.205 | 27.98% | 17 direct-Q4 / 17 staged exact-Q6 layers |
| normalization / final norm | 0.125 | 0.56% | 2,109 RMSNorm dispatches |
| RoPE | 0.048 | 0.21% | Q and K only |
| KV append/write | 0.017 | 0.08% | direct F16 writes; no copy/reformat |
| embedding | 0.148 | 0.67% | Q4 lookup |
| logits | 0.003 | 0.01% | Q6, final prompt row only |
| GPU timestamp residual | 0.008 | 0.04% | GPU total minus classified kernels |
| CPU command recording | 0.034 | 0.16% | 31 command buffers, 32,301 dispatches |
| queue submission | 0.003 | 0.01% | 31 synchronous submissions |
| frontend/driver/first-token boundary | 0.092 | 0.41% | profiler wall reconciliation |
| **Total** | **22.179** | **100%** | **99.96% GPU-kernel coverage** |

Descriptors consume 16.82 ms but are a subset of recording. Fence wait is
22.054 seconds and overlaps GPU execution; it is not added. Steady prefill
allocates, resizes, maps, and creates zero resources. The measured 213-second
budget therefore no longer exists at current production source. The dominant
current 15K costs are down projection, exact attention, gate, and up—not CPU
fallback, KV management, score materialization, or synchronization.

The first-token tail performs one final normalization, one Q6 logits dispatch
for the final prompt row, one GPU argmax dispatch, and one four-byte token
readback before public emission. It does not compute logits for all prompt rows
or run an extra decode token. Logits consume 2.61 ms GPU and the entire
frontend/driver/first-token reconciliation is 91.99 ms, so this tail has no
material hidden candidate.

## 28K Attribution And Routing

Fresh 512-row timestamps measure 48.622 seconds GPU time with 99.97% classified
kernel coverage:

| Component | GPU seconds | Share |
|---|---:|---:|
| Exact attention | 19.691 | 40.50% |
| Q | 1.984 | 4.08% |
| K | 0.656 | 1.35% |
| V | 0.932 | 1.92% |
| O | 1.981 | 4.07% |
| Gate | 5.420 | 11.15% |
| Up | 5.421 | 11.15% |
| Down | 11.234 | 23.11% |
| Norm / RoPE / KV / elementwise / embedding / logits | 1.289 | 2.65% |
| GPU residual | 0.014 | 0.03% |

The request records 55 command buffers, 57,922 dispatches, and 50,442 graph
barriers. CPU recording/submission is 61.02/4.64 ms, descriptor work is
29.73 ms inside recording, and the fence wait is 48.629 seconds. GPU execution
therefore owns the critical path; dispatch-count fusion cannot deliver a
material whole-request gain. The separate profiler wall is 48.799 seconds and
reconciles as 48.622 seconds GPU, 0.061 seconds recording, 0.005 seconds
submission, and 0.111 seconds frontend/driver/first-token boundary. Thus the
28K wall attribution is complete as well as 99.97% classified within GPU time;
the overlapping fence wait is not added twice.

Every complete 512-row chunk routes attention through NVIDIA Q64 because the
device exposes the required workgroup Matrix2, reductions, conversions, tensor
addressing, F16/F16/FP32 types, 128-invocation property, and shared-memory
budget. The 28K 352-row tail is not divisible by 64 and correctly uses Q32
Matrix2; the 15K 75-row tail uses the cooperative-QK fallback. Every recurrent
Q4/Q6 shape is Matrix2 eligible:

| Operation | M x N x K per request | Format | Expected / actual path | Fallback condition |
|---|---|---|---|---|
| Q | rows x 4,096 x 4,096 | Q4_K | direct Q4 Matrix2 M128/N32/K128 | capability, pipeline, or shape absent |
| K | rows x 1,024 x 4,096 | Q4_K | direct Q4 Matrix2 | same |
| V | rows x 1,024 x 4,096 | 17 Q4 / 17 Q6 layers | direct Q4 / staged Q6 Matrix2 | same |
| O | rows x 4,096 x 4,096 | Q4_K | direct Q4 Matrix2 | same |
| Gate / up | rows x 14,336 x 4,096 | Q4_K | direct Q4 Matrix2 | same |
| Down | rows x 4,096 x 14,336 | 17 Q4 / 17 Q6 layers | direct Q4 / staged Q6 Matrix2 | same |
| Attention | 32 Q heads, 8 KV heads, head 128 | F16 KV, exact 4:1 GQA | Q64 full chunks; Q32/cooperative tail | row divisibility or capability absent |

Routing is vendor/capability + format + shape + row-count based. It contains no
product-name or model-name predicate, and all portable paths remain registered.
The hybrid row policy selects 512 only when placement is all-GPU on NVIDIA and
both the direct-Q4 and Q64 Matrix2 capability/resource contracts are live;
otherwise it retains the established 32-row capacity fallback. Thus 8K uses 16
complete 512-row chunks, 15K uses 30 plus a 75-row tail, and 28K uses 54 plus a
352-row tail. Q4 requires Q4_K, batch rows greater than one, aligned measured
N/K shapes, subgroup shuffle, and the Matrix2 shared-memory budget. Q6 requires
Q6_K, the measured N/K shapes, and its exact FP32 Matrix2 staging resources.
Attention additionally requires F16 KV, head dimension 128, exact 4:1 GQA,
WG128 Matrix2 reductions/conversions/tensor addressing, and row divisibility by
64. Failure of any predicate selects the next Q32, cooperative, tiled, or
generic Vulkan route rather than CPU execution.

## Residency, Memory, And Materialization

All 34 8B layers remain on GPU across the complete curve. The immutable
32,768-capacity plan is:

| Device-memory category | Bytes |
|---|---:|
| Model weights | 5,189,935,104 |
| Reserved F16 KV | 4,563,402,752 |
| Prefill/decode scratch | 75,644,928 |
| Fixed logits/reduction/MMVQ buffers | 17,563,648 |
| Explicit reserve | about 606,398,054 |
| GPU placement total | about 10,452,944,486 |
| CPU bounded staging + fixed | 440,926,208 |

NVIDIA telemetry peaks at 9,786 MiB allocated. During long rows utilization is
98--100%, memory clock 7,501 MHz, graphics clock about 1,942--1,972 MHz, power
159--169 W against a 170 W cap, and temperature 71--73 C. There is no observed
thermal clock collapse.

The KV allocation is capacity-fixed; logical occupancy is 0.266 GiB at 2K,
1.063 GiB at 8K, 2.002 GiB at 15K, and 3.632 GiB at 28K. Context growth causes
no allocation churn, resizing, migration, layer eviction, host/device copy, or
route loss. The 14B/16,384 plan likewise stays all-GPU (0 CPU / 40 GPU layers)
at about 11.626 GB planned GPU bytes. Larger 14B capacities require ordinary
mixed placement because weights plus exact F16 KV exceed 12 GiB; that is a
physical capacity boundary, not hidden fallback in the primary 8B matrix.

Capacity is fixed at model initialization, so the reserved allocation and
total pressure do not grow with logical occupancy:

| Context | Model weights | Reserved KV | Logical KV used | Temporary scratch | Fixed/persistent | Global attention intermediates | Planned GPU pressure |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2K | 5,189,935,104 B | 4,563,402,752 B | 285,212,672 B | 75,644,928 B | 17,563,648 B | 0 B | about 10,452,944,486 B |
| 8K | 5,189,935,104 B | 4,563,402,752 B | 1,140,850,688 B | 75,644,928 B | 17,563,648 B | 0 B | about 10,452,944,486 B |
| 15K | 5,189,935,104 B | 4,563,402,752 B | 2,149,539,840 B | 75,644,928 B | 17,563,648 B | 0 B | about 10,452,944,486 B |
| 28K | 5,189,935,104 B | 4,563,402,752 B | 3,899,392,000 B | 75,644,928 B | 17,563,648 B | 0 B | about 10,452,944,486 B |

The remaining roughly 606 MB in the planned total is the explicit device
reserve. Attention keeps online state inside Matrix2/register state and writes
only its ordinary output tensor, so it owns no global score/probability buffer.
Resource lifetimes are: weights, pipelines, scratch, fixed buffers, descriptors,
and bounded CPU staging per model; KV capacity per initialized engine and
reused serialized request state; command buffers and submissions per prefill
chunk; tensor views per graph recording without allocation. The profiler
records zero steady allocations, resizes, mappings, or resource creation.

There is no NxN score or probability buffer. QK, online softmax, and AV are one
matrix-native dispatch with live FP32 state. K/V append writes in place; there
is no separate transpose, packing, conversion, or copy. The largest materialized
round trips remain Q/K/V, attention output, and the three wide MLP tensors, but
acquired byte ceilings put eliminating them below 1% whole prefill because the
surrounding quantized Matrix2 work dominates.

## Traffic, Floors, And Amdahl Ranking

At 28K the canonical recurrent matrices contain about 3.702 GiB payload plus
0.439 GiB metadata across layers. Shader-visible logical movement is about
1,086 GiB weight/metadata traversal, 3,036 GiB activation reads, and 76 GiB
outputs. Direct Q4 traverses canonical weights about 219 times; staged Q6 about
438 times. These are cache-visible logical counts, not physical DRAM
transactions. Fresh per-operation inventory, sorted approximately by removable
time, is:

| Operation | M x N x K (all 34 layers) | Format | Canonical payload / metadata | Logical weight / activation / output | GPU time | Invocations | Logical rate / useful compute | Ownership |
|---|---|---|---:|---:|---:|---:|---:|---|
| Down | 28,000 x 4,096 x 14,336 | 17 Q4 / 17 Q6 | 1.162 / 0.123 GiB | 448.5 / 1,830.4 / 7.2 GiB | 11.234 s | 1,870 | 203.5 GiB/s / 9.95 TFLOP/s | direct Q4 M128; staged Q6 M64 |
| Gate | 28,000 x 14,336 x 4,096 | Q4 | 0.930 / 0.116 GiB | 229.1 / 406.7 / 25.4 GiB | 5.420 s | 1,870 | 122.0 GiB/s / 20.63 TFLOP/s | direct Q4 M128 |
| Up | 28,000 x 14,336 x 4,096 | Q4 | 0.930 / 0.116 GiB | 229.1 / 406.7 / 25.4 GiB | 5.421 s | 1,870 | 122.0 GiB/s / 20.62 TFLOP/s | direct Q4 M128 |
| Q | 28,000 x 4,096 x 4,096 | Q4 | 0.266 / 0.033 GiB | 65.4 / 116.2 / 7.3 GiB | 1.984 s | 1,870 | 95.2 GiB/s / 16.10 TFLOP/s | direct Q4 M128 |
| O | 28,000 x 4,096 x 4,096 | Q4 | 0.266 / 0.033 GiB | 65.4 / 116.2 / 7.3 GiB | 1.981 s | 1,870 | 95.4 GiB/s / 16.13 TFLOP/s | direct Q4 M128 |
| V | 28,000 x 1,024 x 4,096 | 17 Q4 / 17 Q6 | 0.083 / 0.009 GiB | 32.1 / 130.7 / 1.8 GiB | 0.932 s | 1,870 | 176.7 GiB/s / 8.57 TFLOP/s | direct Q4 M128; staged Q6 M64 |
| K | 28,000 x 1,024 x 4,096 | Q4 | 0.066 / 0.008 GiB | 16.4 / 29.1 / 1.8 GiB | 0.656 s | 1,870 | 72.1 GiB/s / 12.18 TFLOP/s | direct Q4 M128 |

`M128` and `M64` are token-row ownership; both use N32/K128 Matrix2 fragments
inside a 256-thread workgroup. The 352-row tail makes useful row utilization
99.89%; recurrent N/K dimensions are exact, so padding is not a material
candidate. Logical rates include shader-visible weights, tiled activation
loads, and outputs and can include cache-served bytes. Vulkan exposes no
executed DRAM/L2 transaction, register, spill, or Matrix utilization counters
for these pipelines, so none are inferred.

Q4 and Q6 have deliberately different dataflows. Q4 reads each canonical
144-byte block as 128 payload plus 16 metadata bytes. Its tensor callback views
payload as packed 16-bit words and obtains scale/min metadata through aligned
16-byte words, extracts one nibble, applies FP32 scale/min arithmetic, converts
the callback value to FP16, feeds a direct
Matrix2 operation, accumulates in FP16, and stores FP16 output. Its 16,512-byte
shared scale/min table plus 8,192-byte driver reserve lets one decoded weight
serve 128 token rows without expanded-weight staging. Q6 reads each 210-byte
block as 192 payload plus 18 metadata bytes; 32-bit packed loads combine the
low four and high two bits, decode the signed scale and FP16 multiplier in
FP32, narrow one exact staged K128/N32 tile to FP16 shared memory, reuse it for
64 token rows, accumulate Matrix2 in FP32, then use a shared FP32 epilogue and
FP16 store. Its declared staging/output shared memory is 16,384 bytes plus the
same 8,192-byte reserve. Q6 therefore has more unpack, staging, barriers, live
FP32 state, and half the row reuse; copying Q4 geometry is not valid.

The one-copy weight minimum is 4.141 GiB, a 12 ms idealized bound at nominal
360 GB/s. Current shader-visible weight traversal is 1,086 GiB (3.24 seconds at
that nominal rate), while all seven matmuls take 27.628 seconds and sustain
39.3 GiB/s on weight bytes alone. Including logical tiled activations and
outputs raises the logical rate to about 152 GiB/s. Acquired probes bound Q4
metadata/shared/barrier work at 0.200 seconds and nibble/load/unpack at 0.183
seconds on 2K; comparator and geometry results show Matrix2 feed, accumulator
resources, cache behavior, and ownership dominate rather than a missing wider
load. Full Q4 FP16 expansion is 3.555x and cannot coexist with the 4.25-GiB KV
allocation. Lossless native Q6 requires a 1.121-GiB sidecar (0.304 GiB above
canonical replacement, or the full premium when canonical remains for decode),
yet exact FP32 screens gain only about 1.5% or regress at 256 rows. No persistent
representation has a useful break-even under the current contract.
The rejected Q6 report did not retain model-load conversion timing; the
economic screen therefore grants conversion a favorable zero-second cost. Its
best exact 0.727-second 28K saving would break even after
`conversion_seconds / 0.727` requests, but even the impossible one-request,
zero-conversion bound is only 1.5% whole prefill. Measuring or optimizing load
conversion cannot reverse the decision. The FP16 native variant has a larger
7.03% opportunity but fails quality, while full Q4 expansion exceeds capacity.
The native-Q6 implementation class is also a moderate-to-large durable change:
it needs load-time repacking/upload, an additional weight offset and allocation,
a new pipeline/kernel ABI, format/shape routing, and continued canonical bytes
for exact decode. That complexity and attack-surface cost is disproportionate
even before its failed performance/quality screens.

Exact Q64 attention performs the same dense causal pairs with no score
materialization. Across 34 layers, 28K contains 392.014 million causal position
pairs per head. Useful QK and AV work is 109.187 TFLOP each, 218.374 TFLOP
total. A naïve per-query-head K/V read would be about 218 TB, but that is not
the current shader traffic: GQA and Q64 tile reuse reduce shader-visible K and
V loads to about 1.71 TB each, plus 7.799 GB Q loads and 7.799 GB output stores.
The unique K+V byte minimum is 3.899 GB, but attaining it would require keeping
all query states live across a KV-oriented traversal. Current live online state
is 33,280 bytes per Q64 workgroup (`64 * (128 output + max + sum) * 4`) in
Matrix2/register state; materializing equivalent state for every row/head/layer
would be 15.841 GB and add a forbidden global handoff.

At 19.691 seconds, the retained path sustains about 174 GB/s of logical K+V
loads and 11.09 useful TFLOP/s. Q64 already loads Q once, keeps FP32 online
state live, removes score/probability shared round trips and per-tile explicit
barriers, and uses >99.7%-useful Matrix2 tiles at 28K. Its 255 registers/thread
and 48,640 bytes driver shared memory allow two WG/SM (128 useful query rows,
16.7% modeled thread occupancy, zero reported stack). The dense 61.1-TFLOP/s
MMA-only physical bound is 3.57 seconds and excludes tensor loads, exact
softmax dependencies, and state. The acquired practical floor is 15.990
seconds on this exact device/driver; the fresh current time is 19.691 seconds.

| Rank | Target | Current | Physical / observed lower bound | Practical floor | Realistically removable | Amdahl result / decision |
|---:|---|---:|---:|---:|---:|---|
| 1 | Exact attention | 19.691 s | 9.186 s llama observed; 3.57 s dense-MMA-only bound excludes loads/state | 15.990 s | <=3.701 s | `f=0.4065`, `S_local=1.231`, final 44.735 s / 625.9 tok/s, 1.083x whole; below 10% complex gate |
| 2 | Direct Q4 Matrix2 | about 20.0 s | 11.828 s llama observed | current | 0 economic | M128/N256, M256/N256, K128, integer, and staged neighbors regress 93--148% |
| 3 | Exact staged Q6 Matrix2 | about 7.6 s | 2.455 s llama observed | current under numeric contract | 0 economic | FP16 accumulation is about 1.3% whole; exact native variants are neutral/regressive |
| 4 | Support / control | 1.289 s | zero impossible-delete bound | about 1.2 s | well below 1 s practical | even complete deletion is only 2.66% whole |

The mandatory formula for the leading candidate is:

```text
T = 48.436 s
f = 19.691 / 48.436 = 0.4065
S_local = 19.691 / 15.990 = 1.231
S_total = 1 / ((1 - f) + f / S_local) = 1.083
removable = 48.436 * f * (1 - 1 / S_local) = 3.701 s
```

This optimistic attention floor would still leave 44.735 seconds, 1.81 times
llama.cpp and above the comparable implementation. It does not satisfy the
10% gate for a new exact-attention primitive. The Q6 comparator gap is larger
than the result of any exact candidate: FP16 accumulation improved local Q6 by
8% but only 2.02% at the 2K whole-request screen; lossless native FP32 gained
about 1.5%; native FP16 gained 7.03% but failed the unchanged numeric oracle;
exact 256-row native ownership passed parity and regressed performance.

The same `T=48.436 s` Amdahl screen for every non-trivial remaining/acquired
candidate is explicit below. Q6 uses the approximately 7.6-second current
component (`f=0.1569`); its rows are optimistic 28K projections from acquired
2K local/whole-request screens, not claims of a retained 28K implementation.
Native local speedups are the effective Q6 speedups implied by those measured
whole-request deltas. The llama Q6 observation is a 2.455-second theoretical
comparator bound, not an attainable Graph Horizon floor.

| Candidate | Practical candidate floor | S_local | Removable | Predicted final / tok/s | S_total | Decision |
|---|---:|---:|---:|---:|---:|---|
| New exact-attention mechanism | 15.990 s | 1.231 | 3.701 s | 44.735 s / 625.9 | 1.083x | below 10% complex gate |
| Q6 FP16 accumulator | 6.992 s | 1.087 | 0.608 s | 47.828 s / 585.4 | 1.013x | below 5% moderate gate |
| Lossless native Q6 FP32 | 6.873 s | 1.106 | 0.727 s | 47.710 s / 586.9 | 1.015x | below gate; persistent memory cost |
| Native Q6 FP16 | 4.195 s | 1.812 | 3.405 s | 45.031 s / 621.8 | 1.076x | unchanged numeric oracle fails |
| Exact native Q6, 256 rows | above current | <1 | negative | slower than 48.436 s | <1x | parity passes, performance rejects |
| Q4 M128/N256, M256/N256, or K128 | above current | <1 | negative | measured 93--148% local regression | <1x | ownership endpoints closed |
| Delete all support/control | 0 physical; current practical | infinity | 1.289 s impossible | 47.147 s / 593.9 | 1.027x | impossible and below gate |

QKV/gate-up/activation-down fusion has a measured byte ceiling below 1% whole,
and dispatch/barrier edits have at most 0.2% measured control ownership. They
are trivial screens rather than omitted redesign candidates. Sparse/windowed
attention and lower-precision weights change the approved dense/numerical
contract and therefore have no valid `S_local` inside this mission.

## Acquired Candidate Registry

These classes were read from current reports and Git history and were not
repeated without new evidence:

| Frontier | Acquired result | State |
|---|---|---|
| Hybrid 512-row NVIDIA routing | 14B/15K `230.888 -> 33.549 s`; capacity-accounted fallback | KEEP, present |
| Q64 Matrix2 attention | attention -37.6%, whole 28K -22.4% | KEEP, present |
| Direct canonical Q4 Matrix2 | 8B/28K -25.0% | KEEP, present |
| Q4 ownership M128/N256, M256/N256, K128 | +132.9%, +147.6%, +93.0% latency | REJECT |
| Q6 direct canonical | only -1.17% at screen | REJECT below gate |
| Q6 FP16 accumulator | local -8%, whole 2K -2.02% | REJECT below gate |
| Lossless native Q6 FP32 | about 1.5% whole | REJECT below gate |
| Native Q6 FP16 | 7.03% whole, numeric oracle failed | REJECT quality |
| Exact native Q6 256-row | parity passed, TTFT regressed | REJECT performance |
| Attention split/persistent/hierarchical/staged/multi-Q/tile families | no exact resource-feasible candidate >=5% | CLOSED |
| QKV, gate/up, activation/down fusion | measured/byte ceiling below 1% | CLOSED |
| Dispatch, barriers, allocation lifetime | <=0.2% measured control; zero steady allocations | CLOSED |
| Sparse/windowed or lower precision | changes dense exact/numerical contract | OUT OF SCOPE |

## Regression And Real-Request Qualification

No production source changed, so baseline and final performance, decode,
memory, arithmetic, and routing are identical by construction. Fresh decode
guards from the public 8B matrix are:

| Prompt | Baseline TTFT | Final TTFT | Baseline tok/s | Final tok/s | Delta |
|---:|---:|---:|---:|---:|---:|
| 128 | 205.06 ms | 205.06 ms | 624.23 | 624.23 | 0% |
| 512 | 582.77 ms | 582.77 ms | 878.57 | 878.57 | 0% |
| 2K | 2,238.81 ms | 2,238.81 ms | 914.77 | 914.77 | 0% |
| 8K | 10,033.02 ms | 10,033.02 ms | 816.51 | 816.51 | 0% |
| 15K | 22,061.52 ms | 22,061.52 ms | 699.63 | 699.63 | 0% |
| 28K | 48,436.05 ms | 48,436.05 ms | 578.08 | 578.08 | 0% |

| KV / prompt | Baseline model tok/s | Final model tok/s | Delta |
|---:|---:|---:|---:|
| 128 | 35.94 | 35.94 | 0% |
| 2K | 34.53 | 34.53 | 0% |
| 8K | 31.31 | 31.31 | 0% |
| 15K | 28.41 | 28.41 | 0% |
| 28K | 24.47 | 24.47 | 0% |

The substantial real request produced all 954 requested tokens:

| Metric | Qualified baseline | Final | Delta |
|---|---:|---:|---:|
| Prompt tokens | 15,435 | 15,435 | 0 |
| Completion tokens / public deltas | 954 / 954 | 954 / 954 | 0 |
| TTFT / prefill boundary | 21.702 s | 21.702 s | 0% |
| Prefill throughput | 711.23 tok/s | 711.23 tok/s | 0% |
| Decode throughput | 27.62 model tok/s | 27.62 model tok/s | 0% |
| Decode time | about 34.54 s | about 34.54 s | 0% |
| Total request | about 56.24 s | about 56.24 s | 0% |

The stale 213.24-second symptom is not used as the `before` column: it is not a
qualified current-source baseline and could not be reproduced. The table uses
the fresh source-identical baseline/final pair required for a valid regression
claim.

The decode begin/middle/end rates are 27.85/27.58/27.36 tok/s; the gain from
the retained prefill architecture survives full inference and does not consume
decode throughput.

For this 954-token completion, decode now consumes about 34.54 seconds versus
21.70 seconds prefill, so it is the larger end-to-end phase by 12.84 seconds.
The global inference re-rank therefore closes prefill priority under the
current gates and hands future decode work to the separate NVIDIA long-context
decode checkpoint; it does not justify continuing below-gate prefill tuning.

Fresh final qualification passes formatting and diff checks, warning-denied
Clippy for both `vulkan` and `vulkan-hybrid`, 356 executed pure-Vulkan workspace
tests, and 428 executed Vulkan-hybrid workspace tests. The two opt-in Q64
boundary and long-context numeric qualifications also pass across F16 and int8
KV at 2K, 8K, and 28K tails (`623.31 s`). Acquired teacher-forced and six-model
qualification remains applicable because production code, shaders, routing,
and arithmetic are unchanged.

## Final 15K Closure

The final attribution groups the fresh 22.179-second profiler wall boundary
into the remaining optimization domains. The attention floor applies the
acquired 1.231x exact local ceiling to 15K, so it is an optimistic screen, not
a physical bound. "Economic" removal means improvement still available under
the existing exactness contract and mission gates.

| Domain | Current | Practical screen floor | Economically removable | Reason closed |
|---|---:|---:|---:|---|
| Exact attention | 6.045 s | >=4.910 s | <=1.135 s | 5.12% whole-request ceiling at 15K; acquired exact architecture families exhausted |
| Q4/Q6 projections and MLP | 15.294 s | current | 0 | exact neighboring geometries regress or remain below gate; FP16 Q6 fails numeric contract |
| GPU support kernels and residual | 0.712 s | current | 0 | fusion/materialization and elementwise candidates are below the whole-request gate |
| CPU/frontend/control boundary | 0.129 s | current | 0 | even impossible deletion is 0.58% of wall; steady allocations are zero |
| **Total** | **22.180 s** | **>=21.045 s** | **<=1.135 s** | **rounding differs from measured wall by 1 ms** |

The full competitive curve is the fresh matrix above: final Graph Horizon is
1.90--2.10x llama.cpp latency across 128--28K. For the substantial request,
baseline and final are both 21.702 seconds TTFT plus about 34.54 seconds decode,
or about 56.24 seconds total, because this investigation retained no source
candidate.

## Final Decision And Reopening Conditions

Final production state equals baseline `e3d23f2102661c5cb95310a11873ad974ab2c831`. The motivating 213-second
symptom is stale or belongs to the pre-fix hybrid route. Current 8B/15K and
14B/15K measurements are 22.06 and 33.95 seconds. The largest present 28K
component is exact attention, but its optimistic exact removal is below the
complex-design gate; Q4/Q6 and control have no economic exact candidate.

Reopen this NVIDIA prefill priority only if at least one premise changes:

- a new Vulkan/NVIDIA cooperative-matrix capability removes the measured
  resource/ownership cliff;
- a new exact attention resource mechanism is not represented by the acquired
  split, persistent, staging, softmax, multi-Q, and tile experiments;
- hardware capacity makes a lossless persistent execution representation
  practical; or
- the task explicitly approves a new numerical/model-quality contract for
  native low precision, sparse, or windowed attention.

Absent one of those changes, another local kernel, tile sweep, barrier edit,
packing variant, or fusion repeats a closed class or has an Amdahl ceiling
below the mission gate.

There is no next candidate under the current exact dense-attention, canonical
Q4_K/Q6_K, 12-GiB capacity, and unchanged-quality contract. The next action is
therefore a measurement refresh only when a reopening condition changes; it is
not another speculative kernel experiment.
