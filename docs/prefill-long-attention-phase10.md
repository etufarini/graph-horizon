<!--
This report owns Phase 10 evidence for exact Vulkan long-context attention:
fresh attribution, traffic/resource models, architecture screening, and the
decision to retain the source-identical Phase 9 kernel. It defines no runtime
behavior or general support contract.
-->

# Vulkan Long-Attention Phase 10

## Result

Phase 10 stops without a production change. Fresh profiling identifies QK and
AV as the largest physical regions, but every exact-attention architecture that
could reduce them either has a measured whole-request ceiling below 5% or
crosses a known register/shared-memory residency penalty. The final production
state is therefore the Phase 9 Q32/KV64 kernel from `d6c9a38`, with its FP32 AV
accumulator persistent across KV tiles.

The investigation starts from clean commit `387210a` on
`perf/vulkan-long-prefill-phase9` and runs on
`perf/vulkan-long-prefill-phase10`. Nothing is pushed. The model is
`Ministral-3-3B-Instruct-2512-Q4_K_M.gguf`, 2,147,023,008 bytes, SHA-256
`9ed150d4367e68df0ac8e1540f6ddc65b42d0ee26378329d1ecbca60f93fc5f8`.
The device is an RTX 3060 (`0x2487`), driver 595.84, Vulkan 1.4.329, 12 GiB,
7,501 MHz memory clock and 170 W board limit. Context is 32,768, KV is F16,
sampling is greedy, and synthetic prompts are `" a"` repeated until the exact
requested token count.

The inherited reference is 49.864 s diagnostics-free TTFT, 26.373 s attention
at 28K, and 114.61 ms at 128. The fresh session is slower in absolute wall time,
so all causal comparisons below use same-session profiler pairs and no result is
claimed against an earlier clock/thermal session.

## Invariants and method

The invariant is exact causal online attention with the existing F16 Q/K/V and
output formats, FP32 QK/softmax/AV accumulation, GQA mapping, and no global
score matrix. Decode, INT8, public APIs, dependencies, routing and numeric
tolerances remain unchanged. Before the first temporary source edit the branch
was clean, the baseline was captured, and the current kernel and all prior
reports were read in full.

Each diagnostics-free TTFT row uses independent processes and the shown raw
samples. The profiler uses one cold request and a separate warm-up plus three
measured requests; stable per-request time is `(four-request total - cold) / 3`.
Downstream-live shader probes use four-request totals from the same session,
because every probe and its baseline have identical warm-up/request counts.
All probes were temporary and are absent from the final tree.

Nsight Compute 2024.1.1 reported `No kernels were profiled` for this Vulkan
workload. Nsight Systems GPU metrics failed with `ERR_NVGPUCTRPERM`. Vulkan
exposes pipeline executable properties but no usable performance-query path on
this system. Consequently this report labels shader-visible logical bytes,
causal hot-address probes, and hardware counters separately; it never presents
logical traffic as physical DRAM traffic or infers a cache hit rate.

## Fresh diagnostics-free baseline

| Tokens | Samples | Mean | Median | SD | CV |
|---:|---|---:|---:|---:|---:|
| 128 | 126.538, 116.226, 115.646, 115.247, 115.160 ms | 117.76 ms | 115.646 ms | 4.92 ms | 4.18% |
| 512 | 438.176, 440.401, 440.679, 441.882, 441.738 ms | 440.58 ms | 440.679 ms | 1.49 ms | 0.34% |
| 2K | 1853.373, 1854.191, 1858.227, 1864.756, 1864.916 ms | 1859.09 ms | 1858.227 ms | 5.56 ms | 0.30% |
| 8K | 9095.610, 9137.874, 9188.034 ms | 9140.51 ms | 9137.874 ms | 46.27 ms | 0.51% |
| 16K | 23052.525, 23074.883, 23101.994 ms | 23076.47 ms | 23074.883 ms | 24.77 ms | 0.11% |
| 28K | 50631.205, 50665.560, 50681.243 ms | 50659.34 ms | 50665.560 ms | 25.59 ms | 0.05% |

The first 128 process is a visible initialization tail; the last four span only
1.07 ms. The long rows have enough stability to distinguish the requested 3%
screening threshold.

| Tokens | GPU prefill | Attention | Seven matmuls | Other | Coverage |
|---:|---:|---:|---:|---:|---:|
| 128 | 118.410 ms | 1.135 ms | 109.132 ms | 8.022 ms | 99.90% |
| 512 | 444.183 ms | 10.029 ms | 413.195 ms | 20.633 ms | 99.93% |
| 2K | 1884.871 ms | 146.370 ms | 1666.347 ms | 70.854 ms | 99.93% |
| 8K | 9316.994 ms | 2282.759 ms | 6755.885 ms | 273.103 ms | 99.94% |
| 16K | 23221.686 ms | 9101.978 ms | 13561.747 ms | 546.819 ms | 99.95% |
| 28K | 50801.893 ms | 26720.289 ms | 23135.868 ms | 927.161 ms | 99.96% |

At 8K/16K/28K, active telemetry samples show 99.67/99.70/99.73% GPU
utilization, 1957/1930/1929 MHz mean core clock, 7501 MHz memory, and
168.25/168.79/169.24 W. Mean temperatures are 64.2/72.3/73.6 C. No run is
VRAM-capacity, idle, power-transition, or thermal-throttle limited.

## Attention attribution

The additive causal chain removes successively Q/K load plus QK, score-state
and softmax, then probability/V load plus AV. It retains loop geometry,
required outputs, and downstream observability. The buckets are order-dependent
causal deltas, not simultaneous hardware counters, but they reconcile exactly.

| Phase | 8K | 8K % | 16K | 16K % | 28K | 28K % |
|---|---:|---:|---:|---:|---:|---:|
| Q load + K load + QK Matrix2 | 825.694 ms | 36.26% | 3316.542 ms | 36.41% | 9893.916 ms | 37.03% |
| score max/exp/sum + online state | 317.645 ms | 13.95% | 1238.425 ms | 13.60% | 3581.754 ms | 13.40% |
| probability + V load + AV Matrix2 | 795.349 ms | 34.93% | 3209.889 ms | 35.24% | 9352.393 ms | 35.00% |
| loop, address, conversion, output and synchronization | 338.307 ms | 14.86% | 1343.871 ms | 14.75% | 3890.922 ms | 14.56% |
| **Total** | **2276.995 ms** | **100%** | **9108.726 ms** | **100%** | **26718.985 ms** | **100%** |

The 2K total is 146.370 ms. Component probes were intentionally concentrated
at 8K/16K/28K, where their signal is large enough; no false-precision 2K split
is extrapolated. A standalone AV-off probe gives 908.232/3572.945/10453.072 ms
at 8K/16K/28K, or 39.89/39.23/39.12%. Its difference from the additive AV
bucket quantifies overlap and demonstrates why direct ablations must not be
summed.

At 28K the residual bucket separates further as follows:

| Residual region | ms | % attention | % GPU prefill |
|---|---:|---:|---:|
| Required internal barriers in the downstream-live skeleton | 436.489 | 1.63% | 0.86% |
| Geometry, addressing, score/probability shared writes, accumulator rescale, final conversion/write | 3454.434 | 12.93% | 6.80% |
| **Residual total** | **3890.922** | **14.56%** | **7.66%** |

Removing both barriers from the full kernel, knowingly introducing races, is
a favorable non-additive upper bound: 205.721/779.145/2350.960 ms at
8K/16K/28K, or 9.03/8.55/8.80% of attention and only 4.63% of 28K GPU
prefill. This is not a valid candidate. A design that merely halves barriers
has at most about 2.3% whole-request headroom before its own overhead.

The compiler output contains four cooperative loads, two MatrixMulAdd sites,
two cooperative stores, two per-element operations, four static barriers, 35
loads, 72 stores, 30 access chains, three integer divisions, 13 integer
multiplications and two exponential sites. SPIR-V shows the FP32 output
accumulator live through the whole KV loop and the QK score accumulator only
through its inner dimension loop. NVIDIA ISA/SASS and instruction-issue/stall
counters are unavailable, so no unsupported claim about pipeline bubbles is
made.

## Resource and occupancy audit

| Resource | Current | Two resident WG | Three-WG threshold | Distance |
|---|---:|---:|---:|---:|
| Registers | 124/thread, 31,744/WG | 63,488/SM | at most 85.33/thread, 21,845/WG | reduce 38.67/thread (31.2%) |
| Driver shared | 38,272 B/WG | 76,544 B/SM | at most 34,133 B/WG | reduce 4,139 B/WG (10.8%) |
| Threads | 256/WG | 512/SM | 512 is below 1,536/SM | not limiting |

The kernel has two resident workgroups, eight subgroups/WG, 16 active warps and
33.3% modeled occupancy. Registers and shared memory independently prevent a
third workgroup. Phase 9 changed 125 to 124 registers/thread and retained the
same residency, so persistent FP32 state did not cause a register cliff.
Executable properties report zero stack. The reported 64 GiB local-memory
value is the known driver sentinel; spills cannot be measured, but no stack or
runtime pathology is evidenced.

An impossible resource-only 2-to-3-WG scaling ceiling is 1.5x, or 33.3%
attention and about 17.5% whole prefill. There is no concrete path to it:
removing the entire persistent-output bookkeeping cannot credibly eliminate
39 registers/thread, and shared memory must also fall by 4 KiB. Phasing or
spilling the output rescans QK or restores the Phase 9 shared round-trip. The
register-cliff experiment gate is therefore closed before coding.

## Global traffic, reuse and cache evidence

The logical traversal is layer, query head, Q32 workgroup, then causal KV64
tile. Every visit tensor-loads one 8,192-byte Q tile, one 16,384-byte K tile and
one 16,384-byte V tile. Final output is written once per Q workgroup. These are
source-exact shader-visible bytes; physical cache/DRAM transactions are not
available.

| Context | Visits | Q actual/min | Q amp | K actual/min | K amp | V actual/min | V amp |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2K | 878,592 | 7.197/0.436 GB | 16.5x | 14.395/0.109 GB | 132x | 14.395/0.109 GB | 132x |
| 8K | 13,737,984 | 112.542/1.745 GB | 64.5x | 225.083/0.436 GB | 516x | 225.083/0.436 GB | 516x |
| 16K | 54,738,944 | 448.421/3.490 GB | 128.5x | 896.843/0.872 GB | 1028x | 896.843/0.872 GB | 1028x |
| 28K | 159,614,208 | 1307.560/5.964 GB | 219.25x | 2615.119/1.491 GB | 1754x | 2615.119/1.491 GB | 1754x |

At 28K output is 5.964 GB actual and minimum. Total logical global traffic is
6.544 TB versus a 14.909 GB unique minimum, 438.9x amplification. The large
logical amplification does **not** establish equally large DRAM
amplification: temporally adjacent workgroups can be served by cache.

Within a workgroup, each K and V tile is explicitly reused by 32 Q rows. Two
independent Q32 fragments could raise that to 64. Ministral GQA has 32 query
heads and eight KV heads, so the cross-head theoretical factor is another 4x,
but current query-head workgroups do not communicate. Wider explicit reuse
requires more independent FP32 online state and output accumulators.

| Context | Unique Q/layer | Unique K/layer | Unique V/layer | Active Q/K/V tiles |
|---:|---:|---:|---:|---:|
| 2K | 16.78 MB | 4.19 MB | 4.19 MB | 8/16/16 KiB |
| 8K | 67.11 MB | 16.78 MB | 16.78 MB | 8/16/16 KiB |
| 16K | 134.22 MB | 33.55 MB | 33.55 MB | 8/16/16 KiB |
| 28K | 229.38 MB | 57.34 MB | 57.34 MB | 8/16/16 KiB |

Timing grows quadratically without a discrete 8K/16K/28K cliff. A K hot-address
probe retains the same K tensor-load instructions and occupancy but redirects
all K loads to tile zero. It saves 65.103/151.132/573.940 ms, only
2.86/1.66/2.15% attention at 8K/16K/28K and 1.13% of 28K GPU prefill. This is
a favorable cache-sensitive K ceiling, not a physical K-byte measurement.

The corresponding V hot-address probe regresses attention by 35.3/34.8/33.8%.
Repeated-address tensor/cache behavior is worse than the normal V stream, so
there is no positive measured V cache-miss ceiling. Data values do not change
the fixed Matrix2 instruction count. The result rejects the assumption that
logical V-byte elimination converts directly into elapsed-time reduction.

At 28K, logical K and V rates are each 97.9 GB/s and total Q/K/V is 244.7
GB/s. Remaining shared traffic is 5.254 TB, 196.6 GB/s logical: 24,576 bytes
per visit for score write and two reads, 8,192 bytes for probability write/read,
plus a final 32,768-byte output round-trip per Q workgroup. The 320,684,416
dynamic internal barriers are two per KV visit plus initialization/finalization
per Q workgroup; 62,322 graph-level barriers are unchanged.

Nominal 360 GB/s gives an 18.16 s logical Q/K/V floor, while K+V at 300 GB/s
gives 17.44 s. Those rates mix cache-served and overlapping logical traffic and
are deliberately not called DRAM rooflines. The hot-address experiments show
that cache-sensitive K removal is worth only 0.574 s and V has no positive
ceiling. The kernel is best described as a low-occupancy tensor-load/MMA/
dependency pipeline, not generically as DRAM-bandwidth bound.

## Matrix2 and roofline

QK is physically 32x64x128 and AV is 32x128x64. Both map exactly to supported
Matrix2 fragment dimensions. Causal boundary padding is the only wasted MMA:
useful utilization is 97.017% at 2K, 99.237% at 8K, 99.617% at 16K and
99.775% at 28K for both QK and AV. Tile reshaping has no material padding win.

At 28K QK and AV each execute 83.684 TFLOP, 167.368 TFLOP total. Relative to
the additive causal buckets, effective QK compute is 8.46 TFLOP/s, AV is 8.95
TFLOP/s and combined is 8.70 TFLOP/s. The dense 61.1 TFLOP/s device-rate floor
is 1.37 s each or 2.74 s combined; it is not attainable here because tensor
loads, online dependencies, barriers and only 33.3% occupancy are included in
the buckets.

| Region | Current 28K | Physical lower bound | Credible practical floor | Removable | Whole-prefill ceiling |
|---|---:|---:|---:|---:|---:|
| Q/K load + QK | 9893.916 ms | 1369.6 ms dense MMA | no supported faster geometry demonstrated | unproven | unproven |
| Softmax/state | 3581.754 ms | nonzero exact max/exp/sum | prior simple rewrites regress; complete deletion impossible | <3581.754 ms | <7.05% |
| probability/V + AV | 9352.393 ms | 1369.6 ms dense MMA | V hot-address has no positive gain | unproven | unproven |
| Valid remaining barriers | 436.489 ms additive | 0 | 218 ms if halved | about 218 ms | 0.43% |
| K cache-sensitive work | at most 573.940 ms | 0 | half removed by 2x reuse | about 287 ms | 0.56% |

Complete deletion of exact softmax/state would save only 7.05% of GPU prefill;
the mandatory max, exponential, sum, normalization and online rescale make its
practical removable fraction far smaller. Static exponentials are only two
sites and prior direct softmax/reduction variants did not improve. It does not
justify a major redesign.

## Architecture screening and experiment registry

| ID | Architecture | Quantitative model or probe | Predicted 28K TTFT | Decision |
|---|---|---|---:|---|
| P10-QK-OFF | downstream-live Q/K/QK ablation | 9893.916 ms additive causal region | diagnostic only | remove |
| P10-AV-OFF | downstream-live probability/V/AV ablation | 9352.393 ms additive; 10453.072 ms standalone | diagnostic only | remove |
| P10-SKELETON | loop/output skeleton and barrier-free variant | barriers 436.489 ms additive; invalid full upper 2350.960 ms | at most 4.63% for impossible removal | remove |
| P10-K-HOT | same K loads, constant hot address | saves 573.940 ms attention | 1.13% | remove |
| P10-V-HOT | same V loads, constant hot address | regresses 33.8% | negative | remove |
| P10-MULTIQ2 | two independent supported Q32 fragments sharing K/V | half K-hot + half invalid barrier upper is about 2.9% before overhead | <5% | reject before code |
| P10-SUBGROUP | subgroup-partitioned K/V sharing | needs standard shared loads; acquired variant was 67% slower and reduces residency | negative risk | reject before code |
| P10-KV-PERSIST | invert traversal around K/V tiles | needs many `(m,l,O)` states or a forbidden global score matrix | <5% measured byte ceiling | reject before code |
| P10-SPLIT | separate QK and AV geometries | both are >99.7% utilized; handoff writes at least FP32 `(m,l,O)` state and adds dispatch | no positive bound | reject before code |
| P10-PHASED | shorten persistent accumulator ownership | must remove 39 registers and 4 KiB shared for 3 WG; implies rescans or spill | no concrete residency path | reject before code |
| P10-DBUF | double-buffer K/V | full invalid barrier upper <5% whole prefill; extra resources threaten residency; old prefetch regressed 4.8–6.3% | <5% | reject before code |

`P10-MULTIQ2` is materially different from the unsupported Phase 8 Q64
cooperative fragment: it would hold two supported Q32 fragments. Its modeled
cost is nevertheless about 24 extra FP32 values/thread plus wider score/
probability shared state. It likely exceeds 128 registers/thread, drops to one
resident WG, and halves active warps. Useful Q ownership per SM would remain
roughly 64 rows while instruction latency hiding worsens. Even the favorable
sum of half the K-hot and half the race-enabled barrier ceilings is only about
5.5% attention and 2.9% GPU prefill before this cost; it fails the mandatory 5%
whole-request coding gate.

Cross-head grouping without shared ownership was also closed by evidence.
Query heads already dispatch contiguously, physical cache counters are
unavailable, and the K-hot upper is 1.13% of prefill while V-hot regresses.
Reopening earlier GQA/shared-load designs cannot reach the gate.

The best candidate is therefore **none**. Final Q_TILE/K_TILE/V_TILE remains
32/64/64, WG 256 with eight subgroups, 32 Q rows/WG, direct per-WG K/V tensor
loads, 32-row explicit K/V reuse, and one FP32 32x128 AV accumulator live across
the KV loop. Resources remain 124 registers/thread, 38,272 B shared/WG, two
resident WG, 33.3% occupancy, zero reported stack, and unmeasurable driver
local-memory sentinel.

## Final state and guards

No production candidate was retained, so baseline and final are source-
identical. The official performance matrix is the diagnostics-free baseline:

| Context | Baseline TTFT | Final TTFT | Delta |
|---:|---:|---:|---:|
| 128 | 117.76 ms mean, 115.646 median | identical source | 0% by construction |
| 512 | 440.58 ms mean, 440.679 median | identical source | 0% by construction |
| 2K | 1859.09 ms mean, 1858.227 median | identical source | 0% by construction |
| 8K | 9140.51 ms mean, 9137.874 median | identical source | 0% by construction |
| 16K | 23076.47 ms mean, 23074.883 median | identical source | 0% by construction |
| 28K | 50659.34 ms mean, 50665.560 median | identical source | 0% by construction |

| Context | Baseline attention | Final attention | Delta |
|---:|---:|---:|---:|
| 2K | 146.370 ms | identical source | 0% |
| 8K | 2282.759 ms | identical source | 0% |
| 16K | 9101.978 ms | identical source | 0% |
| 28K | 26720.289 ms | identical source | 0% |

At 28K the same-session causal result is QK 9893.916 ms, softmax/state
3581.754 ms, AV 9352.393 ms, other 3890.922 ms, total 26718.985 ms, with
50.802 s profiler GPU prefill and 50.659 s diagnostics-free mean TTFT. The
small difference between causal and cold-subtracted attention is normal
same-session protocol variation.

The 128 short guard is source-identical to the qualified 114.61 ms Phase 9
kernel. The baseline cluster median is 115.646 ms, 0.90% above the acquired
control. A final clean rebuild with two warm-ups and 15 repetitions measures
117.49 ms at 0.29% CV, 2.51% above the acquired control; the simultaneous 28K
confirmation is 50.759 s at 0.16% CV versus the fresh 50.659 s baseline. This
is whole-session drift, not a retained-code regression: baseline and final
shader/runtime source are byte-identical and their same-session delta is zero.
The absolute 128 rerun is therefore reported as an environmental guard miss,
not silently called a pass.

Diagnostics-free 32-token decode is 45.52 tok/s at 128, 45.02 at 2K and 28.54
at 28K, with CV at most 0.19%. These absolute values share the session slowdown
and are lower than the acquired 47.19/46.54/29.16 controls, but Phase 10 has no
production change capable of causing a decode delta.

The Vulkan-profile library suite passes 152 tests with four intentional
ignores. The focused ignored numeric qualification passes in 458.20 s for F16
and INT8, at 2K/8K/28K and for both 32-row Matrix2 and 16-row generic tails.
Maximum F16 attention and projected-logit absolute errors are `1.526e-5` and
`4.239e-6`; every INT8 attention and logits comparison is bit-exact. The
authenticated 128-token greedy sequence is exactly
`2757,10637,2479,1636`. Workspace Clippy with all targets and warnings denied,
formatting, `git diff --check`, release shader compilation, and the source diff
against `387210a` pass. No diagnostic branch or benchmark logger remains.

## Stop rationale and next step

Phase 10 satisfies stop conditions 5, 6 and 7:

1. Registers and shared memory independently hold residency at two WG; no
   exact design crosses both three-WG thresholds without restoring spill,
   rescanning QK, or exploding state.
2. At least multi-Q ownership, subgroup/shared reuse, K/V-oriented traversal,
   split QK/AV geometry, phased ownership and double buffering are distinct
   rejected architectures, in addition to the measured hot-address paths.
3. No residual exact-attention candidate has a predicted whole-28K gain of 5%.

The largest remaining bottleneck is the combined direct tensor-load and
Matrix2 work in QK and AV under 33.3% occupancy, not padding, a Phase 9
register cliff, or remaining valid barriers. Large theoretical headroom exists
between 8.70 and 61.1 TFLOP/s, but no supported resource-feasible architecture
isolates it.

The next material step requires a broader contract: lower-precision attention
state or KV representation, a different exact-attention primitive/runtime
mapping with better occupancy, or non-exact sparse/windowed/alternative
attention. Halving Q/K/V representation bytes has high numerical and format
risk and the K/V probes warn that byte removal alone may not improve time.
Windowing or sparsity can remove asymptotic work but changes causal semantics.
A future cross-head primitive could theoretically reuse K/V across four query
heads, but must first demonstrate a supported low-register tensor mapping; the
present cache-sensitive ceiling remains below 5% end-to-end. None is authorized
or implemented in this phase.
