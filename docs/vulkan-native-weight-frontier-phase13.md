<!--
This report owns Phase 13 evidence for the exact Vulkan native-weight frontier:
the post-Phase-12 attribution, marginal VRAM economics, rejected down prototype,
compact-representation screening, and stop decision. It defines no model format,
public support contract, or new runtime behavior.
-->

# Vulkan Native-Weight Frontier Phase 13

## Outcome

Phase 13 retains no production change. The single-group down-projection
prototype is exact and fast locally, but its diagnostics-free 28K A/B/A result
is 46,072.92 ms surrounding control to 43,812.94 ms candidate: -2,259.98 ms,
**-4.90%**. It adds 1.371094 GiB persistent VRAM, reaches 9,664 MiB application
VRAM at 28K, and costs about 6.58 s incremental process initialization. This
misses the mandatory 5% expansion gate; the prototype is removed.

The local result explains the near miss. Down falls from 5,782.249 to
3,313.502 ms (-42.69%), removing 2,468.747 component ms, or 1,801 ms/GiB.
Attention simultaneously rises 149.711 ms (+0.57%), leaving a 2,231.836 ms
GPU-prefill reduction and 4.86% profiled TTFT reduction. Diagnostics-free
marginal efficiency is 1,648 ms removed/GiB and 3.58 whole-request percentage
points/GiB, but the absolute gate takes precedence for an allocation above
1 GiB.

The exact compact alternatives also fail their gate. The retained FP16 layout
has zero padding or duplication; expanded metadata can retain at most 15.4% of
the measured Phase 12 gate/up component gain; and a quantized tile-order cache
still needs the current unpack, FP16 conversion, shared staging, and barriers
before FP16 Matrix2 consumption. A layer-partial FP16 cache reduces runtime
gain approximately with bytes and cannot retain 90--95% of the gain while
removing at least 25% of native residency.

The quantitative knee on this RTX 3060 12 GiB is therefore the Phase 12
gate/up point: +2.742188 GiB for -7.59% 28K TTFT. Exact FP16 expansion beyond
it increases capacity pressure faster than it delivers a qualifying marginal
request gain. The final production source is exactly baseline commit `6cb170f`.
The branch is `perf/vulkan-native-weight-frontier-phase13`; nothing is pushed.

## Baseline and method

- Required baseline branch: `perf/vulkan-predecoded-weights-phase12`.
- Effective baseline commit: `6cb170f` (including `e7cdd28` and `114bb26`).
- Phase 13 branch: `perf/vulkan-native-weight-frontier-phase13`.
- Initial worktree: clean; all three required Phase 10--12 reports were read in
  full before measurement or editing.
- Model: `Ministral-3-3B-Instruct-2512-Q4_K_M.gguf`, 2,147,023,008 bytes,
  SHA-256 `9ed150d4367e68df0ac8e1540f6ddc65b42d0ee26378329d1ecbca60f93fc5f8`.
- Device: RTX 3060 12 GiB (`0x2487`), driver 595.84, 7,501 MHz memory clock,
  170 W board limit.
- Runtime: release pure Vulkan, context 32,768, F16 KV, greedy sampling,
  `GRAPH_HORIZON_PREFILL_PREDECODE_MLP=1`, exact prompts made from `" a"`
  repeated `tokens - 3` times.
- Diagnostics-free long points use one warm-up and three measured repetitions.
  Short points use more repetitions as shown below. Temporary median reporting
  changed only post-request aggregation and was removed after capture.
- Profiler evidence uses a separate `vulkan-profile` build. Stable category
  time is `(warm-up + three requests - independent cold request) / 3`.
- Candidate qualification uses baseline/candidate/baseline recheck in one
  contiguous session at 128 and 28K.

The invariant is byte-identical canonical Q4_K/Q6_K storage at offset zero,
exact existing dequantization followed by the same FP16 narrowing, FP16
activation/output, FP32 Matrix2 accumulation, exact causal attention, unchanged
decode routing, no serialization change, unchanged tolerances, and no native
allocation when Matrix2 is disabled or unavailable.

## Fresh Phase 13 baseline

The first diagnostics-free baseline establishes the Phase 13 denominator.
The final source-identical matrix below supplies medians and the final guard.

| Context | TTFT mean | SD | CV | Prompt tok/s |
|---:|---:|---:|---:|---:|
| 128 | 97.27 ms | 0.45 ms | 0.46% | 1,315.95 |
| 512 | 369.92 ms | 1.36 ms | 0.37% | 1,384.11 |
| 2K | 1,562.99 ms | 3.68 ms | 0.24% | 1,310.31 |
| 8K | 7,930.87 ms | 11.29 ms | 0.14% | 1,032.93 |
| 16K | 20,414.46 ms | 70.87 ms | 0.35% | 802.57 |
| 28K | **46,062.63 ms** | **5.24 ms** | **0.01%** | **607.87** |

At 28K the stable profiler records 46,159.432 ms GPU prefill. Diagnostics-free
CPU wall is 46,062.63 ms; these separate builds are not subtracted. Profiler
CPU record/submit/fence-wait/descriptor totals are 78.613/5.981/46,173.399/
5.956 ms per request. GPU gaps above 50 us remain negligible.

Memory at the Phase 12 point is 1.991741 GiB canonical model weights plus
2.742188 GiB native gate/up weights, 4.733929 GiB total weight residency. The
acquired 28K application peak is 8,260 MiB and measured incremental gate/up
initialization is 9.013 s.

## 28K whole-request attribution

The profiler attributes 99.96% of GPU prefill directly. Attention sub-buckets
remain the acquired source-identical Phase 10 normalized split; the fresh
attention total is direct.

| Component | GPU time | GPU prefill |
|---|---:|---:|
| Embedding/input | 348.561 ms | 0.755% |
| Attention RMSNorm | 60.291 ms | 0.131% |
| Q projection | 2,458.974 ms | 5.327% |
| K projection | 677.254 ms | 1.467% |
| V projection | 721.871 ms | 1.564% |
| RoPE | 72.705 ms | 0.158% |
| QK | 9,804.429 ms | 21.24% |
| Softmax/online state | 3,549.358 ms | 7.69% |
| AV | 9,267.804 ms | 20.08% |
| Attention loop/address/conversion/synchronization | 3,855.730 ms | 8.35% |
| O projection | 2,544.163 ms | 5.512% |
| Residual kernels | 135.374 ms | 0.293% |
| MLP RMSNorm | 60.290 ms | 0.131% |
| Gate projection, native | 3,280.132 ms | 7.106% |
| Up projection, native | 3,274.048 ms | 7.093% |
| SiLU/gate multiply | 217.602 ms | 0.471% |
| Down projection | 5,782.249 ms | 12.527% |
| KV write/copy | 26.348 ms | 0.057% |
| Final norm | less than 0.1 ms | less than 0.001% |
| Logits | 3.877 ms | 0.008% |
| Timestamp/control residual | 18.369 ms | 0.040% |
| **Total** | **46,159.432 ms** | **100%** |

The attention sub-buckets use the acquired Phase 10 proportions
37.03/13.41/35.00/14.56% normalized to 26,477.321 ms; they are a model, not
new simultaneous counters. Macro attribution is direct:

| Macro | Time | Share |
|---|---:|---:|
| Exact attention | 26,477.321 ms | 57.36% |
| Seven recurrent matmuls | 18,738.691 ms | 40.60% |
| Other GPU work | 943.420 ms | 2.04% |

## Gate/up Phase 12 postmortem

Gate/up are remeasured after the native path rather than inherited as a stale
bottleneck. The compressed comparison and controlled decomposition remain the
Phase 12 same-session evidence because the source and representation are
unchanged.

| Property | Compressed gate/up | Native gate/up |
|---|---:|---:|
| Canonical affected storage | 828,112,896 B | same, retained |
| Added execution storage | 0 | 2,944,401,408 B / 2.742188 GiB |
| Values | Q4_K blocks | exact dequantized FP16 |
| Logical executed weights/request | 362.713 GB | 1,289.648 GB |
| Logical executed metadata/request | 40.302 GB | 0 |
| Runtime unpack/dequant | present | absent |
| Runtime numeric conversions | five static sites/variant | one output site |
| Weight staging/barriers | shared staging, three barriers | direct tensor load, one barrier |
| Matrix2 useful-compute floor | 1.350 s pair | 1.350 s pair |
| Registers/thread | 98 | 99 |
| Driver shared/WG | 33,792 B | 34,304 B |
| Resident WG / modeled occupancy | 2 / 33.3% | 2 / 33.3% |
| Measured GPU time | 10,649.916 ms | 6,493.897 ms Phase 12; 6,554.180 ms fresh |

The Phase 12 causal component reduction is 4,156.019 ms. The synthetic
free-Q4-metadata probe accounts for at most 642 ms of gate/up, or 15.4% of the
gain. The remaining 3,514 ms (84.6%) jointly removes payload unpack, scale/min
application, conversion, shared staging, two barriers, and associated issue
pressure. Physical loads, cache hits, and overlapped instruction time are not
available on this device, so the report does not fabricate a finer split among
“better layout”, “better loads”, and instruction removal. The decisive fact is
that native wins despite 3.555x logical weight bytes and unchanged occupancy.

Phase 12 economics are 3,776.36 diagnostics-free ms removed for 2.742188 GiB:
**1,377 ms/GiB** and **2.77 whole-request percentage points/GiB**.

## Updated matmul inventory

All rows use M64/N32/K128, 256 threads, 99.886% useful M utilization at 28K,
FP16 activations/output, and FP32 accumulation. “Executed weight” is logical
shader-visible traversal, not physical DRAM traffic.

| Operation | Canonical / execution | GPU ms | Minimum quant / metadata | Executed weight / metadata | Unpack/conversion/staging | Resources | Useful compute | Effective logical weight rate |
|---|---|---:|---:|---:|---|---|---:|---:|
| Q | Q4 / compressed | 2,458.974 | 0.164/0.020 GB | 71.647/8.956 GB | Q4 runtime | 98 reg, 33,792 B, 2 WG | 7.45 TF/s | 32.8 GB/s |
| K | Q4 / compressed | 677.254 | 0.041/0.005 GB | 17.912/2.239 GB | Q4 runtime | same | 6.77 TF/s | 29.8 GB/s |
| V | 13 Q4 + 13 Q6 / compressed | 721.871 | 0.051/0.005 GB | 22.390/2.379 GB | Q4/Q6 runtime | 98/108 reg, 33,792 B, 2 WG | 6.35 TF/s | 34.3 GB/s |
| O | Q4 / compressed | 2,544.163 | 0.164/0.020 GB | 71.647/8.956 GB | Q4 runtime | 98 reg, 33,792 B, 2 WG | 7.21 TF/s | 31.7 GB/s |
| Gate | Q4 / native FP16 | 3,280.132 | 0.368/0.046 GB canonical | 644.824/0 GB native | none / direct load | 99 reg, 34,304 B, 2 WG | 12.59 TF/s | 196.6 GB/s |
| Up | Q4 / native FP16 | 3,274.048 | 0.368/0.046 GB canonical | 644.824/0 GB native | none / direct load | same | 12.61 TF/s | 196.9 GB/s |
| Down | 13 Q4 + 13 Q6 / compressed | 5,782.249 | 0.460/0.049 GB | 201.507/21.410 GB | Q4/Q6 runtime | 98/108 reg, 33,792 B, 2 WG | 7.14 TF/s | 38.6 GB/s |
| Logits | tied Q6 / compressed decode path | 3.877 | material once | material once | Q6 runtime | decode kernel | immaterial | immaterial |

The Matrix2-only useful-compute floor is 0.300 s for Q/O, 0.075 s for K/V,
and 0.675 s for gate/up/down. Load, unpack, conversion, staging, MMA, and output
overlap in one timestamp. The reported effective rates therefore describe the
whole owning operation and never claim isolated physical bandwidth.

## Candidate ranking before implementation

Native estimates scale the fresh measured 3,277.090 ms average gate/up native
time by exact dense work. This is stronger than applying one local percentage
to every shape, but remains a screening model. Mixed Q4/Q6 candidates have
lower confidence until prototyped.

| Rank | Group | Current | Estimated native | Removable | Extra GiB | ms/GiB | Predicted TTFT | Pre-code decision |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 1 | V | 0.722 s | 0.364 s | 0.358 s | 0.152344 | 2,349 | 0.78% | reject: high ratio, immaterial absolute gain |
| 2 | K | 0.677 s | 0.364 s | 0.313 s | 0.152344 | 2,055 | 0.68% | reject below 3% low-memory gate |
| 3 | Down | 5.782 s | 3.277 s | 2.505 s | 1.371094 | 1,827 | 5.44% | **prototype: only candidate clearing absolute gate** |
| 4 | O | 2.544 s | 1.456 s | 1.088 s | 0.609375 | 1,785 | 2.36% | reject below 3% |
| 5 | Q | 2.459 s | 1.456 s | 1.002 s | 0.609375 | 1,645 | 2.18% | reject below 3% |

V leads raw ms/GiB but cannot approach an allowed whole-request outcome. Down
is the highest-value *economically eligible* group and therefore the required
single-group prototype. Q/K/V are not assumed to move together; each is ranked
individually. Full native expansion is rejected before implementation because
the Phase 12 model requires another 3.645 GiB beyond gate/up and approaches the
12 GiB capacity boundary without a qualifying marginal proof.

## Prototype P13-DOWN

The temporary prototype keeps each canonical Q4/Q6 down tensor and appends one
row-major FP16 `[out=3072][in=9216]` execution region. It extends the existing
exact CPU predecoder to Q6 and routes both formats through the existing direct
FP16 Matrix2 shader only for `ffn_down` under an explicit
`GRAPH_HORIZON_PREFILL_PREDECODE_DOWN=1` flag. No serialization, decode route,
allocation count, or fallback ownership changes.

Focused temporary tests established that Q6 predecode matches the CPU Q6
dequantizer after identical FP16 narrowing and that compressed/native Q6
Matrix2 output buffers are byte-identical. Both tests passed before timing.

| Property | Baseline | Down prototype | Delta |
|---|---:|---:|---:|
| Down canonical bytes | 508,944,384 | same | 0 |
| Down native bytes | 0 | 1,472,200,704 | +1.371094 GiB |
| Total native bytes | 2.742188 GiB | 4.113281 GiB | +1.371094 GiB |
| Total resident weight bytes | 5,083,017,216 | 6,555,217,920 | +1,472,200,704 |
| Total resident weights | 4.733929 GiB | 6.105022 GiB | +1.371094 GiB |
| Process initialization, 128 cold | 11.59 s surrounding | 18.17 s | about +6.58 s |
| Model-resident application trace | about 4,855 MiB acquired | about 6,308 MiB | about +1,453 MiB sampled |
| 28K application peak | 8,260 MiB acquired | 9,664 MiB | +1,404 MiB |
| Persistent allocations | existing 52 gate/up | 26 existing down allocations also grow | no new persistent allocation |
| Transient upload | one native tensor at a time | same | peak one 54 MiB staging buffer |

Incremental startup amortizes after about 2.91 28K requests using the measured
6.58 s process delta and 2.260 s diagnostics-free request saving. Total native
startup over compressed is approximately 15.59 s; this is reported but not
optimized because the runtime candidate fails first.

### Profiler A/B

| Context / component | Baseline | Candidate | Delta |
|---|---:|---:|---:|
| 8K down | 1,691.809 ms | 949.074 ms | -43.90% |
| 8K attention | 2,264.105 ms | 2,283.582 ms | +0.86% |
| 8K GPU prefill | 8,008.914 ms | 7,266.022 ms | -9.28% |
| 8K profiled TTFT | 8,052.35 ms | 7,361.00 ms | -8.59% |
| 28K down | 5,782.249 ms | 3,313.502 ms | -42.69% |
| 28K attention | 26,477.321 ms | 26,627.032 ms | +0.57% |
| 28K seven matmuls | 18,738.691 ms | 16,353.905 ms | -12.73% |
| 28K other | 943.420 ms | 946.660 ms | +3.240 ms |
| 28K GPU prefill | 46,159.432 ms | 43,927.597 ms | -4.84% |
| 28K profiled TTFT | 46,367.49 ms | 44,113.88 ms | -4.86% |

The component gain is 2,468.747 ms / 1.371094 GiB = **1,801 ms/GiB**.
Working-set pressure returns 149.711 ms to attention, so the end-to-end result
must not be justified with the local number.

### Diagnostics-free A/B/A

| Context | Baseline | Candidate | Recheck | Surrounding control | Candidate delta |
|---:|---:|---:|---:|---:|---:|
| 128 | 97.37 ms | 86.85 ms | 97.21 ms | 97.29 ms | -10.73% |
| 28K | 45,964.20 ms | 43,812.94 ms | 46,181.64 ms | 46,072.92 ms | **-4.90%** |

The 28K candidate has 87.15 ms SD / 0.20% CV; controls have 161.13/50.15 ms
SD and 0.35/0.11% CV. Diagnostics-free marginal efficiency is 2,259.98 ms /
1.371094 GiB = **1,648 ms/GiB**, or **3.58 percentage points/GiB**. The result
is reproducibly positive but fails the explicit 5% gate by 0.10 percentage
points. P13-DOWN is **REJECTED and removed**.

## Compact representation experiments

The current native gate/up representation has 1,472,200,704 logical values,
two bytes/value, no duplicated data, zero alignment padding, and dimensions
exactly divisible by N32/K128. The theoretical minimum for the exact currently
consumed FP16 values is 2,944,401,408 bytes; actual allocation is exactly that
size. Direct tensor transpose already maps row-major `[out][in]` storage to
Matrix2 B fragments without a tiled copy.

| ID | Representation | Native GiB | Phase 12 gain retained | Expected TTFT | Decision |
|---|---|---:|---:|---:|---|
| P13-COMPACT-A | Tightly packed FP16 Matrix2 tiles | 2.742188 | 100%, same bytes/path | same | reject: zero padding or duplication exists to remove |
| P13-COMPACT-B | Canonical Q4 payload + expanded scale/min metadata | about 0.086 added | at most 15.4% component gain | at most about 1.3% request gain | reject far below 90--95% retention |
| P13-COMPACT-C | Canonical quants reordered to Matrix2 consumption order | about canonical payload | no credible retained full-native gain | below gate | reject: unpack/conversion/shared staging remain mandatory |
| P13-COMPACT-D | FP16 native cache for 75% of layers | 2.056641 | about 75% under equal layer work | about 5.7% historical gain | reject: cannot retain 90% while reducing 25% VRAM |

P13-COMPACT-B uses the acquired controlled free-metadata upper bound rather
than an optimistic instruction count. P13-COMPACT-C is materially different:
it can simplify addresses, but the current canonical row-major traversal
already consumes blocks in output/K order and has no separate repacking kernel.
Matrix2 on the qualified path consumes FP16, so exact quantized storage must
still reconstruct values and stage them exactly as the compressed shader does.
Lossless compression of the FP16 values merely moves that decode back into the
request and cannot qualify as a direct tensor representation.

No compact prototype reaches the required 90--95% performance retention plus
25% VRAM reduction. None is added to production.

## Native-weight Pareto frontier

Absolute TTFT points come from their named sessions. The down marginal decision
uses only its surrounding Phase 13 control, not a cross-session subtraction.

| Configuration | Added native GiB | Incremental init | 28K TTFT | Gain versus same-session control | Marginal ms/GiB |
|---|---:|---:|---:|---:|---:|
| Historical compressed | 0 | 0 | 49,740.01 ms | control | — |
| Phase 12 gate/up native | 2.742188 | +9.013 s | 45,963.65 ms | -7.59% | 1,377 |
| Phase 13 gate/up recheck | 2.742188 | acquired | 45,879.40 ms final | source-identical | — |
| Phase 13 gate/up + down prototype | 4.113281 | +6.58 s marginal | 43,812.94 ms | -4.90% marginal | 1,648 |

Down has a stronger raw marginal ms/GiB than gate/up, but it misses the
absolute gate while consuming another 1.371 GiB and measurably slowing
attention. The knee is not “maximum performance”; it is the last point meeting
the agreed absolute runtime and capacity economics. That point is gate/up.

## Experiment registry

| ID | Direction | Target / representation | VRAM delta | Init delta | Component delta | Attention | TTFT | Efficiency | Correctness | Decision |
|---|---|---|---:|---:|---:|---:|---:|---:|---|---|
| P13-DOWN | expand | 13 Q4 + 13 Q6 down -> exact FP16 | +1.371094 GiB | +6.58 s | -2,468.747 ms | +149.711 ms | -2,259.98 ms / -4.90% | 1,648 wall ms/GiB | Q6 predecode CPU parity and raw Matrix2 equality pass | reject/remove below 5% |
| P13-COMPACT-A | compact | padding-free tiled FP16 | 0 saved | none | 0 | 0 | 0 | none | exact | reject before code |
| P13-COMPACT-B | compact | expanded Q4 metadata | about +0.086 GiB vs canonical | unmeasured | at most -642 ms gate/up | unmeasured | at most about -1.3% | high ratio but low absolute | arithmetic unchanged | reject before durable code |
| P13-COMPACT-C | compact | quantized Matrix2-order blocks | small/none | unmeasured | no demonstrated unpack/staging removal | unmeasured | below gate | unproven | exact possible | reject before code |
| P13-COMPACT-D | compact | 75% layer FP16 cache | -0.686 GiB vs Phase 12 | about -25% | retains about 75% gain | unmeasured | loses about 25% gain | linear | exact | reject before code |

The down prototype retained canonical payload and metadata, eliminated runtime
unpack/conversion/staging only for down, used the existing native Matrix2
resource profile (99 registers, 34,304 B driver shared, two WG, 33.3%
occupancy), and created no persistent allocation. Rejected source, tests,
flags, and Q6 predecoder are absent from the final tree.

## Final performance and memory

No production candidate is retained, so baseline and final are source-identical
at `6cb170f`. The final diagnostics-free matrix includes the required median.

| Context | Mean | Median | SD | CV | Final delta |
|---:|---:|---:|---:|---:|---:|
| 128 | 96.55 ms | 96.30 ms | 0.88 ms | 0.91% | 0% by construction |
| 512 | 366.10 ms | 366.19 ms | 1.26 ms | 0.34% | 0% by construction |
| 2K | 1,561.73 ms | 1,561.70 ms | 3.98 ms | 0.25% | 0% by construction |
| 8K | 7,910.66 ms | 7,910.79 ms | 23.00 ms | 0.29% | 0% by construction |
| 16K | 20,447.46 ms | 20,460.15 ms | 34.43 ms | 0.17% | 0% by construction |
| 28K | 45,879.40 ms | 45,878.72 ms | 15.93 ms | 0.03% | 0% by construction |

Final memory remains:

| Measure | Final |
|---|---:|
| Canonical weights | 1.991741 GiB |
| Native gate/up | 2.742188 GiB |
| Additional native total | 2.742188 GiB |
| Total resident model weights | 4.733929 GiB |
| Peak native upload staging | 54 MiB |
| 28K application VRAM | 8,260 MiB acquired |
| Incremental native initialization | 9.013 s acquired |

The rejected candidate's 100 ms trace reached about 6,308 MiB after model
load, 7,970 MiB while request state grew, and 9,664 MiB steady at 28K. It had
26 additional sequential native uploads but no extra persistent allocation.

## Decode, capability, lifecycle, and correctness

Decode remains canonical compressed by design. The final 32-token guard is:

| Prompt history | Decode | SD | CV |
|---:|---:|---:|---:|
| 128 | 46.87 tok/s | 0.10 | 0.22% |
| 2K | 46.46 tok/s | 0.04 | 0.09% |
| 28K | 29.19 tok/s | 0.01 | 0.05% |

These match the Phase 12 47.205/46.325/29.155 surrounding controls within
0.71/0.29/0.12%, satisfying the 1% guard. The source has no surviving down
route, so multiple-request lifetime, different prompt lengths, and decode
reuse are exactly Phase 12 behavior.

Capability gating is directly rechecked: with
`GRAPH_HORIZON_PREFILL_PREDECODE_MLP=1` and
`GRAPH_HORIZON_PREFILL_MATMUL_MATRIX2=0`, a 128 cold process completes in
2.30 s rather than the approximately 11.59 s native-enabled control. This
confirms that the opt-in does not convert or allocate the native representation
when Matrix2 is disabled. Unsupported-pipeline and hybrid gates are
source-identical to the Phase 12 resolved-pipeline policy.

The temporary down candidate passed Q6 CPU/predecode parity and raw
compressed/native Matrix2 byte equality before performance measurement. It is
removed, so final correctness is the already-qualified, byte-identical Phase
12 production path:

- Q4 native Matrix2 output bit-exact against compressed output;
- CPU/Vulkan parity at unchanged tolerances;
- F16 and INT8 KV long-context qualification through 28K;
- Matrix2, GQA, logits, and greedy sequence
  `2757,10637,2479,1636`;
- Q6, F16, INT8, fallback, multiple-request, and resource-release paths
  unchanged;
- 154 Vulkan engine tests with four intentional ignores and 165 application
  tests acquired at the source-identical baseline.

No tolerance, precision, model format, public API, dependency, fallback,
allocation lifecycle, or default changes survive Phase 13.

## Stop rationale and remaining bottleneck

Phase 13 stops under conditions 2, 3, 4, 5, 6, 7, and 8:

1. Down is the only remaining group with a modeled 5% opportunity, but measured
   diagnostics-free TTFT is 4.90% for +1.371 GiB.
2. K/V have excellent ratios but less than 1% absolute opportunity; Q/O remain
   below 3%. No remaining exact FP16 expansion is economically valid.
3. Four materially distinct compact strategies fail the Pareto gate: zero-
   padding FP16, expanded metadata, quantized tile order, and layer-partial FP16.
4. Down returns 150 ms to attention and pushes 28K application VRAM to
   9,664 MiB, reducing safe capacity headroom.
5. The fresh final GPU path is 57.36% exact attention, 40.60% matmul, and 2.04%
   other. Attention remains the largest bottleneck and Phase 10 closed its local
   exact-representation tuning.
6. The next credible native-weight step changes numeric representation rather
   than reorganizing the same exact FP16 values.

The exact native-weight frontier is closed. Old Matrix2 sweeps and exact-
attention micro-tuning remain falsified and are not reopened.

## Recommended Phase 14

The next weight-side experiment should explicitly authorize a GPU-native
low-precision tensor representation, not another FP16 copy. A one-byte/value
gate/up cache would reduce native residency from 2.742188 to 1.371094 GiB and
halve its 1.290 TB logical executed native-weight stream. Adding down at one
byte/value would cost another 0.685547 GiB instead of 1.371094 GiB.

This is only a capacity/bandwidth upper bound. The qualified device path
currently consumes FP16 Matrix2 values; native FP8/INT8 tensor compatibility,
conversion placement, accumulation rules, and attainable throughput must be
measured. Such a representation changes numeric values or runtime conversion
and therefore requires a new precision contract, logits/greedy qualification,
and long-context quality evidence. It is Phase 14 scope, not an unmeasured
Phase 13 production change.

If low-precision native weights cannot produce a qualifying whole-request gain,
the next architectural target is a different attention primitive or an
explicitly non-exact attention representation. At 26.477 s, attention offers
the largest Amdahl target, but any windowing, sparsity, or state-precision change
must declare its semantic and numeric trade-off up front.

## Repository state

The down implementation, its Q6 predecoder, temporary parity tests, median
instrumentation, and all rejected compact candidates are absent from the final
tree. The production source diff against `6cb170f` is empty. Only this evidence
report and the documentation index are committed. The final worktree is clean
and nothing is pushed.

Current final-tree verification is:

- `cargo fmt --all --check`: pass;
- workspace Clippy, all targets, warnings denied: pass for pure Vulkan and
  Vulkan-hybrid;
- Vulkan-profile engine library: 154 passed, four intentionally ignored;
- family-agnostic structural/dispatch/placement suite with `docs_contract`
  skipped: five passed, one authenticated-model test intentionally ignored;
- `docs_contract` remains blocked before link validation by the checkout's
  pre-existing missing root `VALIDATION.md`, the same known baseline condition;
- local Phase 13 report target exists and `git diff --check` passes;
- production source is byte-identical to `6cb170f`.
