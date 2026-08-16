<!--
This document owns the Phase 20 Vulkan specialization architecture, capability
and shape qualification, portability inventory, preserved-performance evidence,
and the policies for future GPUs, models, and performance work.
-->

# Vulkan Specialization Architecture — Phase 20

## Outcome

Phase 20 converts the Phase 19 kernel from an implicit current-device route to
an explicit **NVIDIA Matrix2 Q64 attention specialization, performance-qualified
on RTX 3060/Ampere**. Selection contains no device-name or model-name check.
The default path is decided from actual Vulkan capabilities, successful pipeline
construction, datatype, attention shape, and prefill row geometry.

The retained implementation is `1c10022` on
`perf/vulkan-specialization-architecture-phase20`. It is based on the clean
Phase 19 production branch `perf/vulkan-ampere-attention-phase19` at `15f32b7`;
the Phase 19 implementation is `3b9ba6c`. Nothing is pushed.

The exact dense causal numeric contract is unchanged: F16 Q/K/V/output, FP32
QK/online-softmax/AV state, and no global score or probability matrix. No shader
tuning, weight-path optimization, dependency, public API, tolerance, or startup
preprocessing was added.

The main 28K performance guard passes. Fresh pre-edit Phase 19 and final Phase 20
means are 35,493.50 and 35,509.83 ms, a **+0.05%** change with Phase 20 CV 0.13%.
This is inside noise and preserves the acquired ~22% Phase 19 gain.

## Baseline

| Item | Value |
|---|---|
| Phase 20 branch | `perf/vulkan-specialization-architecture-phase20` |
| Phase 19 branch / tip | `perf/vulkan-ampere-attention-phase19` / `15f32b7` |
| Phase 19 implementation | `3b9ba6c` |
| Phase 20 implementation | `1c10022` |
| GPU | NVIDIA GeForce RTX 3060 12 GiB, PCI device `0x2487` |
| Driver | 595.84 |
| Model | Ministral-3-3B-Instruct-2512 Q4_K_M |
| Shape | 26 layers, hidden 3072, FFN 9216, 32 Q heads, 8 KV heads, head dimension 128 |
| Runtime tuple | Vulkan, 32,768 capacity, F16 KV, greedy, optional Phase 12 gate/up native path enabled |

The Phase 19 mission anchor is 35,557.68 ms at 28K, -22.41% TTFT,
-37.36% GPU attention, 2.533x GH/llama, and decode within +/-0.23%.
Its independent source-clean result is -21.84%. Phase 20 uses the fresh pre-edit
35,493.50 ms measurement for the direct preservation comparison below.

## Hardware-family classification

### A. Ampere-specific

No shader instruction, device identifier, tile predicate, or capability check is
intrinsically Ampere-specific. Ampere owns the measured qualification evidence:
RTX 3060 timing, 255 registers/thread, 48,640 driver-reported shared bytes/WG,
two resident workgroups/SM, and the observed context curve. Those measurements
must not be projected onto another family.

### B. NVIDIA-specific

The specialization requires NVIDIA vendor ID `0x10de`,
`VK_NV_cooperative_matrix2`, and the associated NVIDIA SPIR-V/GLSL operations.
The vendor ID is a capability-owner check, not a marketing device-name match.
No string such as `RTX 3060` or model registry entry participates in routing.

### C. Matrix2-specific

The exact device contract is:

- `VK_KHR_cooperative_matrix` and `VK_NV_cooperative_matrix2` advertised;
- KHR `cooperativeMatrix` enabled;
- NV workgroup scope and flexible dimensions;
- NV reductions, accumulator-to-A/B conversions, per-element operations, and
  tensor addressing;
- workgroup-scope, non-saturating F16 A/B and FP32 C/result flexible property;
- exactly 128 workgroup invocations for the Q64 property;
- non-zero M/N/K granularities that each divide 64;
- Matrix2 maximum workgroup size at least 128 and maximum dimension at least 128;
- Vulkan compute invocation/X limits at least 128;
- device shared-memory limit at least the implementation-reserved Matrix2 bytes;
- successful construction of the exact Q64 SPIR-V pipeline.

The conversion bit is required because the shader converts the FP32 probability
accumulator to an F16 MatrixA before AV. Phase 20 now queries and enables it
explicitly, matching the [Vulkan Matrix2 feature definition](https://docs.vulkan.org/refpages/latest/refpages/source/VkPhysicalDeviceCooperativeMatrix2FeaturesNV.html).
Granularity and workgroup checks follow the
[flexible-dimensions contract](https://docs.vulkan.org/refpages/latest/refpages/source/VkCooperativeMatrixFlexibleDimensionsPropertiesNV.html).

Block loads are not used and therefore are not required. The Q64 contract is
independent of the existing WG256 Matrix2 property; lack of WG256 no longer
artificially disables a valid WG128 pipeline.

### D. Shape-specific

The performance-qualified host predicate requires:

- `head_dim == 128`;
- integral GQA with `q_heads / kv_heads == 4`;
- non-zero Q/KV head counts;
- F16 Q, K, V, and output;
- Q rows laid out `[token][query_head][128]`;
- KV laid out `[layer][token][kv_head][128]`;
- equal key/value head width;
- a non-zero query-row count divisible by 64.

Absolute head counts are not required. Production exercises 32/8, while the
focused GPU oracle exercises 4/1; each workgroup owns one query head, so total
head count is only dispatch extent. Layer count, hidden size, FFN size, weight
format, model name, and sequence capacity are absent from the predicate.

The backend's existing Vulkan buffer-binding alignment invariant remains
required. No stronger shader-specific alignment is introduced: Matrix2 tensor
layouts consume contiguous F16 elements from already valid storage bindings.

### E. Workload-specific

Only dense causal F16 **prefill** is eligible. Decode and INT8 use their existing
paths. Query tails use the next qualified fallback. Phase 19 found no useful
extra context crossover: 128 was neutral inside the 1% guard and benefit began
at 512, while a 64-row tail at a long base is valuable. Consequently the route
has no `base + rows` threshold; one complete Q64 tile is the only workload floor.
The threshold is host policy, not shader logic.

## Required, preferred, and incidental assumptions

| Assumption | Status | Reason |
|---|---|---|
| F16 Q/K/V/output; FP32 QK/softmax/AV | required | Numeric and Matrix2 component-type contract |
| `head_dim=128` | required | QK and AV cooperative matrix dimensions are compiled at 128 |
| GQA ratio 4 | required for current qualification | Production 32/8 and focused 4/1 are tested; other ratios fall back |
| 32 Q heads / 8 KV heads | incidental | Head counts only determine dispatch extent; ratio is the qualified shape |
| Q64 / KV64 | required | Compiled cooperative-matrix dimensions and online-state layout |
| WG128 | required | Advertised flexible property and shader local size |
| four 32-lane subgroups | incidental on RTX 3060 | No subgroup-size predicate is needed by the WG Matrix2 shader |
| contiguous token/head/dimension strides | required | Tensor layout and direct Q/K/V addressing |
| valid Vulkan storage-buffer offset alignment | required upstream invariant | All bound views must already meet device alignment |
| explicit shader shared arrays | absent by design | Online state is compiler-managed Matrix2 state |
| reserved Matrix2 shared bytes fit device limit | required | Device property plus Vulkan resource limit |
| 48,640 driver shared bytes/WG | incidental measurement | Current driver executable property, not a portable constant |
| 255 registers/thread / two WG per SM | preferred observed resource point | No portable register predicate exists; pipeline build is the safety gate |
| long context | preferred for larger gain | Not required for correctness or route safety |

## Central eligibility and routing policy

The pure policy in `vulkan/kernels/attention/policy.rs` is conceptually:

```text
can_use_nvidia_q64 =
    datatype == F16                         // enforced by caller split
    AND workload == dense causal prefill    // enforced by operation
    AND exact Q64 pipeline exists           // capability + build success
    AND head_dim == 128
    AND kv_heads > 0
    AND q_heads % kv_heads == 0
    AND q_heads / kv_heads == 4
    AND rows > 0
    AND rows % 64 == 0
```

The full default hierarchy is:

```text
portable Vulkan attention core
└── F16 prefill shape/workload policy
    ├── NVIDIA Matrix2 Q64/KV64/WG128
    │   └── performance-qualified: RTX 3060 / Ampere
    ├── existing Matrix2 Q32/KV64/WG256
    ├── existing KHR cooperative-QK tiled path
    ├── existing tiled path
    ├── existing wide path
    └── portable runtime-dimension fallback
```

Pipeline absence represents extension/feature/property failure or exact shader
build rejection. It never causes initialization failure. Optional pipelines
allocate no attention-specific runtime storage.

Diagnostic routing is intentionally private:

- `GRAPH_HORIZON_PREFILL_ATTENTION_ROUTE=generic` forces the portable shader;
- `GRAPH_HORIZON_PREFILL_ATTENTION_ROUTE=phase19` requests Q64 but still applies
  every safety predicate, falling back to portable when ineligible;
- unset or `auto` uses the hierarchy above;
- the inherited `GRAPH_HORIZON_PREFILL_MATRIX2=0` remains the broad Matrix2-off
  control for automatic routing.

## Model-independence audit

No fixed layer count, hidden size, FFN size, sequence capacity, or model name is
present in attention capability discovery or routing. Fixed 32/8 head counts were
not added. The only retained shape constants are those actually qualified:
head dimension 128, GQA ratio 4, Q64/KV64, and F16.

Untrusted metadata now rejects fractional GQA grouping before allocation or
dispatch. A different head dimension or GQA ratio takes the existing fallback;
Phase 20 does not pretend to support head-dim 64, a different GQA specialization,
or another datatype.

### Native weight-path audit

The Phase 12 gate/up native path is separate from attention. It is selected from:

- successful `MatmulF16Matrix2F16Out` pipeline construction;
- Q4_K source representation;
- exact tensor shape `[3072, 9216]`;
- gate/up tensor role suffix;
- explicit opt-in environment policy.

It contains no device-name or model-name check. It is capability-, role-, shape-,
and weight-format-driven, but intentionally supports only the measured 3B
gate/up shape. Phase 20 does not optimize or generalize it.

## Qualification matrix

| Scenario | Expected path | Tested | Result |
|---|---|---|---|
| RTX 3060/Ampere + head128/GQA4 + long F16 prefill | NVIDIA Q64 | real GPU, 8K label and 28K aggregate | pass; Q64 label observed, final 96-row 28K tail uses Q32 |
| RTX 3060/Ampere + unsupported GQA or head dimension | fallback | pure route tests | pass |
| RTX 3060/Ampere + 63/64/65, 127/128/129, 255/256/257 rows | tails/Q64 by predicate | real GPU plus pure route tests | pass |
| RTX 3060/Ampere + short 128 context | NVIDIA Q64 | real GPU | pass; preserved inside guard |
| Matrix2 broadly disabled | next established path | pure route test | pass |
| Matrix2 unavailable / Q64 pipeline absent | fallback | pure route test | pass |
| required conversion/reduction/type/scope/tile/resource missing | no Q64 pipeline | pure capability tests | pass |
| non-NVIDIA vendor | portable hierarchy | vendor-ID gate by inspection; no AMD hardware | expected; specialization cannot build or route |
| forced generic | portable `attention_prefill` | real GPU profiler | pass; exact kernel label observed |
| forced Phase 19 | Q64 if safe | real GPU profiler | pass; exact Q64 label observed |
| INT8 | established INT8 path | 2K/8K/28K real GPU | pass; bit-exact |

The same Vulkan sources and generic fallback compile after the vendor gate. No
AMD device was available, so AMD performance or physical execution is not
claimed; the tests prove the decision/fallback behavior without one.

## Correctness and regression evidence

- The complete Vulkan library suite passes 157 tests with five ignored after the
  final policy addition. The earlier full run was 156 passes; the additional
  diagnostic-policy test accounts for the increment.
- `family_agnostic` retains four passes, one ignore, and the pre-existing
  `docs_contract` failure caused by the absent root `VALIDATION.md`.
- Real GPU boundary tests pass at 63/64/65, 127/128/129, and 255/256/257.
- F16 Q64, Q32, and tail rows pass sequential Vulkan and CPU-oracle comparisons
  at 2K, 8K, and 28K. The 28K Q64 CPU-oracle maximum absolute attention error is
  `1.374e-5`; maximum prefill/decode difference remains `1.526e-5`.
- INT8 prefill/decode attention and logits are bit-exact at 2K, 8K, and 28K.
- Forced portable and forced Phase 19 prefill produce the same 128 greedy token
  IDs after the same 2K prompt, with decode held on the same fallback.
- Dense format smoke covers F16, Q4_K, Q6_K, and logits.
- `cargo fmt --check`, Vulkan `cargo check`, CPU `cargo check`, release shader
  compilation, and strict Clippy pass. Strict Clippy allows only the documented
  pre-existing Rust 1.95 `manual_is_multiple_of` lint in `harness/stats.rs`.

No tolerance changed.

## Performance preservation

The pre-edit and final binaries use the same model, process settings, prompt
construction, warm-ups, and repetitions. Short rows use two warm-ups and
15/10/5 repetitions; long rows use one warm-up and three repetitions.

| Context | Phase 19 fresh | Phase 20 final | Delta | Phase 20 CV |
|---:|---:|---:|---:|---:|
| 128 | 97.28 ms | 96.58 ms | -0.72% | 0.54% |
| 512 | 366.79 ms | 362.85 ms | -1.07% | 0.38% |
| 2K | 1,515.91 ms | 1,503.77 ms | -0.80% | 0.05% |
| 8K | 7,057.36 ms | 7,031.50 ms | -0.37% | 0.24% |
| 16K | 16,853.12 ms | 16,831.53 ms | -0.13% | 0.27% |
| 28K | **35,493.50 ms** | **35,509.83 ms** | **+0.05%** | **0.13%** |

The favorable short-row movement is ordinary session drift, not a Phase 20
speedup mechanism. The important result is no material loss and a 28K result
well inside the +/-1% guard.

### Decode guard

| Context | Phase 19 | Phase 20 | Delta |
|---:|---:|---:|---:|
| 128 | 46.79 tok/s | 47.00 tok/s | +0.45% |
| 512 | 46.30 tok/s | 46.63 tok/s | +0.71% |
| 2K | 46.24 tok/s | 46.46 tok/s | +0.48% |
| 8K | 40.57 tok/s | 40.78 tok/s | +0.52% |
| 16K | 34.97 tok/s | 35.00 tok/s | +0.09% |
| 28K | 29.21 tok/s | 29.30 tok/s | +0.31% |

All requested decode guards are within 1%.

### Startup guard

One cold 128-token/two-output process measured:

| Mode | Whole process | Measured cold TTFT | Approximate non-inference remainder |
|---|---:|---:|---:|
| normal compressed startup | 2.11 s | 0.166 s | 1.94 s |
| optional Phase 12 native gate/up | 10.91 s | 0.330 s | 10.58 s |

The optional increment is about 8.8 seconds, consistent with the inherited
Phase 12 ~9.013-second preprocessing cost. Phase 20 attention adds no buffer,
conversion, or preprocessing step.

## Current 28K bottleneck and Phase 21 inventory

A final focused `vulkan-profile` process contains one warm-up and one measured
request. Values below divide the two-request aggregates by two; percentages are
unchanged. Q64 attention is 16.320 s/request. The seven matmuls total
**18.468 s/request, 51.67% of GPU prefill**, confirming the Phase 19 51.49%
bottleneck shift.

| Operation | GPU s/request | Request GPU % | Weight representation | Current kernel |
|---|---:|---:|---|---|
| Q | 2.420 | 6.77% | canonical Q4_K | Q4 Matrix2 M64/N32/K128 |
| K | 0.664 | 1.86% | canonical Q4_K | Q4 Matrix2 M64/N32/K128 |
| V | 0.709 | 1.98% | canonical Q4_K/Q6_K by layer | Q4/Q6 Matrix2 M64/N32/K128 |
| O | 2.506 | 7.01% | canonical Q4_K | Q4 Matrix2 M64/N32/K128 |
| gate | 3.246 | 9.08% | opt-in FP16 execution copy from Q4_K | F16 Matrix2 M64/N32/K128 |
| up | 3.240 | 9.07% | opt-in FP16 execution copy from Q4_K | F16 Matrix2 M64/N32/K128 |
| down | 5.682 | 15.90% | canonical Q4_K/Q6_K by layer | Q4/Q6 Matrix2 M64/N32/K128 |
| **Total** | **18.468** | **51.67%** | | |

The down projection is the largest single residual cost, followed by gate/up,
O, and Q. This is the Phase 21 starting order. Phase 20 performs no matmul
redesign.

## Production optimization portability inventory

Terminology: **supported** has current code/test evidence; **expected** follows
the capability contract but lacks that hardware qualification; **needs
qualification** must be benchmarked before a performance claim; **fallback**
means correctness remains supported but the named optimization is not selected.

| Optimization | Generic Vulkan | AMD | NVIDIA | Ampere | Model/shape dependency |
|---|---|---|---|---|---|
| batched RoPE | portable | expected | supported | supported | runtime head/rope dimensions; no model name |
| final-batch-only logits | portable orchestration | expected | supported | supported | model-independent graph policy |
| request KV retention | portable Vulkan orchestration | expected | supported | supported | capacity/layout dependent, not model name |
| Q4/Q6 Matrix2 prefill matmul | NVIDIA Matrix2 specialization + fallback | fallback | architecturally eligible | RTX 3060 qualified | M64/N32/K128 and operation dimensions; Q4/Q6-specific |
| gate/up FP16-native | NVIDIA Matrix2 specialization + compressed fallback | fallback | architecturally eligible | RTX 3060 qualified | exact `[3072,9216]`, gate/up role, Q4 source, opt-in |
| decode GQA split/reduce | portable Vulkan capability path | expected on subgroup-32 device | supported | RTX 3060 qualified | GQA/head/context shape; no weight dependency |
| Phase 19/20 Q64 attention | NVIDIA Matrix2 specialization + portable fallback | fallback | architecturally eligible | RTX 3060 qualified | F16, head128, GQA4, Q64; no weight/model name |

### Generalization matrix

| Optimization | AMD | NVIDIA generic | Ampere | Other model, same shape | Different shape |
|---|---|---|---|---|---|
| batched RoPE | expected | supported | supported | supported | supported when runtime RoPE invariants hold |
| final-batch-only logits | expected | supported | supported | supported | supported |
| request KV retention | expected | supported | supported | supported | supported when budget/layout checks pass |
| Q4/Q6 Matrix2 matmul | generic fallback | needs qualification | RTX 3060 supported | supported for listed operation dimensions | fallback; new dimensions need qualification |
| gate/up FP16-native | compressed fallback | needs qualification | RTX 3060 supported | supported only for exact tensor shape/role | compressed fallback |
| decode GQA | expected, untested | needs qualification | RTX 3060 supported | supported for qualified GQA/head shape | fallback |
| NVIDIA Q64 attention | generic fallback | architecturally eligible, performance unqualified | RTX 3060 supported; other Ampere unqualified | architecturally eligible for head128/GQA4/F16 | fallback |

Ada and Blackwell are not claimed compatible or performant. If their drivers
advertise the exact contract, the pipeline is architecturally eligible; only a
qualification run can promote that to performance-qualified.

## Current performance maturity

llama.cpp remains a diagnostic and competitive reference, **not a parity
requirement**. Prefill ratios use Phase 20 TTFT divided by the Phase 18 llama.cpp
prefill value. Decode uses the latency-equivalent slowdown
`llama.cpp tok/s / Graph Horizon tok/s`, so every row consistently means
"Graph Horizon time / llama.cpp time" and lower is better.

| Workload | Current GH/llama | Target envelope | Distance to upper target |
|---|---:|---:|---:|
| decode short, 128 | 2.187x | <=1.3-1.4x | +0.787x |
| decode long, 28K | 1.707x | <=1.3-1.4x | +0.307x |
| prefill short, 128 | 2.247x | <=1.5x | +0.747x |
| prefill 8K | 2.692x | <=1.5-1.7x | +0.992x |
| prefill 16K | 2.616x | <=1.5-1.7x | +0.916x |
| prefill 28K | 2.530x | <=1.5-1.7x | +0.830x |

For completeness, 512 and 2K prefill are 2.755x and 2.778x. Current decode
slowdowns at 2K/8K/16K are 2.046x/1.918x/1.800x.

### Long-context scaling contract

The current prefill curve is:

```text
8K  2.692x
16K 2.616x
28K 2.530x
```

The ratio improves by 6.0% from 8K to 28K rather than deteriorating. Future
changes must preserve this monotonic direction within measurement noise.

### Current-model stop contract

Stop competitive optimization of the current model when all are true:

- decode <=1.3-1.4x llama.cpp;
- short prefill <=1.5x;
- long prefill <=1.5-1.7x;
- the 8K -> 16K -> 28K ratio does not deteriorate materially;
- numeric tolerances and greedy behavior remain intact;
- normal and optional startup remain acceptable.

The current model has not reached those envelopes. This does not authorize
chasing every residual gap: after the threshold, further work is allowed only
for very low engineering cost, broad model/GPU benefit, a memory/startup fix,
or a production workload requirement.

## Future GPU qualification policy

```text
new NVIDIA GPU
  -> confirm exact KHR/NV features, properties, limits, and pipeline build
  -> run generic correctness and boundary matrix
  -> force existing NVIDIA Q64 path
  -> compare generic/Q64 performance at 128, 2K, 8K, and long context
  -> KEEP only when correct and beneficial
  -> otherwise retain generic fallback
  -> create Ada/Blackwell specialization only after evidence requires it
```

Architectural eligibility is not performance qualification. An RTX 3080 is
likely architecturally eligible only if it advertises the exact contract, but
it is not performance-qualified. RTX 40 and Blackwell compatibility/performance
remain unknown. AMD uses the generic fallback.

## Future model policy

```text
new model
  -> generic path first
  -> validate metadata, tensor shapes, and correctness
  -> inspect explicit datatype/head/GQA/weight predicates
  -> reuse a specialization only when every predicate matches
  -> benchmark generic versus specialized path
  -> specialize only when the measured need justifies it
```

This avoids model registries and premature head-dim/GQA variants. A model with
the same attention shape can be architecturally eligible without sharing layer,
hidden, FFN, or weight shapes. A different attention shape falls back first.

## Second-model readiness

The backend now meets the Phase 20 readiness conditions:

- generic fallback is explicit and forceable;
- specialization discovery is capability-based;
- qualified shape constraints are centralized and unit-tested;
- weight constraints remain separate and explicit;
- routing is testable without physical GPU variants;
- unsupported devices never build the Q64 pipeline and unsupported shapes never dispatch it;
- no model-name or RTX 3060 name check exists.

The production structure remains small:

| File | Productive lines, excluding tests | Single responsibility |
|---|---:|---|
| `vulkan/coopmat2/mod.rs` | ~112 | query and enable NVIDIA Matrix2 capability |
| `vulkan/coopmat2/abi.rs` | ~153 | define ABI records and pure capability predicates |
| `vulkan/kernels/attention/policy.rs` | ~106 | select F16 prefill attention route |

All orchestration files remain below 200 productive lines. The Q64 shader stays
one category-K attention kernel and is functionally unchanged.

## Experiment registry

| ID | Change / target | Phase 19 selected | Fallback selected | 28K delta | 128 delta | Decode delta | Correctness | Decision |
|---|---|---|---|---:|---:|---:|---|---|
| P20-A | exact capability contract, independent WG128 gate, centralized shape/workload route, forced endpoints | yes on eligible F16 tiles | yes for missing caps, unsupported shapes, INT8, tails, and forced generic | +0.05% | -0.72% | +0.45% at 128; +0.48% at 2K; +0.31% at 28K | all gates pass | **KEEP**; explicit and within noise |

No shader experiment or rejected production candidate was needed. The only
implementation corrected and generalized the policy around the acquired kernel.

## Recommended Phase 21

Start from the measured projection/MLP matmul inventory, not attention. Down is
5.682 s/request and 15.90% of 28K GPU prefill; gate/up together are 6.486 s and
18.15%; O and Q follow. Preserve FP32 accumulation and the separate weight-format
predicates unless a future quality decision explicitly changes them.

Phase 21 should first compare reusable large-M ownership/dataflow alternatives
against the current M64/N32/K128 kernels, beginning with down and then the shared
gate/up mapping. It must not reopen Q64/KV64/WG128 attention absent a bug,
regression, or a concrete generalization requirement.
