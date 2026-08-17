<!--
This report owns Phase 23 evidence for the Model 2 large-K Vulkan down
projection: baseline, attribution, retained routing change, qualification, and
the next measured bottleneck. It defines no new public support contract.
-->

# Vulkan Model 2 Large-K Matmul — Phase 23

## Decision

**KEEP** the existing compressed Q4_K/Q6_K Matrix2 kernel for the exact
`K=14336, N=4096` prefill down projection. The production change is only a
capability-and-shape policy extension plus focused tests. It introduces no new
shader, allocation, persistent representation, dependency, public API, model
name check, or precision rule.

On the RTX 3060, Model 2 public TTFT improves by **47.60–52.09%** from 128 to
28K tokens. The down-projection GPU time improves by **90.55% at 2K**, **90.66%
at 8K**, and **90.61% at 28K**. All Model 2 and Model 1 decode comparisons stay
within the mandatory 1% regression guard. CPU, generic Vulkan, optimized
Vulkan, F16 KV, INT8 KV, 128-step teacher forcing, and 128-token free-running
generation retain the required correctness.

This is an architectural reuse win, not a tile sweep. The old scalar path
reused each decoded weight across 32 prompt rows and spent 83.71% of its 2K
down time in shared reads plus scalar FP32 accumulation. The retained M64/N32/
K128 Matrix2 path reuses weights across 64 rows, performs FP32 cooperative
accumulation, and removes about 75% of the internal K-loop barriers. Its
measured 8K result, 4.576 s, matches the 4.5–5.0 s pre-code prediction.

The largest remaining 28K bottleneck is now compressed Q4_K gate/up:
59.346/59.283 s, or 35.28/35.24% of GPU prefill. Phase 24 should qualify a
shape-family Matrix2 route for `K=4096, N=14336`; it should not revisit the
now-small down path first.

## Reproducibility and scope

| Item | Value |
|---|---|
| Baseline branch | `perf/second-model-bringup-phase22` |
| Baseline commit | `d1f945979944bc74484fcaf5304776a05524f296` |
| Investigation branch | `perf/model2-largek-matmul-phase23` |
| Production commit | `de9f81291411b3d8b5024d0cec86afbbb59ec1c9` |
| GPU | NVIDIA GeForce RTX 3060, 12 GiB |
| Driver | NVIDIA 595.84, Vulkan |
| Model 2 | `Ministral-3-8B-Instruct-2512-Q4_K_M.gguf` |
| Model 2 bytes | 5,198,911,904 |
| Model 2 SHA-256 | `33e7a72cf5e6e2cfc2f2847075acc013d68bba023e35310cef86b5cf8fdca761` |
| Model 1 guard | `Ministral-3-3B-Instruct-2512-Q4_K_M.gguf` |
| Prompt | exact chat-templated `" a"` repetition producing the requested token count |
| Request | greedy, 32 decoded tokens, context capacity 32,768, F16 KV unless noted |
| Public repetitions | warmup 1; 5/5/3/3/2/2 at 128/512/2K/8K/16K/28K |
| Profile repetitions | warmup 0, one request; Vulkan GPU timestamps |
| External oracle | llama.cpp build 9826, commit `9bebfcb4bc8b12a316e96ae03f33671eac1e72fd` |

All temporary shader ablations, executable-property probes, and expanded parity
lengths were restored before the production commit. Raw ignored artifacts are
under `target/phase23`; the repository contains only the retained source change
and this evidence.

## Fresh public baseline

The fresh baseline is from the exact Phase 22 commit, not copied from the prior
report. CV is `SD / mean`.

| Prompt | Reps | TTFT mean / median ms | SD / CV | Prompt tok/s | Decode tok/s |
|---:|---:|---:|---:|---:|---:|
| 128 | 5 | 1,408.32 / 1,407.38 | 5.65 / 0.0040 | 90.89 | 21.89 |
| 512 | 5 | 5,446.65 / 5,446.53 | 2.86 / 0.0005 | 94.00 | 21.73 |
| 2,048 | 3 | 21,887.54 / 21,850.31 | 100.47 / 0.0046 | 93.57 | 21.25 |
| 8,192 | 3 | 89,235.83 / 89,239.13 | 18.02 / 0.0002 | 91.80 | 19.60 |
| 16,384 | 2 | 182,000.11 / 182,000.11 | 18.26 / 0.0001 | 90.02 | 17.82 |
| 28,000 | 2 | 320,013.67 / 320,013.67 | 110.71 / 0.0003 | 87.50 | 15.80 |

Active baseline telemetry samples, filtered to memory clock at least 7,000 MHz
and memory use at least 9,000 MiB, report 9,610/9,657/9,787 MiB min/mean/max,
1,942/1,982/2,002 MHz core, 70.50/138.32/165.79 W, and 58/66.9/68 C. There is
no thermal or clock collapse matching the long-context cost.

## Baseline GPU breakdown

The profiler accounts for 99.99% of prefill at all three decision points.
Counts and non-down categories scale as expected, while down remains about 52–56%
of total GPU prefill.

| Component | 2K ms | 8K ms | 28K ms |
|---|---:|---:|---:|
| GPU total | 21,877.012 | 89,249.453 | 320,434.916 |
| Q | 337.050 | 1,338.249 | 4,584.325 |
| K | 87.085 | 344.210 | 1,181.633 |
| V | 87.234 | 351.959 | 1,209.562 |
| O | 318.244 | 1,283.238 | 4,379.640 |
| Gate | 4,309.844 | 17,325.168 | 59,314.088 |
| Up | 4,320.943 | 17,359.432 | 59,321.324 |
| Activation | 29.966 | 121.767 | 416.375 |
| **Down** | **12,193.083** | **49,009.510** | **167,776.162** |
| Attention | 123.189 | 1,853.201 | 21,357.867 |

At 8K the remaining normalization/RoPE/KV/residual/embedding/logits total is
256.526 ms. There are no hidden CPU gaps: record plus submit are 31.452 ms,
the GPU wait is 89,251.516 ms, and profiler residual is 6.193 ms.

The down projection has 34 layers, `M x N x K = prompt x 4096 x 14336`, with
17 Q4_K and 17 Q6_K tensors. Each layer contains 229,376 quant superblocks.

| Format | Stored payload/layer | Metadata/layer | 8K baseline time | 28K baseline time |
|---|---:|---:|---:|---:|
| Q4_K, 17 layers | 29.360 MB | 3.670 MB | 24,952.043 ms | 85,388.659 ms |
| Q6_K, 17 layers | 44.040 MB | 4.129 MB | 24,057.467 ms | 82,387.502 ms |
| Mixed total | 1.249 GB payload | 132.579 MB metadata | 49,009.510 ms | 167,776.162 ms |

## Causal down-projection model

### Controlled cumulative ablations

The 2K shader ablations keep dispatch, dimensions, output stores, barriers, and
the rest of the model fixed. Each row includes the previous deletion, so
adjacent differences are additive and close 100% of baseline down time.

| State | Q4_K ms | Q6_K ms | Total ms | Removed from previous state |
|---|---:|---:|---:|---|
| T0: production fallback | 6,202.657 | 5,990.426 | 12,193.083 | — |
| W0: weight/dequant value is zero | 5,328.621 | 5,382.995 | 10,711.616 | weight global/metadata/unpack/scale/conversion/address: 1,481.467 ms |
| WA0: activation loads also zero | 5,276.208 | 5,326.580 | 10,602.788 | activation loads/address: 108.828 ms |
| WAC0: shared reads/scalar FMA also zero | 197.265 | 199.056 | 396.320 | shared reads + scalar FP32 accumulation: 10,206.468 ms |

| Ranked component | Q4_K | Q6_K | Total | Baseline share |
|---|---:|---:|---:|---:|
| Shared reads + scalar FP32 accumulation | 5,078.943 ms | 5,127.524 ms | 10,206.468 ms | **83.71%** |
| Weight global/metadata/unpack/scale/conversion/address | 874.036 ms | 607.431 ms | 1,481.467 ms | **12.15%** |
| Residual loop/index/barrier/epilogue/output | 197.265 ms | 199.056 ms | 396.320 ms | 3.25% |
| Activation load/address | 52.413 ms | 56.415 ms | 108.828 ms | 0.89% |

This instrumentation cannot legitimately split physical weight load,
metadata reconstruction, payload unpack, numeric conversion, and address
arithmetic inside W0. They are therefore reported as one measured component,
with bytes and static instructions below, rather than assigned invented times.
The 2K output is 0.570 GB; even a pessimistic full DRAM write at 360 GB/s is
about 1.6 ms, so output traffic does not explain the residual 396 ms.

### Bytes, reuse, and instruction evidence

The byte counts are shader-visible logical loads derived from the exact source
mapping. They are not physical DRAM counters; cache can satisfy repeated loads.

| Prompt | Old weight payload+metadata | Matrix2 candidate | Old activation loads | Candidate activation loads | Output writes |
|---:|---:|---:|---:|---:|---:|
| 128 | 5.522 GB | 2.761 GB | 7.986 GB | 15.972 GB | 0.036 GB |
| 512 | 22.086 GB | 11.043 GB | 31.944 GB | 63.888 GB | 0.143 GB |
| 2,048 | 88.345 GB | 44.172 GB | 127.775 GB | 255.551 GB | 0.570 GB |
| 8,192 | 353.379 GB | 176.689 GB | 511.101 GB | 1,022.202 GB | 2.282 GB |
| 16,384 | 706.757 GB | 353.379 GB | 1,022.202 GB | 2,044.404 GB | 4.563 GB |
| 28,000 | 1,207.837 GB | 604.609 GB | 1,746.928 GB | 3,493.855 GB | 7.799 GB |

The old path decodes one weight traversal per 32 rows; the candidate does so
per 64 rows. At 8K this halves logical weight movement from 353.379 to 176.689
GB, while Matrix2 N tiling doubles logical activation tensor loads. Weight
metadata alone changes from 15.972/17.968 GB Q4/Q6 to 7.986/8.984 GB. The
candidate's measured mixed logical weight rate is 38.62 GB/s versus 7.21 GB/s
for the old path. These rates diagnose execution structure, not DRAM bandwidth.

Static SPIR-V from the production build configuration gives the following
cost inventory. “Matrix2 ops” counts cooperative multiply-add sites, not
dynamic instructions.

| Variant | SPIR-V words | Loads | Stores | Access chains | Shifts / bit ops | Int mul / div | Conversions | Barriers | Matrix2 load/MMA/store |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Old Q4 fallback | 3,120 | 138 | 85 | 24 | 12 / 12 | 15 / 4 | 5 | 2 | 0 / 0 / 0 |
| Old Q6 fallback | 2,952 | 129 | 74 | 24 | 10 / 10 | 20 / 5 | 4 | 2 | 0 / 0 / 0 |
| Matrix2 Q4 | 2,721 | 106 | 70 | 18 | 12 / 15 | 10 / 3 | 5 | 3 | 2 / 1 / 1 |
| Matrix2 Q6 | 2,795 | 115 | 70 | 21 | 16 / 14 | 18 / 4 | 4 | 3 | 2 / 1 / 1 |

The dynamic internal barrier count is about `2 * (14336 / 32) = 896` per old
workgroup and `2 * (14336 / 128) + 1 = 225` for Matrix2: approximately 75%
fewer barriers. This reduction and cooperative accumulation address the
measured 83.71% component directly.

## Architectural comparison and prediction

The selected architecture is the already-qualified explicit-dequant Matrix2
kernel: M64/N32/K128, 256 threads, eight subgroups, FP16 A/B fragments, FP32
accumulation, FP16 output. It owns one M64 x N32 output tile per workgroup,
loads activations directly, decodes a compressed weight tile into shared memory
once for 64 rows, then issues Matrix2 operations.

The relevant current paths differ as follows. Gate/up executable register and
driver-shared properties were not exposed by the retained profiler; their
source-declared shared memory is exact and occupancy is therefore not guessed.

| Path | Rows/WG / weight reuse | Accumulation | Registers/thread | Shared/WG | Resident WG / occupancy | Logical weight bytes/output at 8K |
|---|---:|---|---:|---:|---:|---:|
| Model 2 gate/up Q4 coopmat | 32 / 32 | subgroup 16x16 FP32 coopmat | unavailable | 3,584 B declared | unavailable | 72 B |
| Old down Q4/Q6 fallback | 32 / 32 | scalar FP32 | 64 / 63 | 20,992 B driver | 4 / 66.7% | 252 / 367.5 B |
| Selected down Q4/Q6 Matrix2 | 64 / 64 | Matrix2 FP32 | 98 / 108 | 33,792 B driver | 2 / 33.3% | 126 / 183.75 B |

The table's final column is a comparative logical executed-weight metric per
scalar output in one layer, not a physical per-scalar fetch. Both down formats
keep eight logical FP32 accumulator scalars per thread. The old and selected paths therefore retain
128 resident output rows/SM despite the candidate's lower workgroup residency.
Zero stack is driver-reported for all down variants; physical spill counters
are unavailable.

Three Matrix2 workgroups require at most 85.33 registers/thread and 34,133 B
shared/WG. Q4 misses the register threshold by 12.67 and Q6 by 22.67; shared
memory has only 341 B of headroom against that rounded threshold. Four
workgroups require at most 64 registers and 25,600 B shared. A register-only
rewrite cannot reach four-WG residency without also reducing shared memory.

Tile utilization is 100% through 16K and 99.886% at 28K (`28000/28032`). N and
K are exact. Useful down work is 32.710 TFLOP at 8K. The candidate sustains
7.149 TFLOP/s mixed, 7.396 Q4 and 6.917 Q6, versus 0.667 TFLOP/s on the scalar
fallback.

Before implementation, the closest same-kernel evidence was Model 1's 8K
mixed Q4/Q6 down result. Scaling its useful work predicted **4.56 s** for Model
2, with a conservative 4.5–5.0 s practical band. That implied about 44.4 s
removable from 8K GPU prefill and about 49.8% public TTFT improvement. The
measured result is **4.576 s**, validating the model.

Phase 21's rejected direct compressed Matrix2-feed callback was not reopened.
It retained 64-row reuse but moved quant decode into tensor-load callbacks,
raised register/shared pressure, and regressed. The current candidate already
removes the scalar accumulation bottleneck without that complexity.

## Experiment register

| ID | Experiment | Architecture / invariant | 2K result | 8K result | 28K result | Decision |
|---|---|---|---:|---:|---:|---|
| P23-T0 | Fresh production baseline | exact Phase 22 commit | down 12,193.083 ms | down 49,009.510 ms | down 167,776.162 ms | baseline |
| P23-W0 | Weight/dequant zero | dispatch/stores/barriers unchanged | -12.15% down | not repeated | not repeated | causal evidence only; restored |
| P23-WA0 | Also zero activations | cumulative controlled deletion | -13.04% down | not repeated | not repeated | causal evidence only; restored |
| P23-WAC0 | Also zero shared/FMA | retains loop/epilogue/store | -96.75% down | not repeated | not repeated | identifies dominant scalar accumulation; restored |
| P23-M64 | Existing compressed Matrix2 | M64/N32/K128, exact shape gate | **1,151.913 ms (-90.55%)** | **4,575.749 ms (-90.66%)** | **15,761.561 ms (-90.61%)** | **KEEP** |
| P23-DIRECT | Phase 21 direct decode callback | same reuse, higher resources | prior measured regression | prior measured regression | prior measured regression | do not reopen |
| P23-M128 | More row ownership | estimated ~166 Q4/~198 Q6 registers and one WG from prior qualification | no distinct >=5% whole-TTFT bound after P23-M64 | not built | not built | reject below gate |
| P23-EXPAND | Persistent FP16 down | would add multi-GiB residency and startup conversion | no longer needed | no longer needed | no longer needed | reject: Matrix2 compressed route wins with 0 B |

Only P23-M64 changes production. The experiment gate was met before code: the
measured 8K down component alone was 54.91% of GPU prefill and the same-kernel
prediction exceeded the required 5% whole-TTFT opportunity by an order of
magnitude.

## Retained performance

### Public request results

| Prompt | Baseline TTFT ms | Candidate TTFT ms | TTFT delta | Candidate SD / CV | Baseline decode tok/s | Candidate decode tok/s | Decode delta |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 1,408.32 | 674.67 | **-52.09%** | 0.93 / 0.0014 | 21.89 | 22.11 | +1.01% |
| 512 | 5,446.65 | 2,668.23 | **-51.01%** | 7.12 / 0.0027 | 21.73 | 21.81 | +0.37% |
| 2,048 | 21,887.54 | 10,804.19 | **-50.64%** | 45.53 / 0.0042 | 21.25 | 21.19 | -0.28% |
| 8,192 | 89,235.83 | 44,602.55 | **-50.02%** | 34.80 / 0.0008 | 19.60 | 19.56 | -0.20% |
| 16,384 | 182,000.11 | 92,835.15 | **-48.99%** | 49.31 / 0.0005 | 17.82 | 17.83 | +0.06% |
| 28,000 | 320,013.67 | 167,697.44 | **-47.60%** | 45.34 / 0.0003 | 15.80 | 15.78 | -0.13% |

The candidate's active telemetry reports 9,602/9,645/9,650 MiB memory,
1,807/1,972/2,010 MHz core, 52.70/162.82/168.41 W, and 50/69.9/71 C. The source
change allocates zero bytes and measured VRAM is unchanged within telemetry
noise. Memory clock remains 7,501 MHz.

### GPU causal result and bottleneck movement

| Component | 8K baseline ms | 8K candidate ms | Delta | 28K baseline ms | 28K candidate ms | Delta |
|---|---:|---:|---:|---:|---:|---:|
| GPU total | 89,249.453 | 44,522.098 | -50.11% | 320,434.916 | 168,228.393 | -47.50% |
| Down | 49,009.510 | 4,575.749 | **-90.66%** | 167,776.162 | 15,761.561 | **-90.61%** |
| Gate | 17,325.168 | 17,248.798 | -0.44% | 59,314.088 | 59,345.754 | +0.05% |
| Up | 17,359.432 | 17,240.711 | -0.68% | 59,321.324 | 59,282.610 | -0.07% |
| Attention | 1,853.201 | 1,834.975 | -0.98% | 21,357.867 | 21,363.111 | +0.02% |

Command, dispatch, and barrier counts are unchanged. At 28K the new shares are
gate 35.28%, up 35.24%, attention 12.70%, and down 9.37%. This clean movement,
together with unchanged neighboring times, is the causal proof that the policy
route—not clocks or a global runtime change—produces the speedup.

## Correctness and guard models

### Model 2

- The local llama.cpp external oracle generated 128 greedy completion IDs.
  CPU teacher forcing, forced-generic Vulkan, and automatic optimized Vulkan
  all matched its top-2 set at every step; all three Graph Horizon top-1 ID
  arrays were exactly identical for all 128 steps.
- Forced generic Vulkan disabled prefill Matrix2/coopmat, specialized attention,
  and decode GQA. It therefore validates the generic unsupported-shape route
  independently of the selected path.
- A 2K-prompt, 128-token free-running Vulkan generation produced exactly the
  same ID sequence with forced decode-GQA fallback and automatic optimized
  decode.
- F16 KV is covered by the 128-step matrix. Optimized Vulkan with INT8 KV also
  passed the external-oracle parity gate for 16 steps with the same IDs.
- The new focused large-K CPU/Vulkan oracles passed at unchanged tolerances:
  Q4_K max/mean absolute 23.601563/9.415780 and max/mean relative
  0.000495196/0.000194875; Q6_K 0.010269/0.003455 and
  0.00462842/0.000691067.
- The exact policy test accepts `14336 -> 4096` and rejects `14336 -> 3072`.
  No tolerance or oracle vector was changed.

### Model 1 non-regression

Model 1 uses the established `GRAPH_HORIZON_PREFILL_PREDECODE_MLP=1` profile.
The new predicate is false for every Model 1 shape, so this is a guard against
incidental compiler/runtime movement rather than a changed route.

| Workload | Phase 22 guard | Phase 23 | TTFT delta | Decode delta | Result |
|---|---:|---:|---:|---:|---|
| 128 | 96.59 ms / 46.92 tok/s | 96.97 ms / 46.86 tok/s | +0.39% | -0.13% | pass |
| 28K | 35,305.85 ms / 29.46 tok/s | 35,532.66 ms / 29.26 tok/s | +0.64% | -0.68% | pass |

Both are within the mandatory 1% regression threshold.

## Portability boundary

The policy is keyed to format, exact tensor shape, the existing Matrix2
capability contract, and the existing global enable switch. It does not inspect
model name, GPU marketing name, vendor string, or driver version. Unsupported
formats, shapes, devices, explicit disablement, and hybrid placement keep the
existing fallback.

NVIDIA-specific Matrix2 property discovery and SPIR-V feature gating remain in
the capability layer established in Phase 20. Generic Vulkan compiles and is
the safe endpoint; AMD was compile-checked but was not runtime-qualified on
this NVIDIA-only host. This phase therefore makes no AMD performance claim.
The durable interface remains narrow: one exact measured shape joins an
existing architecture family.

## Competitive snapshot and maturity gates

llama.cpp used the same Model 2 GGUF, `Vulkan0`, all GPU layers, no split, flash
attention, F16 K/V, batch 2,048, microbatch 512, and three repetitions. Its
prompt-only boundary excludes Graph Horizon tokenization and first sampling;
the remaining ratios are too large for that small boundary difference to
change classification.

| Workload | GH baseline | GH candidate | llama.cpp mean | Baseline ratio | Candidate ratio | Target | Status |
|---|---:|---:|---:|---:|---:|---:|---|
| Prefill 8K | 89,235.83 ms | 44,602.55 ms | 5,217.067 ms | 17.10x | **8.55x** | <=1.7x | RED |
| Prefill 28K | 320,013.67 ms | 167,697.44 ms | 24,470.459 ms | 13.08x | **6.85x** | <=2.5x | RED |

llama.cpp SD is 16.536 ms at 8K (0.32%) and 37.177 ms at 28K (0.15%). Phase 22
already classified decode as RED at roughly 2.1–2.3x slower than llama.cpp; the
candidate does not change decode. The performance maturity gate is therefore
still not met despite the large retained improvement.

## Verification

| Check | Result |
|---|---|
| `cargo fmt --all -- --check` | pass |
| Workspace CPU check | pass |
| Vulkan-profile check | pass |
| Strict CPU clippy, all targets | pass |
| Strict Vulkan clippy, all targets | pass |
| CPU engine library | 159 passed, 2 ignored |
| Vulkan engine library | 158 passed, 5 ignored |
| Vulkan-hybrid engine library | 230 passed, 4 ignored |
| CPU/Vulkan family-agnostic excluding `docs_contract` | 4 passed, 1 ignored each |
| Vulkan-hybrid family-agnostic excluding `docs_contract` | 5 passed, 1 ignored |
| Full CPU workspace | 165 passed; inherited installer fixture failure remains |

The repository-wide failure is unchanged from Phase 22:
`support_scripts::installer_reports_missing_prerequisite_before_build` observes
exit 1 instead of its environment-fixture expectation of exit 2. The root
`VALIDATION.md` file required by `family_agnostic::docs_contract` is also absent
on the baseline branch, so that known contract test remains excluded. Neither
condition touches Vulkan matmul code.

Conceptual complexity increases only by one exact predicate clause and two
focused test calls. This is smaller than introducing a second shader family and
keeps all routing, capability, resource, and fallback invariants in their
existing owners.

## Phase 24 recommendation

Target Model 2 gate/up as one shape-family investigation:

```text
Q4_K prefill matmul
K = 4096, N = 14336, M = prompt rows
current = subgroup coopmat M32 reuse
candidate = existing compressed Matrix2 M64/N32/K128, capability gated
```

The first gate should compare the current gate/up shader's executable resources
and exact logical traffic with the existing Matrix2 Q4 path, then benchmark 2K
and 8K before a full sweep. At 8K, gate+up are 34.490 s, 77.46% of GPU prefill;
at 28K they are 118.628 s, 70.52%. A same-family result near the selected down
kernel's throughput would materially improve TTFT and is much larger than any
remaining down micro-optimization.

Do not begin with persistent FP16 expansion: Model 2 gate/up would add about
7.438 GiB and exceed the intended memory economics. Do not reopen Phase 21's
direct decode callback without new compiler or counter evidence. Preserve the
same correctness matrix, Model 1 1% guard, generic fallback, and no model-name
routing rule.
