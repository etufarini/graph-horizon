<!--
This report owns Phase 24 evidence for Model 2 Vulkan gate/up prefill:
eligibility, causal attribution, retained routing, qualification, guards, and
the next measured bottleneck. It defines no new public support contract.
-->

# Vulkan Model 2 Gate/Up — Phase 24

## Decision

**KEEP** the existing compressed Q4_K Matrix2 kernel for the measured
`K=4096, N=14336` prefill gate/up shape. The production change adds that shape
to the existing capability/format/workload policy and adds focused routing and
CPU/Vulkan tests. It adds no shader, allocation, native representation,
dependency, public API, model-name check, GPU-name check, or precision rule.

On the RTX 3060, the exact public Vulkan F16 runner improves by **57.62% at
8K** and **52.80% at 28K** in same-session baseline/candidate/baseline tests.
The full qualification matrix improves by 52.02–59.27%, has CV at or below
0.30%, and preserves Model 2 decode and Model 1 within 0.3%. At 28K, gate and
up each become about four times faster: 58.962/59.045 s become 14.706/14.697 s.

The result crosses the 20% strong-signal gate, so Phase 24 stopped local tuning
after this one architecture. Attention is now the largest individual 28K GPU
category at 21.465 s (27.07%). Gate+up remains the largest aggregate domain at
29.403 s (37.07%), but no second gate/up architecture was attempted after the
mandatory stop.

## Reproducibility and scope

| Item | Value |
|---|---|
| Baseline branch | `perf/model2-largek-matmul-phase23` |
| Baseline commit | `364ca52b6cb8b55217ab008c8825312906e6aab9` |
| Phase 23 implementation ancestor | `de9f81291411b3d8b5024d0cec86afbbb59ec1c9` |
| Investigation branch | `perf/model2-gate-up-phase24` |
| Production commit | `b86e2a51cfad16d354d4d0a36828b3ccff5846b1` |
| GPU / driver | NVIDIA GeForce RTX 3060 12 GiB / 595.84 |
| Nominal memory system | 3 MiB L2, 360 GB/s DRAM |
| Model 2 | `Ministral-3-8B-Instruct-2512-Q4_K_M.gguf` |
| Model 2 bytes / SHA-256 | 5,198,911,904 / `33e7a72cf5e6e2cfc2f2847075acc013d68bba023e35310cef86b5cf8fdca761` |
| Model 1 | `Ministral-3-3B-Instruct-2512-Q4_K_M.gguf` |
| Model 1 bytes / SHA-256 | 2,147,023,008 / `9ed150d4367e68df0ac8e1540f6ddc65b42d0ee26378329d1ecbca60f93fc5f8` |
| Prompt | exact chat-templated `" a"` repetition producing the requested token count |
| Request | greedy, 32 decoded tokens, context capacity 32,768, F16 KV unless noted |
| Public repetitions | warmup 1; 5/5/3/3/2/2 at 128/512/2K/8K/16K/28K |
| Profile | warmup 0, one request, Vulkan timestamps, 256-row public batching |
| External oracle | retained authenticated Phase 23 llama.cpp prompt/completion vectors |

The immutable baseline executable is
`0fe755c92a1edd46cd75d964f4cd55cc73268f6de69b1843b22e48398832f26d`.
The candidate public executable is
`15950a247b79e4c64520eb8580120094e3585b274307318d027c8fe8221831c2`;
the exact candidate profiler is
`ae06fe2eff5d71511e9746bee5d443f8bc685f9ced55b68490c375e78fdbaef1`.
Ignored raw evidence is under `target/phase24`.

One early screen accidentally compared the pure Vulkan baseline with a hybrid
candidate. The hybrid runner uses 32-row batches instead of the public pure
Vulkan runner's 256-row batches, so that tuple was invalidated and excluded.
The corrected A/B/A uses the same `vulkan` feature, prompt, capacity, KV format,
binary profile, and device state for all three legs.

## Baseline and eligibility audit

The fresh Phase 23 baseline at 28K is 165,067.82 ms public TTFT and 167,448.67
ms timestamped GPU prefill. Gate is 58,961.53 ms (35.21%) and up is 59,045.41
ms (35.26%): 118.007 s and 70.47% together. The actual path is the legacy
`matmul_q4k_coopmat_f16` kernel for both operations.

| Operation | Logical M | N | K | Stored format | Baseline execution |
|---|---:|---:|---:|---|---|
| gate | prompt rows | 14,336 | 4,096 | canonical Q4_K | M32/N16/K16 cooperative matmul |
| up | prompt rows | 14,336 | 4,096 | canonical Q4_K | M32/N16/K16 cooperative matmul |

Each operation has 34 tensors. One tensor stores 29,360,128 payload bytes plus
3,670,016 metadata bytes, or 33,030,144 bytes. One complete projection is
1,123,024,896 bytes; gate+up is 2,246,049,792 bytes. The working set is hundreds
of times larger than L2, and gate time per row is nearly flat from 2K to 28K,
so no later cache-residency regime appears.

The Matrix2 shader already accepts canonical Q4_K, requires the discovered
Matrix2 capability, uses M64/N32/K128, and has no tensor-layout dependence on a
model identity. For this shape, N is divisible by 32 and K by 128; M utilization
is 100% through 16K and 99.886% at 28K. The sole missing condition was the
policy's measured-shape list. This was a performance qualification exclusion,
not a shader-safety invariant.

| Operation | Expected after audit | Actual | Predicate | Result |
|---|---|---|---|---|
| gate | compressed Q4 Matrix2 | `matmul_q4k_matrix2_wg256_m64_n32_k128` | enabled + Q4_K + Matrix2 capability + `4096 -> 14336` + pipeline | pass |
| up | compressed Q4 Matrix2 | `matmul_q4k_matrix2_wg256_m64_n32_k128` | same | pass |
| unsupported `3072 -> 14336` | legacy fallback | excluded by exact shape | capability alone is insufficient | pass |
| unsupported `4096 -> 12288` | legacy fallback | excluded by exact shape | unmeasured N | pass |

This follows the Phase 20 architecture: capability + shape + quant format +
prefill workload. Unsupported devices, formats, shapes, or an explicit disable
retain the established fallback.

## Causal gate decomposition

A controlled 2K shader ablation closed 100% of both gate and up time. Each step
kept the later loop structure and output path while removing one source of work.

| Component | Gate ms / share | Up ms / share |
|---|---:|---:|
| compressed weight load, decode, staging | 2,918.912 / 69.26% | 2,915.602 / 69.19% |
| activation reads | 258.416 / 6.13% | 266.040 / 6.31% |
| cooperative tensor load/MMA | 440.097 / 10.44% | 445.004 / 10.56% |
| loop, barriers, addressing, epilogue, output | 596.818 / 14.16% | 587.143 / 13.93% |
| total | 4,214.243 / 100% | 4,213.789 / 100% |

This identifies a combined compressed-decode/feeding/mapping regime. It is not
a native-FP16 bandwidth regime: Model 2 has no native gate/up allocation.

## Representation, traffic, and floors

| Metric | Canonical compressed | Hypothetical FP16 native |
|---|---:|---:|
| bytes/tensor | 33,030,144 | 117,440,512 |
| gate+up, all layers | 2.092 GiB | 7.438 GiB |
| runtime metadata/dequant | 3.670 MB/tensor; in shader | none |
| canonical fallback retained | yes | yes, so bytes are additive |
| Matrix2 execution | existing compressed M64/N32/K128 | existing native M64/N32/K128 |
| 28K measured gate | 14.706 s | not built: capacity/economics reject |

The 7.438 GiB native expansion plus the retained canonical weights, other model
weights, KV, and scratch is disproportionate on a 12 GiB device. It was rejected
before implementation. The compressed candidate adds zero persistent bytes.

For one gate or up operation across 34 layers:

| Context | Minimum unique weights | Shader-visible weight traffic | Minimum activation | Shader-visible activation | Output | Effective logical BW | Useful compute |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 8K | 1.123 GB | 143.747 GB | 2.282 GB | 1.022 TB | 7.986 GB | 275.0 GB/s | 7.663 TFLOP/s |
| 28K | 1.123 GB | 491.885 GB | 7.799 GB | 3.494 TB | 27.296 GB | 272.9 GB/s | 7.603 TFLOP/s |

“Shader-visible” is a logical access model, not a DRAM-counter claim; caches can
change physical traffic. The useful-compute floor at the RTX 3060's nominal
61.1 TFLOP/s FP16 rate is 0.535/1.830 s at 8K/28K. The unique compressed-weight
DRAM floor is only 3.12 ms, but is unattainable without cross-workgroup reuse.
The practical pre-code floor from the already-measured compressed Matrix2 path
was 4.42/15.12 s; actual gate time is 4.27/14.71 s. That agreement justified
stopping rather than inventing a new representation.

## Resource and ISA model

Pipeline executable properties were captured with temporary profile-only
instrumentation and then removed.

| Kernel | Threads / subgroups | Registers/thread | Driver shared/WG | Resident WG | Modeled occupancy | Stack/spills |
|---|---:|---:|---:|---:|---:|---:|
| legacy Q4 coopmat | 64 / 2 | 42 | 3,584 B | 24 | 100% | 0 |
| compressed Q4 Matrix2 | 256 / 8 | 98 | 33,792 B | 2 | 33.3% | 0 |

The candidate deliberately exchanges occupancy for reuse and tensor work. One
workgroup owns 64 rows and 32 output columns, has eight accumulator scalars per
thread, and shares each weight tile across 64 rows. At 28K it launches 438 × 448
workgroups per layer instead of 875 × 896, four times fewer. Its K loop has 32
K128 iterations rather than 256 K16 iterations; the dynamic barrier model falls
from about 513 to 65 per workgroup.

| Static SPIR-V measure | Legacy | Matrix2 |
|---|---:|---:|
| bytes / words | 11,176 / 2,794 | 10,884 / 2,721 |
| `OpLoad` / `OpStore` / `OpAccessChain` | 117 / 71 / 21 | 106 / 70 / 18 |
| integer multiply / divide | 17 / 5 | 10 / 3 |
| shifts / bitwise | 12 / 13 | 12 / 15 |
| control barriers | 3 | 3 |
| cooperative loads / MMA / store | 2 / 1 / 1 | 2 / 1 / 1 |

Static counts do not multiply loop trips; the K16-to-K128 iteration reduction
and fourfold workgroup reduction are the important dynamic differences.

## Candidate ranking and prototype

| Rank | Candidate | Removable seconds / predicted TTFT | Added VRAM | Risk / portability | Decision |
|---:|---|---:|---:|---|---|
| 1 | admit compressed M64/N32/K128 Matrix2 | about 88 s / 52–58% | 0 | low; existing capability path | built, KEEP |
| 2 | FP16-native Matrix2 | lower instruction floor, but 7.438 GiB pair | 7.438 GiB plus canonical | capacity and load-time risk | reject before build |
| 3 | compressed M128 variant | possible extra reuse | 0 | estimated ~166 registers, one WG | not built after strong-signal stop |
| 4 | gate/up fusion | activation sharing only | scratch/ABI change | ideal saving below the 15% gate | reject before build |
| 5 | hybrid metadata representation | removes only part of the 69% decode term | persistent bytes | prediction below selected route | reject before build |

The one prototype changes only the policy predicate. It uses canonical Q4_K,
M64/N32/K128, 256 threads, eight subgroups, reuse 64, 98 registers, 33,792 B
shared memory, two resident workgroups, zero stack, and no new representation.
The experiment registry identifies it as `P24-A`: hypothesis “the excluded
shape can reuse proven compressed Matrix2”; affected shape `M x 14336 x 4096`;
measured output 4.27/4.27 s gate/up at 8K and 14.71/14.70 s at 28K; effective
logical bandwidth 273 GB/s; useful compute 7.60–7.66 TFLOP/s; no VRAM delta;
attention/down/decode/Model 1 guards pass; correctness pass; **KEEP** because
the exact public 28K request improves by 52.80%.

## Public performance

### Same-session decision runs

| Context | Baseline A ms | Candidate ms | Baseline B ms | Delta vs bracket mean |
|---:|---:|---:|---:|---:|
| 8K | 44,370.38 | 18,862.62 | 44,642.32 | **-57.62%** |
| 28K | 167,474.32 | 79,049.50 | 167,447.35 | **-52.80%** |

Active-clock telemetry for the decision session covered 530 samples: memory
clock 7,501 MHz, graphics clock 1,807/1,971.5/2,010 MHz, temperature
49/69.3/71 C, and power 38/159.2/169 W (min/mean/max). The baseline telemetry
was comparable at 1,807/1,970.5/2,002 MHz, 54/70.8/72 C, and
46.5/164.2/169.3 W. Baseline bracketing excludes thermal drift as the cause.

### Full F16 qualification

| Prompt | Reps | Baseline TTFT ms | Candidate mean / median ms | SD / CV | Delta | Prompt tok/s | Decode tok/s |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 5 | 668.39 | 277.25 / 277.25 | 0.32 / 0.12% | -58.52% | 461.67 | 22.03 |
| 512 | 5 | 2,629.94 | 1,071.30 / 1,072.73 | 2.20 / 0.21% | -59.27% | 477.92 | 21.78 |
| 2,048 | 3 | 10,594.72 | 4,366.60 / 4,363.15 | 13.09 / 0.30% | -58.79% | 469.02 | 21.23 |
| 8,192 | 3 | 43,889.38 | 18,837.10 / 18,843.75 | 16.83 / 0.09% | -57.08% | 434.89 | 19.57 |
| 16,384 | 2 | 91,369.77 | 41,229.17 / 41,229.17 | 10.89 / 0.03% | -54.88% | 397.39 | 17.80 |
| 28,000 | 2 | 165,067.82 | 79,200.61 / 79,200.61 | 19.11 / 0.02% | -52.02% | 353.53 | 15.77 |

INT8 KV retained oracle correctness. Its public 128 and 2K checks were stable
(296.44 ms, CV 0.23%; 8,939.39 ms, CV 0.17%). A 28K multi-repetition INT8
performance run was terminated after 27 minutes because the pre-existing INT8
long-attention path made it disproportionate and unrelated to the Q4 gate/up
decision; no incomplete number is used here.

## Reprofile and guards

| Category | 8K baseline / candidate ms | Delta | 28K baseline / candidate ms | Delta |
|---|---:|---:|---:|---:|
| attention | 1,815.93 / 1,846.63 | +1.69% | 21,360.86 / 21,464.77 | +0.49% |
| Q | 1,248.06 / 1,278.14 | +2.41% | 4,334.47 / 4,406.11 | +1.65% |
| K | 340.55 / 350.10 | +2.80% | 1,182.16 / 1,203.04 | +1.77% |
| V | 351.55 / 359.65 | +2.30% | 1,223.84 / 1,238.64 | +1.21% |
| O | 1,251.33 / 1,282.91 | +2.52% | 4,376.84 / 4,429.28 | +1.20% |
| gate | 16,956.12 / 4,268.47 | **-74.83%** | 58,961.53 / 14,705.84 | **-75.06%** |
| up | 16,972.17 / 4,271.85 | **-74.83%** | 59,045.41 / 14,697.24 | **-75.11%** |
| activation | 121.08 / 123.69 | +2.16% | 416.98 / 426.02 | +2.17% |
| down | 4,501.17 / 4,617.64 | +2.59% | 15,669.96 / 15,846.11 | +1.12% |
| GPU prefill | 43,814.59 / 18,664.14 | -57.40% | 167,448.67 / 79,301.10 | -52.64% |

The attention and down paths are still the exact Phase 23 kernels. Their small
timestamp drift is not a routing regression, and their 28K changes consume only
0.28 s versus 88.60 s removed from gate/up. Phase 23 down remains about 90.6%
faster than its pre-Phase-23 path.

Gate/up times after KEEP are:

| Context | Gate ms | Up ms |
|---:|---:|---:|
| 2K | 1,105.56 | 1,101.92 |
| 8K | 4,268.47 | 4,271.85 |
| 16K | 8,583.13 | 8,579.05 |
| 28K | 14,705.84 | 14,697.24 |

Model 2 same-session decode changes are +0.18% at 128, -0.12% at 2K, and
-0.22% at 28K. Model 1 uses the established predecoded gate/up profile; its
shape was already admitted before Phase 24.

| Model 1 guard | Baseline bracket | Candidate | TTFT delta | Decode delta |
|---|---:|---:|---:|---:|
| 128 | 96.72 ms / 46.91 tok/s | 96.65 ms / 46.95 tok/s | -0.07% | +0.09% |
| 28K | 35,556.44 ms / 29.50 tok/s | 35,595.30 ms / 29.51 tok/s | +0.11% | +0.05% |

## Model 1 versus Model 2

| Metric | Model 1 native gate/up | Model 2 compressed gate/up |
|---|---:|---:|
| N / K | 9,216 / 3,072 | 14,336 / 4,096 |
| native gate+up bytes | 2.742 GiB | 7.438 GiB hypothetical |
| rows/WG / weight reuse | 64 / 64 | 64 / 64 |
| registers / driver shared | 99 / 34,304 B | 98 / 33,792 B |
| residency / occupancy | 2 WG / 33.3% | 2 WG / 33.3% |
| 28K M/N/K tile utilization | 99.886% / 100% / 100% | 99.886% / 100% / 100% |
| effective logical BW | 198.6 GB/s native | 272.9 GB/s compressed |
| useful compute | about 12.4 TFLOP/s | 7.60 TFLOP/s |

Model 1 can afford its 2.742 GiB opt-in expansion and benefits from removing
dequantization. The analogous Model 2 expansion does not fit the same economic
contract, so reuse of the compressed kernel is the simpler solution.

## Correctness

- The focused `4096 x 14336`, three-row Q4_K CPU/Vulkan oracle passed without
  changing tolerances: max/mean absolute 7.914063/2.052788 and max/mean relative
  0.000480458/0.000148999.
- CPU, forced-generic Vulkan, and automatic optimized Vulkan produced identical
  local top-1 arrays for all 128 teacher-forced steps. Every external-oracle
  token was in the local top two.
- Forced generic disabled prefill cooperative/Matrix2 matmul, specialized
  attention, and decode GQA, independently exercising the fallback.
- Optimized INT8 KV passed the external oracle for 16 steps with identical IDs.
- Forced-fallback and optimized Vulkan exactly matched the retained 128-token
  greedy sequence after a 2K prompt.
- Full CPU, Vulkan, and Vulkan-hybrid library suites preserve Q4, Q6, Matrix2,
  logits, lifecycle, and routing coverage. No tolerance or reference changed.

## Competitive maturity

Fresh llama.cpp build 9826 measurements use the same GPU, model, F16 KV, flash
attention, full offload, and three repetitions.

| Context | GH before | GH after | llama.cpp | Before ratio | After ratio | Target | Status |
|---:|---:|---:|---:|---:|---:|---:|---|
| 8K | 43.889 s | 18.837 s | 5.231 s | 8.39x | **3.60x** | <=1.5x | RED |
| 28K | 165.068 s | 79.201 s | 24.527 s | 6.73x | **3.23x** | <=1.5–1.7x | RED |

llama.cpp prompt rates were 1,566.08 tok/s (SD 4.02) and 1,141.58 tok/s
(SD 1.15). Phase 24 roughly halves the competitive ratio but does not reach
maturity, as predicted by the original 70.47% Amdahl fraction.

## Verification and known baseline checks

`cargo fmt`, CPU/Vulkan/Vulkan-profile/Vulkan-hybrid checks, and CPU/Vulkan
clippy with the single pre-existing lint allowed all pass. Library results are
159 CPU, 158 Vulkan, and 231 Vulkan-hybrid tests passed. Family tests pass when
the local validation-register contract test is skipped.

Two repository-baseline issues are recorded rather than repaired out of scope:

- strict current-toolchain clippy fails in unchanged `harness/stats.rs` on the
  pre-existing manual `% 2 == 0` form;
- `docs_contract` requires the intentionally local, gitignored `VALIDATION.md`,
  which is absent in this worktree.

Neither issue touches Phase 24 source. Temporary shader ablations, pipeline
resource probes, 32-row diagnostic profiler builds, and 128-step parity edits
were restored. No temporary instrumentation remains.

## Final stop and future reuse

The final decision is **KEEP / STOP Phase 24**. The result satisfies the >=10%
28K KEEP threshold, the >=20% strong-signal stop, correctness, decode, Model 1,
attention, and Phase 23 down guards. Conceptual complexity increases only by
one exact measured policy tuple and its tests; it reuses every existing runtime
and shader invariant.

A future model can reuse this route without code changes when it has canonical
Q4_K prefill projections with `K=4096`, `N=14336`, large M divisible or safely
tailed by M64, exact N32/K128 tiling, and a device satisfying the existing
Matrix2 capability contract. New N/K widths remain deliberately unqualified.

The recommended Phase 25 target is long-context attention: it is now the
largest individual 28K category and grows superlinearly. The updated profile,
not Phase 23's obsolete 70.47% gate/up fraction, must seed that investigation.
