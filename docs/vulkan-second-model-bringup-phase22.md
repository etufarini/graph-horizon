<!--
This document owns the Phase 22 second-model bring-up: the complete weight and
shape audit, generic and optimized correctness evidence, inherited-route
qualification, performance baseline, regression guard, and Phase 23 decision.
-->

# Vulkan Second-Model Bring-up — Phase 22

## Outcome

**MODEL SUPPORTED.** `Ministral-3-8B-Instruct-2512-Q4_K_M.gguf` loads, runs
prefill and decode, and generates deterministically on both the CPU reference
and Vulkan backends. No production source change was required. Selection is
still driven exclusively by architecture metadata, tensor shape, quantization,
workload, and device capabilities; no model-name rule was added.

The model first passed the canonical-weight CPU path, then a deliberately
portable Vulkan configuration, and finally the automatic production route.
CPU and optimized Vulkan produced the same local top-1 IDs for 128
teacher-forced accumulated decode steps. A separate 128-token free-running
Vulkan sequence was identical with decode GQA forced off and automatically
selected. The external llama.cpp oracle was exactly greedy-identical for the
first 32 steps and remained in Graph Horizon's top two for all 128
teacher-forced steps.

The previous architecture generalizes substantially. Final-batch logits,
request KV retention, batched YaRN RoPE, eligible Q4_K/Q6_K Matrix2 matmuls,
packed Q6_K interpretation, decode GQA, and NVIDIA Q64 attention are reused.
The optional Phase 12 gate/up representation is shape-ineligible and would add
7.438 GiB for this model, so it is neither built nor benchmarked.

Performance maturity is **RED / CASE B**. Correctness and fallback behavior are
sound, and Q64 attention removes 65.168 s at 8K, but the 14,336-wide MLP falls
outside the measured Matrix2/native shape set. Recurrent matmul consumes 97.51%
of 8K GPU prefill; down projection alone is 48.900 s, or 54.96%. Phase 23 should
evaluate a reusable Q4_K/Q6_K large-K down-projection shape family. Phase 22
stops here without adding that specialization.

## Baseline and artifacts

| Item | Value |
|---|---|
| Date | 2026-08-17 |
| Starting branch | `perf/vulkan-matmul-architecture-phase21` |
| Starting commit | `69570f5aa37cec9b4670b63c0ad26c5ac67cba53` |
| Phase 21 implementation ancestry | `b1744a38f0394f60e2ca1864c7e39ff02e5c7bbf` |
| Investigation branch | `perf/second-model-bringup-phase22` |
| Production baseline | Phase 20 source; Phase 21 retained no production change |
| GPU | NVIDIA GeForce RTX 3060 12 GiB, Ampere, PCI device `0x2487` |
| Driver / Vulkan API | 595.84 / 1.4.329 |
| Runtime | Vulkan, 32,768-token capacity, F16 KV, greedy, 32 decoded tokens |
| Prompt for performance | chat-templated `" a"` repetitions adjusted to exact token count |

The initial worktree was clean. The dedicated branch was created before the
first experiment. Temporary routing, startup, and first-logit diagnostics were
restored byte-for-byte. Nothing was pushed.

| Artifact | Bytes | Parameters | SHA-256 |
|---|---:|---:|---|
| Model 1 `Ministral-3-3B-Instruct-2512-Q4_K_M.gguf` | 2,147,023,008 | 3,429,006,336 | `9ed150d4367e68df0ac8e1540f6ddc65b42d0ee26378329d1ecbca60f93fc5f8` |
| Model 2 `Ministral-3-8B-Instruct-2512-Q4_K_M.gguf` | 5,198,911,904 | 8,489,553,920 | `33e7a72cf5e6e2cfc2f2847075acc013d68bba023e35310cef86b5cf8fdca761` |

`support/profiling/validate-weights.sh` qualified all six artifacts in the
local catalog. No external checksum substitution or weight conversion was
used.

## Architecture discovery

Both files declare `mistral3` and the same dense causal GQA semantics:
RMSNorm with epsilon `1e-5`, a SiLU-gated MLP, and GPT-2/Tekken tokenization
with BOS 1 and EOS 2. RoPE uses adjacent pairs over 128 dimensions, theta
1,000,000, YaRN factor 16, beta-fast 32, beta-slow 1, log multiplier 1, and an
original context of 16,384. The attention temperature scale is 0.1.

| Property | Model 1 | Model 2 | Classification |
|---|---:|---:|---|
| Layers | 26 | 34 | already generic |
| Hidden | 3,072 | 4,096 | already generic |
| FFN | 9,216 | 14,336 | generic correctness; new fast-path shape |
| Q heads | 32 | 32 | already generic |
| KV heads | 8 | 8 | already generic |
| GQA ratio | 4 | 4 | already generic |
| Head / key / value dimension | 128 | 128 | already generic |
| Q / K / V widths | 4,096 / 1,024 / 1,024 | 4,096 / 1,024 / 1,024 | already generic |
| Context limit | 262,144 | 262,144 | already generic; device capacity is lower |
| Vocabulary | 131,072 | 131,072 | already generic |
| RoPE / norm / activation | same YaRN / RMSNorm / SiLU gate | same | semantically compatible |
| Output head | tied to embedding | dedicated Q6_K | already generic |
| Tensor formats | F32, Q4_K, Q6_K | F32, Q4_K, Q6_K | directly supported |

No unsupported semantic difference was found. Model 2's larger configuration
and dedicated output head were already expressible by the generic metadata and
tensor-role contracts. Its only material mismatch is fast-path qualification,
not model execution.

## Complete tensor inventory and map

The Model 2 GGUF is version 3 with 53 metadata entries, 32-byte alignment, a
data offset of 8,411,552, and 309 tensors. A complete enumeration found zero
unknown names and zero misaligned tensor offsets. The table uses GGUF extents
`[input, output]`; brace patterns cover every numbered tensor, not samples.

| Graph Horizon role | External tensor | Count | Shape | Format | Bytes each / distribution | Support |
|---|---|---:|---|---|---|---|
| Token embedding | `token_embd.weight` | 1 | `[4096, 131072]` | Q4_K | 301,989,888 | direct, canonical |
| Attention norm | `blk.{0..33}.attn_norm.weight` | 34 | `[4096]` | F32 | 16,384 | direct; narrowed to FP16 on GPU |
| Q projection | `blk.{0..33}.attn_q.weight` | 34 | `[4096, 4096]` | Q4_K | 9,437,184 | direct |
| K projection | `blk.{0..33}.attn_k.weight` | 34 | `[4096, 1024]` | Q4_K | 2,359,296 | direct |
| V projection | `blk.{0..33}.attn_v.weight` | 34 | `[4096, 1024]` | 17 Q4_K + 17 Q6_K | 2,359,296 / 3,440,640 | direct |
| O projection | `blk.{0..33}.attn_output.weight` | 34 | `[4096, 4096]` | Q4_K | 9,437,184 | direct |
| MLP norm | `blk.{0..33}.ffn_norm.weight` | 34 | `[4096]` | F32 | 16,384 | direct; narrowed to FP16 on GPU |
| Gate projection | `blk.{0..33}.ffn_gate.weight` | 34 | `[4096, 14336]` | Q4_K | 33,030,144 | direct |
| Up projection | `blk.{0..33}.ffn_up.weight` | 34 | `[4096, 14336]` | Q4_K | 33,030,144 | direct |
| Down projection | `blk.{0..33}.ffn_down.weight` | 34 | `[14336, 4096]` | 17 Q4_K + 17 Q6_K | 33,030,144 / 48,168,960 | direct |
| Final norm | `output_norm.weight` | 1 | `[4096]` | F32 | 16,384 | direct; narrowed to FP16 on GPU |
| Dedicated logits | `output.weight` | 1 | `[4096, 131072]` | Q6_K | 440,401,920 | direct, required |

The Q6_K V/down layers are 0, 1, 2, 3, 6, 9, 12, 15, 18, 21, 24, 27,
29, 30, 31, 32, and 33. All other V/down tensors are Q4_K. Format totals are
69 F32 tensors (282,624 elements), 205 Q4_K tensors (6,882,852,864 elements),
and 35 Q6_K tensors (1,606,418,432 elements).

Unlike Model 1, Model 2 has a required dedicated output tensor. The loader maps
it to the internal logits role without duplicating the embedding. There are no
optional, derived, shared, or architecture-specific tensors beyond Model 1's
supported semantic role set.

## Weight format and loader audit

| On-disk class | Use | Backend disposition |
|---|---|---|
| F32 | 68 block norms + final norm | supported directly; existing GPU loader performs checked FP16 narrowing |
| Q4_K | embedding and most projections | canonical 256-value / 144-byte block support |
| Q6_K | mixed V/down and dedicated logits | canonical 256-value / 210-byte block support, including packed loads |
| F16, Q8/INT8, other | absent | no conversion or fallback needed |

The existing parser validates GGUF version, bounds, dimensions, type-specific
block divisibility, and aligned offsets before exposing mapped bytes. The
semantic tensor contract rejects missing roles, duplicates, unexpected shapes,
unsupported formats, inconsistent head geometry, and metadata arithmetic
overflow. Upload ownership remains guarded until the complete selected set is
resident. This file is little-endian GGUF v3 and requires no byte-order path.

The configuration audit found no generic first-model constants for layer
count, hidden/FFN width, head counts, GQA, vocabulary, context, RoPE dimension,
or role names. The explicit shape lists in Vulkan matmul policy are legitimate
measured specialization predicates. The exact `[3072, 9216]` native-weight
predicate is likewise a deliberate Phase 12 specialization. Both fall back
correctly rather than corrupting a new shape.

| Audited assumption | Finding | Classification |
|---|---|---|
| Layers, hidden, FFN, heads, KV heads, GQA, head dimension | derived from checked metadata and tensor shapes | generic configuration |
| Vocabulary, context, RoPE dimensions/parameters | derived from metadata and cross-validated | generic configuration |
| External tensor names | mapped once into internal semantic roles | legitimate format contract |
| Matmul widths `3072`, `4096`, `9216` | measured policy allow-list with mandatory fallback | legitimate kernel specialization |
| Native `[3072, 9216]` | exact Phase 12 economic/shape predicate | legitimate kernel specialization |
| Q64 head dimension 128, GQA 4, Q64 workload | centralized capability/shape predicates | legitimate kernel specialization |
| Generic backend bug | none found | no change required |

### Generic support changes

None. The smallest viable change is therefore no runtime change. Adding a
model-specific fixture or production branch would increase complexity without
correctness value; the existing environment-gated real-model tests exercise
the local authenticated artifact without committing it or its path.

## Correctness

The deterministic semantic prompt was an empty system message and user text
`Quanto fa 17 × 19?`. It renders as:

```text
[SYSTEM_PROMPT][/SYSTEM_PROMPT][INST]Quanto fa 17 × 19?[/INST]
```

Its token IDs are:

```text
1,17,18,3,6605,4842,3456,1032,1049,1055,10039,1032,1049,1057,1063,4
```

The external oracle was local llama.cpp build 9826 at commit
`9bebfcb4bc8b12a316e96ae03f33671eac1e72fd`. The older support-script pin was
not present locally, so the current same-weight implementation was recorded
explicitly instead of claiming that pin.

| Qualification | Result | Evidence |
|---|---|---|
| CPU load / forward / prefill / decode | PASS | canonical Model 2 weights, F16 KV |
| CPU 16-step oracle | PASS | oracle ID in local top two at every step |
| Portable Vulkan 16-step oracle | PASS | Q64, Matrix2, cooperative matmul, Q4 metadata, and decode GQA forced off |
| Automatic Vulkan 16-step oracle | PASS | production routing, zero host/device crossings |
| CPU vs Vulkan teacher forcing | PASS | identical local top-1 IDs across 128 accumulated steps |
| External greedy generation | PASS | exact first token, first four, and first 32 tokens |
| External 128-step teacher forcing | PASS | oracle token in local top two at all steps; two later top-1 crossings |
| Vulkan 128-token generation | PASS | automatic decode GQA exactly matches forced fallback after a 2,048-token prompt |
| 32-token public generation | PASS | deterministic completion and terminal lifecycle |
| Cancellation / ownership lifecycle | PASS | exactly one terminal event; no leaked request KV |

The external first 32 greedy IDs, also produced by CPU and Vulkan, are:

```text
8396,2615,127607,2007,1032,1049,1055,1617,8379,1032,1049,1057,1617,1574,3417,5317,
1650,1667,2101,84118,4578,1603,33301,4464,2390,6734,1651,1265,1730,2161,16580,1438
```

### First logits and numeric scope

| Runtime | Top 1 | Score | Top 2 | Score |
|---|---:|---:|---:|---:|
| CPU | 8,396 | 16.965942383 | 6,319 | 16.604520798 |
| Vulkan automatic | 8,396 | 16.970687866 | 6,319 | 16.608581543 |

For these two recorded scores, max absolute error is 0.004745483, mean absolute
error 0.004403114, RMSE 0.004416405, and max relative error 0.000279706. The
retained synthetic dense intermediate/logit parity hook reports max and mean
absolute and relative error of exactly zero; the retained YaRN intermediate
test also passes.

The production real-model harness exposes ranked final logits, not raw
per-operation tensors. It therefore cannot honestly supply full-vector
max/mean/RMSE for real Q/K/V, attention, MLP, or layer outputs. Those values are
recorded as **not instrumented**, not zero. A layer-by-layer localization was
not triggered because generic and optimized token/logit gates passed. This is
the boundary requested by “where existing instrumentation permits,” and avoids
adding a permanent diagnostic graph solely for this bring-up.

## RoPE, attention, GQA, and KV audit

Model 2's YaRN parameters, adjacent-pair layout, head dimension, attention
temperature, causal mask, and dense GQA mapping are semantically identical to
Model 1. The exact mapping is `kv_head = q_head / 4`; Q heads 0–3 consume KV
head 0, 4–7 consume KV head 1, and so on through Q 28–31 to KV 7. Existing
tests cover batched RoPE and prefill/decode attention consistency.

For F16 KV, each token requires:

```text
K = 34 layers * 8 heads * 128 values * 2 bytes = 69,632 bytes
V = 69,632 bytes
total = 139,264 bytes/token
```

| Capacity | F16 KV bytes | MiB / GiB |
|---:|---:|---:|
| 128 | 17,825,792 | 17 MiB |
| 512 | 71,303,168 | 68 MiB |
| 2,048 | 285,212,672 | 272 MiB |
| 8,192 | 1,140,850,688 | 1.0625 GiB |
| 16,384 | 2,281,701,376 | 2.125 GiB |
| 28,000 | 3,899,392,000 | 3.631 GiB |
| 32,768 qualification capacity | 4,563,402,752 | 4.250 GiB |
| 262,144 metadata maximum | 36,507,222,016 | 34.000 GiB |

Allocation, reuse, retention, and final release passed at the 32K runtime
capacity. The metadata maximum is semantically supported but cannot fit this
model plus F16 KV on the 12 GiB device; 28K is the maximum practical measured
point.

## Optimization compatibility and reuse score

| Production optimization | Model 1 | Model 2 | Correct? | Beneficial? | Reason |
|---|---|---|---|---|---|
| Generic Vulkan | REUSED | REUSED | yes, forced oracle pass | required baseline | shape-driven graph and canonical Q4_K/Q6_K fallbacks |
| Final-batch-only logits | REUSED | REUSED | yes | yes, avoids non-final vocab projection | dedicated 131,072-vocab Q6_K head satisfies generic contract |
| Request KV retention | REUSED | REUSED | yes through 128-token generation | yes, no KV rebuild | layout derives 34 layers, 8 KV heads, dimension 128 |
| Batched YaRN RoPE | REUSED | REUSED | yes, intermediate test | inherited; not separately re-ablated | identical semantics and 128 rotary dimensions |
| Q4_K Matrix2 | REUSED | REUSED | yes | yes, stack ablation | Q/K/O and Q4 V shapes qualify; MLP 14,336 shapes fall back |
| Q6_K Matrix2 | REUSED | REUSED | yes | yes, included in stack ablation | Q6 V shape qualifies; Q6 down falls back |
| Packed Q6_K loads | REUSED | REUSED | yes | inherited; not separately re-ablated | existing canonical interpretation serves V, down, and logits |
| Decode GQA | REUSED | REUSED | yes, 128-token fallback equality | yes, +1.43% | head dimension 128 and ratio 4 satisfy predicate |
| Gate/up FP16-native | REUSED opt-in | INELIGIBLE | n/a | no; 7.438 GiB hypothetical cost | exact Phase 12 shape is `[3072, 9216]`, not `[4096, 14336]` |
| NVIDIA Q64/KV64 attention | REUSED | REUSED | yes, oracle pass | yes, -42.74% 8K TTFT | Ampere Matrix2, head dimension 128, GQA 4, F16 KV, full Q64 batches |

The reuse score is **8/8 applicable production mechanisms**. Native gate/up is
excluded from the denominator because its measured shape predicate is false.
Matrix2 reuse is operation-granular: qualifying projections use it, while MLP
shapes retain their correct fallback.

Canonical gate and up tensors total 2,246,049,792 bytes. An exact FP16 native
copy would require 7,985,954,816 bytes (7.438 GiB), before canonical weights or
KV. It cannot be economic on this device, so startup increase, request
break-even, and native benchmark are **not applicable**. The gate prevents both
preprocessing and persistent allocation.

## Effective route stack

`M` below is the prompt-row count. Matrix2 owns 64 rows/workgroup and reuses a
decoded weight tile across those rows; legacy cooperative Q4 owns 32 rows and
shares its decoded B tile across them. Every selected route has the named
canonical fallback.

| Operation | Logical `M x N x K` / format | Selected production path | Why | Fallback |
|---|---|---|---|---|
| Embedding | lookup, width 4096 / Q4_K | canonical lookup | generic role and width | same generic path |
| Q | `M x 4096 x 4096` / Q4_K | Matrix2 Q4, 64 rows/WG | measured K/N and Ampere Matrix2 | batched Q4 |
| K | `M x 1024 x 4096` / Q4_K | Matrix2 Q4, 64 rows/WG | measured K/N | batched Q4 |
| V | `M x 1024 x 4096` / Q4_K or Q6_K | Matrix2 Q4/Q6, 64 rows/WG | both formats and shape qualify | batched Q4/Q6 |
| RoPE | `M x 32/8 x 128` | batched YaRN | semantics match | portable batched kernel |
| Attention prefill | 32 Q, 8 KV, dim 128 | NVIDIA Matrix2 Q64 | centralized capability/shape/workload predicate | portable dense attention; tails fall back |
| O | `M x 4096 x 4096` / Q4_K | Matrix2 Q4, 64 rows/WG | measured K/N | batched Q4 |
| Gate / up | `M x 14336 x 4096` / Q4_K | legacy cooperative Q4, 32 rows/WG | K 4096 qualifies; N 14,336 is outside Matrix2 list | batched Q4 |
| Activation | `M x 14336` / FP16 | batched SiLU multiply | generic width | same generic kernel |
| Down | `M x 4096 x 14336` / Q4_K or Q6_K | generic batched Q4/Q6 | K 14,336 is outside cooperative/Matrix2 predicates | selected path is fallback |
| Decode projections | one row / Q4_K or Q6_K | tiled quant kernels | decode workload | scalar-compatible quant kernel |
| Decode attention | dim 128, GQA 4 | split/reduce GQA | exact capability and layout predicate | generic decode attention |
| Logits | one final row, 131,072 / Q6_K | dedicated Q6_K logits | explicit output role, final-batch only | generic Q6_K logits |
| Weights | 5.190 GB tensor data | canonical | bring-up invariant | no native representation |

## Generic versus specialized qualification

All ablations use the same Model 2 weights, F16 KV, and deterministic request.
They isolate one route family where possible; profiler and public wall clocks
remain distinct.

| Operation / workload | Generic or control | Optimized | Delta |
|---|---:|---:|---:|
| Attention, 128 public TTFT | portable: 1,423.57 ms | Q64: 1,402.62 ms | -20.95 ms, -1.47% |
| Attention, 8K GPU kernel | portable: 66,998.834 ms | Q64: 1,830.845 ms | -65,167.989 ms, -97.27% |
| Attention, 8K public TTFT | portable: 155,574.80 ms | Q64: 89,085.79 ms | -66,489.01 ms, -42.74% |
| Matmul stack, 128 public TTFT | fully generic: 2,836.02 ms | production: 1,402.62 ms | -1,433.40 ms, -50.54% |
| Matrix2 increment, 128 TTFT | legacy cooperative: 1,495.40 ms | production: 1,402.62 ms | -92.78 ms, -6.20% |
| Decode GQA, 128 | fallback: 21.74 tok/s | GQA: 22.05 tok/s | +1.43% |

The forced-generic run also passed all 16 oracle steps, proving that model
support does not depend on RTX 3060 specialization. Automatic routing then
passed the same gate. No predicate was weakened.

## Official Model 2 performance baseline

Public values are uninstrumented runs after one warm-up. Repetitions are five
through 512, three through 8K, and two at 16K/28K. CV is sample SD divided by
the mean. GPU values come from separate one-sample timestamp-profile builds;
they are not substituted for public TTFT.

| Context | TTFT mean ms | Median ms | SD ms | CV | GPU prefill ms | Prefill tok/s |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 1,402.62 | 1,400.51 | 7.51 | 0.54% | 1,418.530 | 91.26 |
| 512 | 5,447.89 | 5,447.66 | 9.02 | 0.17% | 5,447.231 | 93.98 |
| 2,048 | 21,892.36 | 21,888.02 | 9.79 | 0.04% | 21,936.490 | 93.55 |
| 8,192 | 88,864.57 | 88,828.44 | 96.53 | 0.11% | 88,961.193 | 92.19 |
| 16,384 | 181,271.05 | 181,271.05 | 1.41 | 0.00% | 181,986.709 | 90.38 |
| 28,000 | 318,631.16 | 318,631.16 | 41.45 | 0.01% | 319,814.568 | 87.88 |

| Context | Decode tok/s | ms/token | Decode CV |
|---:|---:|---:|---:|
| 128 | 22.05 | 45.35 | 0.21% |
| 512 | 21.75 | 45.98 | 0.26% |
| 2,048 | 21.23 | 47.10 | 0.21% |
| 8,192 | 19.67 | 50.84 | 0.29% |
| 16,384 | 17.91 | 55.83 | 0.08% |
| 28,000 | 15.85 | 63.09 | 0.09% |

The curve is smooth: prefill declines from 94 to 88 token/s as context grows,
while decode declines from 22.05 to 15.85 token/s with the retained KV. No
allocation threshold, failed growth, or instability appears before 28K.

## Memory and startup

Model 2 tensor payload is 5,190,500,352 bytes. Narrowing the 69 F32 norm tensors
from 1,130,496 disk bytes to 565,248 GPU bytes yields 5,189,935,104 canonical
GPU bytes (4.833 GiB).

| Account | Bytes | GiB | Notes |
|---|---:|---:|---|
| Canonical weights | 5,189,935,104 | 4.833 | persistent |
| 32K F16 KV | 4,563,402,752 | 4.250 | request-owned and retained through generation |
| Main scratch | 147,456 | 0.00014 | shape-derived |
| Device logits | 524,288 | 0.00049 | 131,072 FP32 values |
| Reduce scratch | 131,072 | 0.00012 | max of top-k and GQA partials |
| MMVQ fixed scratch | 40,960 | 0.00004 | allocated although opt-in route remains disabled |
| Optional native | 0 | 0 | gate rejects Model 2 |
| Largest upload staging | 440,401,920 | 0.410 | dedicated Q6_K output head |
| Accounted initialization peak | 5,631,180,800 | 5.244 | weights, fixed buffers, scratch, and largest staging; no request KV yet |
| Active 32K peak / steady state | 9,754,181,632 | 9.084 | staging is already released; excludes budget reserve and host mirror |
| Non-coincident arithmetic bound | 10,194,583,552 | 9.494 | steady plus staging; not presented as observed residency |

The standalone GPU budget also reserves the greater of 256 MiB or 5% of its
heap; that is admission headroom, not an allocated resident buffer. The FP32
host logits mirror is 524,288 bytes. Observed process maximum RSS during the
startup probe plus a short request was 5,347,848 KiB.

| Startup component | Time ms |
|---|---:|
| GGUF parse | 29.327 |
| Contract, tokenizer, and tensor validation | 138.833 |
| Vulkan device | 187.422 |
| Pipeline creation | 56.318 |
| Buffer creation and canonical upload | 2,858.367 |
| Placement plan / finish | 0.348 |
| Vulkan backend total | 3,102.505 |
| Parse + contract + backend | **3,270.665** |
| Native preprocessing | **0** |

This accounting is a temporary timestamp probe removed after collection.
Startup is kept separate from public request TTFT.

## Competitive snapshot

llama.cpp used the same GGUF, Vulkan device, all GPU layers, flash attention,
F16 KV, and three repetitions. Its benchmark boundary is not byte-identical to
Graph Horizon's public chat TTFT, which also includes tokenization and first
sampling; the ratios are large enough that this small boundary difference does
not change classification.

| Workload | Graph Horizon | llama.cpp | Ratio | Maturity target | Status |
|---|---:|---:|---:|---:|---|
| Prefill 128 | 1,402.62 ms | 98.358 ms | 14.26x | <=1.5x | RED |
| Prefill 8K | 88,864.57 ms | 5,260.781 ms | 16.89x | <=1.7x | RED |
| Decode after 128 | 22.05 tok/s | 50.17 tok/s | 2.28x slower | <=1.4x | RED |
| Decode after 8K | 19.67 tok/s | 40.69 tok/s | 2.07x slower | <=1.4x | RED |

The llama.cpp standard deviations were approximately 0.48% at 128 and 0.25%
at 8K prefill. This is a lightweight maturity snapshot, not a new Phase 18
gap analysis.

## 8K bottleneck attribution

The production profile accounts for 99.99% of its 88,961.193 ms GPU prefill.

| Model 2 category | GPU ms | GPU prefill |
|---|---:|---:|
| Attention Q64 | 1,830.845 | 2.06% |
| Q | 1,326.762 | 1.49% |
| K | 345.865 | 0.39% |
| V | 349.409 | 0.39% |
| O | 1,278.533 | 1.44% |
| Gate | 17,263.368 | 19.40% |
| Up | 17,285.552 | 19.43% |
| Down | **48,899.591** | **54.96%** |
| Activation | 122.571 | 0.14% |
| Norm, RoPE, KV, residual, embedding, logits | 258.695 | 0.29% |
| **All recurrent matmul** | **86,749.080** | **97.51%** |
| **Other excluding attention and matmul** | **381.268** | **0.43%** |

Attention is large only on the forced portable route: 66,998.834 ms, or 43.10%
of that 155,452.177 ms GPU run. Q64 removes 97.27% of attention time, leaving
matmul—not attention—as the largest remaining and removable class. Within it,
down projection is the first target because gate/up native expansion is both
shape-ineligible and memory-prohibitive.

Decode GPU time at 8K is 1,644.351 ms for the profiled 32-token request.
Attention is 161.630 ms (9.81%), down 370.985 ms (22.56%), gate/up 611.356 ms
(37.18%), and logits 175.165 ms (10.65%). The same MLP family therefore matters
to both prefill and decode, though Phase 23 should keep the workload predicates
separate.

## Model 1 regression guard

Production source is identical to Phase 20/21, but Model 1 was still rerun with
its retained Phase 12 opt-in and automatic Q64/Matrix2/GQA routes.

| Model 1 guard | Phase 21 | Phase 22 fresh | Delta |
|---|---:|---:|---:|
| 128 TTFT | 96.88 ms | 96.59 ms | -0.30% |
| 128 prefill | 1,321.25 tok/s | 1,325.26 tok/s | +0.30% |
| 128 decode | 46.80 tok/s | 46.92 tok/s | +0.26% |
| 28K TTFT | 35,381.53 ms | 35,305.85 ms | -0.21% |
| 28K prefill | 791.37 tok/s | 793.07 tok/s | +0.21% |
| 28K decode | 29.39 tok/s | 29.46 tok/s | +0.24% |

A separate 2,048-prompt, 32-token automatic GQA run exactly matched its forced
fallback sequence. All performance deltas are within 1%; the Phase 19 Q64 path
remains selected and qualified.

## Experiment registry

| ID | Task / subsystem | Generic path | Fast path | Correctness | Performance / VRAM | Classification | Decision |
|---|---|---|---|---|---|---|---|
| P22-01 | Full GGUF inventory | canonical parser | n/a | 309/309 mapped | 5.190 GB tensor payload | REUSED DIRECTLY | keep evidence |
| P22-02 | CPU bring-up | CPU canonical | n/a | 128-step pass | reference only | REUSED DIRECTLY | keep |
| P22-03 | Vulkan fallback | portable attention/matmul/decode | forced off | 16-step pass | 2,836.02 ms at 128 fully generic | REUSED DIRECTLY | keep qualification |
| P22-04 | Production routing | canonical fallbacks | automatic stack | 128-step CPU/Vulkan ID equality | 1,402.62 ms at 128 | REUSED DIRECTLY | keep |
| P22-05 | Prefill attention | portable dense | NVIDIA Q64 | pass | -65.168 s GPU at 8K | REUSED DIRECTLY | keep |
| P22-06 | Prefill matmul | batched Q4/Q6 | eligible Matrix2/coop | pass | -50.54% TTFT vs fully generic at 128 | REUSED DIRECTLY | keep |
| P22-07 | Decode attention | generic | GQA split/reduce | 128-token free-running equality | +1.43% at 128; no VRAM delta | REUSED DIRECTLY | keep |
| P22-08 | Gate/up native | canonical Q4 | exact FP16 native | predicate false | would add 7.438 GiB | SHAPE INCOMPATIBLE | reject for Model 2 |
| P22-09 | Context scaling | production fallback stack | eligible paths | all requests pass | 128 through 28K baseline | REUSED DIRECTLY | keep evidence |
| P22-10 | Competitive snapshot | same weights/precision | llama.cpp Vulkan | deterministic | maturity ratios RED | n/a | classify only |
| P22-11 | Model 1 regression | forced fallback reference | automatic routes | exact 32-token equality | all guards within 0.3% | REUSED DIRECTLY | pass |

## Phase 23 recommendation

```text
Target:
    Q4_K/Q6_K large-M down projection with K=14,336 and N=4,096

Current fraction:
    54.96% of Model 2 GPU prefill at 8K

Current time:
    48,899.591 ms of 88,961.193 ms GPU prefill

Architecture/shape difference:
    Model 1 down is K=9,216 -> N=3,072; Model 2 is K=14,336 -> N=4,096

Existing fast path reusable?
    No. Canonical Q4_K/Q6_K semantics and fallback are reusable, but the
    measured cooperative/Matrix2 K predicate excludes 14,336.

Predicted removable:
    A conservative 10% down reduction would remove about 4.890 s.

Predicted whole-request:
    About 5.49% of profiled 8K GPU prefill. The candidate must demonstrate at
    least a 9.2% down reduction to clear the 5% whole-request keep gate.

Generalization potential:
    A quant-format- and shape-family predicate for large-K, N=4,096/5,120 down
    projections can serve other larger dense Ministral shapes; it must not be
    keyed to the 8B model name.
```

This is a recommendation for a separate measured phase, not evidence that a
specific shader architecture will win. Phase 21's negative results still rule
out assuming that direct compressed Matrix2 feed or a local tile sweep is
automatically beneficial.

## Verification and final decision

The full CPU workspace suite reported 165 passes and the unchanged installer
fixture/environment failure. The engine CPU library reported 159 passes, zero
failures, and two ignored tests; its integration suite retained only the known
root `VALIDATION.md` docs-contract absence. The Vulkan library reported 158
passes, zero failures, and five ignored tests, including live-device dense and
RoPE parity. Real CPU, forced-generic Vulkan, automatic Vulkan, 32-token, and
128-token Model 2 gates passed as described above.

No new dependency, public API, persistent representation, shader, loader rule,
or model-specific test path was introduced. Conceptual complexity is
unchanged. All required stop conditions are satisfied:

1. weights and metadata load correctly;
2. CPU/reference correctness passes;
3. generic and optimized Vulkan pass;
4. free-running Vulkan generation passes through 128 tokens, and CPU/Vulkan
   teacher-forced parity passes through 128 steps;
5. every production optimization is classified;
6. every eligible specialization is qualified;
7. prefill/decode baseline covers 128 through practical 28K;
8. Model 1 correctness and performance remain within 1%;
9. the Model 2 down-projection bottleneck and Phase 23 gate are explicit.

Final decision: **MODEL SUPPORTED; Phase 22 complete; no production change.**
