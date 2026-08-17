<!--
This checkpoint owns the current cross-phase optimization state: qualified
baseline, competitive gaps, practical floors, global ranking, active candidate,
reproduction commands, and resume point. Phase reports retain detailed evidence.
-->

# Current Optimization State

## Status

**CHECKPOINT — MISSION INCOMPLETE.** Phase 29 closed the exact/same-numeric
Vulkan search, but the performance contract remains unmet and the broader
representation, selective-numerical, model-adaptation, and specialization
spaces are not globally exhausted.

| Item | Current value |
|---|---|
| Date | 2026-08-17 |
| Branch / HEAD | `perf/systematic-llamacpp-gap` / `eb05b06` |
| Production baseline | `e21fc12` plus Phase 29 source restoration |
| GPU / driver | RTX 3060 12 GiB / 595.84 |
| llama.cpp | `9bebfcb4bc8b12a316e96ae03f33671eac1e72fd`, build 9826 |
| Runtime | pure Vulkan, full offload, F16 KV, greedy |
| Supported artifacts | Ministral 3B, 8B, and capacity-guarded 14B instruct/reasoning Q4_K_M |
| Retained latest changes | packed Q4 loads; vectorized GQA decode; FP16 M128 Q4 Matrix2 |

The authenticated 3B artifact remains 2,147,023,008 bytes with SHA-256
`9ed150d4367e68df0ac8e1540f6ddc65b42d0ee26378329d1ecbca60f93fc5f8`.

## Competitive matrix

The first row is the fresh same-session revalidation. Remaining rows are the
latest qualified Phase 28/29 matrix and must be bookended again after a retained
architectural change.

| Model / workload | Graph Horizon | llama.cpp | Latency ratio | Contract | Status |
|---|---:|---:|---:|---:|---|
| 3B prefill / 128, fresh | 90.78 ms, CV 0.15% | 41.97 ms, CV 0.64% | 2.16x | 1.5x | miss |
| 3B decode / 128, fresh | 56.55 tok/s, CV 0.48% | 106.25 tok/s, CV 1.02% | 1.88x | 1.3–1.4x | miss |
| 3B prefill / 28K | 35.15 s | about 14.06 s | 2.50x | 1.5–1.7x | miss |
| 3B decode / 28K | 36.52 tok/s | 50.02 tok/s | 1.37x | 1.3–1.4x | broad-band pass |
| 8B prefill / 128 | 228.67 ms | 109.44 ms | 2.09x | 1.5x | miss |
| 8B prefill / 2K | 3.585 s | 1.19 s | 3.03x | 1.5–1.7x | miss |
| 8B prefill / 28K | 69.145 s | 24.84 s | 2.78x | 1.5–1.7x | miss |
| 8B decode / 128 | 26.64 tok/s | 50.74 tok/s | 1.90x | 1.3–1.4x | miss |
| 8B decode / 28K | 19.52 tok/s | 27.69 tok/s | 1.42x | 1.3–1.4x | near miss |
| 14B decode / capacity guard | 17.23 tok/s | 36.25 tok/s | 2.10x | 1.3–1.4x | miss |

Fresh 3B/128 telemetry began idle at 210/405 MHz, 19.34 W, 54 C and ended at
1,995/7,501 MHz, 156.83 W, 61 C. Both tools used the same artifact, Vulkan
device, full offload, F16 K/V, three measured repetitions, and warm-up. Graph
Horizon includes public first-token work; llama.cpp pure `pp` does not, an
unavoidable boundary difference already bounded at roughly 11 ms for 128.

## Current attribution and floors

The retained Cycle 31b 8B/28K profile attributes 99.97% of 69.31 GPU seconds:

| Domain | Current | Share |
|---|---:|---:|
| Attention | 21.150 s | 30.51% |
| Gate + up | 23.090 s | 33.32% |
| Down | 14.283 s | 20.61% |
| Q/K/V/O | 9.477 s | 13.67% |
| Other | 1.292 s | 1.86% |
| **Total** | **69.292 s** | **99.97% accounted** |

The 1.5x target permits 37.26 s, leaving 32.03 s to remove. The old 65.30 s
same-numeric floor is retired rather than silently reused: Cycle 31 crossed its
numeric boundary and removed another 5.60 profiled seconds. At 8B/128 decode,
the target requires 39.03 tok/s; the measured eight-lane direct Q4 DP4A route
reaches 36.76 tok/s but changes broad-policy greedy sequences.

## Global Amdahl ranking

| Rank | Workload / component | Current | Practical candidate floor | Removable | Predicted whole gain | Level | Portability | Quality risk | Complexity |
|---:|---|---:|---:|---:|---:|---:|---|---|---|
| 1 | 8B/128 decode, Q4 projections/MLP | 26.25 tok/s control | 36.76 tok/s all-Q4 measured | 10.9 ms/token | 40.0% throughput; 28.6% latency | 6 | capability + quant family | high | moderate |
| 2 | 8B/28K prefill, execution representation | 52.31 s Matrix2 | below same-numeric floor only with compact/lower precision | at least 28.04 s needed beyond exact floor | target-sized only if ownership and numeric floor both change | 5–6 | shape + quant family | high | high |
| 3 | 3B/28K prefill, INT8 MLP | 45.92 s historical control | 40.88 s measured | 5.05 s | 10.99% | 5–6 | capability + tensor role | high | high |
| 4 | 3B/28K sparse attention with model adaptation | 45.96 s dense | about 40.79 s W2K model | 5.17 s | 11.23% | 7 | architecture family | high | very high |
| 5 | remaining exact runtime/kernel candidates | workload dependent | acquired Phase 29 bounds | below 5% individually | below economic gate | 1–4 | portable/capability | low | varies |

For rank 1, `T = 38.095 ms/token`, the all-Q4 affected fraction is effectively
the measured route delta, and `S_total = 36.76 / 26.25 = 1.400`. The resulting
latency is 27.204 ms/token, removing 10.891 ms/token. A layer-window candidate
covering one half to three quarters of Q4 projection work predicts roughly
17–30% throughput improvement before interaction costs; this clears the 10%
lossy-candidate gate and requires measurement.

## Selected candidate: layer-window Q4 DP4A decode

Restore the measured eight-lane packed DP4A Q4 decode kernel and add a strictly
diagnostic layer-window eligibility bit at weight upload. Sweep contiguous
windows rather than repeating Phase 29's already rejected expanding,
contracting, or balanced shape-family policies.

Invariant: canonical Q4_K bytes, Q6 paths, KV, attention, logits, sampling,
prefill, and the exact generic Q4 fallback remain unchanged. The experiment may
change only Q4 decode activation quantization and accumulation for explicitly
selected layer weights. Main risk: small per-layer errors compose into ranking
or sequence degradation even when the local matmul oracle passes.

Quality screening requires finite outputs, the unchanged MMVQ CPU oracle,
multiple layer windows, exact-ID agreement measurement over 128 greedy tokens,
teacher-forced reference presence in the existing top-two gate, and multiple
prompts before any retention decision. Exact ID equality is evidence, not the
sole possible lossy-quality contract; no weaker contract will be invented just
to retain a failing candidate.

### Production experiment structure

No new directory is warranted. Productive line estimates exclude tests:

```text
crates/graph_horizon_engine/src/backend/vulkan/
├── mem/buffers.rs (~175 productive lines): carry one diagnostic eligibility bit
├── mem/weights/
│   ├── mod.rs (~175): upload and assemble validated weight buffers
│   └── mmvq.rs (~65): parse diagnostic role/layer eligibility
├── kernels/mmvq.rs (~130): include eligibility in the complete route predicate
├── kernels/matmul/decode.rs (~45): match the eight-lane dispatch geometry
└── shaders/matmul/matmul_q4_k_mmvq_f16out.comp
    (category K; ~85 lines): eight-lane packed Q4 DP4A kernel
```

The weights domain split became necessary after strict selector validation and
tests pushed the former single file above 200 productive lines. The shader
remains one dense numeric kernel and carries its existing category-K
architectural marker. All orchestration files remain below 200 productive
lines. Rejected experiment code will be removed; retained routing must not
depend on a model name.

## Reproduction

```sh
GRAPH_HORIZON_PREFILL_PREDECODE_MLP=1 target/phase29-final-bench \
  /home/emanuele/Documenti/models/Ministral-3-3B-Instruct-2512-Q4_K_M.gguf \
  --context 32768 --kv f16 --prompt '<" a" repeated 125 times>' \
  --max-tokens 32 --warmup 1 --reps 3

/home/emanuele/Documenti/llama.cpp/build/bin/llama-bench \
  -m /home/emanuele/Documenti/models/Ministral-3-3B-Instruct-2512-Q4_K_M.gguf \
  -dev Vulkan0 -ngl 99 -sm none -mg 0 -ctk f16 -ctv f16 -fa on \
  -b 2048 -ub 512 -t 6 -p 128 -n 32 -r 3 -o json
```

## Resume point

### Cycle 30 experiment register

The Q4 DP4A layer/role sensitivity experiment is **REJECTED**. All measurements
use 8B/128, F16 KV, canonical Q4 weights, three repetitions, and the same
diagnostics-free candidate binary. The surrounding control is 26.69–27.05
tok/s with CV at most 0.27% for the directly compared rows.

| ID | Selection | Decode | Change | Quality | Decision |
|---|---|---:|---:|---|---|
| C30-1 | layers 0–16, all Q4 roles | 27.45 tok/s | invalid, CV 22.81%, only 20 deltas | sequence collapses at token 2 | reject |
| C30-2 | layers 17–33, all Q4 roles | 30.95 tok/s | +15.96% | sequence collapses at token 2 | reject |
| C30-3 | layers 8–25, all Q4 roles | 31.67 tok/s | +18.66% | sequence collapses at token 1 | reject |
| C30-4 | layers 0–25, all Q4 roles | 34.27 tok/s | +28.40% | sequence diverges at token 1 | reject |
| C30-5 | gate + up, all layers | 33.15 tok/s | +22.64% | teacher-forced top-two fails step 8 | reject |
| C30-6 | gate only, all layers | 29.50 tok/s | +9.14% | top-two passes 16; free run diverges token 5 | reject below lossy economic gate |
| C30-7 | up only, all layers | 29.33 tok/s | +8.51% | top-two passes 16; free run diverges token 39 | reject below lossy economic gate |
| C30-8 | up plus exact-screen layers 15/24/28/29 | 30.23 tok/s | +11.76% | teacher-forced top-two fails step 3 | reject |

An individual eight-token exact screen passed only layers 15, 24, 28, and 29
of 34. Their proportional all-role ceiling is about 4.7% throughput, below the
10% numerical-policy gate before interaction costs. This closes layer windows,
gate/up separation, and exact-safe layer selection for the current row-scaled
Q8_1 activation/Q4 DP4A representation. The production experiment and its
temporary `mem/weights` split are removed after this record.

Next, re-rank a more accurate large-M numerical representation. The leading
unresolved candidate is FP16 Matrix2 accumulation with a full-model
teacher-forced/logit quality screen rather than the exact local-oracle-only
rejection used previously. If it fails the established lossy quality gate,
evaluate block-scaled partial accumulation before escalating to model
adaptation.

### Cycle 31 candidate: FP16 Q4 Matrix2 accumulation

The recovered Phase 28 prerequisite changed only the cooperative accumulator
and its shared output staging from FP32 to FP16. It was rejected after four of
five focused Q4 CPU-oracle shapes exceeded unchanged exact-path tolerances, but
was never benchmarked or evaluated against the repository's lossy full-model
quality contract. The observed failures were finite and bounded: representative
sample errors were about 0.4--0.8% at large output magnitudes.

Current Q4 Matrix2 work is about 47.8 s of the 75.1 s 8B/28K request. A
conservative 1.25x local speedup removes about 9.6 s, or 12.8% request latency;
FP16 accumulation is also the prerequisite for the competitive reference's
larger `M256xN128` ownership. This clears the numerical-candidate experiment
gate, although the minimal accumulator change alone cannot close the full gap.

No new file or directory is warranted:

```text
crates/graph_horizon_engine/src/backend/vulkan/shaders/matmul/
└── matmul_q4_k_matrix2_f16out.comp
    (category K; ~155 productive lines): Q4_K Matrix2 matmul with FP16 output
```

Invariant: canonical Q4_K decoding, tile ownership, dispatch, FP16 output,
fallback routing, KV/attention, and sampling remain unchanged. The experiment
changes only K-loop accumulation and output-staging precision. Main risk:
cumulative MLP/projection error changes ranked logits or generated sequences.
Screen in this order: finite focused outputs, smallest-model teacher-forced
top-two quality, diagnostics-free 3B/128 performance, then 8B and longer
contexts only if both gates pass.

Result: **REJECTED as a standalone candidate.** The focused result reproduced
the prior finite failures (one of five strict Q4 shapes passed). The authenticated
3B 16-step teacher-forced test nevertheless passed with every oracle token in
the local top two and no crossing. Diagnostics-free 3B/128 timing was 91.62 ms
candidate versus 92.47 ms same-session control, only 0.92% lower latency;
decode was neutral at 56.68 versus 56.53 tok/s. This is far below the 10%
numerical-policy gate. FP16 accumulation remains eligible only as part of a
larger-ownership experiment whose measured gain pays for its precision risk.

### Cycle 31b candidate: combined FP16 M128 Q4 Matrix2

This combination is not covered by the prior experiments. The staged M128 Q4
prototype used FP32 accumulation, was modeled at about 166 registers/thread and
one resident workgroup, and regressed its Q4 component 4.43% at 28K. The FP16
experiment above retained M64 ownership. Combining FP16 accumulation with M128
halves compressed Q4 tile traversals while reducing accumulator and output-stage
footprint; the hypothesis is that the precision change restores enough
residency to make the larger ownership useful.

At a 63.6% Q4 Matrix2 request share, a conservative 1.2x local speedup predicts
10.6% whole-request latency reduction. Structure and productive estimates:

```text
crates/graph_horizon_engine/src/backend/vulkan/
├── kernels/coopmat.rs (~100 lines): dispatch Q4 Matrix2 on its declared M tile
├── pipeline/kernel.rs (~155 lines): expose the selected M128 kernel name
├── exec/profile/summary.rs (~180 lines): classify the M128 profile label
└── shaders/matmul/matmul_q4_k_matrix2_f16out.comp
    (category K; ~155 lines): combined M128 ownership and FP16 accumulation
```

Q6 and direct-FP16 Matrix2 remain M64. Canonical Q4 decoding, N32/K128,
bindings, dispatch topology class, output type, fallbacks, and model routing are
unchanged. The main risks are the same accumulated rounding already screened
above and an M128 register/shared-resource cliff. Focused outputs, the 3B
teacher-forced gate, and 3B/128 timing decide whether it advances to 8B.

Interim result: the combined tile passes the 16-step authenticated 8B oracle
exactly and keeps every retained teacher-forced token in the local top two for
128 steps on 3B/8B Instruct and Reasoning. Free-running 8B Instruct is exact
through token 41 before a top-1 crossing; under teacher forcing only two of 128
local top-1 choices cross. The other three model rows retain their full local
top-1 reference sequence. Synthetic compressed-Matrix2 errors are finite and
bounded at 3.42% maximum / 1.18% mean relative error in the worst large-K
shape; fallback and predecoded FP32 routes retain the prior 0.5% per-element
contract.

Diagnostics-free TTFT is 90.78 versus 92.54 ms on 3B/128 (-1.90%), where
native MLP weights limit the Q4 share; 8B improves 254.76 -> 228.67 ms at 128
(-10.24%) and 4,009.34 -> 3,584.59 ms at 2K (-10.59%). Candidate CV is at
most 0.40% and decode is neutral. The candidate advances to 8B/28K and causal
profiling before retention.

The 8B/28K same-session result is 69.145 s candidate versus 74.244 s control,
both at 0.05% CV: 5.099 s / 6.87% latency removed and 7.38% throughput gained.
Decode is unchanged at 19.70--19.71 tok/s. At 2K the paired profiler attributes
the change to Q4 Matrix2: 3,150.1 -> 2,905.0 ms (-7.78%, 245.1 ms), while its
prompt-grid X dimension halves from four to two. Dispatch/barrier counts,
allocations, Q6, attention, and decode topology are unchanged. The profiled
whole prefill falls 4,011.3 -> 3,820.7 ms despite run-to-run movement in
untouched categories.

Decision: **KEEP.** The retained path clears the numerical economic gate at
8B/128 and 2K, removes 5.099 s at 28K, passes the explicit four-model top-two
quality contract, and adds no allocation, dependency, public API, or
model-name policy. Exact free-running identity is intentionally not claimed.
The full validation gate must pass before the production checkpoint commit.
