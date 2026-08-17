<!--
This checkpoint owns the current cross-phase optimization state: qualified
baseline, competitive gaps, practical floors, global ranking, active candidate,
reproduction commands, and resume point. Phase reports retain detailed evidence.
-->

# Current Optimization State

## Status

**GLOBAL STOP — CONTRACT PARTIALLY MET; QUANTITATIVE ECONOMIC EXHAUSTION.**
Cycle 50 retains direct canonical-Q4_K Matrix2 dataflow, removing 25--36%
across the 8B prefill matrix and bringing 14B/128 inside the strict contract.
Cycles 51--54 close the newly exposed Q6 representation and direct-Q4 device
geometry. No technically available candidate remains above its economic gate;
the only target-sized combined opportunity now requires externally unavailable
model adaptation resources.

| Item | Current value |
|---|---|
| Date | 2026-08-18 |
| Branch / retained production HEAD | `perf/systematic-llamacpp-gap` / `dff4bd8` |
| Production baseline | `e21fc12` plus Phase 29 source restoration |
| GPU / driver | RTX 3060 12 GiB / 595.84 |
| llama.cpp | `9bebfcb4bc8b12a316e96ae03f33671eac1e72fd`, build 9826 |
| Runtime | pure Vulkan, full offload, F16 KV, greedy |
| Supported artifacts | Ministral 3B, 8B, and capacity-guarded 14B instruct/reasoning Q4_K_M |
| Retained latest changes | direct Q4 Matrix2 prefill; per-8 Q8/DP4A Q4 decode; 14B Matrix2 eligibility |

The authenticated 3B artifact remains 2,147,023,008 bytes with SHA-256
`9ed150d4367e68df0ac8e1540f6ddc65b42d0ee26378329d1ecbca60f93fc5f8`.

## Competitive matrix (superseded pre-Cycle-49 snapshot)

This snapshot preserves the resume-time evidence. The fresh post-Cycle-49
matrix near the end of this document supersedes it.

| Model / workload | Graph Horizon | llama.cpp | Latency ratio | Contract | Status |
|---|---:|---:|---:|---:|---|
| 3B prefill / 128, fresh | 90.78 ms, CV 0.15% | 41.97 ms, CV 0.64% | 2.16x | 1.5x | miss |
| 3B decode / 128 | 70.65 tok/s, CV 0.21% | 106.25 tok/s, CV 1.02% | 1.50x | 1.3–1.4x | near miss |
| 3B prefill / 28K | 35.15 s | about 14.06 s | 2.50x | 1.5–1.7x | miss |
| 3B decode / 28K | 36.52 tok/s | 50.02 tok/s | 1.37x | 1.3–1.4x | broad-band pass |
| 8B prefill / 128 | 228.67 ms | 109.44 ms | 2.09x | 1.5x | miss |
| 8B prefill / 2K | 3.585 s | 1.19 s | 3.03x | 1.5–1.7x | miss |
| 8B prefill / 28K | 69.145 s | 24.84 s | 2.78x | 1.5–1.7x | miss |
| 8B decode / 128 | 35.07 tok/s, CV 0.14% | 50.74 tok/s | 1.45x | 1.3–1.4x | near miss |
| 8B decode / 28K | 23.89 tok/s | 27.69 tok/s | 1.16x | 1.3–1.4x | pass |
| 14B decode / capacity guard | 23.55 tok/s, CV 0.15% | 36.25 tok/s | 1.54x | 1.3–1.4x | miss |

Fresh 3B/128 telemetry began idle at 210/405 MHz, 19.34 W, 54 C and ended at
1,995/7,501 MHz, 156.83 W, 61 C. Both tools used the same artifact, Vulkan
device, full offload, F16 K/V, three measured repetitions, and warm-up. Graph
Horizon includes public first-token work; llama.cpp pure `pp` does not, an
unavoidable boundary difference already bounded at roughly 11 ms for 128.

## Current attribution and floors

The retained Cycle 50 8B/28K profile attributes 99.96% of 53.96 GPU seconds:

| Domain | Current | Share |
|---|---:|---:|
| Attention | 21.272 s | 39.42% |
| Direct Q4 Matrix2 | 22.690 s | 42.05% |
| Staged Q6 Matrix2 | 8.702 s | 16.13% |
| Other | 1.297 s | 2.40% |
| **Total** | **53.961 s** | **99.96% accounted** |

The fresh llama.cpp endpoint makes the broad 1.7x target 41.689 s and the strict
1.5x target 36.785 s, leaving 10.248/15.152 s to remove from the 51.937 s public
time. The old 65.30 s
same-numeric floor is retired rather than silently reused: Cycles 31 and 50
crossed its numeric/dataflow boundaries. At 8B/128 decode, the broad target now
requires 35.60 tok/s; retained per-8 Q8/DP4A reaches 34.89 tok/s, leaving a
2.0% throughput gap to that band and a larger gap to the strict endpoint.

## Historical pre-Cycle-43 Amdahl ranking

| Rank | Workload / component | Current | Practical candidate floor | Removable | Predicted whole gain | Level | Portability | Quality risk | Complexity |
|---:|---|---:|---:|---:|---:|---:|---|---|---|
| 1 | 8B/28K prefill | 69.145 s | 37.26 s target | 31.885 s | 46.1% latency required | 5–7 | shape/family | high | high |
| 2 | 8B/128 decode | 35.07 tok/s | 39.03 tok/s threshold | 2.89 ms/token | 11.3% throughput | 4–6 | capability + quant family | medium | moderate |
| 3 | 14B decode / capacity guard | 23.55 tok/s | 25.89 tok/s threshold | 3.84 ms/token | 9.9% throughput | 4–6 | capability + quant family | medium | moderate |
| 4 | 3B/128 decode | 70.65 tok/s | 75.89 tok/s threshold | 0.98 ms/token | 7.4% throughput | 4–6 | capability + quant family | medium | moderate |
| 5 | model-adapted sparse attention | dense retained | about 11% historical 3B gain | workload dependent | target-sized only in combination | 7 | architecture family | high | very high |

Cycle 36 removes 8.21 ms/token from the 8B/128 float-GEMV control but still
misses the threshold by 2.89 ms/token. Decode is no longer the largest absolute
gap: the retained 8B/28K prefill remains 31.885 s above the 1.5x target and must
drive the next candidate ranking.

## Historical selected candidate: layer-window Q4 DP4A decode

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
target/phase36-retained-bench \
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
gate/up separation, and exact-safe layer selection for the current per-32-block
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

### Cycle 32 candidate: FP16 M256/N32 Q4 Matrix2

The retained M128 candidate leaves Q4 Matrix2 at 38.126 s, 55.0% of the 28K
profile. Doubling only prompt-row ownership halves Q4 compressed-tile
traversals again. At a conservative 1.1x Q4-local gain, Amdahl predicts 5.3%
whole-request speedup. This clears the exact-geometry gate because accumulator
precision, K-loop order, per-output arithmetic, N32/K128, and routing remain
identical to Cycle 31b.

No structural change is required:

```text
crates/graph_horizon_engine/src/backend/vulkan/
├── kernels/coopmat.rs (~100 lines): dispatch the declared Q4 prompt tile
├── pipeline/kernel.rs (~155 lines): expose the M256 profile name
├── exec/profile/summary.rs (~180 lines): classify the M256 profile label
└── shaders/matmul/matmul_q4_k_matrix2_f16out.comp
    (category K; ~155 lines): M256/N32/K128 Q4 Matrix2 operation
```

Main risk is the same accumulator/register-residency cliff that made historical
FP32 M128 slower. The smallest 8B/128 paired timing is decisive; a regression
or sub-5% result rejects the candidate before long-context work.

Result: **REJECTED.** Focused error metrics are identical to M128, but 8B/128
TTFT is 566.20 ms versus 228.14 ms for the retained M128 control (+148.2%).
Decode is neutral. M256 therefore crosses the accumulator/resource cliff; no
long-context or repeated quality work is justified. Production returns exactly
to checkpoint `45319e9`.

### Cycle 33 candidate: fused direct Q4 callback with FP16 M128

The earlier direct-callback experiments used FP32 accumulation at M64/M128.
Their short Q4 screen improved locally, but the long FP32 path crossed a
register/resource cliff and regressed. Cycle 31 changes that prerequisite: the
qualified FP16/M128 accumulator has the same per-thread accumulator footprint
as the old FP32/M64 baseline. The new combination lets Matrix2 decode canonical
Q4 bytes during its tensor load, removing explicit FP16 weight staging while
retaining the current accumulator and ownership.

Q4 Matrix2 is 38.126 s / 55.0% of the post-Cycle-31 28K profile. A conservative
1.1x local gain predicts 5.3% whole-request speedup; the old short direct result
shows a much larger optimistic ceiling. The experiment clears the exact-
representation gate because callback and staged paths reconstruct the same
FP16 weights and use the same FP16 K-loop accumulation.

```text
crates/graph_horizon_engine/src/backend/vulkan/
├── shaders/matmul/matmul_q4_k_matrix2_direct_f16out.comp
│   (category K; ~145 lines): direct canonical-Q4 Matrix2 callback
├── kernels/matmul/prefill/{mod.rs,policy.rs} (~180/~55): opt-in route
├── kernels/coopmat.rs (~105): dispatch the M128 direct variant
├── pipeline/{mod.rs,kernel.rs} (~175/~160): capability-gated pipeline
└── exec/profile/{category.rs,summary.rs} (~180/~165): causal attribution
```

The shader directory already owns multiple matmul variants, so no new folder is
warranted. Main risk is callback-generated register/address pressure. The
existing focused Q4 oracle must stay within the retained FP16 bounds; 8B/128
paired timing rejects a regression before any long-context run.

Result: **REJECTED.** All five focused error distributions are byte-for-byte
identical to staged M128, but 8B/128 TTFT is 277.10 ms versus 229.29 ms control
(+20.85%); decode is neutral. Callback address/decode pressure still outweighs
shared FP16 staging even after the FP16 accumulator change. The new shader,
pipeline, route, flag, and profile wiring are removed completely.

### Cycle 34 candidate: FP16 M128/N16 Q4 Matrix2

This exact-geometry candidate probes the opposite side of the M256 resource
cliff. A 128-thread workgroup retains M128 prompt reuse but owns 16 output rows.
Compared with M128/N32 it halves per-workgroup weight/output shared state and
doubles N workgroups, leaving total dispatched threads per output width
unchanged. Better residency can win; repeated activation tile loads can lose.

The candidate changes the existing category-K shader plus the fixed Q4 N tile
in `kernels/coopmat.rs` and two profile labels; all orchestration files remain
below 200 productive lines. FP16 accumulation, K128, canonical decoding,
routing, and outputs are unchanged. A 1.1x Q4-local gain predicts 5.3% whole
speedup. The 8B/128 pair rejects any regression or sub-5% result.

Result: **REJECTED.** Focused errors are identical to M128/N32, but 8B/128
TTFT is 304.32 ms versus 227.18 ms control (+33.96%). Decode is neutral.
Duplicated activation loads and smaller Matrix2 work dominate any residency
benefit. Production returns exactly to checkpoint `45319e9`.

### Cycle 35 candidate: per-16 Q8_1 activation blocks with fast DP4A

Cycle 30's Q8_1 representation was initially described as row-scaled; source
inspection corrects that description. The quantizer already writes one `(d, s)`
pair per 32 activations, matching each Q4_K scale/min group. The unresolved
quality hypothesis is therefore finer activation scaling, not introducing block
scaling for the first time.

This experiment halves the Q8_1 activation block to 16 values. The packed int8
payload size, canonical Q4_K bytes, eight DP4A calls per 32 weights, FP32 output
accumulation, and eight-lane row ownership remain unchanged. Each Q4_K group
instead reads two `(d, s)` pairs and applies its scale/min correction to each
16-value half. Metadata scratch grows from 8 to 16 KiB at the maximum supported
input width; it remains a fixed backend allocation and adds no per-token
allocation.

No new file or directory is warranted. Productive estimates exclude tests:

```text
crates/graph_horizon_engine/src/backend/vulkan/
├── mod.rs (~155 lines): size the existing fixed MMVQ metadata scratch
├── init/setup.rs (~145): allocate the declared fixed scratch layout
├── kernels/mmvq.rs (~125): dispatch per-16 quantization and eight-lane GEMV
├── kernels/matmul/decode.rs (test-only edit): match eight-lane geometry in the oracle
├── shaders/quant/quant_a_q8_f16.comp
│   (category K; ~35 lines): quantize one 16-value activation block
└── shaders/matmul/matmul_q4_k_mmvq_f16out.comp
    (category K; ~75 lines): consume two Q8_1 halves per Q4_K group
```

Invariant: default routing remains off; FP16 fallback, retained prompt kernels,
canonical weight storage, output shape/type, sampling, KV state, and attention
are unchanged. Smallest viable change is the two shader index contracts, fixed
metadata sizing, and matching dispatch geometry. Main risks are insufficient
quality recovery, doubled metadata pressure erasing the fast kernel's gain, and
an indexing mismatch between the quantizer and GEMV. The focused CPU oracle
checks the contract before any model test. If it passes, 8B/128 performance and
the authenticated 16-step teacher-forced top-two gate decide whether longer
quality screening is justified.

Result: **REJECTED.** The focused Vulkan/CPU oracle passes and 8B/128 decode
improves 26.65 -> 36.82 tok/s (+38.2%) with neutral TTFT (230.35 -> 228.75 ms).
The authenticated 8B 16-step oracle is exact, and the 8B Instruct 128-step
teacher-forced top-two gate passes. The same 128-step gate fails on 3B Instruct
at step 17, so per-16 scaling does not satisfy the retained cross-model quality
contract. Longer performance work is not justified.

### Cycle 36 candidate: per-8 Q8 activation blocks with fast DP4A

The per-16 result shows that scale resolution, rather than the eight-lane
reduction itself, materially restores quality: the former per-32 path failed the
8B sequence almost immediately, while per-16 passes all 128 8B steps. Cycle 36
halves the activation block once more to eight values and screens the specific
3B step-17 failure before repeating broad tests.

The structure and invariants are identical to Cycle 35. The packed int8 payload,
canonical Q4_K weights, eight DP4A calls per 32 weights, and FP32 row reduction
remain fixed. Each Q4_K group now reads four `(d, s)` pairs and applies four
corrections. Fixed metadata scratch becomes 32 KiB at the maximum input width,
still with no per-token allocation. Only the existing two category-K shaders,
the two scratch-size expressions, host block count, and focused test geometry
change; the productive estimates above remain below their declared limits.

Main risk is that metadata/correction overhead consumes the decode gain before
quality fully recovers. The failing 3B Instruct 128-step gate is first. If it
passes, 8B/128 must retain at least a 10% decode improvement before the other
model rows are run.

Interim result: **KEEP candidate.** The focused oracle passes. All six 3B, 8B,
and 14B Instruct/Reasoning artifacts retain every saved reference token in the
local top two for 128 teacher-forced steps; the 14B rows use their established
512-token capacity geometry. The authenticated 8B 16-step oracle is exact.

Diagnostics-free paired decode improves 56.65 -> 70.65 tok/s on 3B (+24.7%),
26.89 -> 35.07 on 8B (+30.4%), and 17.53 -> 23.55 on 14B (+34.3%). TTFT is
neutral in all three comparisons. This clears the numerical economic gate at
every supported size. The DP4A/Q4_K/shape capability gate becomes the default;
the strict diagnostic values `0`, `false`, or `no` restore the float GEMV, and
invalid values also fail closed to fallback. Long-context performance and the
full validation suite remain before the checkpoint decision.

The retained default also improves 8B decode at longer prompts: 26.02 -> 33.86
tok/s at 2K (+30.1%, three repetitions, candidate CV 0.47%) and 19.80 -> 23.89
tok/s at 28K (+20.7%, one measured bookend). The 28K TTFT values are 69.339 s
fallback and 69.122 s candidate, a 0.3% difference consistent with an unchanged
prefill path. Attention therefore dilutes but does not erase the decode gain.

The 8B/128 query profiler attributes 306.70 ms of total decode-kernel reduction
(1,123.65 -> 816.95 ms over 31 decode steps). The five affected Q4 projection
categories fall from 629.39 to 366.49 ms; 6,324 quantizer invocations cost only
13.71 ms, leaving a 249.18 ms net Q4+quant reduction. Candidate dispatch and
barrier counts rise by exactly those 6,324 quantization commands; allocation
count remains zero. Untouched Q6 timing also moved between these one-shot
profiled runs, so the Q4+quant subtotal—not the whole one-shot delta—is the
causal attribution.

Decision: **KEEP.** Cycle 36 clears the numerical economic gate on every model
size, passes the focused kernel oracle and six-model 128-step top-two contract,
and retains a 20.7% decode gain at 28K. The only added persistent storage is
24 KiB of activation metadata at the maximum supported input width (32 KiB
instead of 8 KiB); there is no per-token allocation, dependency, public API, or
model-name policy. The eight-lane reduction is more involved than the rejected
serial kernel, but its ownership and scratch contracts remain local to the two
category-K shaders and one dispatch module. Full workspace tests, formatting,
diff hygiene, documentation contracts, and warning-free clippy pass before the
checkpoint commit. Exact free-running identity is not claimed; the explicit
teacher-forced top-two quality contract is the retained numerical boundary.

### Cycle 37 candidate: tiled per-8 Q8/Q4 DP4A prefill

Cycle 36 reopens one numerical premise from Phase 14: per-8 activation scales
qualify where whole-row scales did not. The obvious integer Matrix2 extension
is nevertheless closed by hardware granularity. This device exposes signed
INT8 Matrix2 only at K32; preserving four independent per-8 scales would require
four masked K32 MMAs and four weight traversals. The measured direct-INT8 local
ratio was 0.729 of FP16, so multiplying its dominant work by four cannot beat
the retained compressed Matrix2 path.

The remaining untested representation is a scalar-integer tiled kernel. One
256-thread workgroup owns 16 prompt rows by 32 output rows. It quantizes a
16x256 activation tile to shared per-8 Q8 once, assigns eight lanes to each
canonical Q4_K output row, reuses each lane's packed Q4 words across all 16
prompt rows, accumulates in FP32, and reduces the eight lanes with subgroup
shuffle. It uses no persistent scratch or new weight representation.

The experimental route is opt-in and isolated in the existing prefill domain.
Productive estimates exclude tests:

```text
crates/graph_horizon_engine/src/backend/vulkan/
├── shaders/matmul/matmul_q4_k_dp4a_batch_f16out.comp
│   (category K; ~125 lines): tiled per-8 Q8/Q4 DP4A batched matmul
├── kernels/matmul/prefill/
│   ├── mod.rs (~190): capability-gated route before retained Matrix2
│   └── dp4a.rs (~60): strict experiment flag and 2D dispatch
├── pipeline/kernel.rs (~165): declare the shader ABI
└── pipeline/mod.rs (~198): build it only on DP4A devices
```

Invariant: canonical weights, retained default Matrix2, Q6, output layout/type,
attention, KV, sampling, and allocation policy remain unchanged. The smallest
viable experiment adds one category-K shader and narrow capability/dispatch
wiring; rejection removes all of it. Main risks are scalar integer throughput,
the 8x smaller prompt ownership increasing workgroup count, register pressure
from 16 FP32 accumulators per lane, and composed activation quantization error.
The focused Q4 oracle runs first. A regression or less than 10% 8B/128 TTFT
gain rejects the initial M16 geometry before longer contexts or quality work.

M16 result: **REJECTED.** All five focused shapes pass with worst relative
error below 0.087%, but 8B/128 TTFT is 738.14 ms versus 228.05 ms control
(+223.7%, candidate CV 0.10%). Decode is unchanged. M16 reloads canonical
weights eight times per retained M128 ownership, so one upper-bound geometry is
still warranted: M64 with subgroup reduction and two weight traversals per
retained tile. M64 uses 32 KiB shared activation state and 64 FP32 accumulators
per lane. If it does not beat control, M32 is dominated by twice the weight
traffic and the scalar DP4A prefill family is closed.

M64 result: **REJECTED; family closed.** Focused numerical errors are unchanged,
but 8B/128 TTFT rises to 917.31 ms versus 229.46 ms control (+299.8%). The
larger tile is slower than M16 because 64 FP32 accumulators per lane and 32 KiB
of shared activation state create a resource cliff. M32 is dominated: it keeps
the same scalar DP4A count while traversing weights twice as often as M64. The
shader, route, flag, pipeline, trace, and profile hooks were removed; production
code is restored exactly to checkpoint `905dc70`.

### Cycle 38 candidate: canonical Q4_K with K32 INT8 Matrix2 partials

The rejected Phase 14 route quantized each complete output row of weights and
therefore changed the model. This candidate keeps every canonical Q4_K nibble,
block scale, and block minimum. For each K32 block it quantizes only the FP16
activation row, multiplies signed Q8 activations by the original unsigned
0--15 Q4 codes with native SINT8 Matrix2, then converts the I32 partial to FP32
with the exact Q4 scale/minimum correction before accumulation. The identity is
`d_a * (d_w * dot(q_a,q_w) - m_w * sum(q_a))`; only activation quantization is
approximate. Unlike per-8 scales, one per-K32 scale matches the advertised INT8
Matrix2 granularity without masked MMAs.

The clean production baseline is checkpoint `905dc70`; the immediate 8B/128
controls surrounding Cycle 37 are 228.05 and 229.46 ms TTFT. The experiment is
strictly opt-in through `GRAPH_HORIZON_PREFILL_Q4_INT8_PARTIAL=1`. Productive
estimates exclude tests:

```text
crates/graph_horizon_engine/src/backend/vulkan/
├── shaders/matmul/matmul_q4_k_i8_matrix2_f16out.comp
│   (category K; ~155 lines): K32 activation quantization, canonical Q4 decode,
│   INT8 Matrix2, scale/min correction, and FP32 accumulation
├── coopmat2/
│   ├── abi.rs (~190): exact SINT8 WG256 capability predicate
│   └── mod.rs (~150): expose the detected SINT8 contract
├── kernels/matmul/prefill/
│   ├── mod.rs (~160 productive lines before tests): isolated candidate route
│   └── policy.rs (~55): strict experiment flag
├── kernels/coopmat.rs (~120): admit the candidate's existing M128/three-buffer ABI
├── pipeline/kernel.rs (~165): candidate shader ABI
├── pipeline/caps.rs (~180 productive lines before tests): shared-memory gate
└── pipeline/mod.rs (~200): capability-gated pipeline construction
```

Invariant: model bytes, Q4_K semantics, FP32 accumulation, output layout/type,
decode, Q6, attention, KV, sampling, persistent memory, and default routing do
not change. The smallest viable change is one category-K shader plus narrow
capability, pipeline, dispatch, trace, and profile wiring. Main risks are four
K32 Matrix2 operations and partial epilogues per retained K128 step, activation
quantization error, and compiler support for cross-type per-element conversion.
The focused five-shape Q4 oracle runs first, followed by 8B/128 economics. A
compile/pipeline failure, numerical failure, or less than 10% short-prefill gain
rejects the candidate before long-context qualification.

Result: **REJECTED; representation closed.** The compiler and driver accept and
select the exact SINT8 pipeline. All five focused Q4 tests pass without a
tolerance change; the three Matrix2 shapes have worst relative error below
0.0790%, confirming the canonical scale/minimum identity. Performance is
terminal: 8B/128 TTFT is 766.02 ms versus 226.56 ms control (+238.1%), with
0.18--0.19% CV, while decode remains neutral (34.76 versus 34.90 tok/s). Four
K32 MMAs, activation quantization, I32-to-FP32 conversion, and four per-block
epilogues cost far more than the saved payload/dequant work. Larger K cannot
represent independent canonical Q4 scales; smaller M/N increases workgroup and
weight traffic. The capability, shader, flag, pipeline, dispatch, trace,
profile, and tests are removed; production returns exactly to `905dc70`.

### Cycle 39 candidate: canonical Q6_K per-8 Q8 DP4A decode

After Cycle 36, Q6_K V/down remain the largest decode-only format family. The
Phase 26 8B/128 profile attributed about 377 ms across 31 decode steps to these
two roles, or roughly 12.2 ms/token; the nearest target needs 2.89 ms/token.
The existing per-8 Q8 activation representation is already qualified across
all six models for Q4. Q6_K has no minimum term: each canonical signed 6-bit
code is multiplied by its original per-16 scale, so one eight-lane output row
can consume four per-8 activation blocks per lane with DP4A and FP32
accumulation. No weight conversion or new scratch is required.

The production baseline is clean checkpoint `905dc70`; the fresh 8B/128
control is 34.90 tok/s. Q4 MMVQ stays at its retained default, while Q6 is
strictly opt-in through `GRAPH_HORIZON_DECODE_MMVQ_Q6=1`. Productive estimates
exclude tests:

```text
crates/graph_horizon_engine/src/backend/vulkan/
├── shaders/matmul/matmul_q6_k_mmvq_f16out.comp
│   (category K; ~85 lines): canonical Q6 unpack, per-8 DP4A, FP32 reduction
├── kernels/mmvq.rs (~145): format-specific eligibility and kernel selection
├── pipeline/kernel.rs (~165): Q6 MMVQ shader ABI
└── pipeline/mod.rs (~200): build the Q6 kernel on DP4A devices
```

Invariant: Q6 bytes/scales, per-8 activation quantization, output type/layout,
FP32 accumulation, Q4 routing, prefill, attention, KV, sampling, scratch size,
and default Q6 behavior remain unchanged. The smallest viable experiment is
one category-K shader plus the existing MMVQ route/pipeline/profile wiring.
Main risks are Q6 bit-plane indexing, fourfold lower row ownership than the
float Q6 kernel, and compounded Q6/activation error. A focused CPU oracle gates
8B/128 performance; less than 5% whole-decode gain rejects before model quality.

M8 result: the CPU oracle passes and 8B/128 improves from 35.02 to 36.33 tok/s
(+3.74%, candidate CV 0.15%), below the 5% gate. The kernel's 32 rows/WG are
half the float Q6 dispatch ownership. One bounded M4 endpoint keeps 64 rows/WG
and assigns two adjacent 32-value spans to each lane; it changes only constants,
indexing, and dispatch geometry in the same recorded files. If M4 also misses
5%, M16 is dominated by twice M8's workgroups and the Q6 DP4A family closes.

M4 result: **REJECTED; family closed.** The same focused Q6 oracle passes, but
8B/128 falls to 34.22 tok/s versus the 35.02 control and M8's 36.33. Preserving
64 rows/WG makes each lane's serial decode too long; M8 reduces that work but
still misses the gate, while M16 is dominated by twice M8's workgroups. The Q6
shader, opt-in flag, route changes, pipeline, trace, and profile hooks are
removed; production returns exactly to `905dc70`.

### Cycle 40 candidate: 16-lane per-8 Q8/Q4 DP4A decode

The retained Q4 MMVQ assigns one lane to each 32-value canonical scale group;
each lane serially consumes four per-8 activation blocks. A 16-lane variant
assigns two lanes to each scale group and two activation blocks per lane. It
doubles workgroups from 32 to 16 output rows each, but halves per-lane decode
and dot-product depth while keeping weight traffic, arithmetic, activation
representation, FP32 accumulation, and scratch unchanged. This geometry was
not tested by the older per-32 activation experiments.

The clean baseline is evidence commit `c8335a2` with retained production at
`905dc70`; fresh 8B/128 control is 35.02 tok/s and the nearest contract requires
39.03 tok/s. The candidate is strict opt-in through
`GRAPH_HORIZON_DECODE_MMVQ_L16=1`. Productive estimates exclude tests:

```text
crates/graph_horizon_engine/
├── build.rs (~95 Vulkan productive lines): compile one Q4 MMVQ macro variant
└── src/backend/vulkan/
    ├── shaders/matmul/matmul_q4_k_mmvq_f16out.comp
    │   (category K; ~90 lines): cohesive L8/L16 variants of Q4 MMVQ
    ├── kernels/mmvq.rs (~135): strict variant flag and dispatch geometry
    ├── pipeline/kernel.rs (~165): variant ABI
    └── pipeline/mod.rs (~200): DP4A-gated variant pipeline
```

Invariant: canonical Q4 bytes/scales/minimums, per-8 activation quantization,
FP32 accumulation, output layout/type, Q6, prefill, attention, KV, sampling,
scratch, and default L8 routing remain unchanged. Main risk is that doubled
workgroup traffic outweighs shorter lane dependency chains. The existing MMVQ
CPU oracle runs through the selected variant first; failure to improve 8B/128
by at least 5% rejects L16, and L32 is then dominated by another doubling of
workgroups.

Result: **REJECTED; geometry closed.** The selected L16 variant passes the
existing Q4 CPU oracle, but 8B/128 is 34.72 tok/s versus 34.84 control (CV
0.32% versus 0.24%). Doubled workgroup traffic offsets the shorter per-lane dot
chain; L32 is dominated by another doubling. The build variant, shader branch,
flag, dispatch, pipeline, trace, and profile hooks are removed. The retained
eight-lane kernel remains the measured optimum and production returns exactly
to `905dc70`.

### Cycle 41 candidate: Q6 DP4A including FP32 logits

Cycle 39 measured Q6 MMVQ only through the generic FP16 matmul interface, so it
affected V/down but not the separate Q6 logits route. The Phase 26 8B profile
attributes about 2.7 ms/token to logits, nearly the complete 2.89 ms/token gap,
and the M8 V/down candidate already supplied +3.74% whole decode. A combined
candidate reuses the identical canonical-Q6/per-8-Q8 DP4A arithmetic with an
FP32 output variant for logits. This is a new affected domain, not another Q6
lane sweep.

The clean production state is `905dc70` plus evidence commit `c8335a2`; fresh
8B/128 control is 34.84 tok/s. Both Q6 routes are strict opt-in through
`GRAPH_HORIZON_DECODE_MMVQ_Q6=1`. Productive estimates exclude tests:

```text
crates/graph_horizon_engine/
├── build.rs (~95 Vulkan productive lines): compile the FP32-output shader variant
└── src/backend/vulkan/
    ├── shaders/matmul/matmul_q6_k_mmvq_f16out.comp
    │   (category K; ~90 lines): cohesive FP16/FP32 Q6 MMVQ variants
    ├── kernels/mmvq.rs (~190): Q6 matmul and logits eligibility/dispatch
    ├── kernels/matmul/mod.rs (~75): try Q6 logits MMVQ before exact fallback
    ├── backend.rs (category I): pass existing fixed scratch to the delegator
    ├── pipeline/kernel.rs (~170): two Q6 MMVQ ABIs
    └── pipeline/mod.rs (~200): DP4A-gated pipelines
```

Invariant: canonical Q6 bytes/scales, per-8 activation quantization, FP32
accumulation, exact FP32 logits storage, Q4, prefill, attention, KV, sampling,
scratch size, and default Q6 routes remain unchanged. Main risks are vocabulary-
wide workgroup amplification and the same composed activation error now touching
token ranking directly. The existing Q6 matmul+logits CPU oracle gates 8B/128;
the candidate must improve decode by at least 5% before full-model quality.

M8 combined result: the matmul+FP32-logits oracle passes and 8B/128 improves
35.02 -> 36.74 tok/s (+4.91%, candidate CV 0.41%), just below the gate and far
below the 39.03 tok/s contract. One logits-specific M4 endpoint is warranted:
vocabulary-wide logits launch 4,096 M8 workgroups, so retaining 64 rows/WG can
matter more there than on V/down. The next binary keeps M8 for V/down and adds
an opt-in M4 FP32 logits variant in the same shader/pipeline files. If it does
not materially exceed M8, the combined Q6 DP4A family closes.

M4-logits result: **REJECTED; combined family closed.** The focused FP16/FP32
oracle passes, but M8 V/down plus M4 logits reaches 36.36 tok/s, below the M8
combined candidate's 36.74. M8 combined improves 35.02 -> 36.74 (+4.91%) but
misses both the 5% experiment gate and the 39.03 tok/s contract. Finer logits
row ownership is dominated; coarser ownership already regressed. All Q6 MMVQ
shaders, flags, scratch delegation, dispatch, pipeline, trace, and profile hooks
are removed; production returns exactly to `905dc70`.

### Cycle 42 candidate: four-lane per-8 Q8/Q4 DP4A decode

Cycle 40 closes the finer-than-L8 side but not the coarser endpoint. L4 assigns
two canonical 32-value Q4 groups to each lane and restores 64 output rows/WG.
It halves workgroups and the reduction width while doubling each lane's serial
DP4A/scale/min work. Q6's analogous endpoint regressed, but Q4 has different
payload unpack and minimum-correction costs, so inference alone is insufficient.

This uses the same temporary build/shader/host/pipeline structure and line
estimates recorded for Cycle 40, with strict opt-in
`GRAPH_HORIZON_DECODE_MMVQ_L4=1`. The clean retained baseline is `905dc70` and
fresh 8B/128 control is 35.02 tok/s. Canonical weights, per-8 activation Q8,
scratch, accumulation, outputs, all non-Q4 paths, and default L8 remain fixed.
The existing CPU oracle gates one 8B/128 comparison. A regression or less than
5% gain closes the L4/L8/L16 geometry bracket.

Result: **REJECTED; full Q4 row geometry bracket closed.** The L4 oracle passes,
but 8B/128 collapses to 26.30 tok/s versus the 35.02 control (candidate CV
0.15%). L4 is limited by doubled serial unpack/dot/correction depth; L16 is
limited by doubled workgroups; retained L8 is the measured optimum between
them. Every temporary variant and host hook is removed, restoring production
exactly to `905dc70`.

## Historical Cycle-42 design-space stop (superseded)

At that checkpoint no post-`905dc70` production candidate was retained. Cycles 37--42 close the
remaining integer-prefill and decode-format questions that were still open at
the Cycle 36 checkpoint:

| Frontier | Best measured candidate | Required result | Terminal evidence |
|---|---:|---:|---|
| 8B/28K prefill | retained 69.145 s | <=37.26 s | deleting all 21.150 s attention still leaves 47.995 s; the missing >=10.735 s must come from matmul |
| scalar DP4A prefill | 738.14 ms at M16 | <204 ms at 8B/128 screening gate | M16 +223.7%, M64 +299.8%; M32 dominated |
| canonical block-scaled INT8 Matrix2 | 766.02 ms | <204 ms | +238.1%; K32 scale granularity is irreducible |
| Q4 Matrix2 geometry/representation | retained FP16 M128/N32 | target-sized local speedup | M256, N16, direct callbacks, transient expansion, and integer partials all regress |
| sparse/model-semantic attention | historical <7.82% physical ceiling | 46.1% whole-request reduction | even zero-cost attention cannot meet prefill target; adapted weights would violate same-artifact comparison |
| 8B/128 decode | 36.74 tok/s rejected Q6-combined | >=39.03 tok/s | candidate gains only 4.91%; retained result remains 35.07 tok/s |
| Q4 per-8 DP4A ownership | retained L8 | >=5% incremental | L4 26.30, L8 35.02 control, L16 34.72 tok/s |
| Q6 per-8 DP4A ownership/output | M8 combined 36.74 tok/s | >=39.03 tok/s | M4 V/down 34.22; M4-logits 36.36; M8 below contract |

The prefill impossibility statement is an Amdahl bound, not a claim that
attention has zero optimization headroom. It proves that attention alone cannot
qualify. The required companion matmul breakthrough has been tested across
exact compressed Matrix2, FP16 execution views, direct canonical callbacks,
transient dequantization, scalar DP4A, row-scaled persistent INT8, and canonical
block-scaled INT8 partials. Their retained/rejected evidence is in this file and
the phase reports; no unmeasured combination can add already-overlapping gains.

For decode, the best new representation remains below the nearest threshold
before its required six-model teacher-forced quality suite. Retaining it would
increase complexity and numerical surface without satisfying the task. Q4 and
Q6 lane counts are bracketed on both sides of their optima, quantization costs
only 0.44 ms/token in the retained profile, and CPU/dispatch overhead is below
the missing budget. Further micro-sweeps are not evidence-driven.

The largest remaining bottleneck is still 8B/28K prefill: 69.145 s versus the
37.26 s contract ceiling, with 37.373 s in gate/up/down, 9.477 s in Q/K/V/O,
21.150 s attention, and 1.292 s other. The final source state contains only the
retained FP16-M128 prefill and per-8-Q8 Q4 decode changes; all Cycle 37--42
production experiments were removed exactly.

### Cycle 43 candidate: vector-decoded direct Q4 Matrix2

The pinned llama.cpp commit contains a materially newer direct-load mechanism
than Cycles 21 and 33 tested: `GL_NV_cooperative_matrix_decode_vector`, added to
its Vulkan matmul in May 2026, reconstructs four adjacent Q4 values per tensor-
load callback. The rejected Graph Horizon callbacks reconstructed one value per
callback. This distinction can reduce callback invocations, canonical payload
loads, and address calculations fourfold while also removing the explicit FP16
weight tile and its synchronization. It therefore reopens the direct-callback
frontier without repeating the measured scalar design.

Retained Q4 Matrix2 consumes 38.126 s of the 69.145 s 8B/28K request (55.14%).
A conservative 1.15x local speedup predicts
`31.019 + 38.126 / 1.15 = 64.172 s`, removing 4.973 s or 7.19% end to end. At
1.30x it predicts 60.347 s and 12.72%. The candidate clears the 5% moderate-
redesign gate and directly tests the structural facility used by the reference.

No new directory is warranted. Productive estimates exclude tests:

```text
crates/graph_horizon_engine/src/backend/vulkan/
├── shaders/matmul/matmul_q4_k_matrix2_vector_f16out.comp
│   (category K; ~150 lines): one direct vector-decoded Q4 Matrix2 operation
├── kernels/coopmat.rs (~115): dispatch the M128/N32 vector variant
├── kernels/matmul/prefill/mod.rs (~195): select the opt-in candidate first
├── kernels/matmul/prefill/policy.rs (~65): strict experiment switch
├── pipeline/kernel.rs (~165): variant ABI and embedded SPIR-V
└── pipeline/mod.rs (~195): build the optional Matrix2 pipeline
```

Invariant: canonical Q4_K bytes, FP16 activation/output, retained FP16
accumulation, M128/N32 ownership, all Q6/decode/attention/KV/sampling paths, and
the default staged fallback remain unchanged. Eligibility is capability,
format, and shape based; the experiment is additionally strict opt-in through
`GRAPH_HORIZON_PREFILL_Q4_VECTOR=1`. Main risks are compiler-generated register
pressure and unsupported decode-vector SPIR-V on the installed driver. The five
focused Q4 oracles gate a paired 8B/128 screen. Regression or less than 5%
whole-request improvement rejects the candidate before a long-context run.

Result: **UNAVAILABLE; hardware/compiler path closed.** Both the repository's
bundled shaderc and the installed shaderc/glslang 2026.1 reject
`GL_NV_cooperative_matrix_decode_vector` as unsupported. More decisively,
`vulkaninfo` lists `VK_NV_cooperative_matrix2` but not
`VK_NV_cooperative_matrix_decode_vector` for the RTX 3060/595.84 device. The
pinned llama.cpp CMake cache likewise enables Matrix2 but contains no decode-
vector support flag, so its measured reference did not use this source-level
facility. The candidate cannot create a valid pipeline on the benchmark device;
all temporary shader and host wiring is removed, restoring production exactly
to `905dc70`. Reopening requires a compiler and driver/device advertising the
decode-vector extension, which is a hard external hardware/compiler change.

### Cycle 44 candidate: direct global Matrix2 output store

Architectural comparison exposes an untested dataflow difference independent
of decode callbacks. The retained Q4 kernel cooperatively stores its M128xN32
accumulator to an 8 KiB shared array, executes a workgroup barrier, then has each
thread copy 16 FP16 values to global output. The pinned llama.cpp Matrix2 path
uses `coopMatStoreTensorNV` to write the bounded global tensor directly. Doing
the same removes the full shared-memory round trip, one barrier, scalar index
arithmetic, and 4,096 scalar epilogue iterations per workgroup without changing
the accumulator, weight reconstruction, tile, or output representation.

The affected Q4 component is 38.126 s / 55.14% of 8B/28K. A 1.10x local gain
predicts `31.019 + 38.126 / 1.10 = 65.679 s`, removing 3.466 s or 5.01% end to
end. This is the minimum performance gate; the physical ceiling is higher
because the change also halves shader-declared shared memory from 16 to 8 KiB
for this variant and may improve residency. Complexity and numerical risk are
low, so the candidate outranks model adaptation.

No structural split is warranted:

```text
crates/graph_horizon_engine/src/backend/vulkan/shaders/matmul/
└── matmul_q4_k_matrix2_f16out.comp
    (category K; ~140 lines): store the one Q4 Matrix2 result directly to output
```

Invariant: canonical Q4_K decoding, FP16 accumulation/output, M128/N32/K128,
routing, dispatch, all other formats and operations, and pipeline-construction
fallback remain unchanged. Main risk is a driver-specific direct-store codegen
regression. The five focused Q4 oracles gate a paired 8B/128 benchmark; less
than 5% whole-request improvement rejects it before long-context profiling.

Result: **REJECTED.** All nine focused Q4 tests pass with the retained error
distributions, but diagnostics-free 8B/128 TTFT is 845.04 ms versus a 228.54 ms
bookend control (+269.8%; CV 0.46% versus 0.22%). Decode is unchanged at
34.65 versus 34.64 tok/s. The direct tensor store is pathological at the
retained M128/N32 ownership, so the shared output epilogue is restored exactly.

### Cycle 45 candidate: reference-sized M256/N128/K64 Q4 Matrix2

Cycle 44's isolated direct store does not test the architecture that makes the
reference's store economical. llama.cpp owns M256xN128 output values per
workgroup and uses K64, versus retained M128xN32/K128. Combining direct global
output with that geometry reduces long-M workgroups eightfold. Each workgroup
decodes four times as many canonical values, so total Q4 decode traversal halves;
activation tensor loads fall to one quarter, while total output traffic is
unchanged. Unlike rejected M256/N32, no MxN shared output array remains, and the
K64xN128 decoded-weight tile stays exactly 16 KiB.

The reference reports 247 registers and 63,616 B total driver shared state for
its complete direct-callback variant, proving that M256/N128 is executable on
this device. This staged variant avoids callback address pressure and declares
only the 16 KiB weight tile plus the existing 8 KiB Matrix2 reservation. If it
reaches even 1.5x on the 38.126 s Q4 component, Amdahl predicts
`31.019 + 38.126 / 1.5 = 56.436 s`, removing 12.709 s or 18.38% end to end.

No structural split is warranted; productive estimates exclude tests:

```text
crates/graph_horizon_engine/src/backend/vulkan/
├── shaders/matmul/matmul_q4_k_matrix2_f16out.comp
│   (category K; ~140 lines): staged M256/N128/K64 Q4 Matrix2 with direct output
├── kernels/coopmat.rs (~110): dispatch M256/N128 ownership
├── pipeline/kernel.rs (~165): truthful geometry/profile name
└── exec/profile/summary.rs (~180): classify the candidate path
```

Invariant: canonical Q4_K decoding, retained FP16 accumulation/output, operation
coverage, Q6 and non-Q4 paths, and capability fallback remain unchanged. The
different K grouping may change FP16 accumulation order, so the existing Q4
error bounds and full-model quality contract apply. Main risks are register
pressure and the direct-store driver path. Focused oracles gate one paired
8B/128 screen; regression or less than 5% whole gain rejects the combination.

Result: **REJECTED.** The focused Q4 tests again reproduce the retained error
summaries, but 8B/128 TTFT is 997.54 ms versus the 226.84 ms bookend control
(+339.7%; CV 0.13% versus 0.54%). Decode is neutral within clock movement.
Staging four times as many weights plus the M256/N128 accumulator overwhelms
the eightfold workgroup reduction. The exact retained M128/N32/K128 source and
dispatch are restored.

### Cycle 46 candidate: full scalar-direct M256/N128/K64 architecture

The pinned llama.cpp build does not enable decode-vector support, so its
measured 24.84 s reference uses scalar Q4 tensor callbacks together with
M256/N128/K64 ownership and direct global output. Cycles 33 and 45 tested each
half separately, but neither tested their essential composition: the large tile
amortizes callback work across eight times fewer workgroups, while the callback
removes the staged 16 KiB weight tile that made Cycle 45 pathological.

For long M, canonical Q4 traversal falls to one half of retained M128 staged,
activation tensor loads to one quarter, explicit shared weight/output traffic
to zero, and output traffic stays fixed. The reference's same-device component
times show a roughly 2.3--3.8x advantage across these projections. Using a more
conservative 1.5x local speedup still predicts 56.436 s, 18.38% below retained
8B/28K, and therefore clears the complex-design gate.

The temporary structure matches Cycle 45 plus one distinct category-K shader;
productive estimates exclude tests:

```text
crates/graph_horizon_engine/src/backend/vulkan/
├── shaders/matmul/matmul_q4_k_matrix2_direct_f16out.comp
│   (category K; ~165 lines): scalar canonical decode, M256/N128/K64, direct output
├── kernels/matmul/prefill/mod.rs (~195): strict opt-in route before staged Q4
├── kernels/matmul/prefill/policy.rs (~65): experiment switch
├── kernels/coopmat.rs (~115): variant-specific ownership
├── pipeline/kernel.rs (~170): direct variant ABI
└── pipeline/mod.rs (~200): optional Matrix2 pipeline construction
```

Invariant: canonical Q4 bytes, retained FP16 accumulator/output contract,
operation eligibility, every non-Q4 path, and the staged fallback are unchanged.
Main risk is the 247-register/one-workgroup resource point observed for the
reference. Existing Q4 oracles gate paired 8B/128 timing; a regression or less
than 5% whole gain rejects the full architecture.

Result: **REJECTED.** The direct variant routes and reproduces the retained Q4
error distribution, but 8B/128 TTFT is 1,200.03 ms versus the 229.07 ms
bookend control (+423.9%; CV 0.40% versus 0.10%). Decode is neutral. Combining
large ownership with the direct callback does not help while Graph Horizon keeps
activations as Matrix A and weights as transposed Matrix B.

### Cycle 47 candidate: reference Matrix2 operand orientation

Source inspection corrects the earlier ownership description. In llama.cpp,
canonical Q4 weights are Matrix A with `BM=256` output rows, activations are
transposed Matrix B with `BN=128` prompt tokens, and the accumulator is stored
through a transpose into token-major output. Every Graph Horizon Matrix2 matmul
instead places prompt rows in Matrix A and weights in Matrix B. Cycle 46 copied
the dimensions but not this orientation, leaving the decode callback on the B
operand and creating a driver codegen path the reference does not use.

The corrected candidate keeps K64 and the scalar canonical callback but owns
256 outputs by 128 tokens per workgroup. Relative to retained M128-token/N32-
output staging it still launches eight times fewer workgroups, halves repeated
canonical traversal, quarters repeated activation loads, and removes explicit
weight/output staging. The same-device llama component measurements provide a
direct target-sized prior; a conservative 1.5x Q4-local result still predicts
an 18.38% whole-request gain.

The Cycle 46 file structure and estimates remain valid. Only the category-K
shader's operand layouts and the direct variant's dispatch axes change.
Canonical values, FP16 accumulation/output, operation coverage, fallback, and
all non-Q4 paths remain invariant. The main risk is that Graph Horizon's simpler
shader still misses another reference compiler hint. Focused Q4 oracles gate a
paired 8B/128 run; a regression or less than 5% whole gain rejects the corrected
architecture.

Result: **REJECTED.** Four source-fidelity refinements were measured, and every
one preserved the retained focused-Q4 oracle distribution:

- corrected operand orientation with redundant full-block metadata reads:
  1,043.66 ms TTFT;
- one 16-byte metadata vector per output row and K block: 761.43 ms;
- explicit tensor strides, full-tile layouts, and unrolled K64 traversal:
  539.58 ms;
- reference-style padded scale sharing and packed-16 scalar callback:
  494.79 ms versus a 227.93 ms bookend control (+117.1%).

Decode remains neutral. The final generated shader is 11,596 bytes of SPIR-V,
versus 35,712 bytes for the pinned llama.cpp Matrix2 Q4 shader, confirming that
the upstream generated program contains materially more control/dataflow than
this compact port. The refinements remove 548.87 ms from the initial corrected
prototype but still require an implausible further 2.17x gain merely to tie the
retained kernel, before contributing any end-to-end improvement. Importing the
upstream generator and its shader framework would be a broad subsystem and
dependency change, while the direct architecture itself has now failed at each
isolated and composed stage. The experiment is therefore rejected and every
production edit is removed; the accepted source returns exactly to `905dc70`.

### Cycle 48: model/weight adaptation feasibility gate

The next mandatory escalation level is model adaptation, not another exact
kernel sweep. Two measured lossy execution opportunities could in principle be
taught back toward the existing quality boundary:

| Adaptation target | Measured/ideal whole-request gain | Current quality gap | Runtime prerequisite |
|---|---:|---|---|
| W4K/global sparse layers 19--25 | 7.10% measured; <7.82% physical ceiling | strong B only for the more expensive mask | rejected sparse runtime prototype |
| W2K sparse layers 20--25 | 11.23% predicted ideal | class C; +20 top-10 points plus top-1/retrieval qualification | new adapted artifact and sparse route |
| row-scaled INT8 gate/up/down | 10.99% measured on 3B | 28K top-1 fails, top-10 1/10 | 2.059 GiB native representation |

Even granting independent composition of the optimistic sparse and INT8 gains
gives `1 - (1 - .1123) * (1 - .1099) = 20.99%`, far below the 46.1% reduction
required by retained 8B/28K. The runtime results come from different historical
3B baselines, so 20.99% is deliberately an upper-bound screening estimate, not
a benchmark claim. Sparse adaptation alone is also dominated by the stronger
fact that deleting all retained 21.150 s attention leaves 47.995 s, still
10.735 s above the 37.26 s contract ceiling.

The local adaptation environment is a **HARD EXTERNAL BLOCKER**:

- only Q4_K_M GGUF inference artifacts are present; no trainable source weights
  or model-specific adapter checkpoint exists locally;
- the repository contains neither training code nor a long-context
  distillation/quality corpus, and Python has none of PyTorch, Transformers,
  Datasets, Accelerate, PEFT, TRL, Safetensors, or SentencePiece installed;
- the public, ungated official 3B base checkpoint is 7.70 GB before optimizer
  and activation state, while the device has 12 GiB VRAM and the host only
  14 GiB RAM; the official `mistral-finetune` project is archived and recommends
  A100/H100-class hardware;
- teaching a 16--28K attention mask requires long-context teacher examples and
  mask-aware training, not a short generic instruction LoRA. Neither the data
  nor a quality benchmark capable of accepting the resulting new artifact is
  available here.

Downloading packages and weights would therefore not resolve the missing
training/evaluation inputs or the insufficient performance ceiling. No model
artifact or dependency is added. Reopening Level 7 requires, at minimum, a
trainable Ministral checkpoint, a representative long-context teacher corpus,
an explicit adapted-model quality suite, and enough accelerator/host memory to
train the chosen sparse-plus-execution representation. This closes adaptation
for the current environment only; it is not used as a global-stop argument by
itself.

### Cycle 49 candidate: 14B Matrix2 eligibility

Fresh 14B/128 measures 4,381.96 ms versus 174.94 ms in llama.cpp (25.05x),
while decode is 23.48 versus 36.42 tok/s (1.55x). A current timestamp profile
accounts for 4,707.73 of 4,707.98 GPU ms and reveals that the exact shape
allow-list sends Q/K/V/gate/up/down to scalar batched kernels. Those six
categories consume 4,564.64 ms / 96.96% of GPU prefill; attention is only
1.44 ms. O alone reaches the older cooperative route.

The relevant dimensions are all legal Q4_K/Q6_K and Matrix2 multiples:
hidden 5,120, Q/O 4,096, K/V 1,024, and FFN 16,384. The shaders already accept
runtime dimensions, zero-pad output tails, and require K in 256-value quantized
blocks. The exclusion is policy history, not an ABI or format limitation.

With whole time `T=4,381.96 ms`, affected fraction `f=0.9696`, and a
conservative local speedup `S=4`, Amdahl predicts
`T*((1-f)+f/S)=1,194.7 ms`, removing 3,187.3 ms / 72.74%. Scaling the retained
8B Matrix2 profile by parameter count instead predicts roughly 0.55--0.60 s.
The theoretical zero-cost affected floor is about 133 ms; the practical screen
is the conservative 1.19 s result. This clears the 10% complex-design gate by
a wide margin and outranks every remaining 8B microcandidate in absolute
recoverable time.

No new file or directory is warranted; productive estimates exclude tests:

```text
crates/graph_horizon_engine/src/backend/vulkan/kernels/matmul/prefill/
├── policy.rs (~55 productive lines): admit the exact 14B projection/MLP family
└── mod.rs (~170 productive lines, excluding tests): add Q4/Q6 shape oracles
```

Invariant: canonical Q4_K/Q6_K bytes, FP16 activation/output and retained FP16
Matrix2 accumulation, graph operation order, attention, decode, KV, logits,
sampling, all existing 3B/8B routes, and generic fallback remain unchanged.
Eligibility stays capability/format/shape based and contains no model name.
The smallest viable production change is the exact dimension family in the
pure policy. Main risks are an unmeasured resource cliff at K=16,384 and the
already-qualified FP16-accumulation error composing differently across 40
layers. Focused Q4/Q6 CPU oracles gate a paired 14B/128 screen; regression or
less than 10% TTFT gain rejects before the full quality/regression matrix.

Result: **KEEP.** Focused Q4 and Q6 oracles cover both directions of the
5,120/16,384 family. Q4 FP16-accumulator worst relative/mean-relative error is
3.596%/1.382%; Q6 is at most 0.509%/0.095%, all inside the declared bounds.

Diagnostics-free 14B Instruct A/B/A at 128 tokens is
4,375.47 / **365.60** / 4,403.73 ms. The 4,389.60 ms surrounding baseline
therefore falls by 4,024.00 ms / **91.67%**; candidate CV is 0.28%. Decode is
23.34 tok/s versus 23.50 across the controls (-0.66%, inside the 1% guard).
Against fresh llama.cpp at 174.94 ms, the ratio improves from 25.10x to 2.09x.

The candidate profile routes Q/K/V/O/gate/up/down through the existing Q4/Q6
Matrix2 kernels and reduces accounted GPU prefill from 4,707.73 to 520.47 ms
(-88.95%). Per-category movement is:

| Category | Scalar/legacy baseline | Matrix2 | Local speedup |
|---|---:|---:|---:|
| Q | 371.55 ms | 32.81 ms | 11.33x |
| K | 115.11 ms | 14.24 ms | 8.08x |
| V | 113.05 ms | 14.03 ms | 8.06x |
| O | 127.41 ms | 31.09 ms | 4.10x |
| Gate | 1,338.56 ms | 126.08 ms | 10.62x |
| Up | 1,321.74 ms | 129.02 ms | 10.24x |
| Down | 1,304.63 ms | 150.65 ms | 8.66x |

The 14B Reasoning 251-token row improves from the acquired 8,816.82 to
715.48 ms (-91.88%). Both 14B artifacts pass the established 128-step
teacher-forced top-two contract at context capacity 512; Reasoning retains its
exact local top-1 stream, while Instruct first changes local top-1 at step 20.
This is reported as the qualified FP16-accumulation numerical boundary, not as
bit-exact execution. At a near-capacity 384-token Instruct prompt, pure Vulkan
measures 1,071.32 ms (CV 0.05%) versus 358.10 ms in llama.cpp, or 2.99x.

Fresh 3B/128 and 8B/128 guards are 100.28 and 226.05 ms, respectively, versus
99.59 and 228.12 ms controls (+0.69%/-0.91%); decode is unchanged. Full release
validation passes: root 166/166, engine 233 passed with four intentional
ignores, integration 6 passed with one ignore, semantic 12 passed with one
ignore, formatting clean, and Vulkan plus Vulkan-hybrid all-target Clippy clean
with warnings denied. The diagnostics-free pure-Vulkan executable SHA-256 is
`a6db731ea7d7cae6989dec0323fb3461c710977a80a1ae48e92b11dd2a98b690`.
The retained change is capability-, format-, and shape-based with the existing
fallback; production commit is `9eb83b1`.

## Fresh post-Cycle-50 workload matrix

Graph Horizon rows are diagnostics-free and use the retained Cycle 50 source.
llama.cpp 128/28K endpoints and both 14B points are same-session fresh. Middle
llama.cpp rows reuse the qualified pinned-build values because its source and
tuple are unchanged. Single-repetition 16K/28K Graph Horizon rows report no
SD/CV.

### Prefill

| Model | Tokens | Graph Horizon | llama.cpp | GH/llama | Contract | Status |
|---|---:|---:|---:|---:|---:|---|
| 3B | 128 | 74.08 ms | 42.00 ms | 1.76x | <=1.5x | miss |
| 3B | 512 | 239.83 ms | 130.16 ms | 1.84x | <=1.7x | miss |
| 3B | 2K | 1.011 s | 0.541 s | 1.87x | <=1.7x | miss |
| 3B | 8K | 5.031 s | 2.612 s | 1.93x | <=1.7x | miss |
| 3B | 16K | 12.556 s | 6.435 s | 1.95x | <=1.7x | miss |
| 3B | 28K | 28.995 s | 13.603 s | 2.13x | <=1.7x | miss |
| 8B | 128 | 150.66 ms | 97.51 ms | 1.55x | <=1.5x | near miss |
| 8B | 512 | 568.58 ms | 287.81 ms | 1.98x | <=1.7x | miss |
| 8B | 2K | 2.343 s | 1.185 s | 1.98x | <=1.7x | miss |
| 8B | 8K | 10.631 s | 5.300 s | 2.01x | <=1.7x | miss |
| 8B | 16K | 24.474 s | 12.283 s | 1.99x | <=1.7x | miss |
| 8B | 28K | 51.937 s | 24.523 s | 2.12x | <=1.5--1.7x | miss |
| 14B | 128 | 249.32 ms | 174.94 ms | 1.43x | <=1.5x | pass |
| 14B | 384 | 0.717 s | 0.358 s | 2.00x | <=1.7x | miss |

### Decode

Ratios are latency-equivalent `llama tok/s / GH tok/s`; lower is better.

| Model | KV tokens | Graph Horizon | llama.cpp | Latency ratio | Contract | Status |
|---|---:|---:|---:|---:|---:|---|
| 3B | 128 | 71.24 tok/s | 104.81 tok/s | 1.47x | <=1.3--1.4x | near miss |
| 3B | 512 | 70.35 tok/s | 101.97 tok/s | 1.45x | <=1.3--1.4x | near miss |
| 3B | 2K | 69.85 tok/s | 95.06 tok/s | 1.36x | <=1.3--1.4x | pass |
| 3B | 8K | 60.75 tok/s | 78.21 tok/s | 1.29x | <=1.3--1.4x | pass |
| 3B | 16K | 51.89 tok/s | 63.01 tok/s | 1.21x | <=1.3--1.4x | pass |
| 3B | 28K | 42.64 tok/s | 50.02 tok/s | 1.17x | <=1.3--1.4x | pass |
| 8B | 128 | 34.89 tok/s | 49.84 tok/s | 1.43x | <=1.3--1.4x | near miss |
| 8B | 512 | 34.42 tok/s | 49.37 tok/s | 1.43x | <=1.3--1.4x | near miss |
| 8B | 2K | 33.53 tok/s | 47.15 tok/s | 1.41x | <=1.3--1.4x | near miss |
| 8B | 8K | 30.65 tok/s | 39.32 tok/s | 1.28x | <=1.3--1.4x | pass |
| 8B | 16K | 27.52 tok/s | 35.17 tok/s | 1.28x | <=1.3--1.4x | pass |
| 8B | 28K | 23.93 tok/s | 27.69 tok/s | 1.16x | <=1.3--1.4x | pass |
| 14B | 128 | 23.61 tok/s | 36.42 tok/s | 1.54x | <=1.3--1.4x | miss |

Fresh repeated Graph Horizon TTFT CV is 0.03--0.62% for the 3B/8B rows through
8K, 0.47% for retained 14B/128, and 0.04% for 14B/384. Fresh llama endpoint
SD is 0.17/60.33 ms for 3B 128/28K, 0.17/9.91 ms for 8B, and 2.83/3.12 ms for
14B 128/384. The unavoidable timing boundary remains: Graph Horizon includes
tokenization and first sampling; llama pure `pp` does not.

### Cycle 50 candidate: upstream-fidelity Q4_K Matrix2 dataflow

The global-stop audit leaves one hardware-family architecture untested at full
fidelity. Cycle 47 reproduced its dimensions, orientation, scale staging, and
direct output separately, but retained a compact hand-written callback and
control structure. The pinned reference instead uses typed 144-byte Q4_K block
views, a tensor block size of 256, callback-based cooperative loads, double-
buffered scale fetch/store state, a K64 inner tile unrolled across each full
K256 quant block, tensor-view transposition for activations and output, and
separate clamped/unclamped tensor layouts. Its generated Q4 shader is 35,712
SPIR-V bytes versus 11,596 bytes for the final compact Cycle 47 prototype.

At 8B/28K, compressed Q4 accounts for 38.126 s and everything else for about
31.120 s. Matching a 2.3x reference-side local advantage predicts 47.70 s
(-31.1% whole request); matching the high observed endpoint of 3.8x predicts
41.15 s (-40.6%) and clears the broad 1.7x ceiling of 41.689 s. The exact local
speedups required are 3.61x for the broad ceiling and 6.73x for the strict 1.5x
ceiling. This is high-risk but clears the >=10% hardware-specific economic gate.

No new domain or orchestration split is warranted. The experiment replaces the
existing compressed-Q4 shader and specializes its existing dispatch while the
retained diagnostics-free binary supplies the A control. Productive estimates
exclude tests:

```text
crates/graph_horizon_engine/src/backend/vulkan/
├── shaders/matmul/matmul_q4_k_matrix2_f16out.comp
│   (category K; ~340 lines): one typed Q4_K Matrix2 matmul operation
└── kernels/coopmat.rs (~150): Q4-specific binding order, shape push, and grid
```

Invariant: canonical Q4_K bytes, FP16 activation/accumulator/output, token-major
output, capability/shape eligibility, every Q6/direct-FP16/decode path, and the
existing generic fallback remain unchanged. The main risk is that the earlier
compact port already captured the performance-relevant instructions and that
the larger callback form only increases register pressure. The focused Q4 CPU
oracle gates a diagnostics-free 8B/128 screen. A regression or less than 10%
whole TTFT gain rejects before 28K; a qualifying screen proceeds to profile,
8B/28K, multi-model regression, and the existing quality suite.

Result: **KEEP.** The typed direct callback, M256/N128/K64 ownership, transposed
tensor views, direct clamped output, and scale-only shared state produce 13,912
bytes of SPIR-V. Capacity-equivalent 8B/128 falls from 228.12 to 150.66 ms
(-33.96%, CV 0.43%) and 8B/28K falls from 69.246 to 51.937 s (-25.00%). Decode
is neutral. The long-context result is still 2.12x llama.cpp and misses its
1.7x ceiling by 10.248 s, but the gain is far above the hardware-specific gate.

The 28K profiler accounts for 99.96% of 53.961 GPU seconds. Direct Q4 consumes
22.690 s versus the prior staged 38.126 s (1.68x local, 15.436 s removed),
attention remains 21.272 s, staged Q6 consumes 8.702 s, and all other work is
1.297 s. The profile identifies the causal majority of the separate 17.309 s
public-wall saving; its absolute total is not substituted for diagnostics-free
timing. The final shader is capability-, format-, and shape-specialized and
retains the generic fallback; no model or device name is used.

All six focused Q4 oracle shapes pass without a tolerance change. Maximum
relative error is 3.596% and mean relative error is 1.382% in the established
large-K FP16 boundary. Forced-generic references and the candidate pass 128
teacher-forced top-two steps on all six 3B/8B/14B Instruct/Reasoning artifacts;
local top-one crossings are 2/0/1/1/1/1 respectively. Full validation passes:
166 root tests, 233 engine tests with four intentional ignores, six integration
tests with one ignore, 12 semantic tests with one ignore, formatting, and both
pure-Vulkan and Vulkan-hybrid warning-denied all-target Clippy. Production
commit is `dff4bd8`.

### Cycle 51 candidate: direct canonical-Q6_K Matrix2 dataflow

Cycle 50 changes the ranking. Staged Q6 V/down now accounts for 8.702 s / 16.13%
of the 8B/28K GPU request. Reusing the proven direct Matrix2 architecture at a
conservative 1.5x local speedup predicts 2.901 s / 5.59% whole-request removal;
the measured Q4-local 1.68x predicts 3.522 s / 6.78%. Neither closes the broad
contract alone, but this is the last compressed matmul family above the 5%
moderate-design gate and its incremental architecture is already qualified.

No new file or pipeline variant is required. Productive estimates exclude
tests:

```text
crates/graph_horizon_engine/src/backend/vulkan/
├── shaders/matmul/matmul_q6_k_matrix2_f16out.comp
│   (category K; ~200 lines): one typed Q6_K Matrix2 matmul operation
└── kernels/coopmat.rs (~170): shared direct-Q4/Q6 binding, shape push, and grid
```

Invariant: canonical 210-byte Q6_K blocks, FP16 activation/output, the retained
FP32 accumulator, token-major output, current eligibility,
Q4/direct-FP16/decode paths, and the generic fallback remain unchanged. Main
risks are Q6 callback instruction depth
and the smaller affected fraction. The focused Q6 oracle gates diagnostics-free
8B/128. Less than 5% whole TTFT gain rejects; a qualifying result proceeds to
28K profiling, cross-model guards, and the same six-model quality contract.

Result: **REJECTED.** The direct Q6 callback compiles and passes the focused
oracle at 0.420% maximum / 0.058% mean relative error with unchanged FP32
accumulation. Diagnostics-free 8B/128 improves only 150.66 to 148.89 ms
(-1.17%, candidate CV 0.32%); decode movement is ordinary clock noise. This is
below the 5% gate before a costly long-context run. The direct Q6 shader and
binding/grid changes are removed exactly, restoring production to `dff4bd8`.

### Cycle 52 candidate: reference-large Q4 Matrix2 ownership

The pinned reference's large cooperative-Matrix2 quant tile is `BM=128`,
`BN=256`, `BK=64`: 128 canonical-weight output rows by 256 prompt tokens. The
retained direct kernel is the transposed ownership point, 256 output rows by 128
tokens. On Graph Horizon's 256-row prefill batches, the reference-large point
halves repeated canonical weight traversal/callback work and doubles activation
loads while preserving the workgroup count. Q4 decoding is instruction-heavy,
and the measured reference selected this exact endpoint, so it is not dominated
by traffic arithmetic alone.

Direct Q4 is 22.690 s / 42.05% of 8B/28K. A conservative 1.2x local speedup
removes 3.782 s / 7.0% of profiled time; 1.5x removes 7.563 s / 14.0%. The change
is a low-complexity exact-device geometry endpoint and clears the moderate gate.
No new files or functions are required:

```text
crates/graph_horizon_engine/src/backend/vulkan/
├── shaders/matmul/matmul_q4_k_matrix2_f16out.comp
│   (category K; ~210 lines): direct Q4_K operation with M128/N256 ownership
└── kernels/coopmat.rs (~140): matching output/token dispatch grid
```

Canonical bytes and scale reconstruction, FP16 activation/accumulator/output,
K64 order, direct clamped output, eligibility, every non-Q4 path, and fallback
remain invariant. Main risk is activation amplification dominating the saved
callback traversal. The existing Q4 oracle gates 8B/128; less than 5% whole
gain rejects, while a qualifying result proceeds to 28K and the retained
quality/regression matrix.

Result: **REJECTED.** The focused Q4 oracle reproduces the retained error
distribution, but diagnostics-free 8B/128 regresses from 150.66 to 350.88 ms
(+132.9%; candidate CV 0.18%). The extra activation work and smaller output
ownership dominate the saved canonical traversal. The constants and grid are
removed exactly, restoring production to `dff4bd8`. The reference-large and
retained ownership points now bracket the output/token orientation trade-off.

### Cycle 53 candidate: combined M256/N256 Q4 ownership

One adjacent direct-Q4 endpoint is not dominated by Cycle 52. Keeping the
retained 256 output rows while increasing token ownership from 128 to 256
halves workgroups and repeated canonical weight callbacks without increasing
total activation loads. Unlike the reference M128/N256 point, it does not
sacrifice output reuse. The risk is an accumulator/resource cliff: the FP16
M256/N256 accumulator is twice the retained size and the reference does not
offer this combination.

At the 22.690 s direct-Q4 share, even a conservative 1.15x local gain predicts
2.960 s / 5.49% whole removal; an ideal callback-dominated 2x removes 11.345 s.
This clears the final exact-device geometry screen. Only the category-K shader
token constant and existing Q4 grid change; the recorded Cycle 52 files remain
within ~210/~140 productive lines. Canonical Q4 arithmetic, K64 order, output
ownership, types, eligibility, non-Q4 paths, and fallback are invariant. The Q4
oracle and 8B/128 screen reject pipeline absence, regression, or less than 5%
whole gain before any long run.

Result: **REJECTED.** M256/N256 passes the focused oracle but measures 373.04 ms
at 8B/128 versus 150.66 ms retained (+147.6%; candidate CV 0.20%). Doubling the
accumulator crosses a severe Matrix2 resource/code-generation cliff. The token
constant and grid are removed exactly; production returns to `dff4bd8`.

### Cycle 54 candidate: direct Q4 K128 inner tile

The retained direct callback uses the pinned reference's K64 inner matrix. A
K128 endpoint halves cooperative load/MMA statements while leaving M256/N128,
workgroups, canonical traversal, scale staging, accumulator, and all traffic
classes unchanged. The former staged kernel proved that K128 is a legal device
shape, but it did not test K128 with direct canonical callbacks.

At the 42.05% direct-Q4 share, a 1.15x local result predicts 5.49% whole removal.
Only the category-K `BK` constant changes within the existing ~210-line shader;
host orchestration is unchanged. The focused oracle and 8B/128 screen reject
pipeline absence, regression, or less than 5% gain. If K128 regresses, K32 is
dominated by twice as many callback loads and MMAs, closing direct-Q4 K depth.

Result: **REJECTED; direct-Q4 device geometry closed.** K128 preserves the
focused oracle but measures 290.72 ms at 8B/128 versus 150.66 ms retained
(+93.0%; candidate CV 0.23%). Its operand/resource cost overwhelms the reduced
MMA count. K64 is bracketed by this deeper resource cliff and K32's strictly
greater callback/MMA count. The constant is removed exactly and the retained
source is restored.

## Final global-stop proof

The final diagnostics-free rebuild revalidates 8B/128 at 149.68 ms (CV 0.22%)
and 35.06 tok/s. Its SHA-256 is
`5e3f5c3d00e3105072f2054e50972afe4e03ee4709b3c8705767d62b301c9828`.
All absolute comparisons retain the disclosed boundary that Graph Horizon
includes tokenization and first sampling while llama.cpp pure prompt processing
does not. At 128 tokens that boundary was measured at roughly 11 ms, so the raw
8B/128 miss is smaller than the known non-model-side difference.

### Program-entry versus final representative matrix

Program-entry values are the qualified Phase 29 checkpoint named by the
mission. llama.cpp absolute values were refreshed later on the same pinned
build, so the entry ratio column preserves the original same-session ratio and
the final ratio uses the fresh reference.

| Workload | Program entry | Final | llama.cpp fresh | Entry ratio | Final ratio | Contract | Final status |
|---|---:|---:|---:|---:|---:|---:|---|
| 3B prefill / 128 | 92.98 ms | 74.08 ms | 42.00 ms | 2.16x | 1.76x | <=1.5x | miss |
| 3B prefill / 28K | 35.15 s | 28.995 s | 13.603 s | 2.50x | 2.13x | <=1.7x | miss |
| 8B prefill / 128 | 260.05 ms | 149.68 ms | 97.51 ms | 2.38x | 1.54x | <=1.5x | near miss |
| 8B prefill / 2K | 4.07 s | 2.343 s | 1.185 s | 3.44x | 1.98x | <=1.7x | miss |
| 8B prefill / 28K | 75.10 s | 51.937 s | 24.523 s | 3.02x | 2.12x | <=1.7x | miss |
| 14B prefill / 128 | 4.390 s acquired | 249.32 ms | 174.94 ms | 25.10x | 1.43x | <=1.5x | pass |
| 3B decode / 128 | 56.74 tok/s | 71.24 tok/s | 104.81 tok/s | 1.81x | 1.47x | <=1.4x | near miss |
| 3B decode / 28K | 36.52 tok/s | 42.64 tok/s | 50.02 tok/s | 1.37x | 1.17x | <=1.4x | pass |
| 8B decode / 128 | 26.64 tok/s | 35.06 tok/s | 49.84 tok/s | 1.90x | 1.42x | <=1.4x | near miss |
| 8B decode / 28K | 19.52 tok/s | 23.93 tok/s | 27.69 tok/s | 1.42x | 1.16x | <=1.4x | pass |
| 14B decode / 128 | 17.23 tok/s | 23.61 tok/s | 36.42 tok/s | 2.10x | 1.54x | <=1.4x | miss |

The complete short/medium/long final matrix is the post-Cycle-50 table above.
Repeated final rows have TTFT CV 0.03--0.62%; 16K/28K rows are explicitly
single-observation. The established environment telemetry spans idle
210/405 MHz, 19.34 W, 54 C to loaded 1,995/7,501 MHz, 156.83 W, 61 C. Per-row
clock/power/temperature distributions were not collected for Cycle 50 and are
not invented. The 12 GiB capacity guard limits both 14B artifacts to context
512; 3B/8B use context 32,768.

### Remaining gap and practical floor

The largest remaining workload is 8B/28K. Its public competitive gap is
27.414 s and its gap to the broad 1.7x contract is 10.248 s. The final profile
accounts for 99.96% of 53.961 GPU seconds:

| Component | Current | Optimistic runtime floor | Analytical removal | Evidence / further requirement |
|---|---:|---:|---:|---|
| Attention | 21.272 s | 15.990 s | <=5.282 s | inherited measured llama-rate floor; every dense ownership/dataflow candidate closed; lower needs adapted sparse/windowed behavior |
| Direct Q4 Matrix2 | 22.690 s | 22.690 s | 0 economically available | retained direct architecture; M128/N256 350.88 ms, M256/N256 373.04 ms, K128 290.72 ms at the 150.66 ms screen |
| Staged Q6 Matrix2 | 8.702 s | 8.702 s | 0 economically available | direct canonical Q6 improves only 1.17% whole at 128, below the 5% gate |
| Other | 1.297 s | 0 | <=1.297 s impossible-delete bound | already only 2.40%; no component-sized candidate |
| **Profile total** | **53.961 s** | **47.382 s** | **<=6.579 s** | optimistic runtime-only floor remains 5.693 s above the 41.689 s broad ceiling |

The table is deliberately generous: it combines the old attention-rate bound
with deletion of all other work even though no implementation realizes both.
It therefore proves that another runtime-only microarchitecture candidate
cannot close 8B/28K. The corresponding final aggregate gaps are:

| Workload | Current | llama.cpp | Contract ceiling / floor | Gap to nearest contract | Required change |
|---|---:|---:|---:|---:|---|
| 3B prefill / 28K | 28.995 s | 13.603 s | 23.125 s | 5.870 s / 20.25% | adapted representation or lower-work attention |
| 8B prefill / 28K | 51.937 s | 24.523 s | 41.689 s | 10.248 s / 19.73% | combined model-adapted sparse + low-precision execution |
| 14B prefill / 384 | 0.717 s | 0.358 s | 0.609 s | 0.109 s / 15.14% | same adapted representation; longer rows exceed capacity |
| 3B decode / 128 | 14.04 ms/token | 9.54 ms/token | 13.36 ms/token | 0.68 ms / 5.1% throughput | new exact projection representation; ownership bracketed |
| 8B decode / 128 | 28.52 ms/token | 20.06 ms/token | 28.09 ms/token | 0.43 ms / 1.5% throughput | below economic gate; public boundary already near band |
| 14B decode / 128 | 42.35 ms/token | 27.46 ms/token | 38.44 ms/token | 3.91 ms / 10.2% latency | target-sized projection policy fails quality or adaptation is required |

### Final design-space exhaustion

| Design space | Status | Quantitative closure |
|---|---|---|
| Generic backend/runtime | CLOSED | steady-state allocations are zero; final profile accounts 99.96%; record/submit/descriptor work is about 111 ms against 53.96 s GPU and has no target-sized ceiling |
| Routing / eligibility | SUCCEEDED, THEN CLOSED | Cycle 49 routes all legal 14B Q/K/V/O/gate/up/down shapes and removes 91.67% at 128; final audit finds no dominant fallback on measured shapes |
| Kernel architecture/dataflow | SUCCEEDED, THEN CLOSED | direct Q4 removes 15.436 profiled seconds; direct Q6 gains 1.17%; dense attention and decode ownership families are bracketed in Cycles 1--54 |
| Layout / intermediates | CLOSED | direct tensor-view transpose/output is retained; Q staging, KV/GQA layouts, transient expansion, split/merge, and materialized attention alternatives are measured negative or below gate |
| Weight execution representation | SUCCEEDED, THEN CLOSED | native FP16 views retained where capacity permits; direct compressed Q4 retained; metadata-only, persistent/transient expansion, INT8 and sparse runtime forms fail capacity, speed, or quality |
| Selective numerical policy | CLOSED BY QUALITY/ECONOMICS | FP16 Q4 accumulation and Q4 decode DP4A pass their contracts; broad row-INT8/sparse and faster projection subsets fail unchanged top-two/token gates; best rejected Q6 decode is 4.91% |
| Model / weight adaptation | HARD EXTERNAL BLOCKER | optimistic independent sparse+INT8 ceiling is now 20.99%, predicting 41.04 s at 8B/28K, but trainable weights, long-context teacher data, quality corpus/tooling, and sufficient training memory are absent |
| Hardware-family specialization | SUCCEEDED, THEN CLOSED | NVIDIA Matrix2/GQA/DP4A paths retained behind capabilities; the reference dataflow is reconstructed; remaining full-generator branches do not affect this single-batch core and would duplicate a broad subsystem |
| Exact-device specialization | CLOSED | Q4 M128/N256 +132.9%, M256/N256 +147.6%, K128 +93.0%; retained M256/N128/K64 is bracketed; Q4/Q6 decode lane counts and attention splits were bracketed earlier |

### Remaining opportunity register

| Candidate | Level | Expected whole gain | Risk / quality | Portability / cost | Final classification |
|---|---:|---:|---|---|---|
| Adapted W2K sparse + row-INT8 model | 7 | optimistic 20.99%; 51.937 -> 41.04 s | high; new artifact and long-context quality contract | model-family, very high training/data cost | unavailable: hard external blocker |
| Direct Q6 Matrix2 | 8 | measured 1.17% at screen | low numerical risk | hardware-family, moderate code | rejected below 5% |
| Further direct-Q4 geometry | 9 | all measured negative | retained quality | device-specific, low per test but no positive prior | exhausted by Cycles 52--54 |
| Full upstream generator/framework import | 8 | no remaining evidence-backed positive increment over retained core | integration and maintenance risk | roughly 3,920 relevant GLSL lines plus generator/host subsystem | rejected economically; core architecture and exact large geometry measured |
| Q6 combined decode DP4A | 5--6 | 4.91% at 8B/128; would enter fresh broad band | added activation quantization/logit ranking surface | capability-based, moderate | rejected below gate while prefill remains ~2x |
| Different accelerator/model architecture | 7--9 | potentially target-sized, unquantified here | changes comparison platform or artifact | outside current system | unavailable, not a hidden local candidate |

### Retained optimization architecture

| Retained path | Classification | Eligibility | Fallback |
|---|---|---|---|
| Packed/direct Q4 Matrix2 prefill with FP16 accumulation | hardware-family-specific, capability/shape/quantization-specialized | NVIDIA Matrix2 features + Q4_K + measured dimensions | staged cooperative/scalar Q4 paths |
| 14B Matrix2 eligibility | capability- and shape-based, model-name independent | legal 5,120/16,384/4,096/1,024 dimensions | existing Q4/Q6 batch kernels |
| Per-8 Q8/DP4A Q4 decode | capability- and quantization-family-specific | DP4A + Q4_K + scratch/shape contract | exact float Q4 decode |
| Vectorized GQA decode | backend-family capability/shape specialization | exact GQA head relation and device limits | generic attention decode |
| Native FP16 execution views | capability/capacity/shape specialized | memory budget and measured roles | canonical compressed weights |

No retained route names a model or exact GPU, no dependency or public API was
added, and every specialized path preserves a generic fallback. Portability is
tested only on the available RTX 3060; the classification describes routing,
not an unsupported multi-device performance claim.

### Final answer

Within the runtime, artifacts, hardware, data, and tooling available to this
project, **yes: substantially all technically and economically recoverable
performance has been recovered.** The result is not contract success: medium
and long prefill and short 14B decode remain outside the target. It is a global
stop because the optimistic runtime-only 8B/28K floor is still 47.382 s versus
41.689 s, every portable and device-specific candidate above its gate has been
retained or rejected, and the sole remaining target-sized bound is the 20.99%
model-adaptation combination blocked by missing external weights, data,
evaluation infrastructure, and training memory. Reopen only when those inputs
exist or a new candidate presents a quantified >=5--10% whole-request bound not
already represented in the experiment registry.
