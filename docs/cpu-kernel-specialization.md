<!--
This report is the persistent checkpoint for the CPU-only performance program.
It owns reproducible measurements, attribution, Amdahl ranking, experiment
decisions, correctness evidence, and the eventual global-stop proof.
-->

# CPU Kernel Specialization

## Baseline identity

- Production baseline: `main` at `17f77bf948f74608f2352d81b18506603cde14f6`.
- Investigation branch: `perf/cpu-kernel-specialization`.
- Merge-base with `main`: `17f77bf948f74608f2352d81b18506603cde14f6`.
- Initial worktree: clean; the pre-existing NVIDIA audit branch was left intact.
- Model: authenticated `3b-instruct` Q4_K_M, 2,147,023,008 bytes, SHA-256
  `9ed150d4367e68df0ac8e1540f6ddc65b42d0ee26378329d1ecbca60f93fc5f8`.
- Comparator: llama.cpp `9bebfcb4bc8b12a316e96ae03f33671eac1e72fd`.
- No push, pull request, merge, dependency, public API, or non-CPU change.

## Host and protocol

- CPU: Intel Core i5-9600K, x86_64, six physical cores and six logical threads.
- ISA: SSE through SSE4.2, AVX, AVX2, F16C, FMA, BMI1/2; no AVX-512, VNNI,
  BF16, or AMX.
- Cache: 32 KiB L1d and 256 KiB L2 per core; shared 9 MiB L3.
- NUMA: one node; CPUs 0-5. Thread rows use `taskset` on the first N CPUs.
- OS: Linux 7.0.0-29-generic x86_64; governor reported `powersave` throughout.
- Compiler: rustc 1.95.0 / LLVM 22.1.2; Cargo `release`, locked dependencies.
- Graph Horizon: CPU-only, F16 KV, exact 128-token deterministic chat prompt,
  greedy sampling, 32 requested tokens, one warm-up, three repetitions.
- llama.cpp: same model artifact, CPU-only (`-ngl 0`), F16 K/V, flash attention
  off, 128 synthetic prompt tokens, 32 generation tokens, three repetitions.
- Material mismatch: Graph Horizon TTFT includes tokenization, full chat graph,
  logits, first sampling, and first public delta. llama.cpp `pp128` is isolated
  prompt evaluation. llama.cpp `tg32` starts from its benchmark-owned context,
  whereas Graph Horizon decode follows the 128-token chat prefill.
- Hardware counters and sampling are unavailable because
  `kernel.perf_event_paranoid=4`; opt-in CPU boundary timers provide attribution.

## Initial thread scaling

| Threads | Prompt tok/s | TTFT ms | Decode tok/s | TTFT CV | Decode CV | Wall s | User s | Peak RSS KiB |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 4.60 | 27,826.48 | 1.48 | 0.06% | 0.05% | 194.27 | 192.70 | 4,268,060 |
| 3 | 13.19 | 9,705.46 | 4.26 | 0.18% | 0.30% | 68.17 | 192.80 | 4,268,044 |
| 6 | 18.21 | 7,030.85 | 5.71 | 0.62% | 0.76% | 50.49 | 196.49 | 4,267,996 |

One to six cores scales 3.96x for prompt and 3.86x for decode, or 66% and 64%
parallel efficiency. Six cores are the production policy on this host. llama.cpp
decode measured 5.32, 11.19, and 10.81 tok/s at 1, 3, and 6 threads; its best
decode thread count is three. Its batch-thread setting remained at the build
default, so its three prompt rows are not used as a thread-scaling claim.

## Initial competitive baseline

| Model/workload | Graph Horizon | llama.cpp | Ours / llama | Absolute gap |
|---|---:|---:|---:|---:|
| 3B prefill 128, six cores | 7,030.85 ms TTFT | 321.64 ms pp | 21.86x latency | 6,709.21 ms |
| 3B decode after 128, six cores | 5.71 tok/s | 10.81 tok/s tg | 1.89x latency | 80.15 ms/token |

The prefill gap is the first global target. Decode remains material and will be
re-ranked after a retained prefill gain.

## Initial attribution

The profiled row used 128 prompt tokens, two generated deltas, no warm-up, one
repetition, and six cores. Public TTFT was 7,223.60 ms. The timers accounted for
7,377.44 ms including the second decode step; reported fractions therefore use
the internally complete accounted interval.

| Component | Time ms | Fraction |
|---|---:|---:|
| `n=4` quantized batched matmuls | 6,180.07 | 83.77% |
| RoPE | 577.51 | 7.83% |
| `n=1` batched matmuls | 101.70 | 1.38% |
| attention prefill | 98.51 | 1.34% |
| RMSNorm | 92.26 | 1.25% |
| SiLU multiply | 142.30 | 1.93% |
| residual adds | 112.47 | 1.52% |
| logits | 29.36 | 0.40% |
| KV write | 8.89 | 0.12% |
| decode matmuls | 31.93 | 0.43% |
| attention decode | 2.28 | 0.03% |
| embedding | 0.18 | <0.01% |

The profiler attributes more than 99.9% of its timed CPU boundary. Shape-level
matmul attribution is: gate/up Q4_K 37.76%, down Q4_K/Q6_K 22.98%, Q projection
9.22%, O projection 8.97%, K plus Q4_K V 4.21%, and Q6_K V 1.61%. K and V share
one shape/format on half the layers, so the current boundary timer cannot split
that subset without weight-identity instrumentation; it is not hidden in
`other`.

## Experiment registry

### CPU-01 — 32-row prefill batch

- Target: 3B/128 prefill; decode is a control.
- Hypothesis: the fixed four-row CPU batch repeats Q4_K/Q6_K weight unpack and
  dequantization 32 times. A 32-row batch cuts that repetition eightfold while
  keeping the largest per-worker output tile near 192 KiB, within private L2.
- Intentional variable: `CPU_ROWS`, 4 to 32. No kernel or numeric-order change.
- Practical local floor: measured `n=1`/`n=4` call costs imply about 2.3x for
  the affected batched matmuls; larger batches cannot remove the token FMAs.
- Amdahl inputs: whole TTFT 7,223.60 ms, affected fraction 83.77%, realistic
  local speedup 2.3x.
- Prediction: 1.90x whole-workload speedup, predicted TTFT 3,803.36 ms,
  removable time 3,420.24 ms. Recoverable competitive gap is 3,420.24 ms.
- Correctness gate: existing batched-versus-sequential CPU tests plus real
  pinned-oracle parity if the A/B result clears the retention threshold.
- Measured candidate: 29.93 prompt tok/s, 4,277.03 ms TTFT, 5.77 decode
  tok/s; CV 0.67% prompt, 0.68% TTFT, and 1.26% decode; wall 39.01 s,
  user 160.68 s, peak RSS 4,267,976 KiB.
- Delta: +64.36% prompt throughput, -39.17% TTFT, +1.05% decode throughput,
  and no peak-RSS increase at the process maximum.
- Correctness: both focused batched-final-logits tests pass for F16 and INT8
  KV. The local comparator is `9bebfcb4b`, while the repository parity runner
  requires `13f2b28b0`; current status is
  `external verification: unsupported llama.cpp revision` until that pinned
  oracle is built locally.
- State: provisional KEEP; the performance and focused-correctness gates pass.

### Reprofile after CPU-01

The two-token diagnostic measured 4,639.60 ms TTFT and accounted 4,809.20 ms
including the second decode step. The remaining `n=32` Q4_K matmuls consume
3,116.48 ms (64.80%); `n=32` Q6_K matmuls consume 659.61 ms (13.72%); RoPE is
581.50 ms (12.09%); prefill attention is 58.92 ms (1.23%). The wider batch
reduced the targeted matmul total 6,180.07 -> 3,776.09 ms (1.64x local).

The next batching step is closed: at `n=32` the observed call cost already
exceeds the linear dequant-plus-token model, and `n=64` would move the largest
per-worker output tile from about 192 KiB beyond the 256 KiB L2. Its credible
whole-workload upper bound is below the 5% moderate-change gate.

### CPU-02 — AVX2 Q4_K x Q8 activation integer dot

- Target: the remaining `n=32` Q4_K prefill matmuls; decode remains on the
  existing float path initially.
- Hypothesis: quantize each FP16 activation row into the already qualified
  per-eight Q8 representation, then replace four 8-wide FP32 weight-dequant
  FMAs with AVX2 unsigned-by-signed integer dot reduction while reusing each
  unpacked Q4 sub-block across the token batch.
- Traffic: canonical Q4_K remains 144 bytes per 256 weights. Q8 activations use
  eight int8 values plus `d`/`s` metadata per eight, versus widened FP32 input;
  no weight repack or persistent RAM increase is required.
- Amdahl input: Q4_K batched fraction 64.80%. A conservative 2x local kernel
  gain predicts 1.48x total, 1,559 ms removable from the profiled interval, far
  above the complex-change gate and below the remaining competitive gap.
- Eligibility: x86 AVX2, Q4_K, `n >= 8`, and validated block-aligned shapes;
  every other ISA, format, and shape retains the existing float kernel.
- Correctness gate: synthetic Q8-vs-float bounded parity including block/tail
  boundaries, focused batched graph tests, and pinned-oracle teacher parity.
- Prototype result: 12.98 prompt tok/s, 9,860.26 ms TTFT, and 5.64 decode
  tok/s; prompt CV 0.54%, decode CV 0.40%. Against CPU-01 this is -56.63%
  prompt throughput, +130.54% TTFT, and -2.25% decode throughput.
- Diagnosis: canonical weights plus per-call activation quantization and four
  scalar horizontal reductions per 32-value sub-block cost more than the saved
  FP32 FMAs. llama.cpp's integer path relies on a deeper multi-row/repacked GEMM
  architecture; an arithmetic substitution alone does not transfer that gain.
- State: REJECT. All Q8 candidate code and tests were removed; only this result
  remains. No corrective parameter sweep is justified by a 2.3x regression.

### CPU-03 — reuse RoPE coefficients across heads

- Target: 3B/128 prefill; decode is a control.
- Hypothesis: pair/position coefficients are head-independent, so a pair-major
  loop reduces validation, YaRN correction, pow, sin, and cos calculations from
  2,560 to 128 per token/layer without changing element arithmetic.
- Amdahl prediction: RoPE fraction 12.09%, conservative 5x local gain, 1.11x
  total speedup, and about 465 ms removable.
- Baseline bookend: 30.18 prompt tok/s, 4,240.61 ms TTFT, 5.70 decode tok/s;
  CV 0.35% TTFT and 0.67% decode.
- Candidate: 31.03 prompt tok/s, 4,125.17 ms TTFT, 5.92 decode tok/s; CV 0.50%
  TTFT and 0.61% decode.
- Delta: +2.82% prompt, -2.72% TTFT, +3.86% decode. The result shows only about
  115 ms of the profiled RoPE bucket was redundant coefficient calculation;
  F16 conversion/copy and rotation work dominate the remainder.
- Correctness: all seven focused CPU/shared RoPE tests pass.
- State: REJECT because the declared prefill objective is below 3%. The loop
  change was removed; no coefficient-cache or loop-order sweep is justified.

### Compiler profile audit

The existing Cargo `fast` profile (fat LTO, one codegen unit) measured 26.47
prompt tok/s, 4,835.45 ms TTFT, and 5.78 decode tok/s. A same-session release
bookend measured 30.18 prompt tok/s, 4,240.61 ms TTFT, and 5.70 decode tok/s.
Both records were stable; `fast` regresses TTFT by 14.0% while leaving decode
effectively unchanged. State: REJECT for this CPU; retain the release profile.

### CPU-04 — inline F16C quant-scale widening

- Target: Q4_K/Q5_K/Q6_K x86 SIMD kernels, which account for about 78.5% of the
  post-CPU-01 profiled interval.
- Hypothesis: replace each branchy out-of-line software FP16 scale conversion
  with a guarded, inlined F16C widening. Release disassembly confirmed that the
  hot kernels changed from `call f16_at` plus `vzeroupper` boundaries to inline
  `vcvtph2ps`; focused Q4_K/Q5_K/Q6_K parity passed (13 tests).
- Retained bookends: 32.35/31.32 prompt tok/s, 3,957.26/4,086.76 ms TTFT, and
  5.88/5.74 decode tok/s. Their means are 31.84 prompt tok/s, 4,022.01 ms TTFT,
  and 5.81 decode tok/s.
- Candidate: 24.54 prompt tok/s, 5,215.62 ms TTFT, and 6.27 decode tok/s; CVs
  were 1.03%, 1.04%, and 0.83% respectively.
- Delta against mean bookends: -22.91% prompt throughput, +29.68% TTFT, and
  +7.92% decode throughput. Inline expansion increases pressure in the large
  batched kernels enough to dominate the removed software calls; the decode
  gain does not compensate for a decisive regression in the first global target.
- State: REJECT after one attempt under the >2% regression rule. All candidate
  source changes were removed.

### CPU-05 — fixed-size scale blocks

- Target: the checked six-bit scale/min decoder inside the remaining Q4_K/Q5_K
  SIMD loops.
- Hypothesis: pass `scale_min` a fixed `[u8; 12]` block so one safe slice proof
  replaces repeated dynamic tensor bounds checks. Release codegen changed as
  intended: the two-row Q4_K function fell from 380 to 357 assembly lines, and
  the repeated scale-index checks collapsed to two block-level checks.
- Candidate: 27.40 prompt tok/s, 4,672.30 ms TTFT, and 5.52 decode tok/s; CVs
  were 0.37%, 0.37%, and 0.56%. The immediately preceding retained bookend was
  31.32 prompt tok/s, 4,086.76 ms TTFT, and 5.74 decode tok/s.
- Delta: -12.52% prompt throughput, +14.33% TTFT, and -3.83% decode throughput.
  The smaller branch graph did not improve execution; fixed-slice proof and
  resulting loop layout cost more than the checks removed.
- Correctness: all 13 focused Q4_K/Q5_K/Q6_K matmul tests passed.
- State: REJECT after one attempt under the >2% regression rule. The safe
  candidate was removed; an unchecked transcription and parameter sweep are not
  justified.

### CPU-06 — post-batch core-count policy

- Target: determine whether wider prefill batches move the best production
  point below all six physical cores.
- Diagnostic (128 prompt tokens, two generated tokens, one warm-up, three
  repetitions): four cores measured 26.36 prompt tok/s and 4,856.41 ms TTFT;
  five measured 24.20 and 5,289.16 ms; six measured 26.00 and 4,923.81 ms.
  The three-core row was both slower and noisy (19.53 tok/s, 9.68% CV).
- State: KEEP the six-core production policy. Four cores' 1.38% diagnostic lead
  is below the 3% gate and less representative than the full 32-token six-core
  record; no code or default changed.

### Decode attribution after CPU-01

The dedicated row used the 128-token prompt followed by 31 measured decode
tokens. Public decode was 5.63 tok/s; timers accounted for 10,020.03 ms across
prefill and decode. Call cardinalities separate `n=32` prefill from `n=1` decode.
Against the approximately 5.51-second public decode interval, Q4_K gate/up
consumed 2,180.71 ms (39.6%), Q4_K/Q6_K down projections 1,155.77 ms (21.0%),
attention projections 1,071.24 ms (19.5%), logits 508.23 ms (9.2%), and decode
attention 80.78 ms (1.5%). RoPE and elementwise work complete more than 95% of
the interval after subtracting their previously measured prefill contribution.

### CPU-07 — single-token Q4_K latency dispatch

- Target: Q4_K `matmul_batched(n=1)`, about 48.5% of decode from gate/up plus
  Q4_K down projections.
- Hypothesis: route one token through the existing single-token kernel, whose
  four independent accumulators hide FMA latency. The throughput-oriented
  batched kernel deliberately carries only one accumulator per token.
- Amdahl input: 48.5% affected, with a conservative 1.15x local gain inferred
  from normalized single-token projection timings. Prediction: 1.07x decode,
  about 11.3 ms/token removable.
- Candidate: 29.93 prompt tok/s, 4,276.64 ms TTFT, and 6.63 decode tok/s; CVs
  were 1.23%, 1.24%, and 1.43%. Retained decode bookends averaged 5.81 tok/s.
- Delta: +14.11% decode throughput (21.36 ms/token removed). Prompt matches the
  original CPU-01 retained record (29.93 tok/s, 4,277.03 ms) within 0.01%, which
  is the relevant unchanged-path control across the observed session variance.
- Correctness: all four focused Q4_K scalar/SIMD/batched tests pass; the full
  engine CPU suite passes (159 unit tests, 2 ignored; 5 family-agnostic tests,
  1 ignored; 12 semantic tests, 1 ignored).
- State: KEEP; post-change decode reprofile remains.

### Reprofile after CPU-07

Public decode was 6.40 tok/s in the one-repetition diagnostic. The affected
Q4_K `n=1` matmuls fell from 2,671.41 to 2,047.46 ms across 31 tokens: 1.305x
local speedup and 623.94 ms, or 20.13 ms/token, removed. Q6_K down became the
largest remaining single candidate at 660.92 ms (about 13.6% of decode), followed
by logits and the split attention projections.

### CPU-08 — single-token Q6_K latency dispatch

- Target: Q6_K `matmul_batched(n=1)` down projections, about 13.6% of decode
  after CPU-07.
- Hypothesis: use the existing four-quant-stream accumulator latency kernel at
  one token, mirroring CPU-07. Normalized projection timings predicted little
  gain, but transfer of CPU-07's measured local factor left a 3.2% whole-decode
  ceiling, so one isolated attempt remained economically plausible.
- Candidate: 29.98 prompt tok/s, 4,270.21 ms TTFT, and 6.90 decode tok/s; CVs
  were 0.50%, 0.50%, and 0.55%. CPU-07 measured 29.93, 4,276.64, and 6.63.
- Delta: +0.17% prompt throughput, -0.15% TTFT, and +4.07% decode throughput;
  decode latency falls another 5.90 ms/token.
- Correctness: all three focused Q6_K fused/batched tests pass; the full engine
  CPU suite again passes (159 unit tests, 2 ignored; 5 family-agnostic tests,
  1 ignored; 12 semantic tests, 1 ignored).
- State: KEEP; final decode reprofile remains.

### Reprofile after CPU-08

The Q6_K `n=1` down bucket fell from 660.92 to 535.79 ms across 31 tokens:
1.233x local speedup and 125.13 ms (4.04 ms/token) removed. The final ranked
decode groups are existing-latency-kernel Q4_K work at about 44%, attention
projections at about 22%, Q6_K logits at about 11%, Q6_K down at about 11%,
RoPE near 3%, and decode attention near 2%.

### Post-specialization thread Pareto

Five cores measured 24.87 prompt tok/s, 5,147.14 ms TTFT, and 7.64 decode
tok/s. Against CPU-08's six-core 29.98/4,270.21/6.90 record, five cores gain
10.72% decode but lose 17.04% prompt throughput. State: retain the six-core
mixed-workload production policy; five cores are a documented decode-only
configuration, not a global default.

### CPU-09 — F16C only in single-token kernels

- Target: decode-only Q4_K/Q5_K/Q6_K scale widening, isolating CPU-04's decode
  gain from its batched-kernel expansion.
- Candidate: 24.12 prompt tok/s, 5,307.32 ms TTFT, and 7.50 decode tok/s. The
  restored bookend was 26.65, 4,802.86, and 6.89.
- Delta: -9.49% prompt throughput, +10.50% TTFT, and +8.85% decode throughput.
  Even decode-only inline expansion perturbs the release code layout enough to
  reproduce CPU-04's prefill regression.
- Correctness: all 13 focused quantized matmul tests passed.
- State: REJECT under the >2% regression rule. All source changes were removed;
  F16C scale-conversion variants are closed without an implementation sweep.

### CPU-10 — canonical Q4_K × Q8_K integer dot

- Target: Q4_K prefill. This replaced CPU-02's per-eight activation scales and
  four reductions per sub-block with canonical 256-value Q8_K blocks, sixteen
  precomputed sums, AVX2 `maddubs/madd`, one block conversion, and one final
  horizontal reduction.
- Amdahl prediction: 64.8% Q4_K fraction and a conservative 2x local gain would
  yield 1.48x total, well above the complex-change gate.
- End-to-end candidate: 30.20 prompt tok/s, 4,238.90 ms TTFT, and 6.66 decode
  tok/s (CVs 0.33%, 0.33%, 0.58%). The immediate restored bookend was 26.65,
  4,802.86, and 6.89, but cross-session prompt records reached about 30 tok/s.
- Direct attribution: against the stable CPU-07 profile, Q4_K `n=32` fell only
  3,007.33 -> 2,690.56 ms (1.118x local), predicting about 6.9% whole-prefill
  gain. The 3072x1024 shape regressed. This does not clear the 10% complex-path
  gate, and decode regressed 3.34% despite an unchanged decode route.
- Correctness: an `n=8`, two-super-block batched test passed against the float
  kernel within the unchanged quantized tolerance.
- State: REJECT. All Q8_K code and tests were removed. Evidence agrees with
  llama.cpp's architecture: integer arithmetic needs execution-native multi-row
  Q4_K repacking to earn its cost.

## Closed representation candidate

Evaluate execution-native Q4_K row interleaving as the last high-headroom CPU
representation frontier. Canonical Q8_K alone is below the gate, while the local
llama.cpp AVX2 comparator selects an 8-row interleaved Q4_K × Q8_K GEMM/GEMV.
A four-output-row float register block is closed without implementation: two
rows already consume nearly all YMM registers, so four rows would spill weights
and accumulators.

### CPU-11 proposed structure (recorded before implementation)

- `backend/cpu/repack.rs` (~80 productive lines): Q4_K to one 8-row execution
  layout; category K, no I/O or ownership.
- `backend/cpu/buffer.rs` (+~15 productive lines): optional immutable native
  bytes beside canonical bytes during reversible qualification.
- `backend/cpu/weights.rs` (+~15 productive lines): request native bytes only
  for Q4 matrix weights; embeddings remain canonical.
- `backend/cpu/kernels/matmul/q4k.rs` (+~70 productive lines): canonical Q8_K
  activation blocks and native-path routing.
- `backend/cpu/kernels/matmul/q4k_simd.rs` (+~120 productive lines): one AVX2
  8-output-row by up-to-4-token integer kernel; category K.

No new directory is warranted: repacking is one CPU domain file and matmul is
already a multi-file domain. Qualification retains canonical weights for an
immediate fallback; peak-RSS cost is therefore an explicit keep criterion.

### CPU-11 — 8-row execution-native Q4_K × Q8_K

- Target: the final representation frontier identified by CPU-10 and the local
  llama.cpp AVX2 comparator: interleave eight Q4_K output rows, decode scales at
  load, quantize activations to canonical Q8_K, and compute up to four tokens by
  eight output rows per tile.
- Eligibility: x86 AVX2+FMA+F16C, Q4_K matrix with output rows divisible by
  eight, and `n >= 8`; every other path retained canonical execution.
- Correctness: the native `n=8` test passed against canonical float Q4_K across
  two super-blocks and two output groups within the unchanged tolerance.
- Candidate: 22.05 prompt tok/s, 5,803.96 ms TTFT, and 6.67 decode tok/s; CVs
  were 0.79%, 0.79%, and 0.94%. The restored bookend was 26.65 prompt tok/s,
  4,802.86 ms TTFT, and 6.89 decode tok/s.
- Delta: -17.26% prompt throughput, +20.84% TTFT, and -3.19% decode. Peak RSS
  rose 4,267,960 -> 5,746,252 KiB (+1.41 GiB, +34.64%) while canonical and
  native weights coexisted for qualification.
- Diagnosis: row interleaving alone is insufficient. Pair-wise Q8 broadcasts
  and dot setup dominate this compact kernel; llama.cpp's gain depends on its
  substantially larger 4x8 microkernel and deeper activation/weight packing.
- State: REJECT after one attempt under the >2% regression rule. The new repack
  file and every storage/kernel modification were removed; no native bytes
  remain in the final process image.

## Remaining frontier

Portable batching, float SIMD, cache tiling, row blocking, thread policy,
compiler profiles, F16C specialization, canonical Q8_K, and 8-row native Q4_K
have now been exercised. The remaining comparator gap requires importing the
economics of a much larger 4x8 packed GEMM subsystem (including durable format
replacement and multi-row activation packing), not another bounded kernel
specialization. Its compact predecessor regressed 17.3% and added 1.41 GiB, so
that subsystem does not clear this repository's maintainability/economic gate.

### CPU-12 — 64-row long-prompt batch

- Target: 3B/512 prefill after the workload matrix showed llama.cpp gaining
  more from large prompt batches than this runtime.
- Hypothesis: halve the number of weight-decode passes again, from sixteen
  32-row chunks to eight 64-row chunks.
- Cache model: the largest worker tile grows from about 192 KiB to 384 KiB,
  crossing the 256 KiB private L2. This was the only wider portable tile tested.
- Candidate: 26.43 prompt tok/s, 19,370.28 ms TTFT, and 6.48 decode tok/s.
  The retained 32-row baseline was 29.66, 17,264.17 ms, and 6.51.
- Delta: -10.89% prompt throughput and +12.20% TTFT; decode was effectively
  unchanged. The L2-capacity cost exceeds the additional dequant reuse.
- State: REJECT. `CPU_ROWS=32` was restored; wider batching is closed with a
  measured 512-token result rather than the earlier cache model alone.

### CPU-13 — four-query GQA K/V reuse

- Target: F16-KV CPU attention at long context. The 3B artifact has 32 query
  heads, eight KV heads, head dimension 128, and therefore GQA group four.
- Prior route: two x2 SIMD calls per KV head widened every K/V row twice.
- Candidate route: one AVX2/FMA/F16C x4 dot and value mix widens each row once
  and feeds four independent FMA chains. Per-head max, exp, denominator,
  causal range, and accumulation order remain independent and unchanged.
- Eligibility: GQA group at least four uses x4 chunks; pair/single tails remain.
  The shared widening is selected only after AVX2+FMA+F16C detection; portable
  scalar execution delegates to the existing single-head operations.
- Traffic model at 8K: logical K+V consumption falls 7.15 TB -> 3.57 TB
  (6.50 -> 3.25 TiB) across 26 layers. The retained path sustains about
  11.1 logical GB/s including widening and arithmetic.
- Amdahl prediction from the paired 2K profile: attention local speedup 1.306x.
  Applied to the x2 8K attention fraction of 57.44%, this predicts 1.155x total
  and about 116 seconds removable.
- Paired 2K profile: x2 attention 14,002.43 ms; x4 10,725.87 ms, 1.306x local
  and 3,276.55 ms removed. Same-session public TTFT was 78,996.56 ->
  76,668.00 ms (-2.95%), at the prototype boundary.
- Paired 8K profile: x2 attention 495,842.74 ms; x4 321,626.42 ms, 1.542x
  local and 174,216.32 ms removed. Accounted time fell 863,258.81 ->
  663,336.29 ms; public TTFT fell 863,100.75 -> 663,221.90 ms (-23.16%).
  Prompt throughput rose 9.49 -> 12.35 tok/s (+30.14%); one-delta decode rose
  3.20 -> 3.81 tok/s (+19.06%). Peak RSS stayed 4,268,068 KiB and neither run
  incurred a major fault.
- Correctness: group-four decode and batched-prefill outputs match the serial
  reference; `n=1` prefill matches decode; direct x4 AVX2 dot/value-mix tests
  pass at head dimension 128. The full CPU suite and Clippy pass unchanged.
- State: KEEP. Commit `80eadd2`; reusable by ISA capability and GQA shape, not
  by model name.

## Final workload matrix

Graph Horizon uses chat-tokenized prompts and reports end-to-end TTFT. llama.cpp
uses synthetic prompt tokens and reports isolated prompt evaluation, so the
latency ratios include that fixed boundary mismatch. Both use the same GGUF,
six pinned cores, F16 KV, no GPU layers, and no flash attention. `M` is a
measured repeated result, `P` is a paired one-sample profiled result, and `F` is
an explicit diagnostic fit rather than a benchmark.

### 3B prefill

| Tokens | Graph Horizon TTFT | Ours tok/s | llama.cpp time | llama tok/s | Latency ratio | Absolute gap | Evidence |
|---:|---:|---:|---:|---:|---:|---:|:---:|
| 128 | 5.025 s | 25.48 | 0.317 s | 403.52 | 15.83x | 4.707 s | M, CV 0.90% |
| 512 | 17.629 s | 29.04 | 0.487 s | 1,050.66 | 36.17x | 17.141 s | M, CV 0.80% |
| 2,048 | 96.000 s | 21.33 | 2.385 s | 858.78 | 40.26x | 93.615 s | M, CV 0.12% |
| 8,192 | 663.222 s | 12.35 | 12.831 s | 638.46 | 51.69x | 650.391 s | P |
| 16,384 | 2,051.04 s | 7.99 | 35.886 s | 456.56 | 57.15x | 2,015.15 s | F |
| 28,672 | 5,486.03 s | 5.23 | 80.900 s | 354.41 | 67.81x | 5,405.13 s | F |

The long bounds fit `T(N) = -2.857 + 0.037257*N + 5.377e-6*N^2` seconds to
the final 512/2K/8K rows. A single measured 16K or 28K Graph Horizon pass would
take approximately 34 or 91 minutes under that fit, so those cells are not
presented as statistically qualified measurements. All llama.cpp rows are
measured with three repetitions; its reported prompt CV is below 2.8% at 128
and below 0.6% from 512 onward.

### 3B decode by populated KV length

| KV tokens | Graph Horizon tok/s | llama.cpp tok/s | Latency ratio | Absolute gap | Evidence |
|---:|---:|---:|---:|---:|:---:|
| 128 | 6.79 | 10.312 | 1.519x | 50.30 ms/token | M, CV 0.55% |
| 2,048 | 5.73 | 8.957 | 1.563x | 62.87 ms/token | M, CV 1.10% |
| 28,672 | 1.80 | 2.718 | 1.507x | 186.69 ms/token | F |

Measured Graph Horizon points at KV 128/512/2K/8K fit
`T_decode(N) = 145.54 + 0.014268*N` ms/token; only the 28K Graph Horizon row is
derived from that fit. llama.cpp decode is measured with `-d` cache seeding, so
its timed samples exclude the prefill just as Graph Horizon's decode delta does.

### Model-size coverage

| Model | Workload | Graph Horizon | llama.cpp | Ratio | Capacity note |
|---|---|---:|---:|---:|---|
| 3B | prefill 128 | 25.48 tok/s | 403.52 tok/s | 15.83x latency | 4,268,016 KiB RSS |
| 3B | decode KV128 | 6.79 tok/s | 10.312 tok/s | 1.519x latency | repeated |
| 8B | prefill 128 | 12.42 tok/s | 181.76 tok/s | 14.63x latency | clean warmed reference, 10,228,408 KiB RSS |
| 8B | decode KV128 | 3.06 tok/s | 4.868 tok/s | 1.591x latency | clean warmed reference |
| 14B | 128 | not run | not run | n/a | projected ~16.2 GB runtime footprint exceeds 14 GiB RAM |

The final 8B rerun was internally stable but incurred 75,614 major faults and
4.15 GB of file input under host memory pressure; it is retained only as a
capacity diagnostic (11.66 prompt, 3.03 decode), not as the qualified row.
Running 14B through swap would measure page replacement rather than CPU kernels.

## Final attribution and global rerank

The retained 8K profile accounts for 663,336.29 ms and names more than 99%:

| Component | Time ms | Fraction | Next realistic class |
|---|---:|---:|---|
| F16 GQA attention | 321,626.42 | 48.49% | query-row tiling / CPU-native KV layout |
| Q4_K batched matmuls | 237,460.77 | 35.80% | full packed 4x8 GEMM subsystem |
| Q6_K batched matmuls | 51,895.29 | 7.82% | same packed subsystem |
| RoPE | 38,906.41 | 5.87% | prior bounded variant was only 2.72% total |
| norm, activation, residual, KV, logits, embedding | 13,447.40 | 2.03% | below current economic gates |

At 128, quantized matmul remains the dominant ~78.5% component and attention is
small. At 8K, x4 attention is still the largest component, followed closely by
quantized matmul. This is the final model-by-workload ranking; no significant
cost is hidden in `other`.

## Final performance against production baseline

Only the production-baseline 3B/128 row was captured before the investigation.
It improved from 18.21 -> 25.48 prompt tok/s (+39.92%), 7,030.85 -> 5,024.78 ms
TTFT (-28.53%), and 5.71 -> 6.79 decode tok/s (+18.91%, 27.86 ms/token removed).
The paired pre-x4 8K checkpoint improved 863,100.75 -> 663,221.90 ms, but it is
an intermediate retained-branch baseline, not the original production commit.

Retained reasons:

- 32-row prefill: eightfold more weight-decode reuse than four rows while the
  ~192 KiB worker tile remains inside L2; measured 39.17% TTFT reduction in its
  qualification session.
- Q4_K/Q6_K single-token routing: four independent accumulator chains hide FMA
  latency; local gains were 1.305x and 1.233x, removing 20.13 and 4.04 ms/token.
- Four-query attention: halves logical K/V reads and F16C widenings versus two
  pair calls; measured 1.542x local and 23.16% whole-workload gain at 8K.

## Remaining practical floors

For the x4 8K attention bucket, the measured x2-minus-x4 difference assigns
174.22 s to the eliminated second K/V widening/read and about 147.41 s to the
remaining FMA, softmax, traversal, and one mandatory widening. A two-query-row
tile has a best-case attention time near 234.52 s (1.371x local) if it halves
the remaining widening for free. That would be 1.151x end-to-end, but an x8
kernel also halves parallel units, needs eight query/output chains, adds causal
tail handling, and either broadcasts eight weights or spills registers. A
realistic 1.2x local outcome is only 1.088x end-to-end (about 53.6 s removable),
below the 10% gate for that orchestration and numerical-order change.

For short-context matmul, canonical Q8_K reached only 1.118x local and predicted
6.9% total; the native eight-row prototype regressed 17.26% and added 1.41 GiB.
A credible comparator-class gain requires durable repacking plus a large 4x8
GEMM/GEMV family, not another bounded kernel edit.

The only remaining designs with double-digit theoretical headroom are therefore:

| Candidate family | Theoretical gain | Practical estimate | Decision |
|---|---:|---:|---|
| query-row-tiled/online CPU attention | up to 15.1% for a two-row tile; larger for a full flash tile | ~8.8% for the bounded x8 design | STOP: subsystem/scheduling and arithmetic-order cost exceeds bounded estimate |
| durable packed 4x8 Q4/Q6 GEMM/GEMV | comparator shows very large upper bound | compact native predecessor -17.3%, +1.41 GiB | STOP: requires a separate packed-math subsystem, not an economic incremental change |
| RoPE coefficient reuse | 11.1% predicted from stale bucket | 2.72% measured | STOP: below gate |
| 64-row batching | extra 2x decode reuse | -10.9% measured at 512 | STOP: L2 capacity exceeded |
| thread-policy specialization | decode +10.7% at five cores | prefill -17.0% | STOP: Pareto option only, not mixed default |

## Qualification and backend isolation

- CPU: 159 unit tests passed, two authenticated tests ignored; five
  family-agnostic tests passed, one authenticated test ignored; twelve semantic
  tests passed, one authenticated test ignored; doc tests passed.
- Focused group-four attention tests cover decode, prefill, `n=1`, causal range,
  serial parity, AVX2 dot, AVX2 value mix, and 128-wide SIMD execution.
- Workspace CPU Clippy passes with `-D warnings`; formatting and `git diff
  --check` pass.
- Vulkan and Vulkan-hybrid workspace checks pass. Metal correctly stops at its
  explicit Apple-Silicon build guard on this Linux/x86 host. No GPU source,
  shader, scheduling, resource, public API, dependency, or numerical contract
  changed.
- The repository's external parity runner requires llama.cpp `13f2b28b0`; the
  available comparator is `9bebfcb4b`, so the runner reports `unsupported
  llama.cpp revision`. Existing CPU reference/semantic tests remain the quality
  oracle; tolerances were not changed.

## CPU global stop

The performance contract is not met: prefill remains 15.8x to 51.7x slower in
measured cells, while decode remains about 1.5x slower. The stop is instead
economic: routing, batching, SIMD width, compiler profiles, thread policy,
canonical integer dot, native weight representation, cache working sets, GQA
reuse, and long-context attribution were exercised. Every remaining bounded
candidate is below its complexity gate; the remaining large headroom belongs to
new packed-GEMM or query-tiled-attention subsystems whose smallest credible forms
do not clear the measured practical gate in this repository. Further progress
requires approving one of those subsystems as a new scope, or changing model
quantization/semantics or hardware. No microarchitecture product-name routing is
retained.
