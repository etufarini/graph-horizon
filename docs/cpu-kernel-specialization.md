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

## Next candidate

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
