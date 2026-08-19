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

## Next candidate

Audit compiler output and the existing Q4_K float kernel's practical compute
floor before deciding whether an execution-native packed representation clears
the economic gate. Q6_K and RoPE remain globally ranked behind Q4_K.
