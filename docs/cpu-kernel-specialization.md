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
- State: pending.

## Next candidate

Measure CPU-01 immediately. If retained, reprofile before considering the
Q4_K/Q6_K activation-quantized integer-dot architecture.
