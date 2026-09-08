# CUDA Amdahl campaign 2026-09-08 B

## Recovery state

- Campaign ID: `cuda-amdahl-20260908b`.
- Branch: `perf/cuda-amdahl-20260908b`.
- Immutable start: `46f9277b0909ca4d63a41be9a805c4aef878ba67`.
- Current retained checkpoint: the immutable start; no production candidate is applied.
- State: baseline complete; CPU/GPU timeline acquisition in progress.
- Attempts: 0 of at least 10 new countable attempts.
- Local evidence: `benchmarks/cuda-amdahl-20260908b/`.
- Deadline: none; each A/B comparison retains the canonical two-hour limit.

CUDA is selected because the request names exactly that backend. The host is
Linux `x86_64`, CUDA is build-supported, visible ordinal 0 is idle, and the
worktree was clean before this branch was created.

## Fixed tuple and gates

The campaign uses the authenticated `3b-instruct` Q4_K_M artifact, standalone
`cuda` placement on visible ordinal 0, context 4096, f16 KV, greedy sampling,
Cargo `fast`, one warm-up, and three measured repetitions. The public screen
uses 128, 1,024, and 3,584 actual prompt tokens with 32 requested completion
tokens. Artifact bytes, prompt bytes, context, KV, placement, sampling, build,
runtime options, hardware, and environmental controls remain identical across
each A/B; only the declared candidate and revision may differ.

The initial public benchmark and CPU/GPU timeline select the largest measured
fixed-work end-to-end bottleneck among prefill, TTFT, and decode. Keep requires
at least 5% objective improvement, objective CV at most 5% in A and B,
correctness success, and no control regression above 5%. A stable 3–5% signal
is interesting and removed; below 3% rejects. One complete rerun is permitted
only when an objective CV exceeds 5%.

Before performance, every candidate runs formatting, CPU workspace tests, the
CUDA workspace check, CUDA error tests, applicable exact or risk-specific
tests, and authenticated teacher-forced real-model parity. Numeric candidates
must predeclare a bounded gate that covers their risk. Rejected production code
is restored without rewriting history. Only one GPU measurement runs at a time.

## Preflight

| Field | Value |
|---|---|
| Host | Linux `x86_64`; compute capability 7.5; 4 GiB VRAM; visible ordinal 0 |
| Driver / toolkit | 595.84 / CUDA Toolkit 12.4.131 |
| Rust / Cargo | 1.95.0 / 1.95.0 |
| Timeline / counters | Nsight Systems 2023.4.4 / Nsight Compute 2024.1.1 |
| Model catalog ID | `3b-instruct`, Q4_K_M |
| Expected bytes | 2,147,023,008 |
| Expected SHA-256 | `9ed150d4367e68df0ac8e1540f6ddc65b42d0ee26378329d1ecbca60f93fc5f8` |
| Build | `--locked --profile fast --no-default-features --features cuda` |

The artifact under the campaign input directory matches `support/models.tsv`
in byte size and SHA-256. Exact device identity and live telemetry remain in
ignored raw logs so tracked documentation stays hardware-neutral. The pinned
reference oracle is llama.cpp revision
`13f2b28b098623391b1aacfd27995e1c8b7de9a9`; its availability and parity row
will be verified before the first candidate benchmark.

## Historical evidence

The completed `cuda-amdahl-optimization`, `cuda-amdahl-20260905`, and
`cuda-amdahl-20260908` reports are inherited evidence, not current attempts.
The starting revision already contains their retained scheduling, packed
matmul, attention, normalization, sampling, cache, batching, and row-batched
RoPE changes. The most recent compute-7.5 campaign falsified M32-to-M16 routing,
ordinary replacement for paired staging, 64/96-row batches, shared K/V
attention storage, 16-query attention ownership, 32/128-column tensor output
tiles, and eight-warp decode blocks. Its 12-query attention result was
interesting at +3.20% but below the keep threshold. These premises are closed
unless a new measurement changes one of their assumptions.

## Candidate pool

The active pool will be populated and ranked after the new timeline,
critical-path attribution, and targeted counter or causal evidence. No
historical candidate will be counted as a new attempt. The public screen makes
long prefill the initial diagnostic priority: its 16.05 s TTFT dominates the
approximately 0.75 s required for 32 model decode tokens.

## Authenticated prompts and baseline

Repeating `benchmark` 124, 1,020, and 3,580 times with a final period produced
exactly 128, 1,024, and 3,584 rendered prompt tokens with the current engine
tokenizer. The immutable baseline executable SHA-256 is
`fe03b2f5b025fb8132a86315e1f6e487e2b5baaa8b68b1a5f3dc1c96dda194b6`.

```sh
bench-46f9277 MODEL --context 4096 --kv f16 --prompt PROMPT \
  --max-tokens 32 --warmup 1 --reps 3
```

| Regime | Prompt tokens | Prompt t/s mean (CV) | TTFT ms mean (CV) | Model decode t/s mean (CV) | Public delta t/s mean (CV) | Complete wall s |
|---|---:|---:|---:|---:|---:|---:|
| Short | 128 | 276.33 (0.0015) | 463.22 (0.0015) | 52.03 (0.0016) | 50.39 (0.0013) | 5.28 |
| Medium | 1,024 | 260.43 (0.0000) | 3,932.00 (0.0000) | 50.18 (0.0009) | 47.01 (0.0012) | 19.16 |
| Long | 3,584 | 223.31 (0.0053) | 16,049.70 (0.0053) | 42.46 (0.0063) | 39.79 (0.0064) | 67.88 |

All objective and throughput-control CVs are below 5%, so no stability rerun
applies. For fixed work, 32 model decode tokens take approximately 0.615,
0.638, and 0.754 seconds. Prompt/prefill therefore owns about 43%, 86%, and
96% of TTFT-plus-decode elapsed time in short, medium, and long respectively.

The canonical baseline parity gate passed with all 16 local top-1 IDs exactly
equal to the pinned oracle IDs. Two earlier invocations used relative model or
server arguments and failed before meaningful parity; their raw logs are kept
as invocation diagnostics and are not correctness results.

## Commands and results

Preflight commands completed:

```text
git status --short --branch
git rev-parse HEAD
uname -s
uname -m
nvidia-smi --query-gpu=...
rustc --version
cargo --version
nvcc --version
nsys --version
ncu --version
sha256sum MODEL
```

Results: clean immutable start; supported CUDA host; visible ordinal 0 idle;
model byte size and digest match the catalog; all three prompts authenticated;
CUDA workspace check, baseline build, stable public screen, and exact-token
oracle parity passed. The profiler funnel remains in progress.
