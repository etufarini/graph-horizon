# CUDA Amdahl campaign 2026-09-08 B

## Recovery state

- Campaign ID: `cuda-amdahl-20260908b`.
- Branch: `perf/cuda-amdahl-20260908b`.
- Immutable start: `46f9277b0909ca4d63a41be9a805c4aef878ba67`.
- Current retained checkpoint: the immutable start; no production candidate is applied.
- State: C01/C02 interesting and C03–C05 rejected, all restored; C06 selected.
- Attempts: 5 of at least 10 new countable attempts.
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

## Baseline timeline and critical path

Nsight Systems captured CUDA API and GPU activity without CPU sampling. The
host importer converted the original `.qdstrm` files to `.nsys-rep` and SQLite
under the campaign directory; no workload was repeated for conversion. The
profiled prompt rates differ from the unprofiled means by -0.56%, -0.94%, and
-0.13% for short, medium, and long, so the timelines are representative
diagnostics but the unprofiled public records remain authoritative.

| Regime | Prefill span ms | Tensor matmul ms | Tensor attention ms | Decode span ms | Decode matmul ms | Decode attention ms | Prefill gap ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| Short | 463.300 | 442.126 | 0 | 631.123 | 497.304 | 28.074 | 0.348 |
| Medium | 3,966.649 | 3,556.723 | 203.235 | 654.221 | 499.097 | 44.256 | 6.201 |
| Long | 16,066.583 | 12,525.621 | 3,167.210 | 760.612 | 501.799 | 149.485 | 19.256 |

All request work uses one stream. Long prefill is 99.88% attributed to kernels;
the three tensor matmul entries own 77.96% of prefill and tensor attention owns
19.71%. Synchronization time is host waiting on the same GPU critical path, not
additional removable time. There is no request-time host-to-device transfer.
Host-gap, transfer, and command-submission mechanisms therefore cannot reach
the retention threshold.

The trace reports 128-thread tensor matmul blocks with 70/96/72 registers per
thread and 9,792/14,976/19,584 bytes shared for ordinary/wide/paired entries.
Tensor attention uses 64 registers and 13,408 bytes shared. Baseline PTX uses
scalar `ld.global.u16` and `st.shared.u16` for both hot tensor families and no
32-bit load/store equivalents. A targeted Nsight Compute run on one proven-hot
wide launch returned `ERR_NVGPUCTRPERM`; no kernel was replayed and that run is
not performance evidence. Static launch records, PTX, and isolated causal A/B
experiments are the available counter path.

## Candidate pool

The pool is ranked from long-prefill critical-path time, static launch
resources, baseline PTX, and historical closures. `p` is full-request / target
phase. Predicted gain and ceiling apply to the declared phase objective;
absolute saved time uses the conservative local speedup. Predictions are
working bounds, not measured results.

| ID | Premise and one intentional variable | Objective; controls | Full / phase `p` | Credible `s`; `o` | Predicted gain; ideal ceiling; saved | Cost, numeric risk, uncertainty | State |
|---|---|---|---:|---:|---:|---|---|
| C01 | Replace scalar half activation staging in compute-7.5 tensor matmul with aligned half-pair load/store | Long prompt; short/medium and decode/TTFT controls | 0.74437 / 0.77961 | 1.05; 0.001 | 3.75%; 353.74%; 596.5 ms | Low; bit-copy exact; measured local equivalent about 1.0425 | interesting, removed: +3.282% |
| C02 | Add a full-tile tensor-matmul entry for divisible token/output grids, removing repeated row/output tail predicates | Long prompt; short/medium and decode/TTFT controls | 0.74437 / 0.77961 | 1.05; 0.002 | 3.64%; 353.74%; 596.5 ms | Medium; exact; extra entry/dispatch and instruction-share uncertainty | interesting, removed: +3.457% |
| C03 | Read two staged half activations at a time while preserving left-to-right row-sum additions | Long prompt; short/medium and decode/TTFT controls | 0.74437 / 0.77961 | 1.03; 0.001 | 2.22%; 353.74%; 364.8 ms | Low; exact if add order remains fixed; shared-load share uncertain | rejected, removed: +1.755% |
| C04 | Vectorize tensor-attention K/V global-to-shared staging as aligned half pairs | Long prompt; short/medium and decode/TTFT controls | 0.18822 / 0.19713 | 1.08; 0.001 | 1.38%; 24.55%; 234.6 ms | Low; bit-copy exact; repeated history loads make it diagnostic | rejected, removed: +1.908% |
| C05 | Keep eight queries per tensor-attention block but use 256 threads so two 128-thread groups split PV ownership and tile staging | Long prompt; short/medium and decode/TTFT controls | 0.18822 / 0.19713 | 1.15; 0.003 | 2.32%; 24.55%; 413.1 ms | Medium; exact tree per output; occupancy and mapping risk | rejected, removed: +0.202% |
| C06 | Parallelize each tensor-attention 16-score softmax across a fixed lane group | Long prompt; short/medium and decode/TTFT controls | 0.18822 / 0.19713 | 1.15; 0.002 | 2.43%; 24.55%; 413.1 ms | Medium; reordered f32 reductions require bounded numeric gate | ready, selected |
| C07 | Route packed decode matmul to two output warps per 64-thread block | Short model decode; prompt/TTFT plus medium/long controls | 0.45440 / 0.78797 | 1.05; 0.001 | 3.79%; 371.62%; 23.7 ms | Low; exact per-warp dot; smaller blocks may improve scheduling or add grid cost | ready |
| C08 | Route packed decode matmul to three output warps per 96-thread block | Short model decode; prompt/TTFT plus medium/long controls | 0.45440 / 0.78797 | 1.04; 0.001 | 3.02%; 371.62%; 19.1 ms | Low; exact per-warp dot; complements the historical four/eight-warp evidence | deferred after C07 |
| C09 | Vectorize cached-Q6 integer weight staging into adjacent pairs without changing coefficient or MMA order | Long prompt; short/medium and decode/TTFT controls | 0.74437 / 0.77961 upper bound | 1.04; 0.002 | 2.88%; 353.74%; 481.8 ms upper bound | Medium; exact; Q6 share and conversion-codegen benefit need C01 PTX refresh | deferred |
| C10 | Remove host launch/synchronization gaps | Long prompt | <=0.00120 | unbounded; 0 | <=0.12%; <=0.12%; <=19.3 ms | Ideal ceiling below 5% | closed-untried |
| C11 | Fuse only residual/SILU pointwise launches around matmul | Long prompt | <=0.00283 | unbounded; 0 | <=0.28%; <=0.28%; <=45.4 ms | Measured owner excludes unproven matmul-store savings; present ceiling below 5% | closed-untried |

C06 is selected after C05 because eight threads still execute each 16-score
softmax serially. A fixed lane group will parallelize maximum and denominator
work, intentionally changing their f32 reduction trees. Its predeclared numeric
gate is every existing attention reference/tail test, finite half outputs, exact
causal masking, and real-model parity; any failed bound terminates the candidate
before performance.

## Candidate decisions

### C01 — aligned half-pair activation staging

C01 changed only compute-7.5 tensor-matmul activation staging from scalar
16-bit copies to aligned 32-bit pairs. Fresh exact captures covered ordinary,
wide, and paired paths and produced 2,730,004 bytes of output vectors with the
same SHA-256 before and after:
`e7f791637e061a7af0192992b70649844dbc4dc585032253eda4fa864dc22473`.
Formatting, the CPU workspace, CUDA workspace check, all 11 error-matrix tests,
all 44 CUDA tests, focused compute-sanitizer memcheck with zero errors, and
canonical parity passed before performance. All 16 local top-1 IDs equaled the
oracle.

Candidate PTX replaced most scalar activation movement: ordinary/wide/paired
`ld.global.u16` counts fell from 33/33/36 to 5/5/8 while 16/28/28 32-bit loads
appeared. The stable objective result was:

| Metric | Baseline A | Candidate B | Change |
|---|---:|---:|---:|
| Long prompt t/s | 223.31 (CV 0.0053) | 230.64 (CV 0.0035) | +3.2824% |
| Long TTFT ms | 16,049.70 | 15,539.69 | -3.1777% |
| Long model decode t/s | 42.46 | 42.92 | +1.0834% |
| Long public delta t/s | 39.79 | 40.22 | +1.0807% |

Terminal state: `interesting`, below the 5% keep threshold. No stability rerun
or controls apply. The production change and temporary vector capture were
removed with an empty worktree diff. This is one countable attempt. The measured
effect promotes a later bounded variant that also removes scalar row-sum reads;
it does not make the rejected partial implementation retainable.

### C02 — full tensor-matmul tiles

C02 added separate compute-7.5 entries for ordinary, wide, and paired grids
whose row and output dimensions are complete tiles; partial grids continued to
use the inherited entries. Fresh full-tile captures plus the inherited tail
matrix produced 2,662,474 bytes of selected output vectors with the same
SHA-256 before and after:
`f2d6ce5e165104186327897403105ac179420542399028a5718243237114e4c2`.
Formatting, the CPU workspace, CUDA workspace check, all 11 error-matrix tests,
all 44 CUDA tests, focused compute-sanitizer memcheck with zero errors, and
canonical parity passed. All 16 local top-1 IDs equaled the oracle.

| Metric | Baseline A | Candidate B | Change |
|---|---:|---:|---:|
| Long prompt t/s | 223.31 (CV 0.0053) | 231.03 (CV 0.0061) | +3.4571% |
| Long TTFT ms | 16,049.70 | 15,513.82 | -3.3389% |
| Long model decode t/s | 42.46 | 42.67 | +0.4946% |
| Long public delta t/s | 39.79 | 39.98 | +0.4775% |

Terminal state: `interesting`, below the 5% keep threshold. No stability rerun
or controls apply. The extra private entries, dispatch, and temporary vector
capture were removed; the worktree returned to the retained checkpoint. This
is the second countable attempt.

### C03 — paired tensor-matmul row-sum reads

C03 replaced adjacent scalar shared reads in the tensor-matmul row-sum loop
with one aligned 32-bit read while retaining two left-to-right half-to-float
additions. Fresh ordinary, paired, and wide captures produced 2,724,564 bytes
with the same SHA-256 before and after:
`6d50118e48c307de78e9bfb940efd2f4cbe37b639922dfd6a3afb662894fdb5a`.
All standard gates, focused memcheck with zero errors, and exact 16-token parity
passed. PTX confirms the intended mechanism: scalar shared-load counts for the
three entries changed from 96/96/128 to zero and 32-bit counts became
48/48/64.

| Metric | Baseline A | Candidate B | Change |
|---|---:|---:|---:|
| Long prompt t/s | 223.31 (CV 0.0053) | 227.23 (CV 0.0071) | +1.7554% |
| Long TTFT ms | 16,049.70 | 15,773.20 | -1.7228% |
| Long model decode t/s | 42.46 | 42.59 | +0.3062% |
| Long public delta t/s | 39.79 | 39.93 | +0.3518% |

Terminal state: `rejected`, below 3%. No stability rerun or controls apply. The
production and capture changes were removed, returning to the retained
checkpoint. This is the third countable attempt; the result closes row-sum load
width as an independent retention path.

### C04 — aligned half-pair tensor-attention staging

C04 replaced adjacent scalar F16 K/V copies in tensor attention with aligned
32-bit copies. The fresh tensor-route capture held 1,024 output values and had
the same SHA-256 before and after:
`95a062dd8e1d7fc768f0fca95063328a116fc43e33593aa7936800286b9e09c0`.
All standard gates, focused memcheck with zero errors, and exact 16-token parity
passed. PTX verifies that global/shared scalar half counts changed from 15/15
to 1/1 while 32-bit counts became 14/16.

| Metric | Baseline A | Candidate B | Change |
|---|---:|---:|---:|
| Long prompt t/s | 223.31 (CV 0.0053) | 227.57 (CV 0.0056) | +1.9077% |
| Long TTFT ms | 16,049.70 | 15,749.46 | -1.8707% |
| Long model decode t/s | 42.46 | 42.57 | +0.2591% |
| Long public delta t/s | 39.79 | 39.90 | +0.2765% |

Terminal state: `rejected`, below 3%. No stability rerun or controls apply. The
production and capture changes were removed. This is the fourth countable
attempt and closes scalar K/V staging width as an independent retention path.

### C05 — two-group tensor-attention ownership

C05 doubled only tensor-attention blocks to 256 threads. Two 128-thread groups
owned four queries each, reducing per-thread accumulators from eight to four and
splitting Q/K/V staging while preserving each output's arithmetic order. Its
fresh 1,024-value capture matched exactly before and after with SHA-256
`c76a0507f14c5cb2f35e4e8d5fa129a227d3e7b78c9d00851aef80ba9cad7736`.
All standard gates, exact parity, memcheck with zero errors, and racecheck with
zero hazards passed.

| Metric | Baseline A | Candidate B | Change |
|---|---:|---:|---:|
| Long prompt t/s | 223.31 (CV 0.0053) | 223.76 (CV 0.0042) | +0.2015% |
| Long TTFT ms | 16,049.70 | 16,017.53 | -0.2004% |
| Long model decode t/s | 42.46 | 42.29 | -0.4004% |
| Long public delta t/s | 39.79 | 39.65 | -0.3518% |

Terminal state: `rejected`, below 3%. No stability rerun or controls apply. The
production and capture changes were removed. This is the fifth countable
attempt and closes a 256-thread query-ownership split in this geometry.

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
CUDA workspace check, baseline build, stable public screen, exact-token oracle
parity, three timelines, critical-path attribution, PTX inspection, and the
targeted counter attempt completed. C01 through C05 completed and were
restored. C06 is predeclared; no production edit is currently applied.
