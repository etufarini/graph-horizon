# CUDA Amdahl campaign 2026-09-08

## Recovery state

- Campaign ID: `cuda-amdahl-20260908`.
- Branch: `perf/cuda-amdahl-20260908`.
- Immutable start and current retained checkpoint: `f6e481c27f6d9096ad330c6c84657060b193b5af`.
- State: baseline and ranked pool complete; C01 rejected; C02 selected; no
  production candidate is applied.
- Attempts: 1 of the required 10 new countable attempts.
- Private evidence: `target/cuda-amdahl-20260908/`.
- Deadline: none; every A/B comparison retains the canonical two-hour limit.

The worktree was clean on `main` before this branch was created. The selected
backend is CUDA because the request names it explicitly and the current Linux
`x86_64` host satisfies the build-platform boundary.

## Fixed invariants and decision gates

The campaign uses the authenticated `3b-instruct` Q4_K_M artifact, standalone
`cuda` placement on visible device ordinal 0, context 4096, f16 KV, greedy
sampling, the fast Cargo profile, one warm-up, and three measured repetitions.
The short, medium, and long screen targets approximately 128, 1,024, and 3,584
actual prompt tokens with 32 requested completion tokens; prompts will be
calibrated with the engine tokenizer before measurement.

The initial public benchmark and CUDA timeline select the largest fixed-work
critical path among prompt throughput, TTFT, and model decode. Candidate and
baseline may differ only by the declared implementation. A candidate is kept
only when its objective improves by at least 5%, objective CV is at most 5% in
both records, correctness passes, and no control regresses by more than 5%.
Three-to-five percent is interesting evidence whose production code is removed;
less than 3% rejects. One complete rerun is allowed only for an objective CV
above 5%.

Every candidate runs the CPU workspace tests, a matching CUDA feature check,
CUDA local error tests, risk-specific exact or numeric tests, and all available
real-model checks before performance. The campaign built the reference server
at the pinned llama.cpp revision `13f2b28b098623391b1aacfd27995e1c8b7de9a9`;
teacher-forced parity is therefore an active exact-token gate.

## Environment and artifact preflight

| Field | Value |
|---|---|
| Host | Linux `x86_64`; CUDA compute capability 7.5; 4 GiB VRAM; ordinal 0 |
| Driver / toolkit | 595.84 / CUDA Toolkit 12.4.131 |
| Rust / Cargo | 1.95.0 / 1.95.0 |
| Model catalog ID | `3b-instruct`, Q4_K_M |
| Expected bytes | 2,147,023,008 |
| Expected SHA-256 | `9ed150d4367e68df0ac8e1540f6ddc65b42d0ee26378329d1ecbca60f93fc5f8` |
| Model source | Official catalog repository, immutable revision `fc774f009f0c62a186f48e870fd6295b36f63779` |
| Build | `--locked --profile fast --no-default-features --features cuda` |

The physical device name and live telemetry stay in private logs so tracked
documentation remains hardware-neutral.

Authentication passed before inference:

```text
2147023008
9ed150d4367e68df0ac8e1540f6ddc65b42d0ee26378329d1ecbca60f93fc5f8
```

The official artifact matches `support/models.tsv` exactly. Repeating
`benchmark` 124, 1,020, and 3,580 times with a final period produced exactly
128, 1,024, and 3,584 rendered prompt tokens with the current engine tokenizer.

## Historical evidence

The completed `cuda-amdahl-20260905` campaign and
`cuda-amdahl-optimization` are inherited evidence, not current attempts. The
baseline already includes their retained commits through `475f883`, including
128-row standalone prefill, row-batched RoPE, packed/tensor matmul variants,
parallel normalization and attention, buffered long-history decode, and exact
argmax. Their rejected variants are not retried unless the compute-7.5 timeline
changes a measured premise.

## Candidate pool

The initial pool is ranked from the public screen, Nsight Systems intervals,
static launch resources, and source/PTX inspection. Full-request `p` uses
prefill plus 32-token decode; phase `p` uses the candidate's declared
objective. Predictions are conservative working estimates, not measurements.

| ID | Premise and intentional variable | Objective | Full / phase `p` | Credible `s`; `o` | Predicted gain; ideal ceiling | State |
|---|---|---|---:|---:|---:|---|
| C01 | On compute 7.5, route the 9,216-output M32 tensor kernel to M16; 96 registers and 14,976 B shared fall to 70 and 9,792 | Medium prompt | 0.3005 / 0.3496 | 1.30; 0 | 7.45%; 42.96% | rejected: -7.47% objective |
| C02 | On compute 7.5, route paired K stages to ordinary M16; shared allocation falls 19,584 -> 9,792 B | Medium prompt | 0.2281 / 0.2654 | 1.20; 0 | 3.95%; 29.55% | ready, selected |
| C03 | Bound compute-7.5 standalone prefill batches at 64 rows to reduce large-tile resource waves | Medium prompt | 0.8597 / 1.0000 | 1.08; 0.003 | 6.46%; 612.8% | deferred after C01/C02 |
| C04 | Use a 96-row compute-7.5 batch as the smaller launch-count variant if C03 exposes a batching/resource crossover | Medium prompt | 0.8597 / 1.0000 | 1.04; 0.0015 | 3.26%; 612.8% | deferred after C03 |
| C05 | Reuse one shared K/V tile sequentially in tensor attention, trading one barrier for lower shared residency | Long prompt | 0.1889 / 0.1978 | 1.25; 0.003 | 3.60%; 23.29% | ready |
| C06 | Fill all 16 query rows already computed by Turing WMMA instead of discarding half of the QK tile | Long prompt | 0.1889 / 0.1978 | 1.35; 0.005 | 4.60%; 23.29% | ready |
| C07 | Stage 32 history positions per tensor-attention iteration after C05 lowers shared pressure | Long prompt | 0.1889 / 0.1978 | 1.15; 0.004 | 2.10%; 23.29% | deferred after C05 |
| C08 | Use 128 output columns per tensor block to halve duplicated input staging and row sums | Medium prompt | 0.7705 / 0.8966 | 1.10; 0 | 7.53%; 335.7% | deferred after C01/C02 |
| C09 | Use 32 output columns per tensor block to increase residency when 64-column shared tiles remain limiting | Medium prompt | 0.7705 / 0.8966 | 1.07; 0 | 5.31%; 335.7% | deferred after C01/C02 |
| C10 | Pack eight independent quantized decode output warps in a 256-thread block, halving block scheduling without changing dot order | Short model decode | 0.4506 / 0.7813 | 1.08; 0 | 6.16% phase; 82.0% full ceiling | deferred after prefill leaders |
| C11 | Move tensor-attention threshold from base 512 to 384 | Long prompt | 0.0037 / 0.0038 | unbounded; 0 | <=0.37%; <=0.37% | closed-untried: ideal ceiling below 5% |
| C12 | Remove host/GPU gaps from medium prefill | Medium prompt | 0.0010 / 0.0011 | unbounded; 0 | <=0.10%; <=0.10% | closed-untried: timeline is already dense |

C01 instead cost 312.1 ms on the medium request: the measured launch-amortization
loss outweighed its lower static resource use. C02's estimate remains 171.3 ms.
The attention candidates own 3,103.0 ms of long prefill, while threshold tuning
owns only one 60.2 ms fallback batch. The pool is reranked after every result;
deferred is not a terminal state.

## Baseline screen

Command shape, with the calibrated prompt selected per row:

```sh
cargo run --locked --profile fast --no-default-features --features cuda \
  --example bench -- MODEL --context 4096 --kv f16 \
  --prompt PROMPT --max-tokens 32 --warmup 1 --reps 3
```

| Regime | Prompt tokens | Prompt t/s mean (CV) | TTFT ms mean (CV) | Model decode t/s mean (CV) | Public delta t/s mean (CV) |
|---|---:|---:|---:|---:|---:|
| Short | 128 | 282.70 (0.0015) | 452.78 (0.0015) | 53.04 (0.0019) | 51.36 (0.0020) |
| Medium | 1,024 | 264.77 (0.0040) | 3,867.54 (0.0040) | 51.12 (0.0028) | 47.90 (0.0024) |
| Long | 3,584 | 226.56 (0.0054) | 15,819.34 (0.0054) | 43.17 (0.0061) | 40.46 (0.0063) |

All objective and throughput-control CVs are below 5%; no stability rerun
applies. Fixed-work prefill dominates medium and long. Thirty-two model decode
tokens take approximately 603.3, 625.9, and 741.3 ms respectively.

## Baseline timeline

Nsight Systems 2023.4 captured CUDA/NVTX activity without CPU sampling because
the host denies unprivileged `perf_event_open`. Its target CLI could collect
`.qdstrm` but did not locate the separately installed importer; the existing
host importer converted each original trace directly, so no workload was
repeated. Profiled prompt rates differ from unprofiled means by -1.23%, -0.23%,
and +0.81% for short, medium, and long. These traces are diagnostic only.

| Regime | Prefill span ms | Tensor matmul ms | Prefill attention ms | Decode span ms | Decode matmul ms | Decode attention ms |
|---|---:|---:|---:|---:|---:|---:|
| Short | 456.268 | 436.333 | 8.440 | 621.508 | 485.607 | 28.904 |
| Medium | 3,873.960 | 3,473.279 | 331.628 | 632.610 | 479.864 | 46.169 |
| Long | 15,689.505 | 12,220.245 | 3,238.282 | 736.429 | 489.680 | 144.963 |

Medium tensor matmul splits into 1,354.183 ms M32-wide, 1,091.054 ms ordinary
M16, and 1,028.042 ms paired M16. Static launch records report respectively
96/70/72 registers per thread and 14,976/9,792/19,584 bytes shared per
128-thread block. Nsight Compute was attempted only on one proven-hot wide
launch; hardware counters are unavailable with `ERR_NVGPUCTRPERM`, no kernel
was replayed, and the run is not performance evidence.

The timeline shows one stream and dense prefill: unattributed medium prefill
time is about 4.3 ms, so transfer, host-gap, and synchronization-removal
families cannot meet the retention threshold. Model-load transfers occur before
generation and are outside the target.

## Commands and decisions

Preflight:

```text
git status --short --branch
uname -s
uname -m
nvidia-smi --query-gpu=index,uuid,compute_cap,memory.total,... --format=csv,noheader
rustc --version
cargo --version
nvcc --version
```

Result: clean start; supported CUDA host; ordinal 0 is idle; artifact was absent
locally. The identically named Unsloth artifact was rejected during acquisition
because its size and SHA differ from `support/models.tsv`. Only the exact
official catalog artifact is eligible.

Additional completed commands:

```text
cargo check --workspace --locked --profile fast --no-default-features --features cuda
nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none --backtrace=none ...
ncu --kernel-name regex:^cuda_matmul_tensor_wide$ --launch-count 1 --set basic ...
```

CUDA workspace check passed. All three timeline acquisitions and direct imports
completed. Nsight Compute ended with `ERR_NVGPUCTRPERM` and no profiled kernel.

### C01 — M32-to-M16 routing on compute 7.5

The candidate changed only the compute-7.5 dispatch for quantized matmul with
9,216 outputs; compute capability 8 and newer retained the M32 route. Formatting,
the CUDA workspace check, all 11 error-matrix tests, all 44 CUDA backend tests,
and exact 16-token parity passed. The CPU workspace suite had already passed
before the CUDA-only initializer correction; that correction is not compiled in
the CPU configuration.

| Metric | Baseline A | Candidate B | Change |
|---|---:|---:|---:|
| Medium prompt t/s | 264.77 | 245.00 | -7.47% |
| Medium TTFT ms | 3,867.54 | 4,179.63 | +8.07% |
| Prompt t/s CV | 0.0040 | 0.0037 | stable |

Decision: rejected below the 3% lower bound; short and long controls were not
run. The production diff was removed. C01 is one countable attempt because it
passed correctness and reached a stable objective A/B classification.
