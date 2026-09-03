<!--
This report is the recovery source for the 2026-09-04 CUDA optimization
campaign. Exact commercial device identity remains in private raw logs in
accordance with the repository hardware-neutral documentation policy.
-->

# CUDA Amdahl Optimization

## Campaign state

- Branch: `perf/cuda-amdahl-20260904`
- Baseline revision: `2d7cd89380c7d14145fb317a0da06fecf26025eb`
- Started: `2026-09-04T00:39:08+02:00`
- Unattended deadline: `2026-09-04T08:39:08+02:00`
- Current candidate: serial matmul attribution; no production candidate is active
- Current phase: stable short baseline captured, narrow profiling pending

The target is CUDA end-to-end inference performance. The initial public
evidence makes prompt/prefill the provisional objective; decode throughput and
TTFT are controls. The short row is the first objective because it can acquire
a stable current-revision baseline within the two-hour comparison limit. The
medium and largest safe rows are controls when runnable. The matrix is reranked
after attribution and after every retained change.

## Authenticated tuple

| Field | Value |
|---|---|
| Catalog ID | `3b-instruct` |
| Artifact | `Ministral-3-3B-Instruct-2512-Q4_K_M.gguf` |
| Bytes | `2147023008` |
| SHA-256 | `9ed150d4367e68df0ac8e1540f6ddc65b42d0ee26378329d1ecbca60f93fc5f8` |
| Host | Linux x86_64, CUDA compute capability 8.6, 12 GiB VRAM |
| Driver / toolkit | `580.173.02` / `12.4` |
| Rust | `rustc 1.97.1`, `cargo 1.97.1` |
| Build | Cargo `release`, locked, `--no-default-features --features cuda` |
| Placement | standalone CUDA, all GPU, visible ordinal 0 |
| Context / KV | `4096` / `f16` |
| Sampling | greedy |
| Requested generation | 32 tokens |
| Repetitions | one warm-up, three measured |

The machine had ordinary desktop clients resident on the device but no other
inference or compute campaign. Device utilization, temperature, and performance
state are checked around each acquisition. System clock, power, fan, and driver
settings are not changed.

## Calibrated workload screen

Each prompt is the word `benchmark` repeated with a final period. Counts were
measured with the engine tokenizer at the baseline revision, not estimated.

| Regime | Words | Actual prompt tokens | Prompt SHA-256 |
|---|---:|---:|---|
| Short | 124 | 128 | `24bbd008b03e025caf3b6aece0a67c380d27303a33e9d1282fedc1ff1c27a2b8` |
| Medium | 1,020 | 1,024 | `0bdbe2964b9c8129d215ab61bfbf447b1e7f111cd95167096fe2b3f8568343f8` |
| Long | 3,580 | 3,584 | `9bd7645ed8e09684e7b84fc90e735e106595c414cf8ff1e0cc9e5bf14a7be5fc` |

The long row leaves 480 context slots for generation and template reserve. A
row whose baseline cannot finish inside the canonical two-hour per-comparison
limit is `not_verified: time budget exceeded`; its tuple is not shortened or
otherwise substituted.

## Predeclared gates and decisions

- Retain (`keep`) only when the declared objective improves by at least 5%,
  objective CV is at most 5% in both records, correctness passes, and neither
  TTFT nor non-target throughput regresses by more than 5%.
- A 3% to 5% stable gain is `interesting` evidence only; candidate code is
  removed. Less than 3%, a correctness failure, a control regression over 5%,
  an artifact mismatch, or an incomparable tuple is `reject`.
- If an objective CV exceeds 5%, rerun the complete A/B tuple exactly once.
- Scheduling, dispatch, or submission changes require exact parity. Numeric
  reduction changes require existing CUDA kernel error bounds plus authenticated
  real-model top-two parity before performance is interpreted.
- Independent local gates are the CPU workspace tests, the CUDA feature check,
  CUDA `error_matrix`, and focused kernel tests. The 74-row release matrix is
  outside this iterative campaign unless a shared backend contract changes.

## Historical evidence

The newest repository matrix at revision `e6c4572` used the same authenticated
artifact and unchanged numeric CUDA kernels. It recorded about 0.84 prompt
tokens/s and 0.61 decode tokens/s on its short row, while medium prompt
throughput remained about 0.83 tokens/s. CUDA did not complete its long row.
Those records identify a likely linear per-token bottleneck, but loader and
composition changes since that revision make them screening evidence rather
than the current A/B baseline.

The reachable CUDA history was inspected through the standalone implementation,
typed-view safety follow-up, lifecycle and error gates, frozen qualification,
and later hybrid composition work. No prior CUDA performance candidate or
rejected optimization report exists.

## Commands and measurements

Artifact authentication:

```sh
. support/artifact.sh
artifact_size "$MODEL"
artifact_sha256 "$MODEL"
```

Result: catalog size and SHA-256 matched exactly.

Tokenizer calibration:

```sh
cargo build --locked --no-default-features --features cpu --example tokenize
target/debug/examples/tokenize "$MODEL" "$PROMPT"
```

Result: 128, 1,024, and 3,584 prompt tokens for the declared rows.

Baseline, attribution, candidate decisions, correctness results, and exact A/B
records are appended below as the campaign advances.

Short baseline build and benchmark:

```sh
CARGO_TARGET_DIR=target/perf-baseline cargo build --locked --release \
  --no-default-features --features cuda --example bench
target/perf-baseline/release/examples/bench "$MODEL" \
  --context 4096 --kv f16 --prompt "$SHORT_PROMPT" \
  --max-tokens 32 --warmup 1 --reps 3
```

```text
prompt_tokens=128 completion_tokens=32 decoded_tokens=32 prompt_tps_mean=1.02 prompt_tps_median=1.02 prompt_tps_stddev=0.01 prompt_tps_cv=0.0054 ttft_ms_mean=125823.46 ttft_ms_median=125462.05 ttft_ms_stddev=684.45 ttft_cv=0.0054 model_decode_tps_mean=1.08 model_decode_tps_median=1.09 model_decode_tps_stddev=0.00 model_decode_tps_cv=0.0013 decode_tps_mean=1.05 decode_tps_median=1.05 decode_tps_stddev=0.00 decode_tps_cv=0.0013 delta_interval_ms_mean=951.88 delta_interval_ms_median=951.22 delta_interval_ms_stddev=6.24 delta_interval_ms_cv=0.0066 decode_begin_tps=1.05 decode_middle_tps=1.05 decode_end_tps=1.05
```

Wall time including warm-up was 624.23 seconds. Device memory and clocks were
stable during acquisition; objective and control CVs were below 1%. For the
fixed 128-prompt/32-completion work, TTFT contributes about 80.9% of the sum of
TTFT and model decode elapsed time, so prompt/prefill is the selected objective.

## Candidate ledger

| Candidate | Target | Terminal state | Evidence / revision |
|---|---|---|---|
| Baseline | Short prefill; decode and TTFT controls | complete | Stable record at `2d7cd89` |

## Remaining measured bottleneck

Prompt/prefill is the largest public phase. Kernel-level attribution is pending.
