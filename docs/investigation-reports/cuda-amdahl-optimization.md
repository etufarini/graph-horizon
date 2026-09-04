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
- Current candidate: warp-broadcast Q4 scale/min metadata
- Current phase: candidate declared; implementation pending

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

Narrow CUDA attribution used the short prompt, two requested tokens, no warm-up,
one repetition, and `CUDA_LAUNCH_BLOCKING=1`. Temporary dispatch timing grouped
host-return latency by kernel and was removed immediately after acquisition.

```text
instrumented TTFT=125305.60 ms
all timed kernels=126220.296 ms
matmul_batched=124854.456 ms (98.9179%, 806 launches)
logits=628.005 ms (0.4975%, 2 launches)
matmul=267.983 ms (0.2123%, 104 launches)
attention_prefill_f16=234.422 ms (0.1857%, 104 launches)
rmsnorm=112.374 ms (0.0890%, 262 launches)
all remaining kernels=123.056 ms (0.0975%)
```

The instrumented TTFT was 0.4% below the unprofiled baseline mean, inside the
baseline dispersion. Because the backend uses one ordered stream, forcing launch
completion exposed per-kernel time without removing normal kernel overlap. The
single diagnostic repetition is attribution evidence, not an A/B performance
record.

## Amdahl ranking

| Regime / phase | Candidate | `p` | Credible local `s` | Added `o` | Predicted global speedup | Ideal ceiling | Absolute time saved | Uncertainty |
|---|---|---:|---:|---:|---:|---:|---:|---|
| Short / prefill | One block per output row with parallel K reduction | 0.9892 | 8× | 0.005 | 7.17× | 92.4× | about 108.6 s | One profiled repetition; `p` excludes single matmul and logits, so it is conservative |
| Short / prefill after first keep | Reuse each decoded weight across four batch tokens | 0.9207 | 2× | 0.010 | 1.82× | 12.6× | about 2.8 s | One profiled repetition; local speedup discounted from four-token reuse to 2× |
| Short / both after first keep | Broadcast Q4 scale/min metadata within each warp | 0.697 | 1.10× | 0.002 | 1.065× | 3.30× | about 0.38 s | Q4 share derived from authenticated tensor shapes; local gain is conservative but not counter-profiled |

The candidate changes only matmul launch geometry and the category-K matmul
kernel. It replaces one serial K loop per output with 256 partial sums reduced
inside one block. The invariant is unchanged tensor bounds, formats, and one
output per `(token,row)`. The main risk is floating-point reordering, so this is
a numeric candidate: CUDA packed-format error tests and pinned real-model top-two
parity must pass before any candidate benchmark is interpreted.

Post-change attribution repeated the same two-token diagnostic command after
commit `5d52db5`; temporary timing was again removed before the next candidate.

```text
instrumented TTFT=6162.62 ms
all timed kernels=6227.641 ms
matmul_batched=5734.055 ms (92.0743%, 806 launches)
attention_prefill_f16=238.042 ms (3.8223%, 104 launches)
rmsnorm=113.053 ms (1.8153%, 262 launches)
rope=56.011 ms (0.8994%, 6708 launches)
all remaining kernels=86.480 ms (1.3887%)
```

The next candidate changes only batched matmul: one block processes up to four
tokens for one output row, decoding each quantized weight once and applying it
to four inputs. Each token still visits K indices in the same ascending-stride
order and uses the same tree reduction as the retained kernel. The intended
variable is weight reuse; tensor layout, formats, placement, and public APIs are
unchanged. Its predeclared correctness gate is exact batched-versus-single
kernel output plus the same local and pinned real-model gates.

The four-token candidate passed the CPU workspace, CUDA check, 11 selected
error-matrix tests, all 32 CUDA tests, exact f16 batched-versus-single output for
F16/Q4/Q5/Q6, and pinned real-model parity with all 16 top-1 IDs equal to the
oracle. Its short record was:

```text
prompt_tokens=128 completion_tokens=32 decoded_tokens=32 prompt_tps_mean=44.42 prompt_tps_median=44.39 prompt_tps_stddev=0.07 prompt_tps_cv=0.0016 ttft_ms_mean=2881.51 ttft_ms_median=2883.47 ttft_ms_stddev=4.61 ttft_cv=0.0016 model_decode_tps_mean=9.42 model_decode_tps_median=9.42 model_decode_tps_stddev=0.00 model_decode_tps_cv=0.0003 decode_tps_mean=9.13 decode_tps_median=9.13 decode_tps_stddev=0.00 decode_tps_cv=0.0004 delta_interval_ms_mean=109.50 delta_interval_ms_median=109.43 delta_interval_ms_stddev=2.44 delta_interval_ms_cv=0.0223 decode_begin_tps=9.38 decode_middle_tps=9.14 decode_end_tps=8.91
```

Prompt throughput improved 113.5% and TTFT fell 53.1%, but model decode and
public-delta decode regressed 6.08% and 6.17%. Both exceed the 5% control limit.
Objective CV was below 5%, so the canonical rerun rule did not apply. Terminal
state: `reject: decode control regression`; all candidate code and its focused
test change were removed without rewriting history.

The authenticated inventory has 156 Q4_K matrices and 26 Q6_K layer matrices.
Using hidden width 3,072 and FFN width 9,216, Q4 projections account for about
75.7% of per-layer matmul multiply-accumulates. Applied to the measured 92.07%
matmul share, the Q4 metadata candidate owns a conservative `p=0.697` of the
short fixed-work interval.

For each 32-lane Q4 group, all lanes currently decode the same two scale/min
codes and the same two f16 block factors. The third candidate computes those
values in lane zero and broadcasts their exact f32 bits with warp shuffles;
quant nibbles and per-lane multiplication remain unchanged. The intended
variable is redundant metadata arithmetic; no layout, precision, accumulation
order, or public interface changes. The exact kernel gate pins the accepted
Q4/Q5 patterned-output f16 bits to `d4b6,ce2d` and `d93c,d46d`; Q6 control bits
are `51fd,4d1e`. Full local CUDA and real-model parity gates still apply.

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

Candidate correctness gates, run before candidate performance:

```sh
cargo fmt --all -- --check
cargo check --workspace --locked --no-default-features --features cuda
cargo test --workspace --locked --no-default-features --features cpu
cargo test -p graph_horizon_engine --locked --no-default-features \
  --features cuda error_matrix -- --nocapture
cargo test -p graph_horizon_engine --locked --no-default-features \
  --features cuda 'backend::cuda::' -- --nocapture
support/testing/parity-check.sh --models-dir "$MODELS" --model-id 3b-instruct \
  --backend cuda --kv f16 --reference-server "$PINNED_LLAMA_SERVER"
```

Results: formatting and CUDA check passed; CPU workspace suites passed (170 root
and 163 engine tests, plus integration suites); all 11 selected error-matrix
tests and all 32 CUDA tests passed. The parity runner used its exact pinned
`13f2b28b0` oracle and all 16 local top-1 token IDs equaled the oracle token IDs,
which is stronger than its required top-two bound.

Short candidate record:

```text
prompt_tokens=128 completion_tokens=32 decoded_tokens=32 prompt_tps_mean=20.81 prompt_tps_median=20.82 prompt_tps_stddev=0.07 prompt_tps_cv=0.0033 ttft_ms_mean=6149.64 ttft_ms_median=6147.80 ttft_ms_stddev=20.39 ttft_cv=0.0033 model_decode_tps_mean=10.03 model_decode_tps_median=10.02 model_decode_tps_stddev=0.03 model_decode_tps_cv=0.0034 decode_tps_mean=9.73 decode_tps_median=9.71 decode_tps_stddev=0.03 decode_tps_cv=0.0034 delta_interval_ms_mean=102.78 delta_interval_ms_median=102.70 delta_interval_ms_stddev=2.35 delta_interval_ms_cv=0.0228 decode_begin_tps=10.00 decode_middle_tps=9.75 decode_end_tps=9.48
```

Wall time including warm-up was 38.34 seconds. Against the stable baseline,
prompt throughput improved 1,940.2%, TTFT fell 95.1%, model decode throughput
improved 828.7%, and public-delta decode improved 826.7%. All available CVs are
below 1%; this is a provisional passing A/B pending the medium control.

Medium control baseline:

```text
prompt_tokens=1024 completion_tokens=32 decoded_tokens=31 prompt_tps_mean=1.01 prompt_tps_median=1.01 prompt_tps_stddev=0.00 prompt_tps_cv=0.0001 ttft_ms_mean=1012305.08 ttft_ms_median=1012240.63 ttft_ms_stddev=141.85 ttft_cv=0.0001 model_decode_tps_mean=0.87 model_decode_tps_median=0.87 model_decode_tps_stddev=0.00 model_decode_tps_cv=0.0014 decode_tps_mean=0.82 decode_tps_median=0.82 decode_tps_stddev=0.00 decode_tps_cv=0.0014 delta_interval_ms_mean=1224.14 delta_interval_ms_median=1183.26 delta_interval_ms_stddev=213.32 delta_interval_ms_cv=0.1743 decode_begin_tps=0.85 decode_middle_tps=0.84 decode_end_tps=0.77
```

Medium control candidate:

```text
prompt_tokens=1024 completion_tokens=32 decoded_tokens=31 prompt_tps_mean=16.88 prompt_tps_median=16.88 prompt_tps_stddev=0.02 prompt_tps_cv=0.0014 ttft_ms_mean=60658.02 ttft_ms_median=60675.86 ttft_ms_stddev=86.82 ttft_cv=0.0014 model_decode_tps_mean=3.04 model_decode_tps_median=3.04 model_decode_tps_stddev=0.01 model_decode_tps_cv=0.0037 decode_tps_mean=2.85 decode_tps_median=2.85 decode_tps_stddev=0.01 decode_tps_cv=0.0037 delta_interval_ms_mean=350.33 delta_interval_ms_median=338.15 delta_interval_ms_stddev=61.48 delta_interval_ms_cv=0.1755 decode_begin_tps=2.96 decode_middle_tps=2.96 decode_end_tps=2.67
```

The complete A and B wall times were 4,196.92 and 285.45 seconds. Prompt
throughput improved 1,571.3%, TTFT fell 94.0%, model decode throughput improved
249.4%, and public-delta decode improved 247.6%. Objective CVs were 0.01% and
0.14%; no control regressed.

The long-row baseline was not started. Scaling the measured medium baseline by
prompt work predicts about 3,543 seconds per request and about 236 minutes for
the required warm-up plus three measurements. That exceeds the canonical
two-hour comparison limit, so the row is `not_verified: time budget exceeded`.
The workload, context, generation reserve, and placement were not changed to
manufacture a runnable substitute.

## Candidate ledger

| Candidate | Target | Terminal state | Evidence / revision |
|---|---|---|---|
| Baseline | Short prefill; decode and TTFT controls | complete | Stable record at `2d7cd89` |
| Block-parallel matmul reduction | Short prefill; decode and TTFT controls | **keep** | Correctness, short objective, and medium control pass; retained in this commit |
| Four-token batched matmul weight reuse | Short prefill; decode and TTFT controls | **reject** | Objective +113.5%; decode controls −6.08%/−6.17%; code restored |
| Warp-broadcast Q4 scale/min metadata | Short prefill and decode | running | Amdahl prediction +6.5%; correctness pending |

## Remaining measured bottleneck

After the first retained change, batched matmul still owns 92.07% of measured
kernel time. Four-token weight reuse improved that path but violated the decode
control gate. Redundant Q4 metadata decoding is the next candidate above the
retention threshold.
