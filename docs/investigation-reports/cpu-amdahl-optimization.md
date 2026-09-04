<!--
This report is the recovery source for the 2026-09-04 CPU optimization
campaign. It records only architecture and capability data; exact host identity
remains outside tracked documentation.
-->

# CPU Amdahl Optimization

## Campaign state

- Branch: `perf/cpu-amdahl-20260904`
- Baseline revision: `302512d97348d4851c48d972cd297d1fe3c531f3`
- Started: `2026-09-04T17:48:47+02:00`
- Unattended deadline: `2026-09-05T01:48:47+02:00`
- State: attribution
- Retained revisions: none

The selected backend is CPU, as explicitly requested. It is a build-supported
reference backend on this Linux `x86_64` host. The GPU-oriented optimizer
procedure is applied at its backend-neutral boundaries: authenticated public
measurements, CPU timeline/counters, Amdahl ranking, isolated candidates,
correctness before performance, identical A/B tuples, and quantitative stop.

## Declared experiment

- Target: whichever of prompt/prefill or decode owns the largest measured
  fixed-work interval in the initial public screen; TTFT and the non-target
  throughput metric remain controls.
- Initial intentional variable: none; this is the production baseline.
- Candidate variable: exactly one measured CPU implementation change per A/B.
- Correctness: CPU workspace tests plus focused exact or bounded numeric tests
  selected before each candidate. Authenticated parity is required when its
  pinned reference server is available; otherwise it is recorded as an exact
  external-verification item and every independent gate still runs.
- Stability: one warm-up and three repetitions; objective CV must be at most
  5%, with exactly one complete rerun if it is not.
- Retention: canonical `keep` requires at least 5% objective gain, no control
  regression above 5%, stable records, and passing correctness. A 3--5% result
  is `interesting` and candidate production code is removed.
- Per-comparison limit: two hours. Overall campaign limit: eight hours.

The screen holds model, context allocation, F16 KV, greedy sampling, build,
thread policy, hardware, and requested generation at 32 tokens constant. Only
the prompt history changes: 128, 1,024, and 3,584 authenticated tokenizer
tokens inside context 4,096. The exact prompts consist of 124, 1,020, and 3,580
space-separated `benchmark` words followed by a period; the chat template adds
four tokens. The objective row is selected by removable end-to-end percentage,
not by raw duration.

## Authenticated tuple

| Field | Value |
|---|---|
| Catalog ID | `3b-instruct` |
| Artifact | `Ministral-3-3B-Instruct-2512-Q4_K_M.gguf` |
| Bytes | `2147023008` |
| SHA-256 | `9ed150d4367e68df0ac8e1540f6ddc65b42d0ee26378329d1ecbca60f93fc5f8` |
| Host | Linux 7.0, `x86_64`, 6 physical / 12 logical CPUs, one NUMA node |
| ISA | AVX2, FMA, F16C, AVX-512, AVX-512 VNNI |
| Cache | 32 KiB L1d and 1 MiB L2 per core; shared 16 MiB L3 |
| Governor | `performance`; energy preference `performance` |
| Compiler | `rustc 1.96.1`, `cargo 1.96.1` |
| Build | Cargo `release`, locked dependencies, `cpu` feature only |
| Runtime | default 12-worker CPU policy, context 4096, F16 KV, greedy sampling |
| Sampling | public `examples/bench.rs`, 32 requested tokens, warm-up 1, reps 3 |

The artifact size and SHA were checked before model loading. At preflight the
host had 19 GiB available RAM, no swap use, and no competing Graph Horizon,
Cargo, parity, or reference-server process. Private raw environment logs retain
the exact device identity and instantaneous clocks.

## Historical evidence consulted

The integrated CPU lineage from `perf/cpu-kernel-specialization` retained
32-row prefill batching, Q4_K/Q6_K single-token latency routing, and four-query
GQA attention reuse. Its final recorded 3B/128 result on a six-core AVX2 host
was 25.48 prompt tok/s, 5,024.78 ms TTFT, and 6.79 decode tok/s.

Closed candidates are not repeated: 64-row batching (-10.9% prompt), inline
F16C scale widening (-22.9% prompt in the broad form and -9.5% in the
decode-only form), fixed scale blocks (-12.5% prompt), canonical Q8_K (only
1.118x local and below its complexity gate), compact eight-row native Q4_K
repacking (-17.3% prompt and +1.41 GiB RSS), RoPE coefficient reuse (<3%), the
Cargo `fast` profile (-14.0% TTFT), and a five-core mixed default (decode gain
with 17.0% prompt loss). A new attempt must rely on a premise absent there,
principally this host's AVX-512/VNNI capability or new current-revision cost.

## Baseline and attribution

The baseline build completed once in 11.97 seconds. One warm-up plus three
measured requests completed in 27 seconds for short, 143 seconds for medium,
and 529 seconds for long. All objective and decode CVs are below 1.6%, so no
stability rerun is allowed or needed.

| Regime | Prompt tokens | Prompt tok/s | TTFT ms | Model decode tok/s | Public decode tok/s | TTFT CV | Model-decode CV |
|---|---:|---:|---:|---:|---:|---:|---:|
| Short | 128 | 31.80 | 4,025.70 | 13.41 | 12.99 | 1.51% | 1.32% |
| Medium | 1,024 | 30.12 | 34,001.14 | 11.96 | 11.21 | 0.15% | 0.77% |
| Long | 3,584 | 27.81 | 128,854.94 | 8.99 | 8.42 | 0.41% | 1.08% |

For the fixed 32-completion-token work, the approximate model-decode intervals
are 2.39, 2.68, and 3.56 seconds. TTFT therefore owns about 62.8%, 92.7%, and
97.3% of the corresponding TTFT-plus-decode intervals. Prompt/prefill is the
selected objective in every regime; model and public-delta decode are controls.
The long row has the largest objective fraction, while a candidate must still
pass short and medium controls before retention.

Exact public command shape (the prompt word count is selected per row):

```sh
CARGO_TARGET_DIR=/tmp/graph-horizon-cpu-target-baseline cargo build --locked \
  --release --no-default-features --features cpu --example bench
/tmp/graph-horizon-cpu-target-baseline/release/examples/bench "$MODEL" \
  --context 4096 --kv f16 --prompt "$PROMPT" --max-tokens 32 \
  --warmup 1 --reps 3
```

The public benchmark is the end-to-end authority. Narrow one-repetition,
two-token diagnostics used below are attribution only and are never compared
as public performance results.

Hardware counters and sampling are unavailable because
`kernel.perf_event_paranoid=4`; changing host policy is outside scope. The
declared temporary attribution structure is:

- root and engine `Cargo.toml`: one diagnostic feature line each;
- `backend/cpu/profile.rs` (~80 productive lines): CPU boundary aggregation
  and one stderr report, with no runtime responsibility when disabled;
- `backend/cpu/backend.rs` and `backend/cpu/buffer.rs` (~35 added productive
  lines total): narrow operation and transfer timer scopes.

This instrumentation reuses the repository's removed 2026-08 CPU profiler
design, is built only under the diagnostic feature, and will be removed before
any candidate correctness or A/B measurement. Its overhead is measured against
an identical unprofiled diagnostic row.

## Candidate ledger

Pending baseline attribution and Amdahl ranking.

## Final verification and result

Pending.
