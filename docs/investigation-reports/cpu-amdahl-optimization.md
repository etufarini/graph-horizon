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
- First pass completed: `2026-09-04T18:20:38+02:00`
- Resumed: `2026-09-04T19:27:55+02:00`
- Current unattended deadline: `2026-09-05T03:27:55+02:00`
- Completed: `2026-09-04T20:06:49+02:00`
- State: complete; retained improvement and quantitative stop
- Retained production revisions: `c76c8b9` (CPU-02 cache-sized token tiling)

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
- `backend/cpu/backend.rs` (~35 added productive lines): narrow operation
  timer scopes.

This instrumentation reuses the repository's removed 2026-08 CPU profiler
design, is built only under the diagnostic feature, and will be removed before
any candidate correctness or A/B measurement. Its overhead is measured against
an identical unprofiled diagnostic row.

## Candidate ledger

### Continuation after the first quantitative stop

The first pass stopped too early because it carried forward the 256 KiB L2
premise of the historical six-core host. The current host has a 1 MiB unified
L2 per physical core (shared only by its two hardware threads), while
`token_tile` still targets a 192 KiB activation working set. For the dominant
`in_dim=3072`, `n=32` Q4_K calls this forces two 16-token tiles and therefore
two complete weight decode passes even though all 32 activation rows occupy
384 KiB and fit in the current L2. This is independent of the rejected
AVX-512 row block: it uses the existing AVX2 arithmetic and cannot alter the
decode route.

CPU-02 pre-declaration:

- Target: short prompt/prefill throughput; medium and long prompt throughput,
  TTFT, model decode, and public decode are controls.
- Intentional variable: size the existing activation tile from the CPU's
  reported L2 capacity instead of the historical fixed 192 KiB budget, retaining
  the existing `[16, 64]` bound.
- Invariants: weight and activation layout, arithmetic and reduction order,
  batch size, worker count, output transpose, decode path, and public API stay
  unchanged. Unsupported cache discovery retains the current 256 KiB premise.
- Smallest change: the existing category-K `token_tile` policy and a focused
  policy test; no new file, dependency, or persistent allocation.
- Main risk: a larger activation tile can evict weights or worker scratch and
  lose more locality than the eliminated decode pass saves.
- Correctness before performance: formatting, focused Q4_K/Q6_K batched parity,
  then the locked release CPU workspace suite. Authenticated parity remains
  external verification while the required pinned reference-server revision is
  unavailable.
- Stability and retention: the existing canonical CV, 5% objective, 5%
  control, and A/B rules remain unchanged.

From the current profile, Q4_K and Q6_K batched work own conservatively 75% of
the short fixed-work interval. The affected 3072-wide Q4_K shapes alone own
about 51%. Eliminating one of two weight-decode passes has a conservative local
speedup of 1.12x for those shapes, giving `p=0.45`, `s=1.12`, negligible added
overhead, a predicted global speedup of 1.046x, and an ideal ceiling of 1.82x.
That is slightly below the keep threshold, but the broader affected shapes and
the exact current-host cache premise make one bounded experiment warranted;
it must still clear 5% in the public benchmark to remain.

The resumed same-session short baseline at revision `f1e3669` completed with
all objective and control CVs below 5%:

```text
prompt_tokens=128 completion_tokens=32 decoded_tokens=32 prompt_tps_mean=30.68 prompt_tps_median=30.97 prompt_tps_stddev=1.00 prompt_tps_cv=0.0325 ttft_ms_mean=4174.92 ttft_ms_median=4133.11 ttft_ms_stddev=137.61 ttft_cv=0.0330 model_decode_tps_mean=13.13 model_decode_tps_median=13.15 model_decode_tps_stddev=0.51 model_decode_tps_cv=0.0391 decode_tps_mean=12.71 decode_tps_median=12.73 decode_tps_stddev=0.50 decode_tps_cv=0.0391 delta_interval_ms_mean=78.73 delta_interval_ms_median=78.23 delta_interval_ms_stddev=3.85 delta_interval_ms_cv=0.0489 decode_begin_tps=12.91 decode_middle_tps=12.73 decode_end_tps=12.52
```

The profiled short diagnostic reported 4,411.37 ms TTFT versus 3,761.54 ms
unprofiled (+17.28%); whole-process wall time was 5.73 versus 5.05 seconds
(+13.47%). Dynamic labels and boundary locking therefore make these timers
diagnostic only. They account for 4,497.76, 37,160.28, and 149,642.67 ms in the
short, medium, and long two-token rows, including the second model step.

| Regime | Q4_K batched | Q6_K batched | RoPE | Prefill attention |
|---|---:|---:|---:|---:|
| Short | 68.47% | 15.05% | 10.20% | 0.81% |
| Medium | 67.18% | 14.91% | 10.32% | 4.12% |
| Long | 62.72% | 12.87% | 9.10% | 12.39% |

The Q4 share combines all `n=32` projection/MLP shapes. Temporary profiling
feature lines, module, timers, and report hook were removed immediately after
this attribution; no candidate benchmark contains them.

### Initial Amdahl ranking

| Candidate | Regime / phase | Conservative `p` | Credible local `s` | Added `o` | Predicted global gain | Ideal ceiling | Uncertainty |
|---|---|---:|---:|---:|---:|---:|---|
| Four-row AVX-512-register Q4 block | short prefill | 0.60 | 1.10 | <0.001 | 5.7% | 2.50x | profile overhead; local gain unmeasured |
| Same Q4 block | medium prefill | 0.60 | 1.10 | <0.001 | 5.7% | 2.50x | same |
| Same Q4 block | long prefill | 0.60 | 1.10 | <0.001 | 5.7% | 2.50x | same |
| Wider exact GQA attention | long prefill | 0.12 | 1.20 | <0.001 | 2.0% | 1.14x | x8 scheduling cost omitted |
| Wider Q6 vector path | all prefill | 0.12 | 1.10 | <0.001 | 1.1% | 1.14x | no local prototype |
| RoPE coefficient reuse | all prefill | 0.09 | historical | 0 | 2.7% measured | 1.10x | closed historical result |

The Q4 candidate is selected because it alone clears the 5% conservative gate
and benefits all regimes. Short is the objective because it has the largest
measured candidate-owned fraction; medium and long are controls. For the long
baseline, the conservative model removes about 7.0 seconds of TTFT; short and
medium estimates are about 0.22 and 1.85 seconds.

### CPU-01 — four-row AVX-512-register Q4 block

- Intentional variable: process four adjacent Q4 output rows together on CPUs
  with AVX-512F+VL in the existing batched float kernel, reusing each activation
  load across four rows instead of two. The arithmetic for each row retains the
  existing AVX2/FMA instruction sequence and horizontal reduction.
- Eligibility: `x86_64`, AVX2, FMA, AVX-512F, and AVX-512VL; batched Q4_K
  only. Existing two-row and scalar paths remain the fallback.
- Invariants: canonical weights, activation/output layout, precision, worker
  partition, per-row accumulation order, placement, and public API are unchanged.
- Smallest change: one category-K SIMD function, a narrow dispatch branch, and
  a focused exact-output test. No file, dependency, or orchestration change.
- Main risk: compiler register allocation may spill or wide-ISA execution may
  reduce clocks, eliminating the intended activation-load saving.
- Correctness gate before performance: formatting; focused Q4 scalar/SIMD and
  batched-versus-single tests including exact four-row versus two-row output;
  CPU workspace suite; authenticated parity if the pinned server is available.

The focused five-test Q4 gate passed, including exact four-row versus two-row
f32 output bits. The full release CPU workspace passed: 170 root tests (three
external/live tests ignored), 164 engine tests (two authenticated tests
ignored), documentation, four family-agnostic tests (one authenticated test
ignored), and 12 semantic tests (one authenticated test ignored). The local
`llama-server` is revision `9bebfcb4b`, not the required pinned `13f2b28b0`, so
real-model parity is `external verification: unsupported llama.cpp revision`.

```sh
CARGO_TARGET_DIR=/tmp/graph-horizon-cpu-target-candidate cargo test \
  -p graph_horizon_engine --locked --release --no-default-features \
  --features cpu 'backend::cpu::kernels::matmul::q4k' -- --nocapture
CARGO_TARGET_DIR=/tmp/graph-horizon-cpu-target-candidate cargo test \
  --workspace --locked --release --no-default-features --features cpu
CARGO_TARGET_DIR=/tmp/graph-horizon-cpu-target-candidate cargo build \
  --locked --release --no-default-features --features cpu --example bench
/tmp/graph-horizon-cpu-target-candidate/release/examples/bench "$MODEL" \
  --context 4096 --kv f16 --prompt "$SHORT_PROMPT" --max-tokens 32 \
  --warmup 1 --reps 3
```

Short candidate record:

```text
prompt_tokens=128 completion_tokens=32 decoded_tokens=32 prompt_tps_mean=36.48 prompt_tps_median=36.52 prompt_tps_stddev=0.50 prompt_tps_cv=0.0137 ttft_ms_mean=3509.36 ttft_ms_median=3504.64 ttft_ms_stddev=48.14 ttft_cv=0.0137 model_decode_tps_mean=11.58 model_decode_tps_median=11.58 model_decode_tps_stddev=0.01 model_decode_tps_cv=0.0009 decode_tps_mean=11.22 decode_tps_median=11.22 decode_tps_stddev=0.01 decode_tps_cv=0.0010 delta_interval_ms_mean=89.15 delta_interval_ms_median=88.81 delta_interval_ms_stddev=2.48 delta_interval_ms_cv=0.0279 decode_begin_tps=11.22 decode_middle_tps=11.21 decode_end_tps=11.17
```

Against baseline, prompt throughput improved 14.72% and TTFT fell 12.83%, but
model decode regressed 13.65% and public-delta decode regressed 13.63%. All
candidate objective/decode CVs are at most 1.37%, so the canonical instability
rerun does not apply. Disassembly confirms use of EVEX-encoded `ymm16`--`ymm28`
registers rather than spills; the intended register expansion occurred. Its
wide-ISA/code-layout effect still harms the unchanged decode path beyond the
5% control limit.

State: `reject: decode control regression`. Medium and long controls were not
run after the terminal short failure. All CPU-01 production and test code was
removed without rewriting history.

### CPU-02 — cache-sized token tiling

The implementation reads the per-core L2 capacity from architectural CPUID
leaf `0x80000006` on `x86_64`, falls back to the historical 256 KiB premise
when the leaf is absent or reports zero, and targets three quarters of that
capacity. The existing 16--64 bounds remain. No arithmetic, data layout,
allocation lifetime, or public interface changed.

The predeclared focused gate passed five Q4_K tests and three Q6_K tests,
including batched parity and explicit 256 KiB/1 MiB tile-policy cases. The
locked release CPU workspace passed 170 root tests (three live tests ignored),
164 engine tests (two authenticated tests ignored), documentation, four
family-agnostic tests (one authenticated test ignored), and 12 semantic tests
(one authenticated test ignored).

| Regime | Metric | Baseline | Candidate | Delta | Baseline CV | Candidate CV |
|---|---|---:|---:|---:|---:|---:|
| Short | Prompt tok/s | 30.68 | 33.20 | +8.21% | 3.25% | 2.92% |
| Short | TTFT ms | 4,174.92 | 3,858.00 | -7.59% | 3.30% | 2.95% |
| Short | Model decode tok/s | 13.13 | 13.72 | +4.49% | 3.91% | 4.81% |
| Short | Public decode tok/s | 12.71 | 13.29 | +4.56% | 3.91% | 4.80% |
| Medium rerun | Prompt tok/s | 25.61 | 26.79 | +4.61% | 2.02% | 1.29% |
| Medium rerun | TTFT ms | 39,998.39 | 38,227.34 | -4.43% | 2.01% | 1.30% |
| Medium rerun | Model decode tok/s | 9.92 | 10.51 | +5.95% | 2.24% | 0.32% |
| Medium rerun | Public decode tok/s | 9.30 | 9.85 | +5.91% | 2.24% | 0.33% |
| Long | Prompt tok/s | 27.32 | 31.09 | +13.80% | 0.22% | 2.37% |
| Long | TTFT ms | 131,196.13 | 115,338.56 | -12.09% | 0.22% | 2.40% |
| Long | Model decode tok/s | 9.11 | 9.24 | +1.43% | 0.55% | 1.50% |
| Long | Public decode tok/s | 8.54 | 8.66 | +1.41% | 0.55% | 1.51% |

The first medium pair was 37.44 -> 28.06 prompt tok/s, but the candidate
objective CV was 5.30% and its decode CV was 6.96%. The one complete rerun
allowed by the protocol produced the stable paired row above; no selective
third run was made. All three accepted pairs used the same authenticated model,
context, KV, prompt construction, requested generation, warm-up, repetitions,
worker policy, and build profile.

State: `keep`. The short objective clears 5%; every objective and control CV in
the deciding records is at most 4.81%; medium and long do not regress; and
correctness passed. Pinned real-model parity remains
`external verification: unsupported llama.cpp revision` exactly as for CPU-01.

### CPU-03 — one worker per physical core screen

- Target: short prompt/prefill throughput after CPU-02; decode remains a
  control.
- Intentional variable: process affinity exposes either all 12 logical CPUs or
  one hardware thread from each of the six physical cores. The runtime's
  unchanged `available_parallelism` policy therefore selects 12 or six workers.
- New premise: the historical thread-policy screen ran on a six-core/six-thread
  host; the current host has SMT and has not been screened for sibling
  contention.
- Correctness: no code, arithmetic, data, or runtime option changes. This is a
  performance-only causal screen of the existing supported affinity boundary.
- Stability and retention: one warm-up, three repetitions, 5% CV and objective
  threshold, and 5% decode control limit. A production policy candidate is
  warranted only if this screen clears the objective threshold.

The all-logical-CPU row measured 32.27 prompt tok/s, 3,967.13 ms TTFT,
13.49 model-decode tok/s, and 13.07 public-decode tok/s. One worker per physical
core measured 26.18, 4,889.17 ms, 13.49, and 13.07 respectively: prompt
throughput regressed 18.87% and TTFT regressed 23.24%, while decode was
unchanged. Every objective/control CV was at most 2.51%.

The first all-logical-CPU invocation failed before model loading because the
artifact was moved externally during the campaign. The relocated file was
reauthenticated at the catalog byte count and SHA-256 and the identical numeric
tuple was retried once; the repository and artifact bytes were not changed.

State: `reject: prompt objective regression`. No production policy code was
written; retaining all logical workers is materially better for current-host
prefill.

## First-pass verification and result

No production optimization was retained by the first pass. At that checkpoint,
the runtime tree was identical to baseline revision `302512d`; only this
investigation report and its index entry were present on the branch.

The restored short bookend was:

```text
prompt_tokens=128 completion_tokens=32 decoded_tokens=32 prompt_tps_mean=34.76 prompt_tps_median=34.97 prompt_tps_stddev=0.48 prompt_tps_cv=0.0138 ttft_ms_mean=3683.18 ttft_ms_median=3660.52 ttft_ms_stddev=51.12 ttft_cv=0.0139 model_decode_tps_mean=12.26 model_decode_tps_median=12.29 model_decode_tps_stddev=0.05 model_decode_tps_cv=0.0038 decode_tps_mean=11.88 decode_tps_median=11.90 decode_tps_stddev=0.04 decode_tps_cv=0.0037 delta_interval_ms_mean=84.20 delta_interval_ms_median=83.95 delta_interval_ms_stddev=2.23 delta_interval_ms_cv=0.0265 decode_begin_tps=11.78 decode_middle_tps=11.93 decode_end_tps=11.90
```

This bookend shows session drift relative to the initial baseline: prompt was
9.31% faster while model decode was 8.58% slower. Against the closer restored
bookend, CPU-01 still improves prompt only 4.95%, lowers TTFT 4.72%, and
regresses both decode controls by about 5.55%. It remains a rejection under
both the frozen baseline comparison and the local bookend diagnosis.

The first pass stopped quantitatively: CPU-01 was then the only new current-host
candidate whose conservative predicted whole-request gain cleared 5%, and it
failed the decode control. After removal, the largest measured
bottleneck is Q4_K batched matmul at 62.72--68.47% of accounted time. The next
independent bounded candidates predict only 2.0% for long attention and 1.1%
for Q6; historically closed candidates have no changed premise. A larger
packed-GEMM subsystem remains outside the economic boundary established by its
measured compact predecessor (-17.3% prompt, +1.41 GiB RSS), while a wider
query-tiled attention design remains below its conservative retention gate.

Profiler caveats: hardware counters were unavailable, backend timers added
17.28% to short TTFT, and no claim uses a profiled duration as an end-to-end
result. External verification remains the pinned real-model parity server at
revision `13f2b28b0`; the available `9bebfcb4b` binary was not substituted.

First-pass restored-tree gates:

```sh
cargo fmt --all -- --check
CARGO_TARGET_DIR=/tmp/graph-horizon-cpu-target-final cargo check --workspace \
  --locked --no-default-features --features cpu
CARGO_TARGET_DIR=/tmp/graph-horizon-cpu-target-final cargo test --workspace \
  --locked --release --no-default-features --features cpu
git diff --check
git diff --quiet 302512d -- . \
  ':(exclude)docs/investigation-reports/cpu-amdahl-optimization.md' \
  ':(exclude)docs/investigation-reports/README.md'
```

Formatting, locked CPU workspace check, and diff checks passed. The release
suite passed 170 root tests (three live/external ignored), 163 engine tests
(two authenticated ignored), documentation, four family-agnostic tests (one
authenticated ignored), and 12 semantic tests (one authenticated ignored).
The runtime-tree comparison against the baseline was empty. Added tracked
documentation also passed the hardware-product-name audit.

## Final rerank and stop

Assigning CPU-02's causally removed short TTFT to the previously measured
Q4_K/Q6_K batched bucket gives an observed local speedup near 1.10x and leaves
about 82% of the post-change short interval in quantized batched matmul. This
is an estimate because the boundary profiler added 17.28% overhead; public
unprofiled A/B records remain the performance authority.

| Candidate / regime | Conservative `p` | Local `s` | Added `o` | Predicted/measured global speedup | Ideal ceiling | Uncertainty |
|---|---:|---:|---:|---:|---:|---|
| CPU-02 / short | 0.835 | 1.10 measured | <0.001, included | 1.082x measured | 6.06x | profiled share perturbed |
| CPU-02 / medium | 0.821 | 1.06 inferred | <0.001, included | 1.046x measured | 5.59x | one permitted thermal rerun |
| CPU-02 / long | 0.756 | 1.19 inferred | <0.001, included | 1.138x measured | 4.10x | profiled share perturbed |
| Force one 32-token 9216-wide tile | 0.25 | <=1.10 | uncertain cache spill | <=1.023x | 1.33x | working set exceeds L2 |
| Wider exact long attention | 0.14 | 1.20 | <0.001 | 1.024x | 1.16x | prior scheduling cost omitted |
| Wider Q6 vector path | 0.12 | 1.10 | <0.001 | 1.011x | 1.14x | no local prototype |

The remaining 9,216-wide quantized shape still uses two tiles, but forcing all
32 activation rows into one tile would exceed the reported L2 and predicts only
about 2.3% globally even before spill cost. The rejected AVX-512 row block has
less Q4 share after CPU-02 and no new premise that removes its >5% decode
regression. Packed integer GEMM and wider attention remain the only high ideal
ceilings, but their bounded predecessors regressed or their credible global
effects remain below 5%. CPU-03 also closes the current-host SMT premise.

The campaign therefore stops at the canonical quantitative boundary. The
largest remaining measured bottleneck is quantized batched matmul, estimated at
about 82% of short prefill after CPU-02; no independent bounded candidate has a
conservative predicted gain of 5% or more.

Final retained-tree gates:

```sh
cargo fmt --all -- --check
CARGO_TARGET_DIR=/tmp/graph-horizon-cpu-target-l2-candidate cargo check \
  --workspace --locked --no-default-features --features cpu
CARGO_TARGET_DIR=/tmp/graph-horizon-cpu-target-l2-candidate cargo clippy \
  --workspace --locked --release --no-default-features --features cpu \
  -- -D warnings
CARGO_TARGET_DIR=/tmp/graph-horizon-cpu-target-l2-candidate cargo test \
  --workspace --locked --release --no-default-features --features cpu
git diff --check
```

Formatting, locked CPU check, warning-denied Clippy, diff checks, and the release
suite passed. The hardware-product-name audit found no new product name in
tracked changes. The final branch contains retained production commit
`c76c8b9`, reproducible report checkpoints, and no rejected production code.
The untracked pre-existing development Markdown was preserved untouched and is
not part of the branch.

External verification: a runnable `llama-server` at pinned revision
`13f2b28b0` is unavailable. The only discovered copies are in the desktop
trash, and one fails dynamic loading before it can report a revision. They were
not substituted for the required oracle.
