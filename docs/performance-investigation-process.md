<!--
This page owns model-neutral performance-investigation process and policy. It
contains no measured result and makes no runtime or support claim.
-->

# Performance Investigation Process

Use this process for performance-sensitive changes after correctness and scope
have been approved. `examples/bench.rs` is the only retained iterative
performance executable. It measures one end-to-end public-event tuple; the
operator authenticates artifacts, records environment metadata, compares
revisions, and assigns the terminal state.

`support/profiling/profile.sh` may provide a placement and memory snapshot.
`support/profiling/validate-kv.sh` checks both public KV schemes, and
`support/testing/parity-check.sh` provides pinned-oracle parity. None of these
tools compares A/B records or makes a performance decision.

## Declare The Experiment

Before changing code, declare:

- the target: `prefill`, `decode`, or `both`;
- the intentional code variable;
- the changed runtime profile;
- the narrow correctness gate selected for that path.

### Autonomous Category-K Loop

AGENTS deroga E applies only to an existing category-K kernel, its directly
related tests, and this document's existing measurement path. Before editing,
verify that `HEAD` is on a dedicated `perf/<experiment>` branch and not on
`main`. Record the baseline and commit each candidate at a reproducible checkpoint.
A rejected candidate remains in branch history and is removed by a later commit;
do not amend, squash, or reset away experimental evidence during the loop.

The agent may iterate through implementation, checks, measurement, and rollback
without per-candidate approval. This autonomy covers local work and
commits only: pushing, opening a pull request, merging, or expanding the change
beyond the category-K boundary requires normal authorization. The final report
identifies the baseline, candidate, and any restoration revisions.

Prompt throughput is the end-to-end proxy for prefill. It is prompt-token count
divided by time to the first public text delta, so it includes first sampling;
it is not isolated prefill timing. Decode throughput uses intervals between
public text deltas, not raw model-token steps. TTFT runs from
`Engine::generate` entry to the first public text delta.

## Default Iterative Tuple

Define the default tuple once as follows:

- authenticated `3b-instruct` Q4_K_M artifact, including catalog ID, byte size,
  and SHA-256;
- Cargo build profile, repository revision, hardware, driver/runtime, and
  placement;
- context `4096`, KV `f16`, and greedy sampling;
- prompt `Quanto fa 17 × 19?` and 32 requested tokens;
- one warm-up and three measured repetitions;
- the benchmark's prompt-token and decoded-delta counts, TTFT statistics,
  prompt-throughput statistics, and decode-throughput statistics.

Every later reference to the default tuple means this complete definition.
Baseline and candidate may differ only by repository revision and the declared
intentional change. Treat the artifact and every recorded field as untrusted
until catalog authentication and tuple comparison pass.

Run only the profile whose path changed. If the change is shared by standalone
Metal and hybrid Metal, run two selected rows: `metal` and `metal-hybrid` with
`weights_percent=25`. CPU is not an automatic control row. Add `int8`, Vulkan,
8B, or 14B only when the approved change directly depends on that scheme,
backend, size, layout, capacity, or memory pressure. Select a larger model early
only when 3B cannot exercise the relevant property.

## Acquire Matching Records

Build and run baseline A, apply only the declared change, then build and run
candidate B with the default tuple. Store the benchmark's single-line records
beside the tuple metadata; do not add metadata to benchmark stdout. Do not
selectively repeat a favorable row or alter inputs after reading results.

The complete comparison must finish within two hours of elapsed wall time. Once
two hours are reached, do not start another row; allow an already running row to finish,
then assign `not_verified: time budget exceeded` if the comparison remains
unfinished.

For each metric, the benchmark reports the mean and sample standard deviation.
CV is sample standard deviation divided by the positive mean and is a
fractional ratio: `0.0500` means 5%. One measured repetition reports standard
deviation and CV as `n/a`, never zero.

If an objective CV exceeds 5% in either baseline or candidate, allow exactly
one complete rerun of the selected A/B tuple set. If any objective CV remains
above 5% after that rerun, assign
`not_verified: unstable measurement`. Never rerun an individual row alone.

## Compare And Classify

For throughput, improvement is `candidate / baseline - 1`. For TTFT,
regression is `candidate / baseline - 1`. Evaluate only the target declared
before acquisition. For target `both`, prompt and decode throughput must each
meet the selected threshold; a favorable aggregate cannot hide a failing row.

TTFT is always a control for prefill work. The non-target throughput metric is
also a control. Apply exactly one terminal state:

- `keep`: every objective improves by at least 5%, every objective CV in both
  records is at most 5%, correctness passes, and no control regresses by more
  than 5%;
- `interesting`: every objective improves by at least 3% but less than 5%, with
  the same stability, correctness, and control gates; preserve the evidence,
  not disproportionate candidate code;
- `reject`: any objective improves by less than 3%, correctness fails, a
  control regresses by more than 5%, the artifact mismatches, or the tuples are
  incomparable;
- `not_verified`: a prerequisite is absent, the two-hour budget expires, or
  the only complete rerun remains unstable.

Correctness failure or a control regression over 5% is `reject` regardless of
the target gain. A result at or above 5% does not bypass stability or control
gates.

## Mixed Operation Variants

A proposed variant used only under effective `Mixed` placement must declare its
correctness gate, objective, variability limit, and controls before code is
measured. It may remain in production only when those predeclared gates produce
the terminal state `keep`. The Cargo composition profile is not a dispatch
input, and one retained exception belongs to one operation only.

If the terminal state is `interesting`, `reject`, or `not_verified`, remove the
candidate code before completing the change. Its bounded evidence may remain in
the validation or decision record, clearly labeled with that terminal state.

## Prerequisite And Tuple Failures

Apply these conditions before interpreting performance:

- If the cataloged artifact is absent, assign
  `not_verified: artifact unavailable`; do not substitute another model or
  quantization.
- If artifact byte size or SHA-256 differs from the catalog, stop before
  inference and assign `reject: artifact mismatch`.
- If baseline and candidate differ outside revision and the intentional code
  change, assign `reject: incomparable tuple`.
- If required hardware, driver, local oracle, or another external resource is
  unavailable to the implementer, record
  `external verification: <exact missing prerequisite>` for that check and run
  every independent local check. This implementation status is not `keep` and
  is distinct from the A/B state `not_verified`.

## Correctness Gate

Choose the narrow gate before reading performance output:

- exact parity for bit-preserving runtime, scheduling, dispatch, and submission
  changes;
- the existing bounded numeric gate for approved numeric drift;
- KV payload, layout, and quality tests for KV changes;
- event-order and cancellation tests for orchestration changes;
- placement and capacity accounting for ownership or placement changes.

No performance result compensates for a correctness failure. A capacity error
describes that exact tuple and does not authorize a smaller context or a
different artifact.

## Release Qualification Boundary

The 74-row correctness matrix is outside iterative performance Definition of
Done. Run it only for release qualification or for a change to a shared backend
contract, public artifact compatibility, family-wide numeric behavior, KV
layout, or placement policy. Do not run it merely because one selected
performance row changed. Ordinary iterative work does not run separate 8B or
14B performance benchmarks.

## Reporting

The final investigation record contains:

- hypothesis, declared target, intentional change, and terminal state with
  exact reason;
- baseline and candidate revisions and benchmark records;
- artifact ID, size, SHA-256, hardware, driver/runtime, Cargo profile,
  placement, KV, context, prompt, requested tokens, warm-up, and repetitions;
- objective and control calculations, CV gate, rerun use, and elapsed time;
- chosen correctness gate and result;
- every `external verification` item and its exact missing prerequisite.

Do not publish an unlabeled token rate, convert prompt throughput into an
isolated phase claim, or describe an `interesting` signal as a verified
improvement.
