<!--
This page owns model-neutral performance-investigation process and policy. It
contains no measured result and makes no runtime or support claim.
-->

# Performance Investigation Process

Use this process when an explicit task requests performance investigation or
optimization. The request defines the authorized target, invariants, and local
experimental scope; it does not require a second approval before measurement or
implementation. `examples/bench.rs` is the retained end-to-end public-event
benchmark. Temporary test-only instrumentation or focused diagnostic paths may
be used when that benchmark cannot attribute the measured cost; remove them
before completion unless the task requires a reusable profiling facility.

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

### Active Long-Context Prefill Investigation

- Target: `prefill`; decode is a regression control only.
- Intentional variable: feature-gated Vulkan attribution detail, followed by
  isolated prefill-only causal and structural candidates selected from it.
- Runtime profile: standalone Vulkan, F16 KV, 32,768-token context, exact prompt
  lengths from 128 through 28,000 tokens, and greedy sampling.
- Correctness gate: profiler unit tests and release build for diagnostics;
  bounded Vulkan attention oracle plus end-to-end generation for numeric kernels.

### Autonomous Investigation Loop

Before editing production code, verify that the worktree is clean and `HEAD` is
on a dedicated `perf/<experiment>` branch rather than `main`, then record the
requested baseline. The agent may follow measured cost across kernels, cache
layout, orchestration, dispatch, synchronization, shaders, allocations, and
supporting tests or benchmarks. It may iterate through instrumentation,
implementation, checks, measurement, and rollback without per-candidate
approval.

Each candidate must be isolated and measured against the same applicable
baseline. Retained production changes and reproducible checkpoints are committed
locally. Rejected production changes are removed without rewriting history, and
their result is recorded concisely so the same hypothesis is not repeated.
Pushing, opening a pull request, merging, external side effects, unrelated scope,
new dependencies, or public API changes are not authorized implicitly. The
final report identifies baseline, retained candidates, rejected candidates, and
any restoration revisions.

### Task-Directed Measurements

An explicit task's workload matrix, metrics, thresholds, correctness gates, and
completion conditions take precedence over the defaults below. This includes
requests for multiple context lengths, isolated GPU timestamps, dispatch and
barrier counts, hardware telemetry, causal ablations, parameter sweeps, or
continued profiling after an initial improvement. Missing support for a
required metric calls for the smallest reversible instrumentation experiment;
it is not a reason to request approval.

Prompt throughput is the end-to-end proxy for prefill. It is prompt-token count
divided by time to the first public text delta, so it includes first sampling;
it is not isolated prefill timing. Decode throughput uses intervals between
public text deltas, not raw model-token steps. TTFT runs from
`Engine::generate` entry to the first public text delta.

### Vulkan Phase Attribution

Build the existing benchmark with `--features vulkan-profile` when Vulkan GPU
timestamps are required. At device shutdown it reports accumulated `prefill`,
`decode`, and `sampling` records separately. Each phase includes command,
dispatch, and barrier counts; total and classified GPU time; residual and
accounted percentage; CPU record, submit, and wait time; and operation-category
totals. Allocation count and device/host bytes are reported once. The totals
include warm-up and measured repetitions, so divide them only by the matching
command or repetition count from that exact run. The profiler does not replace
the public-event throughput and TTFT record used for A/B retention decisions.

## Default Iterative Tuple

Use this tuple only when the task does not define a more representative
workload or measurement matrix.

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
8B, or 14B only when the task or measured bottleneck directly depends on that
scheme, backend, size, layout, capacity, or memory pressure. Select a larger
model early only when 3B cannot exercise the relevant property.

## Acquire Matching Records

Build and run baseline A, apply only the declared change, then build and run
candidate B with the default tuple. Store the benchmark's single-line records
beside the tuple metadata; do not add metadata to benchmark stdout. Do not
selectively repeat a favorable row or alter inputs after reading results.

Unless the task defines different continuation or completion conditions, the
complete comparison must finish within two hours of elapsed wall time. Once two
hours are reached, do not start another row; allow an already running row to
finish, then assign `not_verified: time budget exceeded` if the comparison
remains unfinished.

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

Use the task's explicit retention threshold when it provides one. Otherwise,
apply the default terminal states below.

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
