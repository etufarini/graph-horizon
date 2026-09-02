---
name: gpu-amdahl-optimizer
description: >-
  Find and implement substantial GPU inference performance improvements using
  critical-path profiling, Amdahl's law, correctness gates, and reproducible
  A/B benchmarks. Use for Graph Horizon decode, prefill, or TTFT optimization
  on CUDA, Vulkan, or Metal; do not use for unmeasured micro-optimization or
  backend-comparison reports without implementation.
---

<!--
This skill owns backend selection and evidence-driven GPU optimization. The
repository's performance process owns benchmark definitions and thresholds.
-->

# GPU Amdahl Optimizer

Obey `AGENTS.md` and the repository's current performance-investigation
documentation. Match the user's language unless they request another one.

## Select One Backend

Before planning, profiling, or editing, inspect the host tuple with read-only
commands such as `uname -s` and `uname -m`, then check the repository's current
build and qualification matrix in `docs/engine/backend-support-status.md`.

- If the invocation already names exactly one build-supported backend, select it.
- On supported Apple silicon, select Metal automatically and tell the user.
- When more than one requested-scope backend is build-supported, ask one concise
  question listing only those choices. On supported Linux x86_64 this is
  normally: **Which backend should I optimize: CUDA or Vulkan?**
- When exactly one backend is build-supported, select it and state why. When
  none is supported, stop after reporting the exact platform prerequisite.

Validate the selected backend and hardware before expensive work. Never infer
a choice from installed tooling, silently switch after a failure, or substitute
another backend or artifact. A missing prerequisite is an external-verification
item, not permission to alter the tuple.

## Establish Scope and Evidence

Optimize the selected backend's contribution to prompt/prefill throughput,
TTFT, model decode throughput, or public-delta throughput. Include memory and
placement only when measurements connect them to a target metric. Follow the
measured cost across kernels, dispatch, command buffers, barriers,
synchronization, readback, KV layout, allocation, and orchestration; a GPU task
does not imply that a shader is the bottleneck.

Read before editing:

- `docs/development/performance-investigation.md`;
- `docs/development/end-to-end-throughput-benchmark.md`;
- `.skills/safe_optimization_loop/references/correctness-oracle.md`;
- `.skills/safe_optimization_loop/references/measuring-prefill-decode.md`;
- `.skills/safe_optimization_loop/references/rust-optimization-catalog.md`.

Inspect reachable history and investigation records for the selected backend.
Include retained changes, corrective follow-ups, reverts, and removed or
rejected experiments. A commit message is a hypothesis, not measurement
evidence. Revisit a rejected idea only when new evidence changes a premise.

Before the first production edit, follow the canonical process: preserve a
clean worktree, use a dedicated `perf/<experiment>` branch rather than `main`,
authenticate the model with `support/models.tsv`, capture the complete baseline
tuple, and predeclare the target, intentional variable, correctness gate,
controls, stability limit, and retention threshold.

## Attribute Cost Before Optimizing

Use a profiling funnel so each more intrusive tool answers a narrower question:

1. Measure the unprofiled public outcome with `examples/bench.rs`.
2. Capture a low-overhead CPU/GPU timeline to locate critical-path GPU work,
   host launch gaps, transfers, waits, and accidental synchronization.
3. Add the smallest backend timestamp or counter range needed to attribute the
   selected interval when the timeline is insufficient.
4. Use deep kernel counters only after a hot kernel is proven material. Use a
   Roofline or equivalent limiter analysis to distinguish compute, bandwidth,
   latency, occupancy, and launch limits before choosing a kernel change.

The unprofiled public benchmark is the authority for end-to-end performance.
Do not compare its duration with a profiler run that replays work, serializes
queues, changes cache state or clocks, or disables overlap. Measure and disclose
instrumentation overhead; remove temporary instrumentation before completion
unless a reusable profiler was explicitly requested.

Backend-specific traps:

- **CUDA:** use Nsight Systems or an equivalent timeline before Nsight Compute.
  Nsight Compute can replay kernels, serialize launches, flush caches, and
  control clocks. CUDA events time work only with correct stream ordering;
  host timers require an appropriate synchronization boundary.
- **Vulkan:** use the profiler for the actual GPU vendor, not one inferred from
  the Vulkan API. Timestamp queries require a queue with nonzero
  `timestampValidBits`; convert with `timestampPeriod`, handle valid-bit wrap,
  and consume only available results. Keep query readback outside the interval.
- **Metal:** use Instruments' Metal System Trace for the CPU/GPU timeline and
  command-buffer GPU times or narrow counters for attribution. Treat Metal
  debugger counter results as diagnostic because counter profiling can run
  passes without their normal overlap.

When using those tools, consult their current official documentation:
[Nsight Compute profiling](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html),
[Vulkan timestamp queries](https://docs.vulkan.org/spec/latest/chapters/queries.html),
and [Metal counter statistics](https://developer.apple.com/documentation/xcode/analyzing-apple-gpu-performance-using-counter-statistics).

## Apply the Amdahl Gate

Let `T` be the baseline end-to-end time for the declared target and `t` the
causally removable critical-path time owned by a candidate. Use `p = t / T`, a
credible local speedup `s`, and any newly introduced normalized overhead `o`:

```text
predicted global speedup = 1 / ((1 - p) + p / s + o)
ideal ceiling            = 1 / (1 - p)
```

For a throughput objective, decompose elapsed time for a fixed amount of work,
not the reciprocal tokens-per-second value.

`t` is not GPU utilization, occupancy, the sum of overlapping kernel durations,
or all activity bearing the same label. CPU and GPU intervals may overlap; use
timeline dependencies, a synchronized boundary, or a reversible causal
ablation to identify removable critical-path time. If attribution remains
uncertain, calculate a range for `p` and use its conservative bound. State the
assumptions behind `s` and include transfer, launch, conversion, synchronization,
and cleanup costs in `o` rather than treating them as free.

Use the task's or canonical process's threshold. If neither defines one, reject
a candidate before implementation when even its ideal ceiling cannot plausibly
improve the target by 5%. Prefer the largest credible global effect, not the
largest local speedup. Recalculate `p` after each retained change and do not
bundle independent candidates.

## Run the Investigation Loop

Use the smallest authenticated model that exercises the path. Default to the
3B Q4_K_M artifact; use a larger model only for a measured size, capacity,
layout, or memory property.

Repeat:

```text
profile -> quantify with Amdahl -> change one thing -> prove correctness
        -> repeat the identical A/B benchmark -> commit or restore
```

Keep artifact bytes, prompt, token counts, context, KV, placement, sampling,
hardware, driver, build profile, runtime options, and environmental controls
identical. Observe competing GPU clients, clocks, power, and thermal state;
discard or classify throttled or incomparable records. Do not change system
clock, power, fan, or driver settings without explicit authorization. Do not
selectively rerun favorable rows.

Use the repository's repetitions, rerun rule, CV limit, controls, and terminal
states exactly as documented. CV is a dispersion screen, not a confidence
interval: report the effect and variability, but do not call a result
statistically significant unless an inferential method was predeclared and its
uncertainty supports the retention threshold.

The smallest viable change may cross backend layers when profiling requires
it, but it must preserve narrow interfaces and repository structure. Do not add
dependencies, change public APIs, weaken validation, or introduce speculative
abstractions. Numeric reordering, precision, quantization, or layout changes
need a risk-specific gate declared before their performance is observed.

## Gate and Decide

Run correctness before candidate performance. Use the applicable canonical
commands, including CPU workspace tests, one matching backend feature check,
the selected backend's local error tests, and authenticated real-model parity.
Add focused gates for KV payload, layout, and quality; placement and capacity;
event ordering and cancellation; or numeric drift when the candidate can affect
them. Never change tests, oracles, references, workloads, or thresholds to make
a candidate pass. If a model, reference server, backend, or hardware
prerequisite is missing, run all independent local gates and report the exact
missing prerequisite.

Apply the canonical `keep`, `interesting`, `reject`, and `not_verified` rules
without weakening them.

One retained optimization is one local commit containing its before/after
evidence. Remove rejected production candidates without destructive history
rewrites and record their negative result concisely so it is not repeated.

Stop at the task's completion condition, the documented budget, an unusable
correctness oracle, or when no measured candidate has a sufficiently large
conservative Amdahl ceiling. Do not continue into micro-optimization after the
measured goal is met.

## Final Report

Report the selected backend and host tuple; baseline revision and complete
benchmark tuple; historical evidence consulted; an Amdahl table with `p`, local
`s`, added overhead, predicted global speedup, ceiling, and uncertainty; every
candidate's terminal state and retained commit; exact before/after target,
variability, and control metrics; correctness and benchmark commands with
results; profiler caveats and external-verification items; remaining risks; and
the largest remaining measured bottleneck.

The final branch must contain only accepted changes or the restored baseline.
