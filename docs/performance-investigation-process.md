<!--
This document owns the model-neutral process for investigating performance. It
keeps measurement subordinate to correctness and current repository tooling.
-->

# Performance Investigation Process

Use this process for changes affecting prompt throughput, time to first token,
decode throughput, memory residency, host-device transfer, kernel dispatch,
batching, KV behavior, or backend placement.

Correctness comes first, measurement second, and optimization third. Performance
cannot justify a changed numeric or public behavior unless that trade-off was
explicitly approved before implementation.

The initial Metal qualification has no minimum speed or standalone-to-hybrid
ratio. Once correctness passes, record finite TTFT milliseconds, prompt
tokens/second, and decode tokens/second; those values are descriptive evidence.

## Freeze A Baseline

Before editing performance-sensitive code, record:

- repository revision and build profile;
- authenticated artifact and weight format;
- backend feature and final placement;
- KV scheme and context;
- prompt text and maximum generated tokens;
- warmup and repetition counts;
- hardware, driver/runtime, operating system, and relevant environment values;
- benchmark output and any placement or profiling record used.

If the baseline cannot be reproduced, repair the measurement setup before
changing code. Do not replace an unavailable artifact or backend with a more
convenient row.

## Classify The Bottleneck

Identify the dominant class before choosing an optimization:

| Class | Typical Evidence |
|---|---|
| Compute | Kernel time dominates and faster math changes wall time |
| Bandwidth | Weight, KV, crossing, or readback traffic dominates |
| Residency | Placement or capacity prevents the intended device execution |
| Submission | Small dispatches or synchronization dominate token time |
| KV/context | Decode cost grows with context and attention dominates |
| Host orchestration | Scheduling, allocation, formatting, or waits dominate |

A faster kernel does not improve a run dominated by transfer or submission.
State the bottleneck as a measured hypothesis, not an explanation inferred only
from end-to-end token rates.

## Current Tools

Use the smallest retained tool that answers the question:

| Tool | Purpose |
|---|---|
| `examples/bench.rs` | End-to-end prompt throughput, TTFT, and public-stream decode throughput |
| `support/profiling/profile.sh` | Immutable placement/memory report and one timing sample |
| `support/profiling/validate-kv.sh` | Same model/backend/context across both public KV schemes |
| `support/testing/parity-check.sh` | Pinned external-oracle parity for an approved catalog row |
| Backend tests and opt-in diagnostics | Narrow kernel, allocation, dispatch, or transfer questions |

There is no retained `validate` or `regression` example. Benchmark tools report
measurements; validation gates decide pass or fail.

## A/B Method

Run comparisons with one intentional variable changed:

1. capture baseline A;
2. apply the change or enable candidate B;
3. keep artifact, backend, KV, context, prompt, generation length, warmup, and repetitions fixed;
4. run the applicable correctness gate;
5. compare TTFT, prompt throughput, decode throughput, memory, and the targeted metric;
6. keep, revise, disable, or revert the candidate.

If two variables change, split the experiment. Record variance rather than
treating one favorable sample as a result.

For the current Metal comparison, freeze the complete tuple: authenticated
`Ministral-3-3B-Instruct-2512-Q4_K_M.gguf`, context 4096, KV f16, prompt
`Quanto fa 17 × 19?`, 32 requested tokens, one warmup, and three repetitions.
The standalone row uses `metal`; the partitioned row uses `metal-hybrid` at 25%
and must report `mixed`. Changing any tuple field creates a different experiment.

## Context Regimes

Use explicit context values. The current benchmark has no context presets and no
model environment fallback:

```sh
cargo run --release --no-default-features --features cpu --example bench -- \
  /path/to/model.gguf --context 2048 --kv f16 \
  --prompt "Hello" --max-tokens 32 --warmup 1 --reps 3

cargo run --release --no-default-features --features cpu --example bench -- \
  /path/to/model.gguf --context 8192 --kv f16 \
  --prompt "Hello" --max-tokens 32 --warmup 1 --reps 3
```

Choose context rows that fit the approved artifact and hypothesis; the values
above are examples, not release presets. A high-context allocation failure is a
capacity result and does not authorize retrying with another context.

For a placement and memory snapshot, use the retained wrapper:

```sh
support/profiling/profile.sh --model /path/to/model.gguf \
  --backend cpu --context 4096 --kv f16
```

## Correctness Gate

Select the gate before reading performance results:

- exact parity for bit-preserving refactors;
- bounded logit, KL, perplexity, or teacher-forced top-k for numeric drift;
- payload, layout, and quality gates for KV changes;
- event ordering and cancellation tests for orchestration changes;
- unchanged placement and capacity accounting unless placement is the variable.

The [oracle validation process](oracle-validation-process.md) defines evidence
and terminal states. A performance path that misses its correctness gate is
removed or remains disabled; its speed is not an acceptance argument.

## Keep Or Revert

Keep a candidate only when:

- the targeted bottleneck improves in the intended regime beyond noise;
- correctness and validation gates pass;
- memory use and failure modes remain explicit;
- unrelated supported regimes remain within the approved budget;
- added complexity is proportionate to the measured benefit.

Revert or keep disabled when the wrong bottleneck was targeted, the result is
noise, the gain disappears at the target context, another required metric
regresses without an approved trade-off, or the path depends on undocumented
hardware behavior.

## Reporting

A performance record includes the change and hypothesis, artifact/backend/KV/
context/hardware tuple, baseline and candidate samples, variance, correctness
gate, capacity changes, and one decision: keep, revert, keep disabled, or
external verification.

Do not publish an unlabeled token rate. TTFT and decode throughput depend on
context, prompt, generation length, backend, build, and hardware.

## External Verification

Missing required hardware, driver, artifact, or memory capacity is external
verification, not pass. Local compile, synthetic, and correctness gates still
run when a hardware-specific measurement cannot be attempted.

## Definition Of Done

An investigation is complete when the baseline is reproducible, the bottleneck
was measured, A/B rows changed one variable, correctness passed, the result was
classified, and the record contains enough fixed inputs to repeat the run.
