---
name: safe-optimization-loop
description: >-
  Iteratively optimize latency or throughput with a measure-change-verify loop
  that preserves the engine's chat output and repository invariants. Use for
  Rust inference hot paths, prefill/decode profiling, or performance tuning
  where correctness, isolated commits, and reproducible evidence are required.
---

<!--
This skill owns the local measure-candidate-gate-decision loop for authorized
optimizations; numeric criteria and measurements live in its references.
-->

# Safe Optimization Loop

Always communicate in English. Keep commands and identifiers in the language of
the code.

Obey `AGENTS.md`. A candidate that violates the approved structure, single
responsibility, or 200-productive-line limit has failed just like a red test.

Repeat:

```text
measure -> change one thing -> prove correctness -> remeasure -> commit or restore
```

## Rules

1. Correctness blocks the commit.
2. Do not change tests, references, or thresholds to make a candidate pass.
3. One logically independent optimization corresponds to one commit.
4. Keep the model, prompt, context, KV, features, profile, and hardware identical
   between the baseline and candidate.
5. Floating-point ordering, layout, or precision changes are not presumed
   bit-exact. Explore them only with an approved oracle and a threshold declared
   before measurement; otherwise do not commit them.
6. Do not introduce dependencies or abstractions for a hypothetical gain.

## Phase 0 — Baseline

Check the tree and target feature:

```sh
git status --short
cargo test --workspace --no-default-features --features cpu
cargo check --workspace --no-default-features --features <cpu|vulkan|vulkan-hybrid|metal|metal-hybrid>
```

Use a 3B Q4_K_M artifact authenticated by `support/models.tsv`. Q8_0 is only a
negative case for the public gate, not a real parity target. Record its SHA,
context, KV, and commit.

Read [references/correctness-oracle.md](references/correctness-oracle.md) for
the exact matrix and
[references/measuring-prefill-decode.md](references/measuring-prefill-decode.md)
for measurement instructions.

## Phase 1 — Profile

Measure the public API:

```sh
support/profiling/profile.sh --model "$GRAPH_HORIZON_MODEL" \
  --backend <backend> --context 4096 --kv f16
```

Identify the dominant cost from observable data: prefill, TTFT, decode, memory,
or placement. Read
[references/rust-optimization-catalog.md](references/rust-optimization-catalog.md)
only after profiling.

## Phase 2 — Classification

- Bit-exact: remove work, allocations, copies, or synchronization without
  reordering arithmetic. This requires a gate that proves exact equality.
- Layout/numeric: change grouping, FMA, SIMD reduction, tiling, precision, or
  quantization. Use an approved bounded numeric gate only when it measures the
  candidate's risk; otherwise define a specific threshold in advance.
- Architectural: change ownership, threading model, or orchestration. Keep this
  inside the loop when the explicit performance mandate includes it and the
  candidate passes targeted event, cancellation, placement, and capacity gates.
  Dependencies and public APIs still require explicit scope.

## Phase 3 — Change

Apply the smallest diff. Do not perform opportunistic cleanup. Check every
touched Rust file against `AGENTS.md` and document invariants at mutation points.

## Phase 4 — Gate

Run correctness first:

```sh
cargo fmt --check
cargo test --workspace --no-default-features --features cpu
support/testing/parity-check.sh \
  --models-dir "$GRAPH_HORIZON_MODELS_DIR" \
  --model-id "$GRAPH_HORIZON_MODEL_ID" \
  --backend <backend> --kv f16 \
  --reference-server "$GRAPH_HORIZON_REFERENCE_SERVER"
```

For a mixed hybrid backend, add `--weights-percent 25 --expect-mode mixed` and
require positive CPU and GPU layer counts. Do not turn Vulkan E15 into CPU
fallback.

Only after a green gate, remeasure with the same baseline command. A gain must
exceed observed variance without degrading non-target metrics or memory beyond
noise.

## Phase 5 — Decision

- All applicable gates green and a real gain: commit immediately with before
  and after metrics.
- Divergence, regression, or a gain within noise: restore only the candidate and
  record why.
- Candidate without an adequate approved gate: do not leave it in the tree;
  report it as an unapplied proposal.

Stop the loop when no measured candidates remain, the oracle is unstable, or
the requested budget is exhausted. Finally repeat the test matrix, parity, and
benchmark, then report the baseline, final state, and retained commits.
