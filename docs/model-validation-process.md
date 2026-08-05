<!--
This document owns the reusable technical and semantic model-validation process.
Concrete model rows, thresholds and results belong to the validation register.
-->

# Model Validation Process

## Purpose

Validation answers two separate questions:

1. can the exact artifact be parsed, planned and executed by the claimed runtime;
2. does it satisfy a fixed semantic contract under one recorded configuration.

Technical compatibility does not imply semantic qualification. Conversely, a
semantic miss under one corpus and sampling profile does not prove that the model
is generally defective.

## Inputs

Start from an explicit catalog row containing:

- stable model identifier, family and profile;
- expected artifact name, byte count and SHA-256;
- architecture, file type and tensor contract;
- backend, placement, KV, context and sampling rows to exercise;
- external oracle or hardware requirements;
- semantic corpus, normalization, stop rules and thresholds.

The catalog authenticates evidence. Runtime detection must still use GGUF
metadata and capabilities, never the catalog filename or hash.

## Phase 1: Freeze The Contract

Before inference, define the smallest matrix that proves the intended claim:

- accepted and rejected formats;
- backend, KV and placement combinations;
- prompts and expected tokenizer behavior;
- exact numeric metrics and thresholds;
- semantic cases, required cases and aggregate gate;
- sampling, context, maximum generation and stop policy;
- terminal states and their precedence;
- resources whose absence prevents an attempt.

Observed output must not authorize retries, prompt tuning, threshold changes,
artifact replacement or fallback to a different backend.

## Phase 2: Authenticate Artifacts

Before a real-model run, verify that each cataloged file is readable, has the
expected byte count and matches its SHA-256. Open it read-only and never replace a
missing or mismatched artifact with a convenient local file.

An unavailable or unauthenticated artifact receives `external-verification` with
a concrete reason. It is still untrusted input: the loader must validate all
metadata, tensor records and offsets independently.

## Phase 3: Enforce Format Boundaries

For the family under test, state accepted architecture values, quantization
profiles, metadata types, tokenizer requirements, tensor names and shapes.
Reject a known unsupported profile before tokenizer construction or backend
allocation whenever possible.

Parser recognition is not execution support. Retaining an enum variant so an
unsupported format can receive a precise error does not add that format to the
public contract.

## Phase 4: Run Synthetic Gates

Run model-free tests before accessing external artifacts. They should cover:

- metadata type/range validation and checked arithmetic;
- tensor shape, offset and byte-span boundaries;
- tokenizer fixtures and prompt rendering;
- KV size/layout and quantizer payloads;
- backend operation contracts and shape errors;
- placement arithmetic and deterministic policy branches;
- event ordering, cancellation and error normalization;
- validation script parsing and hostile/malformed records.

Real-model tests should remain ignored or explicitly gated during ordinary test
runs so editing protocol code cannot trigger accidental inference.

## Phase 5: Run The Technical Matrix

Execute every authenticated supported artifact across the matrix frozen in Phase
1. Every row names one explicit compile-time profile; there is no runtime
backend default. A row must keep its model, profile, KV scheme, context and
placement unchanged after an error.

Each row records at least:

- artifact identifier and digest;
- build feature and revision;
- requested and final placement;
- context and KV scheme;
- load/generation outcome;
- numeric or oracle verdict;
- external limitation when it was not attempted.

Missing hardware, memory or oracle programs are visible external states, not
reasons for an implicit CPU or smaller-context retry.

For a partitioned profile, assert the generic runtime facts: requested mode and
percentage, positive host and device layer counts for `mixed`, endpoint ownership
for 0% and 100%, and the expected number of crossings per pass. These assertions
belong to selected-runtime integration tests; model-family code must not import
Metal, Vulkan, or any backend pair.

## Phase 6: Check Oracle Parity

Use an external oracle only when its executable, revision and arguments are
pinned. Compare the exact rendered prompt IDs first; a generation comparison is
invalid when tokenization already differs.

Depending on the approved contract, compare completion IDs, teacher-forced
logits, top-k membership, KL divergence or perplexity. Record both the metric and
its predeclared threshold. Short-horizon parity is numeric evidence and does not
replace long free-running semantic tests.

## Phase 7: Gate The Validation Protocol

Before semantic inference, test the runner itself:

- exact catalog and case ordering;
- serialized sampling and context configuration;
- completion, context-limit, max-token and error stop classification;
- response normalization;
- optional structured-marker parsing only for profiles that require it;
- score and aggregate calculations;
- terminal-state precedence;
- missing, duplicated, contradictory and malformed records;
- shell syntax and compilation of every backend the runner may invoke.

The protocol output must be bounded and machine-checkable. Do not publish local
absolute paths, raw stack traces or unbounded model output.

## Phase 8: Run Semantic Qualification

For each attempted row, use the authenticated artifact and the exact frozen
configuration. Execute each case once, continue through later cases for a
complete report, and do not retry after seeing an answer.

Only completed responses are scored when completion is part of the contract.
Context or max-token stops remain visible failures if EOS is required. Diagnostic
cases may be non-blocking but cannot be rewritten as passes.

## Phase 9: Classify Terminal States

Every catalog row receives exactly one terminal state. A matrix must not omit a
row, retry it under another profile, or leave it pending:

### `qualified`

The attempted configuration produced structurally valid records and passed every
required technical and semantic gate.

### `not-qualified`

The row was attempted but configuration, protocol, engine, completion or semantic
requirements failed. Define a deterministic reason precedence before the run so
one row cannot acquire contradictory final causes.

### `external-verification`

The row was not attempted because a required artifact, digest, command, device or
memory policy was unavailable. This is neither a pass nor a model-quality failure.

The three counts must sum to the fixed catalog size:

```text
summary: qualified=<n> not_qualified=<n> external_verification=<n> total=<count>
```

## Phase 10: Publish Evidence

The reviewed register contains:

- implementation revision and catalog hashes;
- unsupported-format rejection evidence;
- technical matrix summary;
- oracle runtime and parity results when applicable;
- semantic corpus, configuration and thresholds;
- one terminal row per catalog item;
- aggregate counts, limitations and explicit non-claims.

Preserved historical evidence must identify its originating revision and may be
reused only when a reviewed plan says the relevant runtime contract is unchanged.
Current repository evidence belongs in [VALIDATION.md](../VALIDATION.md); catalog
and runners live under [`support/`](../support/README.md).

## Non-Negotiable Boundaries

Validation does not modify model artifacts, infer on mismatched files, retry to
obtain a pass, hide unsupported fallback, expose private reasoning traces or turn
an unmeasured explanation into a claimed root cause. Any intended runtime change
returns to the model-addition process and receives its own implementation review.
