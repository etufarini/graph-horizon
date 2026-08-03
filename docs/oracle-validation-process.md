<!--
This document owns the model-neutral process for comparing engine behavior with
numeric or external oracles. Validation campaign results belong elsewhere.
-->

# Oracle Validation Process

Use this process when a change can affect tokenizer output, prompt rendering,
logits, token selection, KV contents, weight math, streaming behavior, or backend
parity. It defines evidence requirements and does not claim support for any
model, artifact, backend, or quantization profile.

Current qualification results belong in [VALIDATION.md](../VALIDATION.md). The
broader release workflow is defined in
[model-validation-process.md](model-validation-process.md).

## Oracle Types

Choose the smallest oracle that proves the contract:

| Oracle | Use |
|---|---|
| CPU backend | Local numeric reference for backend and kernel work |
| File oracle | Stable cross-build reference for logits, tokens, or numeric records |
| External runtime | Template behavior, tokenizer IDs, or generation semantics |
| Normative implementation | Byte-level reference for quantizers, layouts, and scalar conversions |

An external runtime is a pinned validation tool, never a runtime dependency.
Every row records its executable identity, version or commit, arguments, context,
prompt, and sampling parameters.

## Fixed Inputs

Freeze these inputs before observing output:

- authenticated model artifact or approved artifact identity;
- backend feature and final placement, when applicable;
- weight format and KV scheme;
- context and maximum generated tokens;
- prompt text, conversation, or prompt token IDs;
- sampling parameters, including greedy mode or seed;
- oracle source and revision;
- acceptance criterion and threshold.

Do not compare rows that differ in prompt rendering, artifact, context, or
sampling unless that difference is the declared subject of the test. Never retry
with a different backend, context, prompt, or artifact after a failed row.

## Prompt And Tokenizer Gate

Prompt IDs are the first gate whenever tokenization or templates matter. Prefer
exact equality. When an external runtime renders the prompt, collect IDs through
a structured API, validate response types and bounds, and retain the exact
conversation used. Do not scrape presentation logs.

If prompt IDs differ, stop the numeric comparison. Later logit or greedy output
does not isolate backend behavior when the model received different tokens.

## Numeric Gates

Use exact equality for contracts that are exact:

- prompt token IDs;
- scalar conversion vectors;
- tensor and KV byte layouts;
- quantizer payload bytes when a normative encoding exists;
- a reference and optimized path that promise bit identity.

Use predeclared bounds when reduction order, floating-point execution, or
quantization makes exact equality inappropriate:

- argmax identity plus maximum relative logit difference;
- top-1 agreement plus `mean_kl` and `max_kl`;
- perplexity ratio against a saved oracle;
- teacher-forced oracle token inside local top-k at every checked position.

The gate must match the risk. A non-lossy kernel refactor normally needs stronger
parity than a lossy KV or weight-format experiment.

Greedy output alone is not always sufficient across different numeric backends
when the leading logits are nearly tied. In that case, record local greedy IDs
but gate on the approved distribution or teacher-forced metric. Exact greedy
equality remains appropriate inside a path that explicitly promises it.

## File Oracles

A file oracle must define:

- a magic value and version;
- model-independent dimensions required to validate the payload;
- exact byte-length checks using overflow-safe arithmetic;
- token IDs or position count;
- explicit byte order and numeric encoding;
- rejection of malformed lengths, vocabulary mismatch, and non-finite values.

The verifier uses the file's stored tokens or positions. It must not silently
re-tokenize another prompt or infer missing configuration.

## External Runtime Procedure

The retained `support/testing/parity-check.sh` demonstrates the current external
runtime boundary for cataloged rows: it authenticates the artifact, pins the
oracle revision, binds the oracle to loopback, obtains structured prompt and
completion IDs, then runs one exact local parity test.

That script has a deliberately narrow catalog contract. A different family or
oracle requires its own approved catalog and test boundary; do not broaden the
existing script through filename guesses or implicit fallback.

## Terminal States

Classify each row as exactly one of:

- **pass**: execution completed and met the criterion;
- **fail**: execution completed or started but violated the protocol or gate;
- **external verification**: an authenticated prerequisite was unavailable.

Missing hardware, driver, artifact, oracle binary, or free loopback port can be
external verification. Malformed oracle output, prompt-ID mismatch, HTTP failure
after successful startup, or numeric mismatch is a failure. External
verification is never counted as pass.

## Evidence Row

Record an explicit row containing the artifact identifier and digest, backend,
placement, weights, KV, context, oracle revision, criterion, measured value, and
terminal state. Avoid statements such as "works on GPU" without the complete
configuration and threshold.

## Safe Handling

Treat artifacts and oracle responses as untrusted:

- quote paths in scripts and reject invalid catalog values;
- validate JSON types and non-negative bounded token IDs;
- validate file lengths before reading slices;
- own and clean up child processes and temporary directories;
- bind local oracle servers to `127.0.0.1`;
- keep absolute paths, raw logs, and stack traces out of published evidence.

## Definition Of Done

Oracle validation is complete when inputs were frozen, artifact identity was
checked, prompt parity preceded numeric comparison, the criterion matched the
risk, every row has one terminal state, and the evidence can be reproduced
without relying on memory of the session.
