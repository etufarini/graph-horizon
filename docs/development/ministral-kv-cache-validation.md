<!--
This process owns the repeatable f16/int8 KV-cache comparison for Ministral. It
defines protocol and required evidence, not an implicit qualification result.
-->

# Ministral KV-Cache Validation

## Purpose

Compare `f16` and `int8` while keeping artifact, backend, placement, prompt,
and context `4096` unchanged. The result must separately demonstrate correct
layout, completed execution, and the numeric criterion chosen before the run.
Recognizing a GGUF format is not KV evidence.

## Prerequisites

- a readable Ministral Q4_K_M artifact authenticated by size and SHA-256;
- recorded Git revision and Cargo profile;
- one buildable `cpu`, `vulkan`, `vulkan-hybrid`, `metal`, or `metal-hybrid` profile;
- context `4096` without a placement change;
- hardware and driver required by that backend;
- a numeric metric and threshold fixed before observing the result.

A missing external resource produces `external-verification`. It does not
authorize another artifact, backend, context, or KV scheme.

## Matrix

Each artifact/backend row executes exactly two cases:

| Case | Context | KV |
|---|---:|---|
| baseline | 4096 | `f16` |
| candidate | 4096 | `int8` |

Every public profile requires both KV rows. Hybrid percentage and reserve stay
identical. Load reports must include at least `cpu_layers` and `gpu_layers`; a
placement difference invalidates the comparison instead of becoming a numeric
result.

## Procedure

1. Authenticate the read-only artifact against the catalog.
2. Record revision, hardware, backend, context, prompt, and output bound.
3. Run synthetic KV layout, quantization, and arithmetic tests.
4. Run one `f16` row and preserve its output and placement.
5. Run one `int8` row with only the KV scheme changed.
6. Compare completion, tokens, and the predetermined numeric metric.
7. Assign one terminal status with a concrete reason.

The operational command is:

```sh
support/profiling/validate-kv.sh \
  --model "/path/to/model.gguf" --backend cpu --context 4096
```

The script accepts all five profiles and runs f16 followed by int8. It does not
authenticate `models.tsv`, compare output or placement, apply a threshold, or
assign a verdict; those remain responsibilities of this procedure and the
reviewed report.

## Required Evidence

Record for each case:

- artifact identifier, byte count, and SHA-256;
- revision, Cargo feature, backend, and hardware;
- KV scheme and context;
- `cpu_layers`, `gpu_layers`, and allocated memory;
- load and generation outcome;
- metric, threshold, and numeric verdict;
- any external limit that prevented execution.

Synthetic tests do not replace an artifact run. A completed generation alone
does not prove that int8 loss satisfies the threshold.

## Terminal States

- `verified`: both cases ran and the criterion passed;
- `failed`: configuration, execution, or numeric criterion failed;
- `external-verification`: a required external resource was unavailable.

Every row receives one state. Reviewed results belong in
[validation evidence](../project-status/validation-evidence.md); this process
does not claim that an unexecuted check passed.
