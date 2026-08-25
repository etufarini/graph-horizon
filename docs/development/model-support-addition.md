<!--
This document owns the model-neutral engineering process for extending model
support. It defines implementation gates, not the current support claim.
-->

# Model Addition Process

This process applies both to a new model family and to a new profile inside an
existing family. A profile is not supported merely because its GGUF parses: the
public claim begins only after loading, generation, backend and validation gates
have passed and the support contract has been updated.

## Scope

Classify the change before editing:

- **new family** when architecture, tokenizer, template, tensor mapping or
  forward graph differs;
- **existing family profile** when the graph is unchanged and only an explicitly
  supported set of dimensions, metadata or tensor formats expands;
- **validation-only artifact** when no runtime behavior or support boundary
  changes.

Do not route a materially different graph through an existing family by adding
filename checks or scattered conditionals.

## Preflight Inventory

Record the following from the candidate artifact and its authoritative source:

- `general.architecture`, context, vocabulary and tokenizer metadata;
- required and optional metadata with accepted types and ranges;
- tensor names, dimensions, byte layouts and quantization types;
- attention, positional encoding, normalization, MLP or expert topology;
- chat-template and special-token behavior;
- expected backend, KV and context combinations;
- reference runtime, revision, prompt, sampling and expected token IDs when an
  external oracle is needed;
- artifact size and SHA-256 for evidence tracking, never for runtime detection.

Treat the GGUF as untrusted. Use checked arithmetic before allocations and keep
parse errors free of local paths, stack traces and raw system details.

## Structure Approval

Before production edits, propose the smallest file tree that gives every file one
responsibility. Orchestration files remain within the repository's 200 productive
line limit; kernel or trait exemptions must use the exact AGENTS marker. Split a
domain into a folder only when it genuinely needs more than one file.

The invariant, smallest viable change and main new risk must be explicit before
implementation.

## Implementation Order

1. **Detection**
   Select the family from architecture and capabilities. Never authenticate a
   model by filename, path, marketing name, size or digest.
2. **Configuration**
   Parse typed metadata, validate dimensions and reject contradictions before
   backend allocation.
3. **Tensor contract**
   Bind required names, shapes and formats explicitly. Optional tensors need one
   narrow fallback rule.
4. **Tokenizer and template**
   Validate vocabulary, special IDs, merge tables and supported template
   behavior. Do not execute untrusted template code implicitly.
5. **Graph**
   Implement the smallest cohesive prefill/decode path. Keep family math outside
   application and transport modules.
6. **Backend operations**
   Reuse the existing `Backend` contract. Add an operation only when the graph
   cannot be expressed correctly with the current interface.
7. **KV and placement**
   Derive sizes from checked model dimensions and document which backend owns
   each layer's state.
8. **Facade**
   Expose only stable, family-neutral concepts through `Engine` unless public
   inspection genuinely requires a typed family object.

## Required Invariants

Every supported model must preserve these properties:

- unsupported artifacts fail before UI, bind or backend allocation where their
  incompatibility is already knowable;
- model files are opened read-only and never modified;
- all tensor spans and allocation sizes use checked arithmetic;
- request state, KV state and placement cannot cross model instances;
- cancellation terminates generation and releases owned resources;
- one generation emits at most one terminal public event;
- public errors do not expose paths, driver internals or stack traces;
- internal kernel formats do not silently expand the public GGUF contract.

## Validation Matrix

Define one row per artifact, backend, KV scheme, context and relevant placement.
Do not compress untested combinations into a broad support statement.

| Gate | Required evidence |
|---|---|
| source structure | single-purpose files, architectural comments and line policy |
| synthetic parsing | metadata, tokenizer, shape, byte-span and rejection tests |
| real load | approved GGUF loads read-only; unsupported variants fail cleanly |
| prompt parity | rendered IDs match the pinned reference when an oracle exists |
| CPU reference | deterministic short generation and backend-level numeric checks |
| accelerated backends | each claimed backend runs without hidden fallback |
| hybrid placement | all-GPU, mixed and CPU-only rows required by the claim |
| KV schemes | layout, payload and quality gates for every public scheme |
| streaming | text, stats, error and cancellation contract |
| performance | benchmark after correctness, without replacing correctness |
| documentation | current support, commands, evidence and limitations |

Synthetic tests protect ordinary development runs. Real-model rows remain the
release evidence and may require external hardware or artifacts.

## Numeric Acceptance

Use exact equality for discrete contracts such as token IDs, tensor byte counts,
shape arithmetic and normative serialized payloads. Use documented tolerances
for floating-point backend drift and lossy KV formats.

Possible bounded gates include top-1 agreement, maximum relative logit error,
KL divergence, perplexity ratio and membership of an oracle token in local top-k.
Choose thresholds before looking at the candidate result and record the greedy
sequence per backend/KV row.

## Oracle Policy

An external oracle must pin runtime name, version or commit, artifact, prompt,
template, context and sampling. Run it offline where possible and collect
structured token IDs rather than scraping presentation text.

Missing binaries, devices or artifacts are external verification gaps. Malformed
oracle output or a failed numeric gate is a validation failure. Oracle evidence
checks behavior; it does not establish model identity.

## KV Validation

For every public KV scheme:

- compare the same artifact with only the KV scheme changed;
- verify allocation arithmetic and per-vector metadata;
- compare payload bytes when a normative host quantizer exists;
- measure numerical drift against the baseline scheme;
- exercise every backend and placement included in the support claim.

## Performance Pass

Run performance measurement only after correctness gates pass. During
implementation and profiling, use the smallest supported model that exercises
the affected path. Validate the other supported models once after the candidate
is complete; do not use them for iterative tuning.

Move a larger model earlier only when model size, capacity, layout, or memory
pressure is itself part of the change. Larger-model performance runs are
required only by an explicit task acceptance criterion. The current
model-neutral harness requires an explicit model, context and KV scheme:

```sh
cargo run --no-default-features --features cpu --example bench -- \
  /path/to/model.gguf --context 4096 --kv f16
```

Repeat with `vulkan`, `vulkan-hybrid`, `metal`, or `metal-hybrid` only when
those rows are claimed. Record context,
KV, build profile, hardware, prompt, generation length, warmup and repetitions.
An allocation failure is capacity evidence, not a correctness result.

## Documentation Checklist

After acceptance, update:

- the engine README support contract;
- `support/models.tsv` and operational scripts when the catalog changes;
- runtime flags only when their behavior changes;
- `docs/project-status/validation-evidence.md` with artifact, matrix, thresholds and results;
- benchmark notes when performance was measured;
- this process only when the reusable method changes.

## Definition Of Done

A conforming family exposes weights through `WeightSource` and execution
through `LayeredGraph`. It thereby gains homogeneous and partitioned profiles
from the runtime without importing a concrete backend or creating backend-pair
modules. Family tests assert the neutral contract; device tests stay in backend
or selected-runtime integration coverage.

The addition is complete when unsupported variants fail deliberately, supported
artifacts load and generate through the public Rust API, tokenizer/template behavior
is verified, every claimed backend and KV row has evidence, performance is
measured or explicitly out of scope, and documentation states the exact support
boundary without relying on filenames or implied compatibility.
