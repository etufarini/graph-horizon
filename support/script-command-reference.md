<!--
This reference owns the command-line interfaces and bounded behavior of the
profiling and validation scripts under `support/`. It does not define runtime
options, installation policy, or current qualification status.
-->

# Support Script Command Reference

Support scripts orchestrate existing builds and checks. They do not download
models, modify GGUF files or user configuration, or retry with a different
backend, context, or model.

| Script | Purpose |
|---|---|
| `profiling/profile.sh` | Model-neutral memory, placement, and throughput snapshot |
| `profiling/validate-kv.sh` | Runs explicit f16 and int8 KV profiles; authentication and verdict remain with the caller |
| `profiling/validate-weights.sh` | Authenticates the six Q4_K_M artifacts and synthetic internal formats |
| `testing/parity-check.sh` | Compares one exact prompt and top-two result with the pinned oracle |
| `testing/matrix-check.sh` | Runs the six Q8 rejections, 60 primary rows, and eight hybrid endpoints |
| `testing/public-readiness.sh` | Verifies a clean installed product, local backends, and one public benchmark |
| `testing/release-integrity.sh` | Verifies one published source release against its immutable version tag |
| `testing/semantic-check.sh` | Runs the terminal Reasoning semantic qualification matrix |
| `testing/run-graph-horizon.sh` | Starts the local terminal with an explicit test tuple |

Installer behavior is documented separately in the
[installation guide](../docs/installation.md).

## External Prerequisites

Real-model campaigns require authenticated, read-only GGUF artifacts plus
`curl`, `jq`, `stat`, and either `sha256sum` or `shasum -a 256`. Oracle parity
requires `llama-server` at revision `13f2b28b0`. Metal profiles additionally
require macOS arm64 and `xcrun metal`/`metallib`.

Artifact identities and SHA-256 records are in [`models.tsv`](models.tsv).
Every model directory is supplied explicitly as `--models-dir DIR`; scripts do
not search personal directories or silently substitute a resource. A missing
external resource produces `external verification: <precise reason>` without
skipping independent synthetic checks.

## Installer Verification

```sh
bash -n install.sh support/install.sh
cargo test -p graph-horizon --no-default-features --features cpu installer_
cargo test -p graph-horizon --no-default-features --features cpu bootstrap_
```

The end-user installer interface and installed paths are documented in the
[installation guide](../docs/installation.md).

## Release Integrity

Run the verifier from a checkout containing the annotated version tag:

```sh
support/testing/release-integrity.sh \
  --repository etufarini/graph-horizon \
  --tag v0.1.4
```

The command resolves the local and anonymous remote tag, downloads the named
archive and adjacent checksum without credentials, verifies the checksum,
embedded Git commit, versioned root, and member paths, then prints the compared
identities and `PASS`. It compares the artifact only with its version tag;
`HEAD` and `main` may contain later development. Any tag/archive identity
mismatch fails even when the two commits have identical Git trees.

## Public Readiness

The canonical week-2 campaign is:

```sh
support/testing/public-readiness.sh \
  --model-id 3b-instruct \
  --model /absolute/path/Ministral-3-3B-Instruct-2512-Q4_K_M.gguf \
  --benchmark-backend vulkan \
  --report /absolute/path/public-readiness.md
```

The runner authenticates the explicitly selected catalog row and exact artifact, resolves
the current commit, and checks out a clean local clone. It installs every
backend proved usable on the host into a separate temporary prefix, verifies
the installed CLI and Web product without fallback, and invokes the retained
benchmark with the fixed public tuple. Unavailable platform backends remain
`external verification`; a usable backend that fails makes the campaign fail.
The trap owns only its `mktemp` tree, and the report excludes local paths,
hostnames, usernames, generated text, and other private machine identity.

## Weight And KV Commands

```text
validate-weights.sh --models-dir DIR

validate-kv.sh --model PATH \
  --backend cpu|vulkan|vulkan-hybrid|metal|metal-hybrid \
  --context N

profile.sh --model PATH \
  --backend cpu|vulkan|vulkan-hybrid|metal|metal-hybrid \
  --context N --kv f16|int8 [--weights-percent 0..100]
```

`validate-kv.sh` executes the public KV profiles but does not authenticate the
artifact or decide whether either row passes a comparison contract. That
procedure is defined by the
[Ministral KV-cache validation guide](../docs/development/ministral-kv-cache-validation.md).

## Oracle Parity

```text
parity-check.sh --models-dir DIR --model-id ID \
  --backend cpu|vulkan|vulkan-hybrid|metal|metal-hybrid --kv f16|int8 \
  --reference-server PATH [--reference-port PORT] \
  [--weights-percent 0..100 \
   --expect-mode all-gpu|all-metal|mixed|cpu-only]
```

The script authenticates one catalogued Q4_K_M artifact, starts one CPU
`llama-server` on loopback, and terminates only that process. `f16` and `int8`
are distinct rows. Protocol, code, parity, placement, or lifecycle mismatches
fail the row; missing infrastructure remains external verification.

The complete sequence of 16 `local_ids` from each available homogeneous
hybrid endpoint must equal the corresponding standalone backend sequence.

## Complete Matrix

```text
matrix-check.sh --models-dir DIR --reference-server PATH \
  [--reference-port PORT]
```

The matrix attempts 74 serial rows: six Q8 rejections, 60 primary rows, and
eight 3B-Instruct hybrid endpoints. Primary hybrid rows use 25%/mixed. Endpoint
rows cover f16 and int8 with accelerator-only and CPU-only placement.

Unavailable independent rows remain external verification. A comparison
mismatch stops the matrix immediately. Model paths remain single quoted
arguments; enums and numbers are validated before Cargo or the binary runs.
Scripts do not use `eval`, construct shell commands from input, or write to
model directories.

## Semantic Qualification

```text
semantic-check.sh --models-dir DIR
```

The runner always constructs six rows. Historical Instruct rows are preserved
without opening artifacts. The three Reasoning rows authenticate one artifact
at a time and run `real_semantic_acceptance` with this exact configuration:

```text
context=4096 max_tokens=4096 temperature=0.7 seed=0
top_p=1 top_k=0 min_p=0 repeat_penalty=1
kv=f16 backend=vulkan-hybrid placement=all-gpu
```

There is no retry, CPU fallback, oracle, tuning, or Instruct inference. A
structurally complete matrix exits zero even when a row is `not-qualified` or
`external-verification`; invalid arguments or catalog input exit with code 2.
Current reviewed results belong in
[validation evidence](../docs/project-status/validation-evidence.md).

Each attempted Reasoning row emits one configuration record, one selection
record, nine case records, one summary, and one timing record. Terminal lines
have this normalized form:

```text
qualification: model_id=<id> profile=<instruct|reasoning> \
  evidence=<preserved|current> \
  status=<qualified|not-qualified|external-verification> \
  reason=<reason> critical=<n/4|not-applicable> \
  semantic=<n/9|not-applicable>
summary: qualified=<n> not_qualified=<n> \
  external_verification=<n> total=6
```

## End-To-End Benchmark

[`examples/bench.rs`](../examples/bench.rs) measures one explicit tuple through
the public engine event stream. It does not authenticate artifacts, compare
revisions, or select a keep/reject verdict. Its full interface is in the
[benchmark reference](../docs/development/end-to-end-throughput-benchmark.md);
comparison and acceptance policy are in the
[performance investigation process](../docs/development/performance-investigation.md).
