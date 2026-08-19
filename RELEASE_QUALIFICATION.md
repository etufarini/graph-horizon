<!--
This report owns v0.1.0 release provenance, qualification results, and the final
determination. VALIDATION.md remains the authoritative support registry.
-->

# Ministral 3 v0.1.0 release qualification

## Release identity

| Field | Value |
|---|---|
| Version / tag | `0.1.0` / annotated `v0.1.0` |
| Initial main | `a8b5a16d0c2197a7bcdc46a72546a71556aa5a2f` |
| Qualified inference RC | `d1bf18f034fd44df5b8e81931e7feea32edeb47f` |
| Packaging RC | `c59ea4eebdd77cfe2712b64bc11245fe4dd12440` |
| Final pre-tag candidate | `fe7a46aec1e1b5904413082d95946a45ce70d8d2` |
| Final source | the sole commit referenced by `v0.1.0` |
| Qualification date | 2026-08-19 |
| Determination | `RELEASE READY` for local immutable tagging/publication |

Final metadata commits do not change inference code. Model evidence from the
qualified inference RC is reused under the explicit unrelated-change policy;
build, packaging, installation, version, and smoke gates are rerun from the
final tag. `git rev-list -n 1 v0.1.0` is the tag-to-source mapping.

## Environment and inputs

- Linux x86_64, kernel `7.0.0-29-generic`; Intel Core i5-9600K, 6 cores.
- NVIDIA GeForce RTX 3060 12 GiB, driver 595.84, Vulkan 1.4.329.
- Rust/Cargo 1.95.0, Node.js 24.15.0, npm 11.12.1, GCC 15.2.0.
- Metal/macOS, AMD Vulkan, and other NVIDIA hardware were unavailable.
- Cargo and npm lock files contain resolved versions; no release dependency is
  taken from a mutable Git branch.
- Exact filenames, byte counts, architecture/profile, quantization, and SHA-256
  values for all six external model inputs are in `VALIDATION.md` and
  `support/models.tsv`.

All six local files matched the catalog before qualification. Different bytes
are a different qualification target.

## Model qualification

| Model | Semantic generation | Teacher-forced | Repetition | Final |
|---|---|---|---|---|
| 3B Instruct | preserved Plan 05, 8/9 | current 16/16 top-1 | current server smoke | QUALIFIED |
| 3B Reasoning | 8/9, 9/9, 9/9 | 16/16 twice | 3 fresh processes | QUALIFIED |
| 8B Instruct | preserved Plan 05, 8/9 | current 16/16 top-1 | current parity | QUALIFIED |
| 8B Reasoning | 9/9, 9/9, 8/9 | 16/16 twice | divergent bytes/lengths | NOT SUPPORTED |
| 14B Instruct | preserved Plan 05, 9/9 | current 16/16 top-1 | current parity | QUALIFIED |
| 14B Reasoning | 9/9, 9/9, 9/9 | 16/16 twice | 3 fresh processes | QUALIFIED |

The oracle is llama.cpp commit
`13f2b28b098623391b1aacfd27995e1c8b7de9a9`; acceptance is exact local/oracle
top-1 equality for 16 steps. Reasoning generation used Vulkan-hybrid all-GPU,
F16 KV, context/max 4096, temperature 0.7, seed 0, top-p 1, top-k 0, min-p 0,
and repeat penalty 1. There were no retries.

### 8B Reasoning resolution

The predeclared gate required three fresh processes to pass critical 4/4,
semantic at least 8/9, complete markers and EOS, teacher-forced parity, and
identical per-case counts plus raw response bytes. Generation met the semantic
floor but S08 lengths were 330, 883, and 3,324 tokens and most response bytes
differed. Two greedy S01 controls were identical and teacher-forced passed.
The sampler and prompt/KV setup are controlled; small backend numerical
differences cross deterministic sampling thresholds and amplify the sequence.
The precise reduction/race/state source was not isolated because this release
excludes the model. Final status: `NOT SUPPORTED`.

## Backend and platform contract

| Backend / placement | Platform/device | Build | Runtime | Quality | Release status |
|---|---|---|---|---|---|
| Vulkan-hybrid, 100% GPU | Linux x86_64 / RTX 3060 | PASS | PASS | five artifacts PASS | SUPPORTED |
| CPU | Linux x86_64 / i5-9600K | PASS | synthetic PASS | real matrix incomplete | EXPERIMENTAL |
| Vulkan standalone | Linux x86_64 / RTX 3060 | PASS | backend tests PASS | real matrix incomplete | EXPERIMENTAL |
| Vulkan-hybrid mixed/CPU | same host | PASS | backend tests PASS | real matrix incomplete | EXPERIMENTAL |
| Metal / Metal-hybrid | macOS arm64 | UNAVAILABLE locally | UNAVAILABLE | UNAVAILABLE | BUILD-ONLY / EXPERIMENTAL |
| Vulkan AMD/other NVIDIA | unavailable devices | capability code only | UNAVAILABLE | UNAVAILABLE | EXPERIMENTAL |

Backend selection is compile-time. Only the first row with the five qualified
models forms the supported v0.1.0 runtime contract.

## Serving and failure paths

The server streams one generation at a time. Three sequential SSE requests
(A, B, C) in one 3B Instruct process completed with `[DONE]`, with no visible
cross-request leakage. `/props` reported context 4096 and SIGTERM ended
promptly. Concurrent scheduling and parallel throughput are not claimed.

Missing and invalid GGUF files, invalid KV/placement values, and unknown flags
returned nonzero with bounded errors. Inspected stderr exposed no developer
path, backtrace, or panic.

## Build and test qualification

The clean final candidate passed formatting and all shell syntax checks. CPU,
Vulkan, and Vulkan-hybrid each passed locked workspace tests, warnings-denied
Clippy over all targets, and optimized release builds in isolated target trees.
Representative counts were 166 application tests and 159 CPU engine tests;
Vulkan-hybrid ran 229 engine tests. Real-artifact tests not selected by the
explicit qualification harness remained intentionally ignored. The frontend
passed 92/92 tests, Svelte check with zero errors/warnings, production build,
and npm audit with zero known vulnerabilities. The documentation contract and
bootstrap rejection/forwarding tests passed.

## Installer and artifact integrity

The bootstrap downloads only `graph-horizon-0.1.0.tar.gz` from the immutable
v0.1.0 release URL and a strict same-name `.sha256` record, verifies SHA-256
before extraction, rejects unsafe/incomplete archives and symlinks, then
delegates locally. Models are never downloaded. The archive and checksum are
generated from the annotated tag; publication must upload both without moving
the tag.

## Performance smoke

3B Instruct on the supported tuple, F16 KV/context 4096, one repetition and no
warmup produced:

| Prompt / decode | Prompt tok/s | TTFT | Decode tok/s | Result |
|---|---:|---:|---:|---|
| 5 / 15 tokens | 16.45 | 303.90 ms | 66.98 | PASS |
| 5 / 62 tokens | 34.51 | 144.87 ms | 70.90 | PASS |
| 83 / 16 tokens | 201.11 | 412.70 ms | 73.71 | PASS |
| 83 / 64 tokens | 201.66 | 411.59 ms | 73.33 | PASS |

This is a catastrophic-regression characterization, not an optimization or
cross-SHA performance claim. All values are finite and no major regression was
observed.

## Clean installation gate

An archive made with `git archive`, the release root prefix, and deterministic
gzip was extracted into a new temporary directory with no build tree. The local
installer rebuilt frontend and Vulkan-hybrid release artifacts, installed into
a new prefix, and reported `graph-horizon 0.1.0`. The installed binary loaded
the authenticated 3B Instruct artifact, served `/props` with context 4096,
streamed `OK`, usage, stop, and `[DONE]`, then shut down cleanly. The root
bootstrap's immutable URLs, checksum mismatch rejection, archive validation,
argument forwarding, and cleanup are covered by focused tests; anonymous
network retrieval can only be repeated after the two release assets exist.

## Known limitations

- 8B Reasoning is not supported.
- Support is limited to five exact model byte sequences and one physical tuple.
- CPU, standalone/mixed Vulkan, Metal, AMD, and other NVIDIA devices are not
  release-qualified even where they compile or have historical evidence.
- Serving is serialized; backend selection is compile-time; no fallback exists.
- Text-only chat; no tools, multimodal input, or separate Reasoning channel.
- Q8, MoE, custom Reasoning prompts, and long-context quality are outside scope.

## Post-v0.1.0 work

Investigate 8B variation; broaden physical qualification; treat concurrent
scheduling, runtime backend selection, packaging, and performance as separate
programs.

## Final determination

`RELEASE READY`. All P0/P1 blockers are resolved for the narrow 5/6 contract.
Create the annotated local `v0.1.0` tag at the final release commit, generate
the source archive and `.sha256` from that tag, verify the mapping and installed
version, and publish only with explicit authorization. The tag must never move;
any later defect requires v0.1.1.
