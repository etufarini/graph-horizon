<!--
This report preserves the preliminary v0.1.0 qualification campaign and the
remaining final-release gates. docs/validation.md remains the authoritative
support registry.
-->

# Ministral 3 v0.1.0 preliminary qualification

> **Status:** superseded as a final release decision. No `v0.1.0` tag or GitHub
> release exists, and later runtime changes require qualification on a new final
> commit. The measurements below remain historical evidence for their recorded
> revisions only.

## Release identity

| Field | Value |
|---|---|
| Version / tag | `0.1.0` / planned annotated `v0.1.0` |
| Initial main | `a8b5a16d0c2197a7bcdc46a72546a71556aa5a2f` |
| Qualified inference RC | `d1bf18f034fd44df5b8e81931e7feea32edeb47f` |
| Packaging RC | `c59ea4eebdd77cfe2712b64bc11245fe4dd12440` |
| Final pre-tag candidate | `fe7a46aec1e1b5904413082d95946a45ce70d8d2` |
| Final source | pending |
| Qualification date | 2026-08-19 |
| Determination | `REQUALIFICATION REQUIRED` |

At the time of this campaign, changes after the qualified inference RC were
classified as packaging or documentation changes. The repository has since
received runtime changes, so that reuse policy no longer establishes a final
v0.1.0 result. The eventual release must record one final commit and bind it to
the immutable tag; `git rev-list -n 1 v0.1.0` will then provide the mapping.

## Environment and inputs

- Linux x86_64, kernel `7.0.0-29-generic`; Intel Core i5-9600K, 6 cores.
- NVIDIA GeForce RTX 3060 12 GiB, driver 595.84, Vulkan 1.4.329.
- Rust/Cargo 1.95.0, Node.js 24.15.0, npm 11.12.1, GCC 15.2.0.
- Metal/macOS, AMD Vulkan, and other NVIDIA hardware were unavailable.
- Cargo and npm lock files contain resolved versions; no release dependency is
  taken from a mutable Git branch.
- Exact filenames, byte counts, architecture/profile, quantization, and SHA-256
  values for all six external model inputs are in `docs/validation.md` and
  `support/models.tsv`.

All six local files matched the catalog before qualification. Different bytes
are a different qualification target.

## Historical model qualification

| Model | Semantic generation | Teacher-forced | Repetition | Final |
|---|---|---|---|---|
| 3B Instruct | preserved Plan 05, 8/9 | RC 16/16 top-1 | RC generation smoke | QUALIFIED |
| 3B Reasoning | 8/9, 9/9, 9/9 | 16/16 twice | 3 fresh processes | QUALIFIED |
| 8B Instruct | preserved Plan 05, 8/9 | RC 16/16 top-1 | RC parity | QUALIFIED |
| 8B Reasoning | 9/9, 9/9, 8/9 | 16/16 twice | divergent bytes/lengths | NOT SUPPORTED |
| 14B Instruct | preserved Plan 05, 9/9 | RC 16/16 top-1 | RC parity | QUALIFIED |
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
The precise reduction/race/state source was not isolated because the candidate
excluded the model. Final campaign status: `NOT SUPPORTED`.

## Historical backend and platform contract

| Backend / placement | Platform/device | Build | Runtime | Quality | Release status |
|---|---|---|---|---|---|
| Vulkan-hybrid, 100% GPU | Linux x86_64 / RTX 3060 | PASS | PASS | five artifacts PASS | SUPPORTED |
| CPU | Linux x86_64 / i5-9600K | PASS | synthetic PASS | real matrix incomplete | EXPERIMENTAL |
| Vulkan standalone | Linux x86_64 / RTX 3060 | PASS | backend tests PASS | real matrix incomplete | EXPERIMENTAL |
| Vulkan-hybrid mixed/CPU | same host | PASS | backend tests PASS | real matrix incomplete | EXPERIMENTAL |
| Metal / Metal-hybrid | macOS arm64 | UNAVAILABLE locally | UNAVAILABLE | UNAVAILABLE | BUILD-ONLY / EXPERIMENTAL |
| Vulkan AMD/other NVIDIA | unavailable devices | capability code only | UNAVAILABLE | UNAVAILABLE | EXPERIMENTAL |

Backend selection is compile-time. Only the first row with the five qualified
models formed the proposed v0.1.0 contract for this candidate; it is not a
qualification of the current source tree.

### Later integrated evidence

These historical cells are intentionally not rewritten as if the 19 August
campaign had executed later hardware. Corrected runtime `e7edc83`, subsequently
merged into `main` by PR #41, passed all locally executable CPU, standalone
Vulkan, and Vulkan-hybrid mixed rows on the AMD host: `40 PASS`, `34 external
verification`, `0 failure`, plus `6/6` semantic qualification. Later Apple M4
evidence qualifies the documented Metal and Metal-hybrid tuple. Missing Q8
artifacts and unmeasured devices remain external rather than PASS. The current
summary is [`validation.md`](validation.md).

## Failure paths

Missing and invalid GGUF files, invalid KV/placement values, and unknown flags
returned nonzero with bounded errors. Inspected stderr exposed no developer
path, backtrace, or panic.

## Historical build and test qualification

The clean final candidate passed formatting and all shell syntax checks. CPU,
Vulkan, and Vulkan-hybrid each passed locked workspace tests, warnings-denied
Clippy over all targets, and optimized release builds in isolated target trees.
Representative counts were 166 application tests and 159 CPU engine tests;
Vulkan-hybrid ran 229 engine tests. Real-artifact tests not selected by the
explicit qualification harness remained intentionally ignored. The frontend
passed 92/92 tests, Svelte check with zero errors/warnings, production build,
and npm audit with zero known vulnerabilities. The documentation contract and
bootstrap rejection/forwarding tests passed.

## Installer and artifact design

The bootstrap is designed to download only `graph-horizon-0.1.0.tar.gz` from
the immutable v0.1.0 release URL and a strict same-name `.sha256` record,
verify SHA-256 before extraction, reject unsafe or incomplete archives and
symlinks, then delegate locally. Models are never downloaded. The tag and
assets do not exist yet; final qualification must generate and test them from
the selected release commit.

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

## Historical clean installation gate

An archive made with `git archive`, the release root prefix, and deterministic
gzip was extracted into a new temporary directory with no build tree. The local
installer rebuilt frontend and Vulkan-hybrid release artifacts, installed into
a new prefix, and reported `graph-horizon 0.1.0`. The installed-binary product
smoke must be repeated for the current CLI and Web UI boundary. The root
bootstrap's immutable URLs, checksum mismatch rejection, archive validation,
argument forwarding, and cleanup are covered by focused tests; anonymous
network retrieval can only be repeated after the two release assets exist.

## Historical Candidate Limitations

- 8B Reasoning was not supported by this candidate's release gate.
- Support is limited to five exact model byte sequences and one physical tuple.
- CPU, standalone/mixed Vulkan, Metal, AMD, and other NVIDIA devices are not
  release-qualified even where they compile or have historical evidence.
- One generation runs at a time; backend selection is compile-time; no fallback exists.
- Text-only chat; no tools, multimodal input, or separate Reasoning channel.
- Q8, MoE, custom Reasoning prompts, and long-context quality are outside scope.

## Post-v0.1.0 work

Investigate 8B variation; broaden physical qualification; treat concurrent
scheduling, runtime backend selection, packaging, and performance as separate
programs.

## Final release gates

The current source is not yet release-qualified. Before publishing v0.1.0:

1. select and record one clean final commit;
2. rerun the applicable build, lint, frontend, runtime, semantic, parity,
   serving, failure-path, and documentation gates on that source;
3. create the annotated local `v0.1.0` tag on that exact commit;
4. generate the deterministic source archive and `.sha256` from the tag;
5. perform a clean-prefix installation, version check, and inference smoke from
   the generated archive;
6. publish the tag and both assets only with explicit authorization, then test
   the anonymous bootstrap if the repository is public.

The tag must never move. A defect found after publication requires v0.1.1.
