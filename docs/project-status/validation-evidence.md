<!--
This record publishes artifact identities, reviewed evidence, and qualification
state only. Procedures live under `docs/development/` and `support/`; raw logs
and experimental decisions remain in Git history and investigation reports.
-->

# Validation Evidence

## Current State

Graph Horizon `0.1.4` is the current release candidate. No local or remote
`v0.1.4` tag or GitHub Release existed at preparation time. Graph Horizon
`0.1.3` remains the current published stable release. Its annotated
local and anonymous remote tag, published source archive, and archive-embedded
commit all resolve to `147f3e2c25aeca1cc1721a101d639d108533d81a`.
The published archive SHA-256 is
`868abcbbccfcf0d0f230c622ed3f0343218f8db4a291e16fa601d52124399aab`,
and its root is `graph-horizon-0.1.3/`.

The release-triggered GitHub workflow failed because checkout fetched the
annotated tag and then rewrote its local ref to the peeled commit before the
verifier ran. A direct anonymous run of the same verifier passes all remote tag,
archive, checksum, commit, and member-path checks. The current development
change makes the workflow check out the default branch while retaining the
complete tag graph; the hosted rerun remains external verification until that
workflow change reaches GitHub.

The previous `v0.1.2` release has an immutable annotated tag and source assets,
but an independent audit found that they identify different commits. The local
and anonymous remote tag resolve to
`7ca9de8be4ea37dc2d24cf75a14f40411819f341`; its published archive embeds
`30f26c1b854dc1e48dc84b22b66f952e451bc967`. This is a historical
release-identity defect even though the commits have the same Git tree.

The `v0.1.0`, `v0.1.1`, `v0.1.2`, and `v0.1.3` tags and assets remain immutable. They
must not be moved, deleted, or replaced; any correction must use a later
version. `main` is the moving development branch and may advance beyond a
release tag. That normal branch movement is unrelated to the v0.1.2 defect.

Current Cargo and frontend versions are `0.1.4`. Historical v0.1.2 release
identity is exclusively the commit resolved by `v0.1.2^{commit}`. The final
v0.1.0 campaign supersedes, for its own contract, preliminary runtime evidence at
`d1bf18f034fd44df5b8e81931e7feea32edeb47f` and
`e7edc8315d397f6eb34c5efb91a9ae20b9b59bc4`; those records remain historical
evidence for their declared revisions only.

Technical compatibility, numeric correctness, semantic quality, and
performance are separate claims. A loadable file is not automatically
qualified, and historical evidence does not qualify later source.

## v0.1.4 Release-Readiness Candidate — 26 August 2026

The compatible release-readiness changes after v0.1.3 select patch version
`0.1.4`. The candidate updates only current package, installer, documentation,
and fixture versions while preserving every v0.1.0–v0.1.3 tag, asset identity,
and historical result.

The candidate simplifies repository guidance while protecting AI-development
material, selects an explicit authenticated catalog model during public
readiness, keeps model-family dispatch closed, and corrects the hosted
release-integrity checkout. It changes no supported runtime behavior, numeric
operation, public API, model profile, or backend policy.

Qualification results belong only to the exact clean candidate commit recorded
in the external release report. The future annotated tag, archive-embedded
commit, versioned archive root, and adjacent checksum must all identify that
same commit. A later `main` commit is unrelated to that immutable identity.

| v0.1.4 preparation gate | Result |
|---|---|
| Rust format and warnings-denied CPU Clippy | PASS |
| Locked CPU workspace, all targets | PASS: app/support 170, engine 163, documentation 1, family integration 4, semantic integration 12; declared model/network cases ignored |
| CPU release build and exact source/documentation contracts | PASS |
| Rust 1.88 CPU suite, Vulkan check, and Vulkan-hybrid check | PASS |
| Frontend install, 135 tests, diagnostics, build, and audit | PASS: 0 warnings, 0 vulnerabilities |
| Tracked shell syntax | PASS |
| Installer, bootstrap, and release-integrity fixtures | PASS: 7 + 2 + 3 |
| Live Web/News providers | PASS: 3/3 tests for three consecutive rounds |
| Direct anonymous v0.1.3 release integrity | PASS: tag, remote, archive `147f3e2c25aeca1cc1721a101d639d108533d81a`; SHA-256 `868abcbbccfcf0d0f230c622ed3f0343218f8db4a291e16fa601d52124399aab` |
| Exact-commit public readiness | external verification: `RELEASE_MODEL_ID`, `RELEASE_MODEL_PATH`, and `RELEASE_BACKEND` were not supplied |
| Metal and Metal-hybrid execution on Linux x86_64 | external verification |

## v0.1.3 Corrective Release — 26 August 2026

The compatible changes after v0.1.2 selected patch version `0.1.3`. The
published correction updates package, installer, documentation, and fixture
versions while preserving every earlier tag, asset identity, and historical
result.

The corrective release invariant is stricter than equal file trees: the
annotated `v0.1.3` tag, local and anonymous remote tag commit, archive-embedded
commit, versioned archive root, and adjacent checksum must all identify the
exact published release commit. The archive must be generated from that
annotated tag commit. A later `main` commit is unrelated and is never part of
this comparison.

Qualification results belong only to the exact clean commit recorded in the
release report. Publication does not transfer older numeric evidence to this
later tag.

| Local release gate | Result |
|---|---|
| Rust format and warnings-denied CPU Clippy | PASS |
| Locked CPU workspace suite | PASS: app/support 170, engine 163, integration 5+12; declared model/network cases ignored |
| CPU release build and exact source-structure gate | PASS |
| Rust 1.88 CPU suite, Vulkan check, and Vulkan-hybrid check | PASS |
| Frontend install, 135 tests, diagnostics, build, and audit | PASS: 0 warnings, 0 vulnerabilities |
| Tracked shell syntax | PASS |
| Installer, bootstrap, and release-integrity fixtures | PASS: 7 + 2 + 3; includes `HEAD` ahead of the release tag |
| Live Web/News providers | PASS: 3/3 tests for three consecutive rounds |
| Metal and Metal-hybrid execution on Linux x86_64 | external verification |

## Week 2 Public Readiness — 26 August 2026

The canonical public-readiness campaign completed at commit
`8a2d62333442938889c46b8fb953f5150d31d439` with overall result **PASS**. It
authenticated `Ministral-3-3B-Instruct-2512-Q4_K_M.gguf` as
`9ed150d4367e68df0ac8e1540f6ddc65b42d0ee26378329d1ecbca60f93fc5f8`,
created a clean clone without generated or untracked state, and used only the
product installed from that clone in separate temporary prefixes.

This campaign applies only to the recorded commit. It is not transferred to a
tag commit, archive commit, or later `main` commit merely because a version or
tree happens to match.

| Installed-product check | Result |
|---|---|
| Version, help, model load, and minimal CLI generation | PASS |
| Installed Web assets, startup, runtime identity, and SSE generation | PASS |
| CPU | PASS |
| Vulkan | PASS |
| Vulkan-hybrid with accelerator placement | PASS |
| Metal / Metal-hybrid on Linux x86_64 | external verification |
| Bounded temporary cleanup | PASS |

The public benchmark used Graph Horizon 0.1.2, Linux 7.0.0-30-generic x86_64,
an AMD Ryzen 5 5500, and an AMD Radeon RX 6750 XT with RADV/Mesa
26.0.3-1ubuntu1. The Vulkan loader exposed instance 1.4.341 and device API
1.4.335. The fixed tuple was Vulkan, f16 KV, context 2048, 64 maximum tokens,
one warm-up, three measured repetitions, and the prompt “Spiega in una frase
che cosa misura un benchmark riproducibile.” (19 prompt tokens).

| Public metric | Median | Sample standard deviation | CV |
|---|---:|---:|---:|
| TTFT | 71.14 ms | 0.02 ms | 0.0003 |
| Prompt throughput | 267.07 tokens/s | 0.07 tokens/s | 0.0003 |
| Model decode throughput | 81.89 tokens/s | 0.15 tokens/s | 0.0018 |

These measurements apply only to this machine, commit, model, and
configuration and do not generalize to other GPUs or platforms. This activity
makes no comparison with llama.cpp. TTFT follows the public Graph Horizon
boundary and includes tokenization and first sampling. The benchmark measures
performance; it does not by itself qualify correctness or output quality. The
generated public report contained no hostname, username, local path, or model
output.

## Canonical Release Identity

Release verification compares a published artifact with its own version tag,
never with current `HEAD` or `main`. A conforming release requires the local
annotated tag, anonymous remote tag, archive commit, checksum, version, and
archive root to identify the same frozen release.

The independent v0.1.2 audit on 26 August 2026 reproduced:

| v0.1.2 identity field | Result |
|---|---|
| Local `v0.1.2^{commit}` | `7ca9de8be4ea37dc2d24cf75a14f40411819f341` |
| Anonymous remote `v0.1.2^{commit}` | `7ca9de8be4ea37dc2d24cf75a14f40411819f341` |
| Published archive commit | `30f26c1b854dc1e48dc84b22b66f952e451bc967` |
| Tag and archive Git tree | same: `02e55f4ddc1c524c2d5f8e48abf5fc340ca8e821` |
| Published SHA-256 | `5099cbd4848ab78998c6b3d37aaffdb5e99ffdc90aca030f98431ba2b0fa19c1`, PASS |
| Archive root | `graph-horizon-0.1.2/`, PASS |

Equal Git trees establish equal file content, not equal commit identity. They
therefore do not satisfy the release invariant: the published archive commit
does not equal `v0.1.2^{commit}`. The valid checksum authenticates the mismatched
published archive as downloaded; it does not make that archive tag-derived.
Later development on `main` is expected and neither causes nor repairs this
historical defect.

`v0.1.1` preserves the packaging-fix identity below. `v0.1.0`, its archive, and
its checksum remain authoritative for the numeric campaign. The v0.1.2 tag and
assets must remain as published, and a corrected release identity requires a
later version.

## v0.1.1 Packaging Correction

This patch changed installation, Web-asset lookup, tests, documentation, and
versions only. Except for its manifest version, the engine crate remained the
v0.1.0 engine; numeric and semantic results were not reassigned.

The installed-product defect was reproducible: v0.1.0 built the frontend but
installed only the executable, which searched `web/frontend/dist` relative to
the current directory. v0.1.1 installs only regular files below
`<prefix>/share/graph-horizon/web`, rejects symlinks, and resolves assets from
the executable.

Environment on 23 August 2026: Linux x86_64, CPU packaging smoke with an AMD
Radeon RX 6750 XT available, Rust/Cargo 1.97.1 plus minimum toolchain 1.88.0,
Node.js 24.18.0, and npm 11.16.0.

| v0.1.1 gate | Result |
|---|---|
| Rust format, shell syntax, and diff | PASS |
| Rust 1.88 and CPU/Vulkan/Vulkan-hybrid checks | PASS |
| CPU Clippy `-D warnings` and complete suite | PASS: app 145, engine 163, integration 5+12 |
| Frontend | PASS: 119 tests, no Svelte errors/warnings, build, no vulnerabilities |
| Bootstrap/installer, nested assets, and symlink rejection | PASS |
| Archive/tag/checksum and clean-prefix CPU installation | PASS |
| Installed product outside checkout: version, Web HTTP, and 3B SSE inference | PASS |
| Anonymous public bootstrap after publication | PASS |

The 3B Instruct smoke artifact matched the catalog byte count and SHA-256. UI,
context, and runtime requests returned HTTP 200, then generation completed
`OK.` with statistics and terminal `done`. This proves packaging and product
boundaries, not a new parity or quality campaign.

## v0.1.2 Public Search And Web Interface

This release adds explicit keyless public search while retaining private local
inference transport. Evidence belongs only to `v0.1.2^{commit}`. Environment on
25 August 2026: Linux x86_64 7.0.0-30, Rust/Cargo 1.97.1 with an additional
1.88.0 check, Node.js 24.18.0, npm 11.16.0, and curl 8.18.0.

DuckDuckGo Lite Web and Google News RSS were exercised over the public Internet
without keys. Three consecutive rounds per adapter passed. The final assertion
run produced `3 passed`: non-empty real results, at most five HTTP(S) URLs,
non-empty titles/excerpts, publishers, and timestamps inside the requested
local interval for dated Web/News cases. These tests remain ignored in the
ordinary suite because they depend on external services and network access.

```sh
cargo test --locked --no-default-features --features cpu -p graph-horizon
cargo test --locked --no-default-features --features cpu -p graph-horizon \
  live_ -- --ignored
cd web/frontend && npm test && npm run check && npm run build
```

| v0.1.2 gate | Result |
|---|---|
| Rust format, shell syntax, and diff | PASS |
| CPU Clippy `-D warnings` and CPU release build | PASS |
| Locked Rust suite | PASS: app 166, engine 163, integration 5+12; model/network tests ignored |
| Rust 1.88 CPU suite; Vulkan and Vulkan-hybrid checks | PASS |
| Live Web, dated Web, News, and dated News providers | PASS: 3/3 for three consecutive rounds |
| Deterministic 429, timeout, oversize, invalid feed/JSON, and no-result failures | PASS |
| Frontend | PASS: 135 tests, no Svelte errors/warnings, build, no vulnerabilities |
| Clean-prefix CPU `fast` installer, binary version, and installed Web assets | PASS |
| Direct dependencies | `scraper` 0.27.0 ISC; `roxmltree` 0.20.0 and `httpdate` 1.0.3 MIT/Apache-2.0 |

Citation structure is covered: source IDs use `[S1]`, URLs stay outside model
text, provenance remains separate, and only IDs present in the visible answer
are marked `Cited`. No GGUF was available in the worktree, so semantic
evaluation of a genuinely generated searched answer remains **external
verification**. Search snippets belong to providers, not original pages; a
well-formed citation alone does not prove every claim.

## Minimum Rust Contract

Graph Horizon supports Rust and Cargo 1.88 or newer. Edition 2024 requires only
1.85, but locked dependencies declare 1.88, so a lower claim would be false.
Both crates inherit `rust-version = "1.88"`, and the installer rejects older or
unparseable toolchains before building.

```sh
cargo +1.88.0 test --locked --workspace --all-targets \
  --no-default-features --features cpu
cargo +1.88.0 check --locked --workspace --all-targets \
  --no-default-features --features vulkan
cargo +1.88.0 check --locked --workspace --all-targets \
  --no-default-features --features vulkan-hybrid
```

Metal requires macOS arm64 and was external verification on the Linux host.
The locked dependency graph, including optional Metal packages, declares no
Rust version above 1.88.

## Final v0.1.0 Campaign

This determination belongs only to the commit resolved by the local immutable
annotated `v0.1.0` tag. If that tag is absent or resolves elsewhere, this
section qualifies no revision.

Environment on 22 August 2026: Linux x86_64 `7.0.0-30-generic`, Intel Core
i5-9600K, NVIDIA RTX 3060 12 GiB, driver 595.84/Vulkan 1.4.329, Rust/Cargo
1.95.0, minimum Rust/Cargo 1.88.0, Node.js 24.15.0, npm 11.12.1, GCC 15.2.0,
six catalog-matching Q4_K_M files, and llama.cpp `13f2b28b0`.

| Final tagged-commit gate | Result |
|---|---|
| Rust format, shell syntax, diff, and worktree | PASS |
| Rust 1.88 CPU suite; Vulkan and Vulkan-hybrid checks | PASS |
| CPU Clippy, full suite, and release build | PASS: app 143, engine 163, integration 5+12 |
| Vulkan Clippy, full suite, and release build | PASS: app 143, engine 162, integration 5+12 |
| Vulkan-hybrid Clippy, full suite, and release build | PASS: app 143, engine 233, integration 6+12 |
| Frontend | PASS: 119 tests, no Svelte errors/warnings, build, no vulnerabilities |
| Repeated `support_scripts` harness | PASS: 20 focused executions, no `ETXTBSY` |
| Q4_K_M authentication | PASS: 6/6 |
| Real/oracle matrix | 37 PASS, 37 external verification, 0 failure, total 74 |
| Terminal semantic matrix | 6 qualified, 0 not-qualified, 0 external verification |

The 37 external rows were six unavailable Q8_0 artifacts, 28 Metal rows not
executable on Linux, and three 14B Vulkan-hybrid mixed rows that exceeded this
host's memory. None is reported as PASS. Available CPU/Vulkan rows, mixed 3B/8B,
one mixed 14B Reasoning row, and four 3B Vulkan-hybrid endpoints passed. Current
Reasoning rows achieved 4/4 critical cases, 9/9 semantic cases, and 9/9 complete
markers; Instruct rows retained their protocol-declared evidence.

Tag/archive Git identity, adjacent checksum, clean-prefix archive installation,
version, terminal/Web UI, and installed-binary inference were also verified.
After publication, the documented anonymous bootstrap downloaded and verified
the immutable assets, built frontend and CPU release, and returned
`graph-horizon 0.1.0`; that smoke did not download models or repeat inference.

## Authenticated Artifacts

The machine-readable authority is [`support/models.tsv`](../../support/models.tsv).
Models are read-only external inputs and are not distributed or downloaded by
the project.

| ID | Profile | Q4_K_M file | Bytes | SHA-256 |
|---|---|---|---:|---|
| 3b-instruct | instruct | `Ministral-3-3B-Instruct-2512-Q4_K_M.gguf` | 2147023008 | `9ed150d4367e68df0ac8e1540f6ddc65b42d0ee26378329d1ecbca60f93fc5f8` |
| 3b-reasoning | reasoning | `Ministral-3-3B-Reasoning-2512-Q4_K_M.gguf` | 2147021472 | `7e9516cc01a039bb3e2d41227cdf388849bc1c942c4624c84567b1684cd9c0fc` |
| 8b-instruct | instruct | `Ministral-3-8B-Instruct-2512-Q4_K_M.gguf` | 5198911904 | `33e7a72cf5e6e2cfc2f2847075acc013d68bba023e35310cef86b5cf8fdca761` |
| 8b-reasoning | reasoning | `Ministral-3-8B-Reasoning-2512-Q4_K_M.gguf` | 5198910368 | `894aa3645ef8708a81dbe201c26105ce37c4c741252c89c5a78f81b49ac438c6` |
| 14b-instruct | instruct | `Ministral-3-14B-Instruct-2512-Q4_K_M.gguf` | 8239593024 | `824e0f3373e69b84f2cae46fdcb9bd1ebc6ab3bfc7acc125d818b7b8178cc613` |
| 14b-reasoning | reasoning | `Ministral-3-14B-Reasoning-2512-Q4_K_M.gguf` | 8239591488 | `fe08ca2158cd7438211ec6a4e5256d31bc980f016e3f5b635fe91fe6848d461c` |

The public runtime accepts Q4_K_M. Catalogued Q8_0 names are negative cases and
must fail before backend allocation. The catalog authenticates evidence; it is
not a runtime whitelist.

## Current Backend Evidence

This section is the canonical current matrix of qualified hardware, driver,
operating-system, and backend tuples. Architecture and installation documents
link here rather than duplicating device models.

| Profile | Reviewed evidence | Technical status |
|---|---|---|
| CPU | Synthetic suite and post-repair real matrix across six artifacts and f16/int8; no performance promise | REFERENCE |
| Vulkan | Suites, numeric oracles, and real NVIDIA RTX 3060 / AMD RX 6750 XT matrices | PRODUCTION |
| Vulkan-hybrid | Qualified NVIDIA all-GPU; complete post-repair AMD mixed/CPU/all-GPU matrix, pending repetition on the selected final commit | QUALIFIED |
| Metal | Suite, oracles, teacher row, and Apple M4/macOS 26.3 measurements | QUALIFIED |
| Metal-hybrid | Suite and mixed path on the same host; claim limited to that tuple | QUALIFIED |

Labels describe current path maturity and do not rewrite v0.1.0 history or
extend to unmeasured hardware. Details are summarized in
[current performance status](current-performance-status.md).

## Integrated Post-Cleanup Evidence

On corrected runtime `e7edc83`, the available AMD matrix produced
`pass=40 external_verification=34 failure=0 total=74`: all 36 CPU, standalone
Vulkan, and mixed Vulkan-hybrid rows across six Q4_K_M artifacts and both KV
schemes passed, as did four endpoints. External rows were six missing Q8_0 files
and 28 Metal rows unavailable on Linux. The same runtime produced
`qualified=6 not_qualified=0 external_verification=0` semantically.

Separate Apple M4 evidence qualifies only its declared Metal tuple. RTX 3060
and RX 6750 XT results likewise do not qualify every NVIDIA or AMD device. See
[AMD cleanup-regression repair](../investigation-reports/amd-deep-clean-regression-repair.md).

## Preliminary v0.1.0 Campaign

The 19 August 2026 campaign used Vulkan-hybrid all-GPU on Linux x86_64, RTX
3060 12 GiB, driver 595.84, f16 KV, context 4096, and llama.cpp
`13f2b28b098623391b1aacfd27995e1c8b7de9a9`. There were no retries.

| Model | Semantic generation | Teacher-forced | Historical result |
|---|---|---|---|
| 3B Instruct | preserved 8/9 | 16/16 top-1 | QUALIFIED |
| 3B Reasoning | 8/9, 9/9, 9/9 | 16/16 twice | QUALIFIED |
| 8B Instruct | preserved 8/9 | 16/16 top-1 | QUALIFIED |
| 8B Reasoning | 9/9, 9/9, 8/9; divergent bytes | 16/16 twice | NOT SUPPORTED |
| 14B Instruct | preserved 9/9 | 16/16 top-1 | QUALIFIED |
| 14B Reasoning | 9/9, 9/9, 9/9 | 16/16 twice | QUALIFIED |

For 8B Reasoning, S08 lengths were 330, 883, and 3,324 tokens. The predetermined
gate required identical counts and bytes across three fresh processes;
teacher-forced parity did not replace that requirement.

## Final Release Invariants

Qualification remains valid only while:

1. the immutable annotated tag resolves to the clean commit containing this record;
2. applicable build, lint, frontend, runtime, parity, semantic, product-surface, failure-path, and documentation gates ran on that commit;
3. `v0.1.0` is never moved;
4. its archive and checksum derive from that tag;
5. checksum, Git header, clean installation, version, and inference are verified from the archive rather than the worktree;
6. remote tag and published assets match the qualified identity.

Campaign detail is in
[v0.1.0 release qualification](v0.1.0-release-qualification.md). Operational
interfaces are in the [support script reference](../../support/script-command-reference.md);
reusable processes are in [model-support validation](../development/model-support-validation.md),
[oracle comparison](../development/oracle-comparison.md), and
[Ministral KV-cache validation](../development/ministral-kv-cache-validation.md).

## Record Updates

New evidence replaces a row only when it records revision, artifact identity,
configuration, criterion, and terminal state. Missing resources remain external
verification; a favorable benchmark never hides a failure. Immutable release
tags must never be moved.
