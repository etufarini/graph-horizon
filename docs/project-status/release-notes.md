<!--
These notes describe unreleased and released user-facing changes. Qualification
mechanics belong in `docs/project-status/validation-evidence.md`.
-->

# Release Notes

## Unreleased

## Graph Horizon v0.1.3

Graph Horizon v0.1.3 is the corrective release for the v0.1.2 source-archive
identity defect. The v0.1.2 tag, archive, and checksum remain immutable. The
v0.1.3 archive is produced from the exact commit resolved by its annotated tag,
and the adjacent checksum authenticates that archive. Release verification
compares those identities with the version tag, never with later `main` work.

The public entry point is shorter and now includes CLI and Web UI media. The
documentation is organized by user, maintainer, project-status, and historical
investigation concerns, with explicit installation, support, contribution,
security, backend, and model-format guidance. The logo and Web palette share a
cyan-and-coral identity while preserving the established pixel-art interface.

Repository guardrails now run locked CPU and frontend checks in CI. Separate
public-readiness and release-integrity automation binds qualification to one
clean commit and verifies an anonymously downloaded release against its own
tag. Cargo package metadata is complete and the Rust 1.88 checks are
warnings-clean. These changes do not alter numeric operations, supported model
artifacts, backend policy, or runtime interfaces.

The published tag, archive, and checksum pass direct anonymous verification.
The first release-triggered workflow run failed earlier because checkout peeled
the annotated tag into a lightweight local ref; the verifier itself detected
that local-ref mismatch before downloading assets.

The release commit passed formatting, warnings-denied CPU Clippy, the
complete locked CPU workspace suite, the CPU release build, minimum Rust 1.88
CPU/Vulkan/Vulkan-hybrid checks, frontend tests and diagnostics, shell syntax,
installer/bootstrap/release-integrity fixtures, and three consecutive live
Web/News provider rounds. Exact commit-bound public-readiness and archive
results are kept in the release report generated for that commit.

After publication, install anonymously with:

```sh
curl --fail --location --silent --show-error https://raw.githubusercontent.com/etufarini/graph-horizon/v0.1.3/install.sh | bash -s -- --backend vulkan-hybrid
```

The model/backend qualification evidence remains attached to immutable
`v0.1.0`; v0.1.3 does not relabel that historical campaign as a new numerical
qualification. Exact release gate results and identity evidence are
recorded in [validation evidence](validation-evidence.md).

## Graph Horizon v0.1.2

Graph Horizon v0.1.2 adds explicit Web and News search to the integrated Web UI.
DuckDuckGo Lite serves Web requests and Google News RSS serves News requests
without setup or API keys. Search is opt-in for each message, returns bounded
snippets, preserves source provenance and citation labels, and never downloads
result pages. Inference, chat history, and attached files remain local.

`--search-url` remains an advanced replacement for both routes, with the same
strict JSON contract and optional protected bearer-token file. `curl` is now an
installed runtime prerequisite and is checked by the local installer.

The Web workspace retains its pixel-art design while improving responsive
layout, compact controls, search options and sources, transcript rendering,
session actions, metrics, status, and collapsible side panels.

The release passed the locked CPU workspace suite, warnings-denied Clippy,
Rust 1.88 compatibility checks for CPU, Vulkan, and Vulkan-hybrid, all frontend
tests and diagnostics, three consecutive live Web/News provider rounds, and a
clean-prefix installed-product CLI/Web asset smoke. No model was present, so
semantic generation with retrieved sources remains external verification.

The repository, tag, and release assets are public. Install anonymously with:

```sh
curl --fail --location --silent --show-error https://raw.githubusercontent.com/etufarini/graph-horizon/v0.1.2/install.sh | bash -s -- --backend vulkan-hybrid
```

The model/backend qualification evidence remains attached to immutable
`v0.1.0`; v0.1.2 does not relabel that historical campaign as a new numerical
qualification. The accepted artifacts remain the six exact Ministral 3 2512
Q4_K_M files listed in [validation evidence](validation-evidence.md).

## Graph Horizon v0.1.1

Graph Horizon v0.1.1 is a packaging patch for the local text-to-text runtime.
It corrects the installed Web UI without changing numeric engine operations,
model formats, backend policy, or the CLI contract qualified for v0.1.0.

The v0.1.0 installer built the frontend but installed only the executable, so
`--mode web` failed outside a source checkout with `web frontend build missing`.
The v0.1.1 installer places compiled assets under
`<prefix>/share/graph-horizon/web`; the executable resolves that directory from
its own location and remains independent of the shell's current directory.
Installer input containing symlinks or other non-regular frontend output is
rejected before the Rust build.

The repository, tag, and release assets are public. Install anonymously with:

```sh
curl --fail --location --silent --show-error https://raw.githubusercontent.com/etufarini/graph-horizon/v0.1.1/install.sh | bash -s -- --backend vulkan-hybrid
```

The bootstrap downloads the immutable v0.1.1 source archive and `.sha256`,
verifies it before extraction, then builds and installs the command plus Web
assets. It never downloads a model. A tagged checkout can instead run
`./support/install.sh --backend vulkan-hybrid`.

The packaging patch passed the locked CPU workspace suite, warnings-denied
Clippy, the Rust 1.88 compatibility check, all frontend tests and diagnostics,
a clean-prefix archive installation, and installed-product CLI/Web smoke. The
Web smoke served the frontend from outside the checkout and completed a real
3B Instruct CPU generation through the private browser transport.

The model/backend qualification evidence remains attached to immutable
`v0.1.0`; v0.1.1 does not relabel that historical campaign as a new numerical
qualification. The accepted artifacts remain the six exact Ministral 3 2512
Q4_K_M files listed in [validation evidence](validation-evidence.md).

Known limitations remain unchanged: one generation at a time; compile-time
backend selection; no runtime fallback; no tools, multimodal input, or separate
reasoning channel; Q8 and MoE are outside scope.
