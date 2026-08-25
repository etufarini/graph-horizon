<!--
These notes describe unreleased user-facing changes and the released v0.1.2
contract. Qualification mechanics belong in docs/validation.md.
-->

# Unreleased

# Graph Horizon v0.1.2

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
Q4_K_M files listed in [validation.md](validation.md).

# Graph Horizon v0.1.1

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
Q4_K_M files listed in [validation.md](validation.md).

Known limitations remain unchanged: one generation at a time; compile-time
backend selection; no runtime fallback; no tools, multimodal input, or separate
reasoning channel; Q8 and MoE are outside scope.
