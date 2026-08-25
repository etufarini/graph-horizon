<!--
This README owns the shortest supported path from discovering Graph Horizon to
installing and running it. Detailed contracts and evidence belong to the linked
documentation.
-->

<p align="center">
  <img src="assets/graph-horizon-logo.svg" alt="Graph Horizon logo" width="220">
</p>

# Graph Horizon

Graph Horizon is a focused local text-to-text runtime with two interfaces: an
interactive CLI and an integrated Web UI. Both run the model locally; there is
no standalone server mode or supported public HTTP API. The Web UI can send an
explicit Web query to DuckDuckGo Lite or a News query to Google News without a
key or additional configuration; inference remains local.

The current model integration is Ministral 3 Instruct and Reasoning 2512 in the
3B, 8B, and 14B sizes. Ministral 3 is the model family currently used by Graph
Horizon; it is not part of the project name.

## Install

The installer builds Graph Horizon from source. Before running it, install:

- Rust and Cargo 1.88 or newer;
- Node.js and npm 22.12 or newer;
- `bash`, `curl`, `tar`, `find`, `awk`, `mktemp`, `uname`, `install`, and either
  `sha256sum` or `shasum`; `curl` remains a runtime prerequisite for search;
- the platform requirements listed below.

The default installation directory is `$HOME/.local/bin`. The installer does
not use `sudo`, modify shell configuration, or download a model.

### macOS

Graph Horizon supports macOS on Apple Silicon (`arm64`). Install the Xcode
command-line Metal tools and check that `xcrun -f metal` and
`xcrun -f metallib` succeed, then run:

```sh
curl --fail --location --silent --show-error \
  https://raw.githubusercontent.com/etufarini/graph-horizon/v0.1.2/install.sh \
  | bash -s -- --backend metal
```

### Linux

Graph Horizon supports Linux on `x86_64`. Install a working Vulkan loader and
driver for a compatible NVIDIA or AMD GPU, then run:

```sh
curl --fail --location --silent --show-error \
  https://raw.githubusercontent.com/etufarini/graph-horizon/v0.1.2/install.sh \
  | bash -s -- --backend vulkan
```

On either platform, replace the backend with `--backend cpu` when GPU execution
is not required. See [installation and backend options](support/README.md) for
hybrid builds, optimized profiles, and custom installation prefixes.

If the installed command is not found, add its directory to the current shell:

```sh
export PATH="$HOME/.local/bin:$PATH"
```

The bootstrap downloads the immutable `v0.1.2` source archive and its published
checksum, verifies SHA-256 before extraction, and then builds the Web UI and the
Graph Horizon executable.

## Download a model

Graph Horizon currently accepts only Ministral 3 2512 `Q4_K_M` GGUF files with
`general.architecture=mistral3`. Every other quantization is unsupported;
BF16, Q8, IQ, Q4_0, and UD-Q4 files from the same repositories are rejected.

Download one of these exact files from Unsloth on Hugging Face:

| Size | Profile | Hugging Face repository | Required GGUF file |
|---|---|---|---|
| 3B | Instruct | [unsloth/Ministral-3-3B-Instruct-2512-GGUF](https://huggingface.co/unsloth/Ministral-3-3B-Instruct-2512-GGUF) | `Ministral-3-3B-Instruct-2512-Q4_K_M.gguf` |
| 8B | Instruct | [unsloth/Ministral-3-8B-Instruct-2512-GGUF](https://huggingface.co/unsloth/Ministral-3-8B-Instruct-2512-GGUF) | `Ministral-3-8B-Instruct-2512-Q4_K_M.gguf` |
| 14B | Instruct | [unsloth/Ministral-3-14B-Instruct-2512-GGUF](https://huggingface.co/unsloth/Ministral-3-14B-Instruct-2512-GGUF) | `Ministral-3-14B-Instruct-2512-Q4_K_M.gguf` |
| 3B | Reasoning | [unsloth/Ministral-3-3B-Reasoning-2512-GGUF](https://huggingface.co/unsloth/Ministral-3-3B-Reasoning-2512-GGUF) | `Ministral-3-3B-Reasoning-2512-Q4_K_M.gguf` |
| 8B | Reasoning | [unsloth/Ministral-3-8B-Reasoning-2512-GGUF](https://huggingface.co/unsloth/Ministral-3-8B-Reasoning-2512-GGUF) | `Ministral-3-8B-Reasoning-2512-Q4_K_M.gguf` |
| 14B | Reasoning | [unsloth/Ministral-3-14B-Reasoning-2512-GGUF](https://huggingface.co/unsloth/Ministral-3-14B-Reasoning-2512-GGUF) | `Ministral-3-14B-Reasoning-2512-Q4_K_M.gguf` |

The final v0.1.0 campaign authenticated all six entries, exercised every locally
available technical-matrix row without failure, and qualified all six artifacts
in the semantic gate. See [validation evidence](docs/validation.md) for the exact
SHAs, environment, external-verification boundaries, and criteria.

A Q8 profile is rejected before backend allocation even though the parser can
diagnose it.

Without an explicit context, the engine uses the lesser of 32,768 tokens and
the GGUF context maximum. On Linux, hybrid automatic RAM capacity is
`floor(MemAvailable × 90 / 100)`, and placement keeps the maximum possible
contiguous GPU suffix that fits its budgets. Reasoning output, including
`[THINK]` and `[/THINK]`, remains ordinary raw text; the CLI and Web UI derive
their visual sections without changing model history.

## Run

Replace `/path/to/model.gguf` with the absolute path of the downloaded model.
CLI and Web UI resolve the model context automatically when the option
is omitted. The examples set `--context-tokens 32768` explicitly so their
memory use and context limit are reproducible. Run `graph-horizon --help` for
the complete option reference.

### CLI

Start an interactive local conversation in the terminal:

```sh
graph-horizon --mode cli \
  --model "/path/to/model.gguf" \
  --context-tokens 32768
```

### Web UI

Start the integrated Web UI:

```sh
graph-horizon --mode web \
  --model "/path/to/model.gguf" \
  --context-tokens 32768 \
  --host 127.0.0.1 --port 8080
```

Open [http://127.0.0.1:8080](http://127.0.0.1:8080) in a browser. The interface
is called Web UI, while its command-line mode is `web`. Search is ready as soon
as the UI starts: the globe button enables one explicit search for the next
request. The query shown in the composer is the only chat text sent to the
selected public provider; inference, history, and files remain local.
`--search-url` and `--search-key-file` remain an advanced override for a
compatible JSON provider. See the [Web search contract](docs/web.md#web-search).

## Current scope

- Local text messages only; tool calling and multimodal input are not
  supported.
- Web search is an explicit, snippets-only Web UI option; it is not an agentic
  tool loop and does not download result pages.
- Reasoning markers remain ordinary streamed text, not a separate reasoning
  channel.
- One generation runs at a time.
- The backend is selected when Graph Horizon is built; there is no runtime
  backend switch.

Current backend labels are evidence-scoped; accepted build targets alone do not
assign these statuses:

| Profile | Status |
|---|---|
| `cpu` | **reference** |
| `vulkan` | **production** |
| `vulkan-hybrid` | **qualified** |
| `metal` | **qualified** |
| `metal-hybrid` | **qualified** |

The precise hardware, software, model, placement, and KV scopes for each label
are maintained in the [backend contract](docs/backend.md#support-status).

## Documentation

- [CLI guide](docs/console.md)
- [Web UI guide](docs/web.md)
- [Runtime configuration](docs/configuration.md)
- [Backend contract](docs/backend.md)
- [Validation evidence](docs/validation.md) and
  [release qualification](docs/release-qualification.md)
- [Contributing](CONTRIBUTING.md)
- [License](LICENSE)
