<!--
This README owns the shortest supported path from discovering Graph Horizon to
installing and running it. Detailed contracts and evidence belong to the linked
documentation.
-->

<p align="center">
  <img src="assets/graph-horizon-logo.svg" alt="Graph Horizon logo" width="220">
</p>

# Graph Horizon

Graph Horizon is a focused local text-to-text runtime with three interfaces: an
interactive CLI, an HTTP server compatible with the OpenAI chat subset used by
the included clients, and an integrated Web UI.

The current model integration is Ministral 3 Instruct and Reasoning 2512 in the
3B, 8B, and 14B sizes. Ministral 3 is the model family currently used by Graph
Horizon; it is not part of the project name.

## Install

The installer builds Graph Horizon from source. Before running it, install:

- Rust and Cargo;
- Node.js and npm 22.12 or newer;
- `bash`, `curl`, `tar`, `find`, `awk`, `mktemp`, `uname`, `install`, and either
  `sha256sum` or `shasum`;
- the platform requirements listed below.

The default installation directory is `$HOME/.local/bin`. The installer does
not use `sudo`, modify shell configuration, or download a model.

> The commands below become available when the repository, the `v0.1.0` tag,
> and its release assets are public. They do not currently work anonymously
> while the repository is private and the release is unpublished.

### macOS

Graph Horizon supports macOS on Apple Silicon (`arm64`). Install the Xcode
command-line Metal tools and check that `xcrun -f metal` and
`xcrun -f metallib` succeed, then run:

```sh
curl --fail --location --silent --show-error \
  https://raw.githubusercontent.com/etufarini/graph-horizon/v0.1.0/install.sh \
  | bash -s -- --backend metal
```

### Linux

Graph Horizon supports Linux on `x86_64`. Install a working Vulkan loader and
driver for a compatible NVIDIA or AMD GPU, then run:

```sh
curl --fail --location --silent --show-error \
  https://raw.githubusercontent.com/etufarini/graph-horizon/v0.1.0/install.sh \
  | bash -s -- --backend vulkan
```

On either platform, replace the backend with `--backend cpu` when GPU execution
is not required. See [installation and backend options](support/README.md) for
hybrid builds, optimized profiles, and custom installation prefixes.

If the installed command is not found, add its directory to the current shell:

```sh
export PATH="$HOME/.local/bin:$PATH"
```

The bootstrap downloads the immutable `v0.1.0` source archive and its published
checksum, verifies SHA-256 before extraction, and then builds the Web UI and the
Graph Horizon executable.

## Download a model

Graph Horizon currently accepts Ministral 3 2512 GGUF models with
`general.architecture=mistral3`. Only the `Q4_K_M` quantization is supported;
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

The loader recognizes all six entries. Reviewed runtime qualification is a
separate claim: the preliminary v0.1.0 campaign did not qualify the 8B
Reasoning artifact. See [validation evidence](VALIDATION.md) for the exact
results.

## Run

Replace `/path/to/model.gguf` with the absolute path of the downloaded model.
Run `graph-horizon --help` for the complete option reference.

### CLI

Start an interactive local conversation in the terminal:

```sh
graph-horizon --mode cli --provider local \
  --model "/path/to/model.gguf"
```

### Server

Start the HTTP server on the local machine:

```sh
graph-horizon --mode server \
  --model "/path/to/model.gguf" \
  --host 127.0.0.1 --port 8080
```

Chat completions are available at
`http://127.0.0.1:8080/v1/chat/completions`.

### Web UI

Start the integrated Web UI:

```sh
graph-horizon --mode web \
  --model "/path/to/model.gguf" \
  --host 127.0.0.1 --port 8080
```

Open [http://127.0.0.1:8080](http://127.0.0.1:8080) in a browser. The interface
is called Web UI, while its command-line mode is `web`.

## Current scope

- Local text messages only; tool calling and multimodal input are not
  supported.
- Reasoning markers remain ordinary streamed text, not a separate reasoning
  channel.
- One generation runs at a time.
- The backend is selected when Graph Horizon is built; there is no runtime
  backend switch.

## Documentation

- [CLI guide](docs/console.md)
- [Server guide](docs/server.md)
- [Web UI guide](docs/web.md)
- [Runtime configuration](docs/configuration.md)
- [Backend contract](docs/backend.md)
- [Validation evidence](VALIDATION.md) and
  [release qualification](RELEASE_QUALIFICATION.md)
- [Contributing](CONTRIBUTING.md)
- [License](LICENSE)
