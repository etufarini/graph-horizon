<!--
This README introduces GH Zero Engine and guides users from installation to
their first run. Internal contracts and development procedures remain in the
technical documentation.
-->

<p align="center">
  <img src="assets/gh-zero-engine-logo.svg" alt="Graph Horizon Zero Engine logo" width="220">
</p>


# GH Zero Engine - Ministral 3 Version

A local text-to-text runtime for Ministral 3 Instruct 2512 and the approved
Ministral 3 3B Reasoning 2512 profile. It provides an interactive console, an
HTTP server compatible with the OpenAI chat subset, and a Web UI. It supports
text messages only, without tool calling or a separate reasoning channel.

## Requirements

- Git, Rust/Cargo, and Node.js/npm;
- platform build dependencies;
- a Vulkan loader and driver for the `vulkan` and `hybrid` backends;
- a compatible GGUF model already present on the computer.

The installer builds from source and does not download models.

## Installation

```sh
git clone https://github.com/etufarini/gh-zero-engine.git
cd gh-zero-engine
./support/install.sh
```

The default command builds the `hybrid` backend with the `release` profile and
installs `$HOME/.local/bin/gh-zero-engine`. If that directory is not in `PATH`:

```sh
export PATH="$HOME/.local/bin:$PATH"
gh-zero-engine --help
```

To make the `PATH` change permanent, add the `export` line to `~/.profile`,
`~/.zshrc`, or your shell configuration file.

### Configuring the installer

```text
./support/install.sh \
  [--backend cpu|vulkan|hybrid] \
  [--profile release|fast] \
  [--prefix PATH]
```

| Option | Default | Effect |
|---|---|---|
| `--backend` | `hybrid` | Builds CPU, Vulkan, or both |
| `--profile` | `release` | `fast` optimizes more but takes longer to build |
| `--prefix` | `$HOME/.local` | Installs the command into `PATH/bin` |

Examples:

```sh
# Portable backend
./support/install.sh --backend cpu

# More optimized hybrid build
./support/install.sh --backend hybrid --profile fast

# Custom prefix (equivalent to --prefix "$HOME/.local")
GH_ZERO_INSTALL_PREFIX="$HOME/.local" ./support/install.sh
```

Run the installer again to rebuild and replace the binary.

## Usage

The model is opened read-only and must declare
`general.architecture=mistral3`.

### Local console

```sh
gh-zero-engine --provider local --model "/path/to/model.gguf"
```

### HTTP server

```sh
gh-zero-engine --mode server --model "/path/to/model.gguf" \
  --host 127.0.0.1 --port 8080
```

It exposes `POST /v1/chat/completions` and `GET /props`.

### Web UI

```sh
gh-zero-engine --mode web --model "/path/to/model.gguf" \
  --host 127.0.0.1 --port 8080
```

Open [http://127.0.0.1:8080](http://127.0.0.1:8080).

### Console connected to a server

Without `--provider local`, the console uses an HTTP provider:

```sh
gh-zero-engine --base-url "http://127.0.0.1:8080/v1"
```

## Runtime configuration

Runtime configuration uses flags only. Run `gh-zero-engine --help` for the
complete list accepted by the program.

| Flag | Default | Effect |
|---|---|---|
| `--mode <cli\|server\|web>` | `cli` | Selects the surface |
| `--model <path>` | none | GGUF for the local console, server, and Web UI |
| `--base-url <url>` | `http://127.0.0.1:8080/v1` | Console HTTP provider |
| `--context-tokens <n>` | local `min(32,768, GGUF maximum)` | Sets an exact explicit context |
| `--max-tokens <n>` | CLI `2048`, server/web `1024` | Limits the response |
| `--system-prompt <text>` | none | Console system prompt |
| `--kv-quant <f16\|int8>` | `f16` | KV-cache format |
| `--cpu-threads <n>` | automatic | Sets CPU workers |
| `--vram-weights-percent <0..100>` | automatic | Limits VRAM for weights |

When `--context-tokens` is absent on a local surface, the engine uses the lesser
of 32,768 tokens and the GGUF context maximum. An explicit positive value is
never reduced and may reach the GGUF maximum when backend memory permits it.
The official/reference maximum for Ministral 3 Instruct 2512 is 262,144 tokens.
The repository source of truth is the private
[Ministral version contract](crates/gh_zero_engine/src/family/mistral/version.rs).

Hybrid placement exposes `floor(MemAvailable × 90 / 100)` bytes—90% of Linux
`MemAvailable`—to the planner. Swap and `MemFree` are excluded, while the
automatic VRAM reserve remains the greater of 256 MiB and 5% of available VRAM.

Examples:

```sh
# Explicit context and a more compact KV cache
gh-zero-engine --provider local --model "/path/to/model.gguf" \
  --context-tokens 4096 --kv-quant int8

# CPU-only with a hybrid build
gh-zero-engine --provider local --model "/path/to/model.gguf" \
  --vram-weights-percent 0
```

Advanced options, including `--vram-reserve-mib` and `--no-attn-simd`, are
described in the [complete configuration](docs/configuration.md).

The CLI uses the bright Mistral palette already represented by the Web UI:
Input `#FF5229`, Response `#44BA82`, Secondary `#55B3FB`, and Hint/status
`#FFAF01`.

## Supported models

Ministral 3 Instruct 2512 support is unchanged for 3B, 8B, and 14B. Reasoning
support is limited to Ministral 3 3B Reasoning 2512 in the `Q8_0` and
`Q4_K_M` public GGUF profiles; Reasoning 8B and 14B variants are rejected.
Reasoning output, including `[THINK]` and `[/THINK]`, remains ordinary raw text
in the existing stream rather than a separate channel.

Internal backends retain FP16 activations, FP32 residual/logits, and
F16/Q4_K/Q5_K/Q6_K/Q8_0 weights. Official real-model verification covers the
3B artifacts listed in the
[validation guide](docs/kv-quant-mistral-validation.md); other Instruct files
matching the capability-based contract remain compatible/unverified. The
Instruct 8B and 14B reference rows are fixtures, not a whitelist. Hybrid startup
chooses all-GPU, the maximum possible contiguous GPU suffix with a CPU prefix,
or all-CPU. Pure Vulkan remains all-GPU-or-error, and development remains
CPU-first. These fixtures do not constitute MoE support.

## Documentation

- [Configuration](docs/configuration.md)
- [Console](docs/console.md)
- [HTTP server](docs/server.md)
- [Web UI](docs/web.md)
- [CPU/Vulkan backends](docs/backend.md)
- [Architecture](docs/architecture.md)
- [Operational scripts](support/README.md)
- [Contributing](CONTRIBUTING.md)
