<!--
This README owns the product-facing introduction and usage. Library details,
operational procedures, and reviewed evidence belong to the linked crate,
support, and validation documents.
-->

<p align="center">
  <img src="assets/graph-horizon-logo.svg" alt="Graph Horizon logo" width="220">
</p>

# Graph Horizon

Graph Horizon is a focused local text-to-text runtime. It provides an
interactive console and Web UI as its primary user interfaces, plus an HTTP
server compatible with the OpenAI chat subset used by the included clients. It
supports text messages only, without tool calling or a separate reasoning
channel.

The current model integration is Ministral 3 Instruct and Reasoning 2512 in the
3B, 8B, and 14B sizes. Ministral 3 is the model family used by the current
implementation; it is not part of the project name or a separate Graph Horizon
edition.

## Project Scope

Graph Horizon is intentionally a single-user chat runtime, not a general model
framework or a high-throughput inference server. One generation runs at a time,
which keeps request lifetime, cancellation, KV-cache ownership, and prefix
reuse explicit and predictable. The HTTP API is a transport for the Web UI,
the remote console, and simple external clients; broad API compatibility is not
the product goal.

Each build contains one backend profile selected at compile time. Separate CPU,
Vulkan, Metal, and hybrid builds keep platform dependencies and runtime policy
narrow while allowing the same source to run on different supported machines.
Operation-level specialization is still selected from device capabilities and
retains its documented fallback.

The project is developed and validated by one maintainer. The reviewed hardware
matrix therefore records the machines physically available for qualification;
it is an evidence boundary, not a claim that every other compatible device was
tested or failed. Performance comparisons with `llama.cpp` provide a stable
reference for optimization work, not a statement that Graph Horizon aims to
replace its broader runtime and ecosystem.

## Requirements

- Bash, `curl`, `tar`, `mktemp`, `find`, `awk`, `uname`, `install`, and either
  `sha256sum` or `shasum` for the public bootstrap; Git is needed only for a
  local checkout;
- Rust/Cargo and Node.js/npm 22.12 or newer;
- platform build dependencies;
- a Vulkan loader and driver for `vulkan` and `vulkan-hybrid`;
- macOS arm64 plus Xcode command-line `metal` and `metallib` tools for Metal;
- a compatible GGUF model already present on the computer.

The installer builds from source and does not download models.

## Current model support

Graph Horizon currently accepts Ministral 3 2512 GGUF models with
`general.architecture=mistral3`. The only supported public quantization is
`Q4_K_M`. Other files offered in the same repositories, including BF16, Q8,
IQ, Q4_0, and UD-Q4 variants, are not accepted.

The official Unsloth GGUF repositories are:

| Size | Profile | Hugging Face repository | Required GGUF file |
|---|---|---|---|
| 3B | Instruct | [unsloth/Ministral-3-3B-Instruct-2512-GGUF](https://huggingface.co/unsloth/Ministral-3-3B-Instruct-2512-GGUF) | `Ministral-3-3B-Instruct-2512-Q4_K_M.gguf` |
| 8B | Instruct | [unsloth/Ministral-3-8B-Instruct-2512-GGUF](https://huggingface.co/unsloth/Ministral-3-8B-Instruct-2512-GGUF) | `Ministral-3-8B-Instruct-2512-Q4_K_M.gguf` |
| 14B | Instruct | [unsloth/Ministral-3-14B-Instruct-2512-GGUF](https://huggingface.co/unsloth/Ministral-3-14B-Instruct-2512-GGUF) | `Ministral-3-14B-Instruct-2512-Q4_K_M.gguf` |
| 3B | Reasoning | [unsloth/Ministral-3-3B-Reasoning-2512-GGUF](https://huggingface.co/unsloth/Ministral-3-3B-Reasoning-2512-GGUF) | `Ministral-3-3B-Reasoning-2512-Q4_K_M.gguf` |
| 8B | Reasoning | [unsloth/Ministral-3-8B-Reasoning-2512-GGUF](https://huggingface.co/unsloth/Ministral-3-8B-Reasoning-2512-GGUF) | `Ministral-3-8B-Reasoning-2512-Q4_K_M.gguf` |
| 14B | Reasoning | [unsloth/Ministral-3-14B-Reasoning-2512-GGUF](https://huggingface.co/unsloth/Ministral-3-14B-Reasoning-2512-GGUF) | `Ministral-3-14B-Reasoning-2512-Q4_K_M.gguf` |

Open the repository for the desired size and profile, then download exactly the
listed `Q4_K_M` file. In particular, do not select the `UD-Q4_K_XL` file that
some Hugging Face examples use. The installer does not download models.

Technical recognition and reviewed qualification are separate: all six rows
describe the current loader boundary, while the preliminary v0.1.0 campaign did
not qualify the 8B Reasoning artifact. See the
[planned support contract](#planned-v010-support-contract) and
[validation register](VALIDATION.md) for that evidence.

## Installation

After the repository, tag, and release assets become public, install the
immutable v0.1.0 source with:

```sh
curl --fail --location --silent --show-error https://raw.githubusercontent.com/etufarini/graph-horizon/v0.1.0/install.sh | bash -s -- --backend cpu
```

The command is unavailable until publication and does not work anonymously
while the repository is private. It downloads the immutable
`graph-horizon-0.1.0.tar.gz` release artifact and its published checksum,
verifies SHA-256 before extraction, validates archive shape, and delegates the
build to `support/install.sh`. Prerequisites and GGUF models are never installed
or downloaded automatically.

The equivalent local-checkout path is:

```sh
git clone --branch v0.1.0 --depth 1 https://github.com/etufarini/graph-horizon.git
cd graph-horizon
./support/install.sh --backend cpu
```

There is no default backend. The installer accepts these build tuples; build
availability is broader than the planned v0.1.0 runtime qualification:

| Platform | Backends |
|---|---|
| macOS arm64 | `cpu`, `vulkan`, `vulkan-hybrid`, `metal`, `metal-hybrid` |
| Linux x86_64 | `cpu`, `vulkan`, `vulkan-hybrid` |

Build acceptance is not a support claim. The current profile labels are:

| Profile | Status |
|---|---|
| `cpu` | **reference** |
| `vulkan` | **production** |
| `vulkan-hybrid` | **qualified** |
| `metal` | **qualified** |
| `metal-hybrid` | **qualified** |

Production Vulkan is scoped to Linux x86_64 on compatible NVIDIA or AMD
devices. Metal qualification is scoped to the documented Apple M4/macOS 26.3
tuple; Vulkan-hybrid still requires a fresh complete real-model mixed-placement
campaign for production promotion. Exact definitions and limits are in the
[backend contract](docs/backend.md#support-status).

The default profile is `release`, and the default prefix is `$HOME/.local`, so
the command is installed as `$HOME/.local/bin/graph-horizon` with
`gh-zero-engine` linked to the same artifact, without `sudo`. If that directory
is not in `PATH`, the installer reports the exact directory but does not edit
shell files:

```sh
export PATH="$HOME/.local/bin:$PATH"
graph-horizon --help
```

Add that `export` manually to the appropriate shell configuration only if you
want the change to persist.

### Configuring the installer

```text
./support/install.sh \
  --backend cpu|vulkan|vulkan-hybrid|metal|metal-hybrid \
  [--profile release|fast] \
  [--prefix PATH]
```

| Option | Default | Effect |
|---|---|---|
| `--backend` | required | Selects exactly one static runtime profile |
| `--profile` | `release` | `fast` optimizes more but takes longer to build |
| `--prefix` | `$HOME/.local` | Installs the commands into `<prefix>/bin` |

Examples:

```sh
# Portable backend
./support/install.sh --backend cpu

# Optimized Metal build
./support/install.sh --backend metal --profile fast

# Custom prefix (equivalent to --prefix "$HOME/.local")
GRAPH_HORIZON_INSTALL_PREFIX="$HOME/.local" ./support/install.sh --backend cpu
```

Run either installer again to rebuild and replace both command names. The
public form always rebuilds the published v0.1.0 source; the local form rebuilds
the current checkout.

## Usage

The model is opened read-only and must satisfy the
[current model support](#current-model-support) boundary.

### Local console

```sh
graph-horizon --provider local --model "/path/to/model.gguf"
```

### HTTP server

```sh
graph-horizon --mode server --model "/path/to/model.gguf" \
  --host 127.0.0.1 --port 8080
```

It exposes `POST /v1/chat/completions` and read-only `GET /props` context
discovery. Direct API clients keep the existing chat contract; capacity
preflight belongs to the included CLI and Web clients.

### Web UI

```sh
graph-horizon --mode web --model "/path/to/model.gguf" \
  --host 127.0.0.1 --port 8080
```

Open [http://127.0.0.1:8080](http://127.0.0.1:8080).

### Console connected to a server

Without `--provider local`, the console uses an HTTP provider:

```sh
graph-horizon --base-url "http://127.0.0.1:8080/v1"
```

The integrated server is discovered automatically through `/props`. For an
external provider that does not expose that route, supply its known positive
window explicitly:

```sh
graph-horizon --base-url "https://provider.example/v1" --context-tokens 32768
```

Requests preserve every message. CLI admission includes its response reserve;
Web admission compares only estimated prompt occupancy with the 90% safe budget
and leaves the engine to enforce the exact rendered limit. The CLI status is
compact, for example `ctx ~5.2k/32.8k tok gen 12.4s`, with no percentage or
progress bar.
The Web UI shows an accessible `Context` usage bar and the current generation
phase or latest generation metrics.
See the [context capacity contract](docs/context.md) for exact admission and
prompt-preservation behavior.

## Runtime configuration

Runtime configuration uses flags only. Run `graph-horizon --help` for the
complete list accepted by the program.

| Flag | Default | Effect |
|---|---|---|
| `--mode <cli\|server\|web>` | `cli` | Selects the surface |
| `--model <path>` | none | GGUF for the local console, server, and Web UI |
| `--base-url <url>` | `http://127.0.0.1:8080/v1` | Console HTTP provider |
| `--context-tokens <n>` | local `min(32,768, GGUF maximum)` | Sets an exact explicit context |
| `--max-tokens <n>` | CLI `2048`, server `1024`; Web ignored | CLI response reserve or server request fallback |
| `--system-prompt <text>` | none | Console system prompt |
| `--kv-quant <f16\|int8>` | `f16` | KV-cache format |
| `--cpu-threads <n>` | automatic | Sets CPU workers |
| `--vram-weights-percent <0..100>` | automatic | Limits VRAM for weights |

When `--context-tokens` is absent on a local surface, the engine uses the lesser
of 32,768 tokens and the GGUF context maximum. An explicit positive value is
never reduced and may reach the GGUF maximum when backend memory permits it.
The official/reference maximum for Ministral 3 Instruct 2512 is 262,144 tokens.
The repository source of truth is the private
[Ministral version contract](crates/graph_horizon_engine/src/family/mistral/version.rs).

The five profiles are `cpu`, `vulkan`, `vulkan-hybrid`, `metal`, and
`metal-hybrid`. Selection is compile-time and there is no runtime backend
switch or fallback. Separate-memory Vulkan planning uses Linux `MemAvailable`
and a VRAM budget: its automatic mixed policy selects the maximum possible
contiguous GPU suffix and budgets host memory as
`floor(MemAvailable × 90 / 100)`. Metal planning treats CPU and Metal
allocations as competing claims on one unified-memory capacity; 0 means
CPU-only, 100 means all-Metal, and an omitted percentage lets the planner
choose.

Examples:

```sh
# Explicit context and a more compact KV cache
graph-horizon --provider local --model "/path/to/model.gguf" \
  --context-tokens 4096 --kv-quant int8

# CPU-only placement with a hybrid build
graph-horizon --provider local --model "/path/to/model.gguf" \
  --vram-weights-percent 0
```

The advanced `--vram-reserve-mib` option is listed by
`graph-horizon --help`.

The CLI uses the bright Azure palette represented by the logo and Web UI:
Input and Hint/status `#0096DC`, Response and Secondary `#41BEF5`. Errors retain
their dedicated red state color.

## Planned v0.1.0 support contract

Version v0.1.0 has not been tagged or published. Its preliminary campaign used
Linux x86_64, an NVIDIA RTX 3060 12 GiB, driver 595.84, `vulkan-hybrid`, F16 KV,
and 100% all-GPU placement. Five exact artifacts passed that candidate: the
Q4_K_M 3B/8B/14B Instruct models and the Q4_K_M 3B and 14B Reasoning models.
Their byte counts and SHA-256 values appear in [VALIDATION.md](VALIDATION.md).
The Q4_K_M 8B Reasoning artifact was `NOT SUPPORTED` because fixed-seed
fresh-process generation did not satisfy its predeclared reproducibility gate.

Later runtime changes require requalification on the final release commit.
Until that gate is complete, these results are historical candidate evidence,
not a published v0.1.0 support claim. Current backend maturity and release
qualification remain separate.

The current public model boundary accepts only Ministral 3 2512 `Q4_K_M` GGUF
files.
Instruct remains dimension-generic while Reasoning loading requires one of the
three recognized names. Recognition means technical load compatibility, not
qualification. A Q8 profile is rejected before backend allocation.

Reasoning output, including `[THINK]` and `[/THINK]`, remains ordinary raw text
in the existing stream. The bundled [Web UI](docs/web.md) and
[CLI](docs/console.md) derive separate THINK and final-answer presentation from
that raw text; transport, model context, and transcript import/export retain
the markers. The runtime does not parse it into a separate Reasoning channel.
For Reasoning models, non-empty explicit system text follows the fixed internal
instruction instead of replacing it.

Web mode ignores `--max-tokens` and sends the loaded context limit as its
generation allowance. Internal numeric formats do not expand the public GGUF
contract. Hybrid startup chooses the device-only endpoint, one contiguous
device suffix with a CPU prefix, or all-CPU. Standalone device profiles are
device-only or error. Reviewed artifacts are not a runtime whitelist and do not
imply MoE support.

Technical load support and reviewed semantic qualification are separate claims.
The validation register records both the historical campaign and the remaining
final-release gates. The server selects the Reasoning sampling policy for a
loaded Reasoning profile while keeping Instruct greedy; only the harness also
fixes context, KV, placement, and semantic corpus for qualification.

Serving streams one generation at a time. Concurrent multi-request scheduling
and a parallel throughput guarantee are outside the planned v0.1.0 contract.
Backend choice is compile-time and there is no runtime fallback.

## Documentation

- [Reviewed validation evidence](VALIDATION.md)
- [Engine library contract](crates/graph_horizon_engine/README.md)
- [Operational scripts](support/README.md)
- [Contributing](CONTRIBUTING.md)
