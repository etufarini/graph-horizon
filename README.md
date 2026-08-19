<!--
This README owns the product-facing introduction and usage. Library details,
operational procedures, and reviewed evidence belong to the linked crate,
support, and validation documents.
-->

<p align="center">
  <img src="assets/graph-horizon-logo.svg" alt="Graph Horizon logo" width="220">
</p>


# Graph Horizon - Ministral 3 Version

A local text-to-text runtime for Ministral 3 Instruct and Reasoning 2512 in the
3B, 8B, and 14B sizes. It provides an interactive console, an
HTTP server compatible with the OpenAI chat subset, and a Web UI. It supports
text messages only, without tool calling or a separate reasoning channel.

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

## Installation

After the repository becomes public, install the immutable v0.1.0 source with:

```sh
curl --fail --location --silent --show-error https://raw.githubusercontent.com/etufarini/graph-horizon/v0.1.0/install.sh | bash -s -- --backend cpu
```

The URL does not work anonymously while the repository is private. The command
downloads the immutable `graph-horizon-0.1.0.tar.gz` release artifact and its
published checksum, verifies SHA-256 before extraction, validates archive shape,
and delegates the build to `support/install.sh`. Prerequisites and GGUF models
are never installed or downloaded automatically.

The equivalent local-checkout path is:

```sh
git clone --branch v0.1.0 --depth 1 https://github.com/etufarini/graph-horizon.git
cd graph-horizon
./support/install.sh --backend cpu
```

There is no default backend. The installer accepts these build tuples; build
availability is broader than the v0.1.0 runtime qualification claim:

| Platform | Backends |
|---|---|
| macOS arm64 | `cpu`, `vulkan`, `vulkan-hybrid`, `metal`, `metal-hybrid` |
| Linux x86_64 | `cpu`, `vulkan`, `vulkan-hybrid` |

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

# Optimized Metal build (runtime qualification is external to v0.1.0)
./support/install.sh --backend metal --profile fast

# Custom prefix (equivalent to --prefix "$HOME/.local")
GRAPH_HORIZON_INSTALL_PREFIX="$HOME/.local" ./support/install.sh --backend cpu
```

Run either installer again to rebuild and replace both command names. The
public form always rebuilds v0.1.0; the local form rebuilds the current
checkout.

## Usage

The model is opened read-only and must declare
`general.architecture=mistral3`.

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
The Web UI shows an accessible `Contesto 63%` bar and `Generazione 12.4s`.
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

Advanced options, including `--vram-reserve-mib` and `--no-attn-simd`, are
listed by `graph-horizon --help`.

The CLI uses the bright Mistral palette already represented by the Web UI:
Input `#FF5229`, Response `#44BA82`, Secondary `#55B3FB`, and Hint/status
`#FFAF01`.

## v0.1.0 support contract

The release-qualified runtime tuple is Linux x86_64, NVIDIA RTX 3060 12 GiB,
driver 595.84, `vulkan-hybrid`, F16 KV, and 100% all-GPU weight placement. The
qualified artifacts are the Q4_K_M 3B/8B/14B Instruct models and the Q4_K_M 3B
and 14B Reasoning models whose exact byte counts and SHA-256 values appear in
[VALIDATION.md](VALIDATION.md). The Q4_K_M 8B Reasoning artifact is explicitly
`NOT SUPPORTED` in v0.1.0 because fixed-seed fresh-process generation did not
satisfy its predeclared reproducibility contract.

CPU, standalone Vulkan, mixed/CPU placement through Vulkan-hybrid, Metal, and
Metal-hybrid remain usable build configurations but are experimental for this
release because the complete v0.1.0 runtime and quality matrix was not run on
them. Metal runtime hardware, AMD Vulkan hardware, and other NVIDIA devices
were unavailable to this campaign. Compilation or capability-based routing is
not a claim that those devices were physically qualified.

The public model boundary accepts only Ministral 3 2512 `Q4_K_M` GGUF files.
Instruct remains dimension-generic while Reasoning loading requires one of the
three recognized release names. Recognition means technical load compatibility,
not qualification: 8B Reasoning is recognized but excluded from the release
support claim. A Q8 profile is rejected before backend allocation.

Reasoning output, including `[THINK]` and `[/THINK]`, remains ordinary raw text
in the existing stream. The bundled [Web UI](docs/web.md) and
[CLI](docs/console.md) derive separate THINK and final-answer presentation from
that raw text; transport, model context, and transcript import/export retain
the markers. The runtime does not parse it into a separate Reasoning channel.
For qualified Reasoning models, an explicit system prompt follows the fixed
release-owned Reasoning instruction instead of replacing it.
Web mode ignores `--max-tokens` and sends the loaded context limit as its
generation allowance.
Internal numeric formats do not expand this public GGUF contract.
Hybrid startup chooses the device-only endpoint, one contiguous device suffix
with a CPU prefix, or all-CPU. Standalone device profiles are device-only or
error. Historical Metal evidence is not v0.1.0 release qualification. The
reviewed artifacts are not a runtime whitelist and do not imply MoE support.

Technical load support and reviewed semantic qualification are separate claims.
The authoritative current evidence is in [VALIDATION.md](VALIDATION.md). The
three Instruct semantic rows reuse the repository's preserved Plan 05 quality
evidence under the explicit unrelated-change policy and add current-source
teacher-forced parity. The 3B and 14B Reasoning rows passed three current-engine
fresh-process semantic runs and two teacher-forced runs each. The 8B Reasoning
row is `NOT SUPPORTED` after its dedicated investigation.
The server now selects that sampling policy for a loaded Reasoning profile while
keeping Instruct greedy. The harness remains the qualification authority because
it additionally fixes context, KV, placement, and the semantic corpus.

Serving streams one generation at a time. Sequential requests and clean
shutdown are qualified; concurrent multi-request scheduling and a parallel
throughput guarantee are not part of v0.1.0. Backend choice is compile-time and
there is no runtime fallback.

## Documentation

- [Reviewed validation evidence](VALIDATION.md)
- [Engine library contract](crates/graph_horizon_engine/README.md)
- [Operational scripts](support/README.md)
- [Contributing](CONTRIBUTING.md)
