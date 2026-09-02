<!--
This guide owns end-user prerequisites, source installation, installer options,
installed paths, and installation verification. It does not document runtime
flags or developer validation scripts.
-->

# Installation

Graph Horizon is built from source. The public bootstrap downloads and verifies
the immutable source archive before delegating to the same local installer used
by a checkout. Neither installer downloads models, invokes `sudo`, or changes
shell configuration.

## Prerequisites

- Bash, `curl`, `tar`, `mktemp`, and `find` for the public bootstrap;
- Rust and Cargo 1.88 or newer;
- Node.js and npm 22.12 or newer;
- a working Vulkan loader and driver for Vulkan builds;
- macOS arm64 and the Xcode `metal` and `metallib` tools for Metal builds;
- Linux `x86_64`, an NVIDIA driver, and CUDA Toolkit `nvcc` for CUDA builds.

Verify the Metal tools explicitly:

```sh
xcrun -f metal
xcrun -f metallib
```

Models are acquired separately. See
[supported models and formats](supported-models-and-formats.md).

## Quick Install

The public bootstrap installs the stable `v0.1.5` release.

### Apple Silicon and Metal

On Apple Silicon macOS:

```sh
curl --fail --location --silent --show-error \
  https://raw.githubusercontent.com/etufarini/graph-horizon/v0.1.5/install.sh \
  | bash -s -- --backend metal
```

### Linux x86_64 and Vulkan

On Linux `x86_64` with Vulkan:

```sh
curl --fail --location --silent --show-error \
  https://raw.githubusercontent.com/etufarini/graph-horizon/v0.1.5/install.sh \
  | bash -s -- --backend vulkan
```

### Linux x86_64 and CUDA

On Linux `x86_64` with an NVIDIA driver and CUDA Toolkit `nvcc`:

```sh
curl --fail --location --silent --show-error \
  https://raw.githubusercontent.com/etufarini/graph-horizon/v0.1.5/install.sh \
  | bash -s -- --backend cuda
```

Use `--backend cpu` when GPU execution is not required.

## Local Installer Options

```text
support/install.sh --backend cpu|vulkan|vulkan-hybrid|metal|metal-hybrid|cuda \
  [--profile release|fast] [--prefix /absolute/install/prefix]
```

`--backend` is required and selects one compile-time profile. There is no
runtime backend switch or fallback.

| Platform | Accepted build backends |
|---|---|
| macOS arm64 | `cpu`, `vulkan`, `vulkan-hybrid`, `metal`, `metal-hybrid` |
| Linux x86_64 | `cpu`, `vulkan`, `vulkan-hybrid`, `cuda` |

The installer accepts CUDA only for Linux `x86_64` and requires `nvcc` before
starting the frontend or Rust build. CUDA compilation embeds PTX targeting
compute capability 7.5; runtime loading uses the NVIDIA driver and visible
device ordinal 0. CUDA has no hybrid placement or prefix-KV reuse. Installer
availability does not broaden its **qualified** status beyond the exact tuple
in [validation evidence](project-status/validation-evidence.md#cuda-implementation-gate--1-september-2026).

`release` is the default Cargo profile; `fast` is the alternative. The prefix
must be absolute, must not be the filesystem root, and must contain no `.` or
`..` component. Precedence is:

1. `--prefix`;
2. `GRAPH_HORIZON_INSTALL_PREFIX`;
3. `$HOME/.local`.

The installer runs `npm ci`, builds the frontend, performs a locked Cargo
build, and installs:

- `graph-horizon` under `<prefix>/bin`;
- the compatibility link `gh-zero-engine` beside it;
- Web assets under `<prefix>/share/graph-horizon/web`.

The npm `allowScripts` policy permits exactly `esbuild@0.28.1` and rejects
`@parcel/watcher@2.5.6`. Any dependency that adds an install script must be
classified before its lockfile change is accepted.

If `<prefix>/bin` is not an exact `PATH` entry, the installer prints an export
command but does not edit any shell file.

Build availability does not by itself assign `production`, `qualified`, or
`reference` status. See [backend support status](engine/backend-support-status.md).

## Verification

```sh
graph-horizon --version
graph-horizon --help
graph-horizon --model /absolute/path/to/model.gguf
```

If the command is not found after a default installation:

```sh
export PATH="$HOME/.local/bin:$PATH"
```

Maintainer-facing installer and validation commands are listed in the
[support script reference](../support/script-command-reference.md).
