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

The commands below are prepared for the authenticated `v0.1.5` bootstrap after
that exact candidate is published. They do not claim publication; `v0.1.4`
remains the current stable release.

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

### Hybrid profiles

On Apple Silicon macOS with the Xcode command-line Metal tools:

```sh
curl --fail --location --silent --show-error \
  https://raw.githubusercontent.com/etufarini/graph-horizon/v0.1.5/install.sh \
  | bash -s -- --backend metal-hybrid
```

On Linux `x86_64` with a working Vulkan loader and driver:

```sh
curl --fail --location --silent --show-error \
  https://raw.githubusercontent.com/etufarini/graph-horizon/v0.1.5/install.sh \
  | bash -s -- --backend vulkan-hybrid
```

On Linux `x86_64` with an NVIDIA driver and CUDA Toolkit `nvcc`:

```sh
curl --fail --location --silent --show-error \
  https://raw.githubusercontent.com/etufarini/graph-horizon/v0.1.5/install.sh \
  | bash -s -- --backend cuda-hybrid
```

The backend argument selects one compile-time profile. Hybrid placement is
chosen only when a model loads and can be constrained with
`--vram-weights-percent` and `--vram-reserve-mib`. No hybrid profile enables
prefix-KV reuse.

Use `--backend cpu` when GPU execution is not required.

## Local Installer Options

```text
support/install.sh --backend cpu|vulkan|vulkan-hybrid|metal|metal-hybrid|cuda|cuda-hybrid \
  [--profile release|fast] [--prefix /absolute/install/prefix]
```

`--backend` is required and selects one compile-time profile. There is no
runtime backend switch or fallback.

| Platform | Accepted build backends |
|---|---|
| macOS arm64 | `cpu`, `vulkan`, `vulkan-hybrid`, `metal`, `metal-hybrid` |
| Linux x86_64 | `cpu`, `vulkan`, `vulkan-hybrid`, `cuda`, `cuda-hybrid` |

The installer accepts both CUDA profiles only for Linux `x86_64` and requires
`nvcc` before starting the frontend or Rust build. Compilation embeds PTX
targeting compute capability 7.5; runtime loading uses the driver and visible
device ordinal 0. `cuda` remains all-GPU-or-error. `cuda-hybrid` composes the
existing CPU and CUDA numeric paths with an immutable separate-memory plan.
Neither profile supports prefix-KV reuse. Installer availability does not
broaden their **qualified** status beyond the exact tuples in
[validation evidence](project-status/validation-evidence.md#current-backend-evidence).

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
