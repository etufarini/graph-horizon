---
name: gpu-backend-benchmark
description: >-
  Benchmark GPU inference backends on one machine and generate reproducible
  Markdown, JSON, and PNG comparisons of prefill throughput, decode throughput,
  and TTFT. Use for CUDA-versus-Vulkan measurements on one NVIDIA GPU or
  Metal-only measurements on Apple silicon.
---

# GPU Backend Benchmark

Measure one model tuple without changing anything except the backend. The
scripts reject or clearly invalidate comparisons when the physical GPU, model,
runtime, token counts, context, batch, sampling, or runtime options differ.

## Invariants

- Compare CUDA and Vulkan only on the same Linux or Windows NVIDIA GPU.
- On macOS, run Metal only and report the human-readable Mac model and unified
  memory. Never invent a CUDA/Vulkan comparison there.
- Use the same readable model file, SHA-256, quantization, prompt text, actual
  prompt tokens, generated tokens, context, batch, seed, temperature, runtime
  version, and fixed adapter arguments for both backends.
- Run at least one warm-up and three measured repetitions per workload/backend;
  publish medians while retaining every measurement in JSON.
- A performance report is measurement evidence, not a correctness or model
  quality qualification.

## Prerequisites

Requires Python 3.10+ and an inference runtime adapter implementing
[the runtime adapter contract](references/runtime-adapter.md). The scripts use
only the Python standard library and installed system tools. Do not install a
framework or plotting package for this skill.

Hardware detection prefers `nvidia-smi` plus `vulkaninfo --summary` on Linux and
Windows, and `system_profiler`, `sysctl`, and `sw_vers`-equivalent platform data
on macOS. Missing optional version tools produce `n/a`; missing proof of a
supported physical GPU makes that backend unavailable.

Before benchmarking, inspect the target runtime and either select its existing
adapter or create a temporary narrow adapter outside `.skills`. Do not shell
interpolate model paths, prompts, or options. The adapter must report actual
token counts and phase timings from the runtime rather than estimating from
string length.

## Inputs

Establish explicitly:

- the readable model file and human-readable model name;
- the model's actual quantization or precision;
- one adapter command and fixed arguments shared by all backends;
- the requested context size, capped automatically by every probed backend's
  reported model context limit;
- batch size, normally `1`, unless another identical value is intended.

Never place credentials in adapter arguments. Only a hash of fixed adapter
arguments is retained, but arguments remain visible to the local process list.

## Run

First inspect hardware detection:

```sh
python3 .skills/gpu_backend_benchmark/scripts/hardware.py
```

Then use a fresh output directory:

```sh
python3 .skills/gpu_backend_benchmark/scripts/benchmark.py \
  --adapter /absolute/path/to/runtime-adapter \
  --model /absolute/path/to/model.gguf \
  --model-name "Model name" \
  --quantization Q4_K_M \
  --context-size 4608 \
  --batch-size 1 \
  --output-dir /absolute/path/to/gpu-benchmark
```

Use repeated `--adapter-arg VALUE` only for fixed runtime options that apply
unchanged to every backend. On Windows, invoke the same script with `py -3` when
that is the installed Python launcher.

The orchestrator:

1. detects hardware and candidate backends;
2. probes runtime support and verifies the selected physical device;
3. hashes the model and caps context to the common reported limit;
4. deterministically calibrates Short (~128/128), Medium (~1024/256), and Long
   (~4096/512) prompt/decode workloads using the runtime's tokenizer;
5. records context-driven adjustments;
6. warms up and runs every available backend at least three times;
7. validates the tuple, calculates medians, and generates all outputs.

An unavailable CUDA or Vulkan backend is skipped with a reason while the other
may still complete. If both complete but any comparable field differs, retain
the measurements but label the comparison invalid and suppress percentages.

## Outputs

The selected output directory receives:

- `benchmark-results.json`: machine, configuration, deterministic prompts,
  every raw run, medians, failures, and comparison validity;
- `benchmark-results.md`: reproducible configuration, the main performance
  table, availability, adjustments, and metric-specific conclusions;
- `benchmark-results.png`: separate grouped sections for prefill tokens/s,
  decode tokens/s, and TTFT milliseconds.

Existing output files are never overwritten. Use a new directory for each run.
The Markdown report intentionally records model name and hash, not its private
absolute path.

## Failures

Stop and report the corrective action for a missing adapter/model, no supported
GPU backend, unreadable or malformed adapter JSON, model load failure, out of
memory, context overflow, timeout, or zero complete measurements. Preserve a
generated partial report when at least one measurement exists; its process exit
is nonzero when no backend completed all three workloads.

Do not replace a failed backend, model, prompt, precision, or context with a
different tuple. Fix the named condition and rerun in a fresh output directory.

## Interpretation and Validation

Higher prefill/decode throughput is better; lower TTFT is better. Read each
metric for its workload rather than declaring an overall winner. Percentage
advantages appear only for a valid CUDA/Vulkan comparison.

After creating or changing this skill, run:

```sh
python3 .skills/gpu_backend_benchmark/scripts/validate.py
git diff --check
```

For a real benchmark, also inspect all three artifacts and confirm the adapter's
runtime/version/device fields against the runtime's own diagnostics.
