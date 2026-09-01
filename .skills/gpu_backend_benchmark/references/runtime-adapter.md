# Runtime Adapter Contract

Use this reference when selecting or implementing the narrow executable passed
to `benchmark.py --adapter`. The adapter translates one installed inference
runtime into three JSON-only modes. It must not silently retry another backend,
model, device, context, precision, prompt, or sampling configuration.

The orchestrator invokes an argument vector directly, never a shell:

```text
ADAPTER [fixed adapter args...] MODE [mode args...]
```

Success writes exactly one UTF-8 JSON object to stdout and exits zero. Keep
diagnostics on stderr. Output over 64 KiB, non-JSON output, or a successful
object containing `error` is malformed.

## Probe

```text
probe --model FILE --backend cuda|vulkan|metal
```

Probe runtime support, validate that the model can be opened, and identify the
device the runtime will actually use:

```json
{
  "available": true,
  "backend": "cuda",
  "runtime": {"name": "Example Runtime", "version": "1.2.3"},
  "device": {
    "id": "GPU-fbe2e4b5-f33d-ab11-9bb8-7ecb4d750ea2",
    "name": "NVIDIA GeForce RTX 2060"
  },
  "model": {
    "identifier": "model-family-q4_k_m",
    "quantization": "Q4_K_M",
    "context_limit": 32768
  }
}
```

Use a stable physical-device UUID where the runtime exposes one. CUDA and
Vulkan IDs may differ in punctuation or the `GPU-` prefix; the orchestrator
normalizes those forms, not unrelated identifiers.

## Count

```text
count --model FILE --backend BACKEND --prompt-file FILE
```

Apply the same chat template/tokenizer as inference and return the actual total
prefill count:

```json
{"prompt_tokens": 128}
```

Do not estimate from bytes, characters, words, or a different tokenizer. This
mode may initialize the tokenizer without allocating the complete GPU model.

## Run

```text
run --model FILE --backend BACKEND --context N --batch-size N \
  --prompt-file FILE --max-tokens N --seed N --temperature 0 --json
```

Perform exactly one request and return one raw measurement:

```json
{
  "backend": "cuda",
  "device_id": "GPU-fbe2e4b5-f33d-ab11-9bb8-7ecb4d750ea2",
  "gpu_name": "NVIDIA GeForce RTX 2060",
  "runtime_name": "Example Runtime",
  "runtime_version": "1.2.3",
  "runtime_config": "greedy; fixed device; no early EOS",
  "model_identifier": "model-family-q4_k_m",
  "quantization": "Q4_K_M",
  "prompt_tokens": 128,
  "completion_tokens": 128,
  "context_size": 4608,
  "batch_size": 1,
  "seed": 1,
  "temperature": 0,
  "prefill_seconds": 0.125,
  "decode_seconds": 1.5,
  "ttft_ms": 130.2
}
```

`prefill_seconds` measures the runtime's actual prompt-processing phase.
`decode_seconds` measures generation of the reported completion tokens. `ttft_ms`
spans request entry through the first generated token and may therefore be
slightly broader than prefill time. All timings use a monotonic clock and must
be positive and finite.

The adapter must force exactly `--max-tokens` tokens when its runtime supports
disabling early EOS. Otherwise an early stop is an incomplete workload rather
than a silently smaller benchmark.

## Bounded Failures

On failure, exit nonzero and write one safe JSON object to stdout:

```json
{
  "error": {
    "code": "out_of_memory",
    "message": "insufficient device memory"
  }
}
```

Supported codes are:

| Code | Meaning |
| --- | --- |
| `unsupported_backend` | Runtime was not built for the requested backend |
| `model_load` | Model is unreadable, corrupt, or unsupported |
| `out_of_memory` | GPU or unified memory allocation failed |
| `context_too_large` | Requested context exceeds the model/runtime limit |
| `runtime_failure` | Benchmark execution failed for another bounded reason |

Do not expose stack traces, absolute paths, prompts, generated text, secrets, or
driver internals in JSON. A developer may run the adapter directly to inspect
its stderr when deeper local diagnostics are authorized.
