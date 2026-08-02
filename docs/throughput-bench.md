<!--
This document owns the current public-API throughput example, its metrics,
arguments and limits. It does not define quality or release thresholds.
-->

# End-To-End Throughput Benchmark

The root example `examples/bench.rs` measures a complete generation through
`Engine`. It requires a supported GGUF, an explicit context, and a KV scheme; it
does not depend on the model name or family.

The benchmark measures and reports. It does not compare against an oracle, apply
quality thresholds, or replace the validation matrix.

## Metrics

The `gh_zero_engine::harness::throughput` harness uses public events:

| Metric | Definition |
|---|---|
| `prompt_tps` | Rendered prompt tokens divided by time to first delta |
| `ttft` | Time from `Engine::generate` to the first `TextDelta` |
| `decode_tps` | Text deltas divided by intervals between the first and last delta; available with at least two deltas |

Each metric is aggregated across repetitions with the mean and sample standard
deviation. With one observation, deviation is `n/a`, not zero. The prompt token
count comes from `GenerationStats`, so it includes the tokenization and template
actually selected by the engine.

The decode counter follows `TextDelta`, not
`GenerationStats.completion_tokens`. A token that does not yet complete UTF-8
may therefore produce no immediate delta: the value describes the exact public
stream measured by the harness.

The harness also calculates throughput for the first and last halves of
generations with at least four deltas. The current CLI example does not print
these fields, but they remain available in `ThroughputReport` to other callers.

## Usage

```text
bench <model.gguf> --context N --kv f16|int8
  [--prompt TEXT] [--max-tokens N] [--warmup N] [--reps N]
```

`--context` and `--kv` are required. Other defaults are:

| Argument | Default |
|---|---|
| `--prompt` | built-in prompt |
| `--max-tokens` | `32` |
| `--warmup` | `1` |
| `--reps` | `3` |

The model must be the first positional argument. There is no `GH_ZERO_MODEL`
fallback and there are no context presets.

CPU example:

```sh
cargo run --no-default-features --features cpu --example bench -- \
  /path/to/model.gguf --context 4096 --kv f16 \
  --prompt "Hello" --max-tokens 32 --warmup 1 --reps 3
```

To measure another backend, replace the feature with `vulkan` or `hybrid`
without changing the artifact, prompt, context, or KV of the compared row.

## Output

The example prints one line similar to:

```text
prompt_tokens=27 prompt_tps=123.45±1.23 ttft_ms=218.70 decode_tps=35.60
```

A decode with fewer than two deltas displays `decode_tps=n/a`.

## Interpretation

- `prompt_tps` includes the first sampling step and is not isolated prefill;
- TTFT and decode depend on context, prompt, backend, KV, build, and hardware;
- warmup and repetition counts must remain identical between measurements;
- an out-of-memory error describes that configuration's capacity, not a numeric regression;
- no reported value alone constitutes a support gate.

## Tool Verification

Model-free tests cover the harness calculations. The example can be compiled
without running inference:

```sh
cargo test -p gh_zero_engine --no-default-features --features cpu harness
cargo check --no-default-features --features cpu --example bench
```

An end-to-end run instead requires an external artifact and resources; record it
with the model, digest, revision, feature, context, KV, and hardware.
