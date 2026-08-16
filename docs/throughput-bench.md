<!--
This page owns the public-event benchmark interface, metrics, output, safe
errors, and limitations. It defines no quality, support, or release threshold.
-->

# End-To-End Throughput Benchmark

`examples/bench.rs` is the retained end-to-end iterative performance
executable. It runs one explicit model tuple through public `Engine` events and
emits one human-readable record. It measures and reports; it does not
authenticate an artifact, compare revisions, or assign a performance verdict.
Focused temporary instrumentation may supplement it under the performance
investigation process when a task requires phase or GPU attribution.

The [performance investigation process](performance-investigation-process.md)
owns tuple authentication, A/B comparability, thresholds, correctness gates,
time budget, and terminal states.

## Interface

```text
bench <model.gguf> --context N --kv f16|int8
  [--weights-percent 0..100] [--prompt TEXT]
  [--max-tokens N] [--warmup N] [--reps N]
```

The model is the first and only positional argument. `--context` and `--kv`
are required; there is no environment fallback or implicit context preset.
`--weights-percent` is optional and is used only by a hybrid build.

Defaults are:

| Argument | Default |
|---|---|
| `--prompt` | `Ciao` |
| `--max-tokens` | `32` |
| `--warmup` | `1` |
| `--reps` | `3` |

All input is untrusted. Recognized options must appear at most once, have one
valid value, and satisfy their documented range. An empty prompt, zero context,
fewer than two requested tokens, more than 100 percent hybrid weights, or zero
repetitions is invalid before model loading.

## Profile Examples

CPU:

```sh
cargo run --locked --release --no-default-features --features cpu \
  --example bench -- /path/to/model.gguf --context 2048 --kv f16 \
  --prompt "Example" --max-tokens 8 --warmup 0 --reps 1
```

Metal:

```sh
cargo run --locked --release --no-default-features --features metal \
  --example bench -- /path/to/model.gguf --context 2048 --kv f16 \
  --prompt "Example" --max-tokens 8 --warmup 0 --reps 1
```

Metal-hybrid:

```sh
cargo run --locked --release --no-default-features --features metal-hybrid \
  --example bench -- /path/to/model.gguf --context 2048 --kv f16 \
  --weights-percent 25 --prompt "Example" --max-tokens 8 \
  --warmup 0 --reps 1
```

These commands demonstrate the interface, not an A/B tuple. For an
investigation, use the task-directed matrix when one is specified; otherwise
use the default iterative tuple by reference from the process page and change
only the selected profile as that procedure permits.

## Metric Contract

The retained throughput harness aggregates public events:

| Field family | Definition | Unit |
|---|---|---|
| `prompt_tps_*` | Prompt-token count divided by time to first public text delta; includes first sampling | tokens/second |
| `ttft_ms_*` | Time from `Engine::generate` entry to first public text delta | milliseconds |
| `decode_tps_*` | Intervals between first and last public text deltas | deltas/second |

`prompt_tokens` comes from terminal `GenerationStats` after the engine's actual
tokenization and template. `decoded_tokens` counts public `TextDelta` events.
It is not the model completion-token count: a model token that leaves an
incomplete UTF-8 sequence produces no immediate public delta.

Each family reports its arithmetic mean, median, sample standard deviation, and CV.
CV is sample standard deviation divided by a positive mean and is printed as a
fractional ratio. With one measured repetition, standard deviation and CV are
`n/a`, never zero. The benchmark does not print the harness's first/last decode
segment statistics.

## Success Output

A successful invocation writes exactly one newline-terminated stdout record in
this field order:

```text
prompt_tokens=<integer> decoded_tokens=<integer> prompt_tps_mean=<fixed-2> prompt_tps_median=<fixed-2> prompt_tps_stddev=<fixed-2|n/a> prompt_tps_cv=<fixed-4|n/a> ttft_ms_mean=<fixed-2> ttft_ms_median=<fixed-2> ttft_ms_stddev=<fixed-2|n/a> ttft_cv=<fixed-4|n/a> decode_tps_mean=<fixed-2> decode_tps_median=<fixed-2> decode_tps_stddev=<fixed-2|n/a> decode_tps_cv=<fixed-4|n/a>
```

`fixed-2` has exactly two decimal places. `fixed-4` has exactly four decimal
places; `0.0500` means 5%. Success output contains no model path, generated
text, artifact identity, revision, environment identity, or verdict.

All means and medians must be finite and positive. Every available standard deviation
must be finite and non-negative, and every calculated CV must be finite. At
least two public text deltas are required. A violation produces no success
record.

## Bounded Failures

Usage failures exit 2 and write exactly one of these stderr lines:

```text
bench: invalid arguments
bench: invalid --context
bench: invalid --kv
bench: invalid --weights-percent
bench: invalid --prompt
bench: invalid --max-tokens
bench: invalid --warmup
bench: invalid --reps
```

Unknown options, invalid command shape, positional errors, and non-UTF-8 input
use `bench: invalid arguments`. A recognized missing, malformed, duplicated, or
out-of-range option uses its option-specific line.

Execution failures exit 1 and write exactly one of:

```text
bench: model initialization failed
bench: invalid measurement
bench: measurement failed
```

The boundary never prints the model path, backend diagnostic, source location,
or stack trace. Initialization errors use the first line; invalid aggregates or
too few deltas use the second; other generation or aggregation failures use the
third.

## Model-Free Verification

The interface and retained statistics compile and test without a model:

```sh
cargo check --locked --no-default-features --features cpu --example bench
cargo test -p graph_horizon_engine --locked --no-default-features \
  --features cpu 'harness::'
```

A successful record requires an external supported artifact and the selected
runtime resources. Record that check with its authenticated tuple; never replace
a missing artifact or backend with another one.
