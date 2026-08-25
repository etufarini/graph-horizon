<!--
This reference defines reproducible prefill and decode measurement hygiene for
the safe optimization loop; it does not choose optimization candidates.
-->

# Measuring Prefill and Decode

Use the maintained public examples, not ad hoc timers.

```sh
support/profiling/profile.sh --model "$GRAPH_HORIZON_MODEL" \
  --backend <backend> --context 4096 --kv f16

cargo run --release --no-default-features --features <backend> \
  --example bench -- "$GRAPH_HORIZON_MODEL" --context 4096 --kv f16 \
  --prompt "Hello" --max-tokens 32 --warmup 1 --reps 5
```

Record:

- prompt tokens and prompt tok/s;
- TTFT;
- decode tok/s;
- mode and CPU/GPU layers;
- weights, KV, scratch, fixed, staging, crossing, and reserve breakdown;
- mean/deviation and repetition count.

## Measurement hygiene

Use identical release or fast builds for the baseline and candidate. Warm up,
keep the machine idle, and check for throttling. Do not compare runs with
different contexts, KV, or placement.

An improvement must exceed baseline noise. Also check non-target metrics and
memory: a decode gain that degrades prefill or changes placement is not
automatically acceptable.

## Diagnosis

- High TTFT: focus the profile on prompt/prefill.
- Slow decode: increase `max_tokens` while keeping the prompt fixed.
- Slow mixed mode: compare crossing and layer count without changing the split
  between baseline and candidate.
- Context growth: compare f16/int8 separately; do not mix results.

Do not add permanent tracing for a single measurement. If attribution is
missing, prefer a narrow temporary counter and remove it before committing.
