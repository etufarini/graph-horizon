<!--
This reference defines the correctness gates used by the optimization loop. It
describes the protocol and limits, not qualification results.
-->

# Correctness Oracle

## Local gate

The mandatory baseline does not require models:

```sh
cargo test --workspace --no-default-features --features cpu
cargo test -p graph_horizon_engine --no-default-features --features vulkan error_matrix
cargo test -p graph_horizon_engine --no-default-features --features vulkan-hybrid error_matrix
```

The synthetic tests cover 3B/8B/14B dimensions, Q8_0 rejection, the Q4_K_M
profile, f16/int8 KV, and internal backend formats.

## Real-model gate

Use only Q4_K_M artifacts whose size and SHA are fixed in `support/models.tsv`.
A capability-compatible file that does not match remains
`compatible/unverified`.

`support/testing/parity-check.sh` invokes the ignored tests with:

- a fixed template and exact prompt IDs against the reference tokenizer;
- sixteen oracle tokens obtained from the reference and applied with teacher
  forcing;
- every oracle token present in the deterministic local top two, with local
  top-1 IDs recorded separately;
- an explicit backend feature and context.

Example:

```sh
support/testing/parity-check.sh \
  --models-dir "$GRAPH_HORIZON_MODELS_DIR" \
  --model-id "$GRAPH_HORIZON_MODEL_ID" \
  --backend cpu --kv f16 \
  --reference-server "$GRAPH_HORIZON_REFERENCE_SERVER"
```

Repeat for f16/int8 when the candidate touches KV. Repeat for both profiles when
it touches formats or dispatch. For Vulkan, E15 is expected when the file does
not fit; it does not authorize switching backends. A mixed hybrid row uses the
same authenticated Q4_K_M, an explicit context, 25%, and requires positive layer
counts on both sides.

## Determinism

Keep these constant:

- GGUF bytes and SHA;
- prompt and system prompt;
- context and `max_tokens`;
- KV scheme;
- feature/backend and Cargo profile;
- reference and commit;
- hardware and thermal conditions.

Do not regenerate a reference to accept a candidate. An oracle token missing
from the local top two fails the row; a difference in recorded top-1 IDs alone
does not fail this bounded gate. If the candidate promises exact identity, it
must also pass the exact gate selected before measurement.

## Numeric candidates

The current protocol retains a bounded top-two oracle for sixteen teacher-forced
steps. This is not a universal tolerance: a candidate that changes ordering,
layout, or precision may use it only when that criterion measures the declared
risk. KV changes also require their own payload, layout, and quality gates; any
other numeric divergence requires an approved metric and threshold before the
result is observed.
