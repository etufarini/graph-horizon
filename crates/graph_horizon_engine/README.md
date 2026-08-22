<!--
  This document owns the contract of the library, backends, and memory;
  operational results belong in the validation documents.
-->

# graph_horizon_engine

`graph_horizon_engine` is the text-to-text inference runtime for
`general.architecture=mistral3`. Its public API accepts chat messages, runs
prefill/decode, and produces incremental events without depending on a console,
HTTP, or Web UI.

## Backend features

The crate does not select a backend by default: the consumer enables exactly
one of the five profiles.

```sh
cargo check -p graph_horizon_engine --no-default-features --features cpu
cargo check -p graph_horizon_engine --no-default-features --features vulkan
cargo check -p graph_horizon_engine --no-default-features --features vulkan-hybrid
cargo check -p graph_horizon_engine --no-default-features --features metal
cargo check -p graph_horizon_engine --no-default-features --features metal-hybrid
```

Build availability does not assign support status. The primary labels are:

| Profile | Status | Execution |
|---|---|---|
| `cpu` | **reference** | Complete portable numeric path |
| `vulkan` | **production** | Entire model on a Linux x86_64 Vulkan GPU or an error |
| `vulkan-hybrid` | **qualified** | CPU plus Vulkan with an immutable all-GPU, mixed, or CPU-only plan |
| `metal` | **qualified** | Entire model on Metal or an error |
| `metal-hybrid` | **qualified** | CPU plus Metal with all-Metal, mixed, or CPU-only modes |

Production Vulkan covers compatible NVIDIA and AMD devices. Metal and
Metal-hybrid are qualified on the documented Apple M4 10-core/macOS 26.3 tuple.
Vulkan-hybrid requires a fresh complete real-model mixed-placement campaign for
production promotion. `reference` is a numeric role rather than a lower
maturity tier and carries no performance promise. The normative definitions
and exact scope are in the [backend contract](../../docs/backend.md#support-status).

The hybrid plan is immutable after loading. Only the mixed plan copies the
residual from CPU to GPU, once per pass; each layer's KV remains on its backend.
The report contains the mode, split, layer counts, and CPU/GPU breakdown of
weights, KV, scratch, fixed, staging, crossing, and reserve memory.

### Semantic Reasoning qualification

Semantic qualification is a separate, test-only policy. The runner retains the
three historical Instruct rows and invokes only the three Reasoning profiles.
It uses `vulkan-hybrid` to observe placement, but a qualifying generation row
requires all-GPU Vulkan; `mixed`, `cpu-only`, or missing resources become
`external-verification` and do not activate a fallback.

The Reasoning run uses `f16` KV, `context=4096`, `max_tokens=4096`,
`temperature=0.7`, `seed=0`, `top_p=1`, `top_k=0`, `min_p=0`, and
`repeat_penalty=1`. The corpus contains only the nine cases S01–S04 and S06–S10.
Each case is attempted once; `context`, `max-tokens`, engine errors, incomplete
Reasoning markers, and semantic misses are terminal results, not reasons for a
retry, tuning, CPU fallback, or an oracle.

The final status of a Reasoning run can be `qualified`, `not-qualified`, or
`external-verification`. A structurally complete six-row matrix is an
operational success for the runner even when a model does not pass the semantic
gate. The runtime continues to emit raw text and does not own this assessment
policy.

The [validation log](../../docs/validation.md) records the most recent reviewed
campaign and its exact source revision. Those historical results do not qualify
later runtime changes or a future release tag. The server selects the same
sampling parameters for a Reasoning profile, but the complete gate remains
owned by the harness because it also fixes context, KV, placement, and corpus.

`Engine::memory()` provides retained model weights and full-context KV capacity
for every backend profile. `Engine::placement()` provides the final hybrid
placement and its complete planned breakdown. Neither method exposes process
RSS, live allocator state, or raw available VRAM. An error after final selection
remains a failure with no retry or fallback. The command and operational
protocol are in the [script guide](../../support/README.md).

## Ministral contract

Architecture, tensor, and quantization recognition is capability-based and does
not authenticate a file by its name, directory, total byte size, or
`general.name`. The latter selects only the chat policy. The exact,
case-sensitive values `ministral-3B-Reasoning-2512`,
`ministral-8B-Reasoning-2512`, and `ministral-14B-Reasoning-2512` enable the
private Reasoning profile; any other name containing `Reasoning` is rejected
before backend allocation. The Instruct configuration remains dimension-generic
for 3B, 8B, and 14B.

The only public GGUF profile is `Q4_K_M`: Q4_K/Q6_K matrices and
one-dimensional F32 auxiliaries. The parser retains `GgmlType::Q8_0` to identify
and diagnose a Q8 file, but the E04 gate rejects it before allocation. The six
real validation rows are reproducible evidence, not a whitelist. The fixed
values for Ministral 3 Instruct/Reasoning 2512 belong to the private
[`family/mistral/version.rs`](src/family/mistral/version.rs) module, which also
owns the fixed Reasoning system prompt.

The backend maintains FP16 activations, FP32 residual/logits, and internal
F16/Q4_K/Q5_K/Q6_K numerical formats. This internal surface does not add public
Ministral profiles.

## Public facade

The crate root exposes only:

- engine/config/placement: `Engine`, `EngineConfig`, `ModelMemory`,
  `BackendMemory`, `PlacementReport`, plus bounded `model_name` and static
  `backend_name` inspection;
- chat and events: `Message`, `Role`, `Request`, `SamplingParams`, `Event`,
  `GenerationPhase`, `GenerationStats`, `EventSink`, `render_chat_prompt`;
- Ministral/GGUF inspection: `MistralConfig`, `TekkenTokenizer`,
  `GgufFile`, `GgufValue`, `GgmlType`, `TensorInfo`;
- KV selection: `KvQuant`;
- `harness::throughput` with `BenchConfig`, `Stat`, `ThroughputReport`, and
  `run`, plus `harness::ParityReport` and `harness::validate_parity` for the
  fixed external-oracle protocol.

Backends, sampling, KV layout, allocation metadata, and the graph remain
internal. The active harness measures only the engine's public stream; it does
not expose buffer profiling or private model access.

The `Engine` methods expose the resolved `context_limit`, immutable `memory`,
optional hybrid `placement`, model-profile `default_sampling`, ordinary
`generate`, and caller-keyed `generate_cached`. The cached form reuses an exact
rendered-token prefix only on standalone Vulkan and Metal and only for the same
16-byte key; CPU and hybrid profiles run the ordinary generation path. One
different key or prefix can replace the retained slot.

## Chat

```rust
use graph_horizon_engine::{Engine, EngineConfig, Message, Request, Role, SamplingParams};

let engine = Engine::new(
    std::path::Path::new("/path/model.gguf"),
    EngineConfig::default(),
)?;
let request = Request {
    messages: vec![Message {
        role: Role::User,
        content: "Hello".into(),
    }],
    sampling: SamplingParams::greedy(),
    max_tokens: 128,
};
engine.generate(request, &mut |event| {
    println!("{event:?}");
    true
});
# Ok::<(), color_eyre::Report>(())
```

A valid sequence contains at most one initial `System` message, followed by
strictly alternating `User` and non-empty `Assistant` messages, and ends with
`User`. The template is fixed: `tokenizer.chat_template` is not executed. In the
Reasoning profile, the fixed internal Reasoning instruction is always rendered
first. Non-empty content from an initial `System` message follows it after a
blank line; an empty message adds nothing. Caller-provided content is ordinary
tokenizer input and cannot inject structural control tokens. The `[THINK]` and
`[/THINK]` markers generated by the model remain raw text in `TextDelta`,
without a new public event or channel.

The public events are:

- `Phase(Prefill)`;
- `Phase(Decode)`;
- `TextDelta(String)`;
- `Finished(GenerationStats)`;
- `Error("generation failed")`.

Phase events are ordered and emitted immediately before their work.
`GenerationStats` separates total prompt tokens from actually-prefilled tokens,
then reports completion tokens and engine-measured prefill/decode milliseconds.
With prefix reuse, cached tokens remain in the prompt total but not the prefill
count. This preserves meaningful phase throughput without exposing cache state.
Each generation emits exactly one terminal event. If the sink returns `false`,
execution is cancelled, resources are released, and no further event is
emitted.

## Memory and context

`EngineConfig` accepts a context, an `f16|int8` KV scheme, CPU threads, and a
device budget. `EngineConfig.context_tokens = None` applies the engine policy:
`min(32,768, maximum GGUF context)`. `Some(N)` instead requests exactly `N`:
every positive value up to the GGUF maximum is valid if it fits in backend
memory and is neither truncated nor reduced. The effective limit is returned by
`Engine::context_limit` to local surfaces. In hybrid profiles:

- no percentage: automatic selection based on available capacity;
- percentage `0`: CPU-only with no device initialization;
- percentage `100`: device-only endpoint (`all-gpu` or `all-metal`).

Vulkan uses separate RAM and VRAM; automatic RAM is
`floor(MemAvailable × 90 / 100)` on Linux, and the automatic VRAM reserve is the
greater of 256 MiB and 5%.
Metal uses unified memory: CPU and Metal compete for the same capacity derived
from physical memory and the recommended working set. Allocation, pipeline,
kernel, transfer, readback, or decoder errors after plan selection do not cause
a retry.

## Verification

The synthetic suite covers 3B/8B/14B dimensions, the public Q4_K_M gate, FP16
activations, FP32 residual/logits, internal numerical formats, f16/int8 KV, and
the three hybrid placements. Separate tests verify that Q8 remains diagnosable
only and is rejected.

Real-model tests are ignored by default and do not look for local fallbacks.
Required resources are explicit:

| Variable | Test-only use |
|---|---|
| `GRAPH_HORIZON_MODEL` | single real GGUF |
| `GRAPH_HORIZON_MODEL_Q4_K_M` | Q4_K_M artifact for the tokenizer test |
| `GRAPH_HORIZON_REFERENCE_CLI` | oracle executable for greedy tokens |
| `GRAPH_HORIZON_REFERENCE_TOKENIZE` | oracle executable for tokenization |
| `GRAPH_HORIZON_REFERENCE_PROMPT_IDS` | Reasoning prompt vector supplied by the script |
| `GRAPH_HORIZON_REFERENCE_COMPLETION_IDS` | 16 oracle IDs supplied by the script |

`GRAPH_HORIZON_CONTEXT`, `GRAPH_HORIZON_KV`, and, in the mixed test,
`GRAPH_HORIZON_VRAM_WEIGHTS_PERCENT` configure the external row without locating
resources. The `GRAPH_HORIZON_REFERENCE_*_IDS` variables are internal to the
parity script and are not stable runtime configuration.

```sh
GRAPH_HORIZON_MODEL="/path/model.gguf" GRAPH_HORIZON_CONTEXT=4096 GRAPH_HORIZON_KV=f16 \
GRAPH_HORIZON_REFERENCE_PROMPT_IDS="..." GRAPH_HORIZON_REFERENCE_COMPLETION_IDS="..." \
  cargo test --release -p graph_horizon_engine --no-default-features --features cpu \
  --test family_agnostic real_selected_runtime_parity_and_lifecycle \
  -- --ignored --nocapture --exact
```

The complete interfaces of the 74-row matrix and semantic acceptance are
described in the [script guide](../../support/README.md); reviewed results belong
in the [validation log](../../docs/validation.md). Maintainer-facing source ownership
is mapped separately in [`src/README.md`](src/README.md).
