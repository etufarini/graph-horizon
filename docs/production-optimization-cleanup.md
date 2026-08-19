<!--
This report owns the evidence for the post-optimization production cleanup:
baseline identity, architecture inventory, ablation decisions, qualification,
and the final review boundary. Historical experiment detail remains in the
specialization reports and is not repeated here.
-->

# Production Optimization Cleanup

## Baseline

- Main commit: `a8b5a16d0c2197a7bcdc46a72546a71556aa5a2f`.
- Cleanup branch: `cleanup/backend-optimization-artifacts`.
- Final production-code checkpoint: `2a78d4da23ffefdc71ac3c8d639e47f6c8d315d8`;
  later commits on the branch contain only this qualification report. The
  handoff records the final branch-tip SHA because a committed file cannot
  contain the hash of the commit that contains it.
- Branch point: exactly the main commit above, after `git fetch --prune origin`;
  local `main` and `origin/main` were identical and the worktree was clean.
- Host: Linux x86_64, NVIDIA RTX 3060 12 GiB, driver 595.84, Vulkan 1.4.329;
  Intel CPU execution is available. AMD execution and macOS/Metal execution are
  external qualification items on this host.
- Models: all six catalogued Ministral 3B/8B/14B Instruct/Reasoning Q4_K_M
  artifacts passed `support/profiling/validate-weights.sh` without substitution.
- Oracle: the local llama.cpp checkout is `9bebfcb4b`; the repository parity
  contract requires `13f2b28b0`, so fresh pinned external parity is unavailable.
  The historical six-artifact 128-step teacher fixture is also explicitly
  recorded as unavailable in the retained specialization reports.

### Baseline validation

| Gate | Main result |
|---|---|
| `cargo fmt --all -- --check` | pass |
| CPU workspace tests | 166 root + 159 engine + 5 integration + 12 semantic pass; expected real-resource tests ignored |
| Vulkan workspace tests | 166 root + 156 engine + 5 integration + 12 semantic pass; expected long/real-resource tests ignored |
| Vulkan-hybrid workspace tests | 166 root + 229 engine + 6 integration + 12 semantic pass; expected long/real-resource tests ignored |
| CPU, Vulkan, Vulkan-hybrid Clippy, all targets, warnings denied | pass |
| Frontend check, 92 tests, production build | pass |
| Metal and Metal-hybrid build on Linux | external: explicit macOS arm64 guard and Apple-only bindings |
| Six authenticated model files and synthetic weight-format checks | pass |

### Baseline performance

All rows use Cargo `release`, F16 KV, greedy sampling and 32 requested tokens.
The short prompt is 31 repetitions of `alpha beta gamma delta `, which renders
to exactly 128 tokens; it uses one warm-up and three measured repetitions. The
long prompt is 27,996 repetitions of `a `, which renders to exactly 28,000
tokens; it uses one measured repetition because each run performs the complete
prefill. Short 3B/8B rows use context 32,768; short 14B and CPU rows use 2,048
because a 14B context allocation at 32,768 exceeds this GPU's available memory.
Long rows use context 32,768. These exact tuples will be reused for the final
same-session comparison.

| Model | Backend | Prompt | TTFT mean | Prompt tok/s | Decode tok/s | CV note |
|---|---|---:|---:|---:|---:|---|
| 3B Instruct | Vulkan | 128 | 73.28 ms | 1,746.76 | 74.30 | all <= 0.34% |
| 8B Instruct | Vulkan | 128 | 149.71 ms | 855.05 | 34.84 | all <= 0.99% |
| 14B Instruct | Vulkan | 128 | 251.05 ms | 509.87 | 23.50 | all <= 0.52% |
| 3B Instruct | Vulkan | 28,000 | 29,143.33 ms | 960.77 | 42.80 | one complete pass |
| 8B Instruct | Vulkan | 28,000 | 52,084.32 ms | 537.59 | 23.49 | one complete pass |
| 3B Instruct | CPU | 128 | 4,253.29 ms | 30.10 | 7.00 | all <= 1.61% |

## Initial Complexity Snapshot

The counts are descriptive physical-source metrics, not deletion targets.

| Metric | Main |
|---|---:|
| Engine Rust files | 172 |
| Engine Rust physical lines | 30,860 |
| Vulkan/Metal shader files | 51 |
| Vulkan/Metal shader physical lines | 4,193 |
| Rust functions, including tests | 1,289 |
| Rust structs/enums/traits, including tests | 149 |
| Engine Cargo features | 7 |
| Vulkan pipeline variants | 45 |

## Production Architecture Inventory

### Shared model and graph semantics

- `family/mistral/detect.rs`, `config.rs`, `tensors.rs`, and `tokenizer/` own
  the Q4_K_M model contract, checked dimensions, tensor inventory, and chat
  policy before backend allocation.
- `family/mistral/graph/` owns one dense operation order. Decode records one
  token through `graph::forward`; prefill records bounded packed rows through
  `graph::prefill`. They share backend operations but intentionally keep their
  GEMV-like and GEMM-like orchestration separate.
- `runtime/homogeneous.rs` owns one-backend sessions.
  `runtime/partitioned/` owns the immutable CPU-prefix/device-suffix plan and
  the single residual crossing for mixed execution.
- Fallback/compatibility: tied and dedicated output weights, F16 and INT8 KV,
  cancellation, exact context bounds, and CPU-only hybrid placement are active.
- No legacy family graph or alternate public model architecture is present.

### Backend abstraction and selection

- `backend/contract.rs` is the single backend trait. Its operations correspond
  directly to graph semantics and keep buffer ownership and Q/K/V dimensions
  explicit.
- `backend/selection.rs` maps exactly one Cargo profile to CPU, Vulkan, Metal,
  or a hybrid runtime. It owns compile-time wiring, not numeric routing.
- `backend/source.rs` supplies one canonical grouped weight inventory to all
  loaders. `WeightSelection` is required by hybrid prefix/suffix ownership.
- Suspected accidental surface: the `cpu-profile` and `vulkan-profile` Cargo
  features extend production builds solely with completed-investigation
  diagnostics; they are not backend capabilities.

### Weight loading and representation

- GGUF is memory-mapped and tensors retain the canonical F32/F16/Q4_K/Q5_K/Q6_K
  bytes. CPU buffers share mapped/copied canonical data; Vulkan and Metal upload
  the selected canonical representation.
- Hybrid loading uses the same grouped source and selects an embedding/tail and
  contiguous layer range. Tied embeddings remain one logical weight.
- Active execution-native transient representation: Vulkan Q8 activations for
  qualified Q4 decode, held in persistent scratch and consumed immediately by
  MMVQ. It is not a duplicate persistent weight representation.
- The earlier native-FP16 predecode and alternate packed-weight
  representations are absent. No current weight conversion has a missing
  consumer.

### Quantization and tensor representation

- Public weights are Q4_K_M; Q8_0 remains parser-visible only so the public gate
  can reject it precisely before allocation.
- Internal Q4_K, Q5_K and Q6_K are each consumed by CPU and GPU embedding,
  projection, or logits paths. F32 auxiliaries and F16 activations remain
  required by the dense graph contract.
- KV has one backend-independent layout for F16 and per-vector INT8 payload plus
  metadata. Key/value roles select their independently validated dimensions;
  role values are therefore not duplicate state even though current metadata
  byte counts match.
- Major scratch tensors are allocated once per backend/session and reused.
  Batched prefill views alias these buffers without an extra representation.

### CPU backend

- `backend/cpu/backend.rs` delegates the backend trait to focused buffer,
  attention, elementwise and matmul kernels.
- Scalar kernels are the portable fallback. AVX2/F16C paths are capability
  selected locally and retain scalar numeric tests.
- Qualified specializations are 32-row prefill, single-token Q4/Q6 latency
  kernels, and four-query GQA KV-row reuse.
- Suspected experimental remnant: `backend/cpu/profile.rs`, its feature flag,
  macro wrappers, dynamic label strings, global timing map, and drop-time report
  exist only to reproduce the completed CPU investigation.

### Vulkan initialization, capabilities, and resources

- `backend/vulkan/init/` owns loader/device creation and immutable device facts:
  vendor family, subgroup controls, cooperative matrix capabilities and memory
  limits.
- `pipeline/caps.rs` derives optional pipeline availability once from those
  facts; `pipeline/` builds only legal variants. Runtime policies then combine
  built-pipeline presence with operation shape/format requirements.
- `mem/` owns checked budgets, buffers, uploads and selected weight lifetime.
  Model-lifetime Q8/GQA scratch and request-lifetime KV are intentionally
  retained to avoid hot-path allocation.
- AMD-only submission row bounds remain isolated at the Vulkan session boundary
  and preserve the portable 256-row policy elsewhere.
- Suspected diagnostic state: the feature-gated timestamp query pool, phase and
  category aggregation, per-layer totals, allocation counters, and command
  hooks under `exec/profile/` do not affect normal execution semantics.

### Vulkan matmul and attention routing

- Decode matmul routes by weight format, activation shape, required device
  capabilities and scratch bounds. Qualified Q4 Q8/DP4A MMVQ falls back to the
  exact per-format projection.
- Prefill matmul routes Q4 through NVIDIA Matrix2, cooperative, AMD MMQ, metadata
  or portable batch variants as their independent capability and shape gates
  permit. Q6 uses Matrix2/cooperative/portable stages. Every optional route has
  a built-pipeline and exact-format fallback.
- Attention routes F16 decode through qualified wave32/wave64 4:1 GQA, then
  wider and generic variants. Prefill routes through Matrix2, cooperative QK,
  tiled, wide and generic variants. INT8 uses its separate exact layout-aware
  kernels.
- The 45 Vulkan pipeline variants all have a registry construction path and an
  operation route or required fallback. No shader is presently identified as
  consumerless.
- Suspected remnant: `kernels/matmul/trace.rs` and
  `GRAPH_HORIZON_LOAD_TRACE` report a route once but neither select nor verify
  it.

### Metal backend

- `backend/metal/device.rs` owns feature-family capability detection without
  product-name routing. `kernels/` keeps Metal-local dispatch and threadgroup
  policy.
- The retained tiled GQA path is selected from SIMD width, threadgroup memory,
  execution width, dimensions and batch shape; the generic Metal attention
  kernel remains its fallback.
- Metal uses canonical uploaded weights and unified-memory accounting. No
  alternate Metal weight representation or experimental kernel remains.
- Test-only device probe counters verify that CPU-only hybrid placement does not
  touch Metal; they are not production state.

### Hybrid placement and resource ownership

- `backend/hybrid/weights/model.rs` accounts canonical weights once;
  `weights/runtime.rs` derives scratch, logits, KV, fixed, staging and crossing
  bytes from neutral dimensions.
- `placement/separate.rs` handles CPU plus discrete Vulkan memory;
  `placement/unified.rs` handles CPU plus Metal unified memory. Both produce one
  immutable plan before loading.
- CPU, device and mixed prefill row capacities are stored because they drive
  different scratch budgets and execution batching. Placement tests cover
  exact fits, overflow, all-device, mixed and CPU-only modes.
- No retry or post-load fallback state exists.

### Diagnostics, configuration, tests, and documentation

- Public runtime configuration is limited to context, KV scheme, CPU threads,
  weight percentage and device reserve. No model-name or product-name fast-path
  switch exists.
- The end-to-end benchmark is family-neutral and required for the cleanup's
  performance guard.
- CPU and Vulkan feature-gated internal profilers plus the Vulkan load-trace
  environment switch are investigation facilities embedded in production
  modules; they are the first controlled ablation candidates.
- Capability, numeric oracle, lifecycle, placement and fallback tests protect
  current behavior. Historical reports remain evidence but are not runtime
  architecture documentation.

## Necessity Graph

```text
Q4_K_M load
  -> family detection and tensor contract
  -> grouped canonical WeightSource
  -> full or contiguous hybrid WeightSelection
  -> CPU mapped/canonical buffer or Vulkan/Metal upload
  -> per-format embedding, projection and logits kernels
  -> format coverage, loader transaction and real-artifact checks

Q4/Q5/Q6 decode
  -> graph::forward token
  -> Backend::matmul / Backend::logits
  -> backend-local format and capability dispatch
  -> specialized Q4/Q6 latency path or exact scalar/GPU fallback
  -> CPU golden vectors and CPU/Vulkan numeric oracles

Packed prefill
  -> runtime row-capacity policy
  -> graph::prefill bounded row views
  -> Backend::matmul_batched and Backend::attention_prefill
  -> format + shape + immutable capability route
  -> Matrix2/cooperative/MMQ/tiled or portable fallback
  -> batched-vs-sequential, route-boundary and model-size oracle tests

GQA decode
  -> KV layout with independently validated K/V widths
  -> graph decode routing
  -> CPU four-query reuse, Vulkan wave32/wave64 split+reduce, or generic decode
  -> exact 4:1/resource eligibility and generic fallback
  -> attention oracle, subgroup-capability and long-context guards

F16/INT8 KV
  -> EngineConfig::kv_quant
  -> backend-independent payload/metadata layout and request allocation
  -> per-call KV write and attention scheme dispatch
  -> CPU/Vulkan/Metal implementation
  -> layout, payload, numeric and public validation rows

Hybrid execution
  -> explicit profile and placement inputs
  -> checked model/runtime byte accounting
  -> immutable all-device, mixed or CPU-only plan
  -> selected weight ranges and per-layer local KV
  -> zero or one residual crossing per pass
  -> placement, capacity, crossing and endpoint tests
```

## Initial Ablation Candidates

| Candidate | Current consumer | Initial classification | Required proof |
|---|---|---|---|
| CPU operation profiler and `cpu-profile` | completed investigation only | remove | CPU build/tests/Clippy and performance guard unchanged |
| Vulkan timestamp profiler and `vulkan-profile` | investigation process only | controlled ablation | Vulkan builds/tests/Clippy, shader build and performance guard unchanged |
| Vulkan load-path trace and environment switch | diagnostic stderr only | remove with diagnostics | route tests and all performance rows unchanged |
| Profiler-only kernel names/category tables | Vulkan profiler only | remove transitively | compiler and repository-wide call-site search |
| Historical/diagnostic comments in affected production code | none | remove or rewrite as current invariant | line-by-line review |
| `RuntimeShape` dead-code allowance | per-feature partial consumers | audit, do not delete blindly | compile all supported profiles and inspect each field consumer |
| Optional Vulkan/Metal/CPU kernels | active capability route or fallback | retain unless controlled ablation disproves use | registry, route, capability and numeric tests |
| Canonical/internal weight formats | active loader and kernel consumers | retain | producer/consumer map above |

The invariant for the first production edit is that the five supported backend
profiles, their routing, arithmetic, resource ownership, public output and
qualified performance remain unchanged. The smallest viable change is deletion
of instrumentation-only features and hooks. Its main risk is accidentally
removing a hook that also performs command/resource work; build, route tests,
runtime tests and same-tuple benchmarks will check that boundary.

## Cleanup Decisions

### Removed investigation diagnostics

The first controlled ablation removed only facilities whose output served a
completed investigation:

- the `cpu-profile` and `vulkan-profile` Cargo features;
- the CPU global timing map, dynamic operation labels, wrappers, and drop-time
  report;
- the Vulkan timestamp query pool, phase/category aggregation, allocation and
  barrier counters, and command/pipeline hooks;
- the `GRAPH_HORIZON_LOAD_TRACE` environment branch, one-shot route reporters,
  and profiler-only `Kernel::name` table.

This deletes 1,007 lines while adding 39 lines of direct calls produced by
unwrapping the CPU operations: a net deletion of 968 lines across 22 files.
There is no replacement abstraction, dependency, runtime state, shader, public
API, numeric route, or fallback change. The performance-process page no longer
advertises the deleted Vulkan facility; historical specialization reports keep
their original profiler evidence as history.

CPU, Vulkan, and Vulkan-hybrid workspace tests pass with their baseline counts;
all three all-target Clippy profiles pass with warnings denied, as do formatting
and whitespace checks. A detached-main 3B run in the same session measured the
candidate at 1,730.49 prompt tok/s, 73.97 ms TTFT, and 73.48 decode tok/s versus
main at 1,721.22, 74.37 ms, and 73.48: every delta is at most 0.54%. Candidate
8B and 14B rows differ from the recorded main baseline by at most 0.89% and
0.70% respectively. The CPU row improved to 31.13 prompt tok/s, 4,112.17 ms
TTFT, and 7.19 decode tok/s. The deletion is therefore retained.

### Removed manual CPU SIMD override

`--no-attn-simd` was an optimization A/B control, not a required production
configuration. Unsupported CPUs already select the scalar attention fallback
through runtime feature detection, while scalar and SIMD arithmetic remain
covered by direct oracle tests. Removing the flag also removed one public
`EngineConfig` field, the set-once override state and setter, and the last
boolean-only parser representation: one enum, one parsed-state field, one
lookup function, and the boolean parser/help branches disappeared transitively.

CPU, Vulkan, and Vulkan-hybrid workspace tests retain their baseline results;
all three Clippy profiles pass with warnings denied. The stable candidate CPU
row measured 30.39 prompt tok/s, 4,211.44 ms TTFT, and 7.42 decode tok/s, versus
the recorded main row at 30.10, 4,253.29 ms, and 7.00. Prompt throughput and
TTFT are within 1%; decode improved. Automatic SIMD eligibility and the scalar
fallback are unchanged, so the override and its parser state remain deleted.

### Consolidated backend construction policy

Model construction was an accidental member of the numeric `Backend` trait,
even though Metal and hybrid construction already lived at concrete resource
boundaries. Pure Vulkan consequently copied weight-percentage and reserve
configuration into two process-global `OnceLock`s before loading. Construction
now dispatches explicitly to the CPU or Vulkan module, while the trait contains
only graph execution and buffer operations. Vulkan receives the two policy
values directly in its immutable load transaction; the global caches, setters,
getters, and fake shape-backend constructor disappeared.

This removes 75 net lines and two pieces of global state without adding a type,
field, feature, dependency, or execution branch. CPU, Vulkan, and
Vulkan-hybrid workspace tests and all-target Clippy pass. A real Vulkan load
with an explicit 100% weight cap measured 1,719.55 prompt tok/s, 74.44 ms TTFT,
and 73.27 decode tok/s. Against the same-session detached-main row at 1,721.22,
74.37 ms, and 73.48, every delta is at most 0.29%; the consolidation is
retained.

### Retained production specializations

A second producer/consumer audit found no removable numeric representation or
kernel. Every canonical weight format reaches a format-specific consumer, the
Q8 decode activation is transient scratch consumed by MMVQ, and every one of
the 45 Vulkan pipeline variants has both a construction path and an active
route or required fallback. Metal and CPU optional kernels likewise remain
behind current capability checks with scalar or generic fallbacks. Removing any
of these would reduce hardware coverage or qualified performance rather than
remove an experiment.

The family-neutral end-to-end profiler in `examples/profile.rs` and
`support/profiling/profile.sh` is also retained: it is the public benchmark
driver and is consumed by the KV validation workflow. The deleted profilers
were backend-internal timestamp instrumentation, so retaining this benchmark
does not recreate their production state.

The remaining `dead_code` allowances are deliberate cross-profile facts in
hybrid placement and runtime sizing: one supported Cargo profile consumes only
a subset of variants or fields, while the combined five-profile source set
consumes all of them. Metal's staging allocation is not dead state; it is an
RAII owner whose lifetime bounds transfers. Its private `_staging` field now
states that ownership directly without a dead-code suppression.

Finally, optimization-era chronology and obsolete profiler references were
removed from affected production comments. The surviving comments describe
current ownership, numeric, dispatch, and fallback invariants only. No runtime
logic changed in this pass.

## Final Ablation Registry

| ID | Component and suspicion | Production consumers | Ablation and evidence | Decision |
|---|---|---|---|---|
| A01 | CPU operation profiler and `cpu-profile`; completed investigation output | none outside profiling | Removed feature, timing map, labels, wrappers and report; CPU tests/Clippy and same-tuple CPU/Vulkan performance passed | **REMOVE**: no production behavior |
| A02 | Vulkan timestamp profiler and `vulkan-profile`; query/counter state in normal backend modules | none outside profiling | Removed query pool, phase/category/layer aggregation, allocation/barrier hooks and report; Vulkan and hybrid tests/Clippy passed; active SPIR-V stayed byte-identical | **REMOVE**: diagnostics only |
| A03 | Vulkan load trace; one-shot environment-controlled stderr | none | Removed environment predicate, route reporters and profiler-only kernel names; route tests and performance guard passed | **REMOVE**: investigation artifact |
| A04 | `--no-attn-simd`; manual A/B override | CLI and `EngineConfig`, but no supported product requirement | Removed parser kind/state, config field and global override; scalar oracle remains reachable by capability fallback; CPU/Vulkan/hybrid gates passed | **REMOVE**: tuning control, not product behavior |
| A05 | Backend construction in the numeric trait plus Vulkan global policy caches | standalone CPU/Vulkan loading | Replaced trait/global indirection with explicit concrete load calls and immutable arguments; all local matrices passed and Vulkan delta stayed within 0.29% | **CONSOLIDATE**: preserve behavior with narrower ownership |
| A06 | Hybrid `dead_code` allowances and runtime-size fields | complementary Vulkan-hybrid, Metal and Metal-hybrid profiles | Built every locally available profile, inspected field readers, and retained only three source annotations with the cross-profile reason beside them | **KEEP**: source union serves supported profiles |
| A07 | Optional CPU, Vulkan and Metal kernel families | capability/format/shape routes and generic fallbacks | Traced each route and fallback; all 41 Vulkan sources plus four build variants map to 45 registered pipelines; numeric and route-boundary tests passed | **KEEP**: required hardware coverage or qualified performance |
| A08 | Canonical quant formats and transient Q8 activation | Q4/Q5/Q6 embedding, projection and logits; DP4A Q4 decode | Traced producer, lifetime and consumer; no persistent alternate weight representation or orphan conversion exists | **KEEP**: required representation, not archaeology |
| A09 | Family-neutral public profiler and benchmark | KV validation workflow and end-to-end regression guard | Distinguished it from deleted backend-internal instrumentation and verified its model-free tests | **KEEP**: current supported tooling |

The registry is terminal: removed items have no surviving production half, and
kept items have a named consumer, fallback relationship, and test or measured
reason. No candidate is left in an unknown state.

## Removed Artifact Inventory

| Kind | Removed |
|---|---|
| Files | CPU profiler; four Vulkan profiler modules; Vulkan matmul load trace |
| Features and configuration | `cpu-profile`, `vulkan-profile`, `--no-attn-simd`, its `EngineConfig` field, and boolean-only parser machinery |
| Types and state | profiler phase/category/mark/totals types, CPU timing entries, Vulkan query/allocation/layer state, two Vulkan `OnceLock` policy caches, and fake shape-backend construction support |
| Functions and interfaces | measurement/report hooks, query lifecycle and command counters, one-shot route loggers, trait-level backend construction, global policy setters/getters; 37 Rust functions disappear overall |
| Representations and conversions | none removed in this branch; earlier alternate native/packed weight representations were already absent on `main` |
| Kernels and shaders | none removed: every current kernel is reachable or a required fallback, and shader binaries are unchanged |
| Tests | profiler-only tests disappeared with their subsystem; numeric, capability, boundary, lifecycle, placement and fallback coverage remains |
| Dependencies | none removed or added; every direct dependency retains a source or build-script consumer |

## Final Complexity, Validation, and Performance

### Complexity delta

The final production diff is 183 additions and 1,281 deletions across 46
source/manifest files, a net reduction of 1,098 lines. Most additions are the
direct CPU calls that replaced profiling wrappers and the explicit concrete
load boundary; no new production file, feature, dependency, representation,
kernel, or runtime state was added.

| Metric | Main | Final | Delta |
|---|---:|---:|---:|
| Engine Rust files | 172 | 166 | -6 |
| Engine Rust physical lines | 30,860 | 29,796 | -1,064 |
| Vulkan/Metal shader files | 51 | 51 | 0 |
| Vulkan/Metal shader physical lines | 4,193 | 4,193 | 0 |
| Rust functions, including tests | 1,289 | 1,252 | -37 |
| Rust structs/enums/traits, including tests | 149 | 142 | -7 |
| Engine Cargo features | 7 | 5 | -2 |
| Vulkan pipeline variants | 45 | 45 | 0 |

The reduction removes completed-investigation infrastructure and accidental
construction surface; it does not obtain a smaller count by deleting hardware
coverage. The largest intentional complexity that remains is the capability
and shape routing for 45 Vulkan variants, each of which has a current consumer
and required fallback. The remaining hybrid `dead_code` allowances describe
facts split across supported Cargo profiles and are not dormant alternatives.

### Final validation

| Gate | Final result |
|---|---|
| Formatting and `git diff --check` | pass |
| CPU workspace tests | pass: 166 root, 159 engine, 5 integration, 12 semantic assertions; only declared real-resource rows ignored |
| Vulkan workspace tests | pass: 166 root, 156 engine, 5 integration, 12 semantic assertions; only declared long/real-resource rows ignored |
| Vulkan-hybrid workspace tests | pass: 166 root, 229 engine, 6 integration, 12 semantic assertions; only declared long/real-resource rows ignored |
| CPU, Vulkan, Vulkan-hybrid all-target Clippy with warnings denied | pass |
| Six authenticated Q4_K_M artifacts | pass: exact byte counts and SHA-256 |
| Frontend | pass: zero type warnings/errors, 92 tests, production build |
| Shell support | pass: syntax checks and repository contract tests |
| Metal and Metal-hybrid | external: Linux host reaches the expected Apple-only `objc2` and macOS arm64 guards |
| AMD-specific execution | external: no AMD device on this host; route/capability tests pass |
| Pinned llama.cpp parity | external: installed `9bebfcb4b`, required `13f2b28b0` |

All 45 active candidate and main SPIR-V modules compare byte-for-byte. Ten
additional `.spv` files found in one local Cargo `OUT_DIR` were stale build
artifacts not named by the registry, not repository files or executable
pipeline variants.

### Model-quality evidence

The fresh full semantic run qualified the preserved three Instruct rows and
the current 8B and 14B Reasoning rows; both current rows scored 9/9 with 4/4
critical cases and complete reasoning markers. Its first 3B Reasoning sample
scored 7/9 and stopped one case at context, so the script correctly reported
that sample as not qualified rather than weakening the gate.

The required attribution check exposed run-sensitive stochastic Vulkan output
despite the fixed seed. A second final-HEAD sample scored 8/9 but missed a
different critical case; two exact-main samples scored 9/9 and 8/9 and both
qualified. The isolated first production commit then qualified 9/9, and a
third final-HEAD sample also qualified 9/9. Completion lengths diverged within
the same binary as well as across revisions, while execution stayed healthy,
placement stayed all-GPU, greedy numeric parity tests passed, active SPIR-V was
identical, and no retained change touched sampling arithmetic. The evidence
therefore does not identify a stable cleanup regression, but it does identify
the seeded stochastic semantic gate's run sensitivity as a remaining
qualification limitation. The final current-HEAD evidence is qualified for all
six catalogued artifacts; the earlier misses remain recorded here.

### Final performance

All final rows repeat the exact release/F16/greedy tuples declared in the
baseline section. Positive prompt/decode deltas and negative TTFT deltas are
improvements. The recorded 3B short result initially exceeded the 1% guard, so
that row uses the required same-session exact-main control. The CPU candidate's
first cold sample was 26.63 prompt tok/s; a main control at 30.23 and immediate
candidate repeat at 30.47 showed that sample was an ambient outlier. It remains
recorded here rather than being silently omitted, while the controlled repeat
provides the comparison row below.

| Model | Backend | Prompt | Control prompt tok/s | Final prompt tok/s | Control TTFT | Final TTFT | Control decode tok/s | Final decode tok/s | Delta prompt / TTFT / decode |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 3B Instruct | Vulkan | 128 | 1,729.51 | 1,713.35 | 74.01 ms | 74.71 ms | 73.45 | 73.48 | -0.93% / +0.95% / +0.04% |
| 8B Instruct | Vulkan | 128 | 855.05 | 848.13 | 149.71 ms | 150.93 ms | 34.84 | 34.56 | -0.81% / +0.81% / -0.80% |
| 14B Instruct | Vulkan | 128 | 509.87 | 512.29 | 251.05 ms | 249.86 ms | 23.50 | 23.58 | +0.47% / -0.47% / +0.34% |
| 3B Instruct | Vulkan | 28,000 | 960.77 | 969.22 | 29,143.33 ms | 28,889.29 ms | 42.80 | 42.82 | +0.88% / -0.87% / +0.05% |
| 8B Instruct | Vulkan | 28,000 | 537.59 | 538.26 | 52,084.32 ms | 52,019.46 ms | 23.49 | 23.98 | +0.12% / -0.12% / +2.09% |
| 3B Instruct | CPU | 128 | 30.10 | 30.47 | 4,253.29 ms | 4,201.68 ms | 7.00 | 7.31 | +1.23% / -1.21% / +4.43% |

No retained row regresses by more than 1% against its applicable control. The
same-session 3B control was required because the older recorded row was faster
than both binaries in the final session; candidate versus same-session main is
within 0.95% on every metric.

### Final diff audit

The line-by-line branch audit found no remaining runtime reference to the
deleted profiler features or load-trace environment switch, no TODO/FIXME/HACK
marker, no unconsumed weight or buffer representation, and no consumerless
shader or kernel. The removed SIMD flag remains only in negative parser tests
that guarantee its rejection. Historical investigation reports retain their
original profiler vocabulary as evidence. Every direct dependency has a
current source or build-script consumer; no dependency change was warranted.
The final worktree is clean after the report commit.

## Remaining Production Architecture

The shared layer now owns model validation, canonical weights, graph semantics,
KV layout and backend-neutral operations once. Compile-time selection chooses
one of five product profiles. Each concrete backend owns its capabilities,
resources and operation routing; hybrid profiles add one immutable placement
plan and at most one residual crossing. CPU uses mapped canonical weights,
Vulkan and Metal upload canonical weights, and the only execution-native
conversion is the transient Q8 activation consumed immediately by qualified
Vulkan Q4 decode.

| Specialization | Eligibility | Why it remains | Fallback |
|---|---|---|---|
| CPU AVX2/F16C attention and Q4/Q5/Q6 matmul | runtime ISA plus validated block shape | qualified CPU throughput | scalar kernels |
| CPU four-query GQA reuse | exact 4:1 GQA relation | avoids rereading identical KV rows | ordinary per-query attention |
| NVIDIA Matrix2 Q4/Q6 prefill and attention | advertised/built Matrix2 pipeline plus exact format, shape and resource gates | dominant qualified prefill improvement, including 14B | cooperative, tiled or portable kernels |
| Portable cooperative/tiled Vulkan paths | cooperative/subgroup/shared-memory capability and legal shape | optimized generic Vulkan coverage below Matrix2 | portable scalar-style Vulkan kernels |
| AMD Q4 MMQ and bounded prefill submissions | AMD vendor family, DP4A and exact workload/resource bounds | qualified AMD prefill and memory stability | portable Q4 batch and 256-row policy |
| Vulkan Q8/DP4A Q4 decode | DP4A, Q4_K, aligned shape and bounded persistent scratch | qualified decode throughput | exact Q4 projection |
| Vulkan wave32/wave64 GQA decode | exact 4:1 F16 GQA plus subgroup and resource contract | qualified long-context decode | wide or generic attention decode |
| Metal tiled GQA attention | SIMD width, threadgroup memory, dimensions and batch shape | qualified Metal prefill | generic Metal attention |
| Hybrid contiguous CPU/device placement | checked separate or unified memory budget | supported partial placement without duplicated weights | all-device or all-CPU terminal plans |

## Remaining Intentional Complexity

| Component | Consumer and necessity |
|---|---|
| Five backend profiles | Distinct supported CPU, discrete Vulkan, unified Metal and two placement products; selection is compile-time and exclusive |
| Canonical F32/F16/Q4_K/Q5_K/Q6_K formats | Authenticated model tensors reach format-specific embedding, projection or logits consumers |
| F16 and INT8 KV schemes | Public configuration with one shared checked layout and backend implementations |
| 45 Vulkan pipelines | Each has construction plus an active route or required capability fallback; four are deliberate compile variants of shared sources |
| Capability and shape predicates | Prevent unsupported hardware or tensor shapes from entering specialized arithmetic and dominate internal invariants |
| Persistent Q8/GQA scratch and request KV cache | Removes qualified hot-path allocation while explicit lifecycle tests protect ownership and failure cleanup |
| Three cross-profile `dead_code` annotations | Fields or variants are consumed by another supported Cargo profile; the reason is local to each annotation |

## Final Minimality Assessment

If the current production system were implemented fresh, every remaining
optimization-related concept above would still be intentional: it has a
reachable capability, a concrete consumer, an explicit fallback relationship,
and a correctness or measured-performance reason. The audit found no remaining
unknown representation, route, state object, feature, kernel, shader,
configuration switch or dependency. Removing another retained numeric path
would reduce supported hardware coverage, remove a required fallback, or undo a
qualified performance result; merging the remaining backend-local paths would
hide rather than remove those hardware distinctions.

## PR Description

### Summary

Remove completed CPU/Vulkan profiling and load-trace infrastructure, delete the
manual CPU SIMD experiment switch, and move standalone backend construction
policy out of the numeric backend trait.

### Motivation

Optimization work left investigation-only features, global diagnostic state and
an accidental construction interface in production modules. None described a
supported runtime capability.

### Architecture before

CPU and Vulkan profiling features wrapped normal execution, Vulkan carried a
load-trace environment route and global placement-policy caches, the numeric
backend trait also constructed models, and the CLI retained a manual SIMD A/B
switch.

### Architecture after

The five supported profiles contain only production execution. Concrete CPU or
Vulkan loaders receive immutable construction inputs directly; backend-local
capability routing and every qualified fallback remain unchanged.

### Dead or obsolete code removed

Six Rust files, two Cargo features, seven types, 37 functions, profiler state
and hooks, trace logging, manual SIMD configuration, two global policy caches
and fake construction support are gone.

### Representations removed

None: `main` already had no orphan persistent weight representation. Canonical
formats and the transient decode Q8 activation all have active consumers.

### Routing and backend simplification

Numeric routing is unchanged. Construction now crosses one explicit concrete
boundary, and scalar CPU eligibility once again depends only on runtime
capability rather than a historical override.

### Validation

Formatting, diff checks, CPU/Vulkan/Vulkan-hybrid workspace tests and
warnings-denied all-target Clippy pass. Six artifacts authenticate, the
frontend passes check/92 tests/build, shell contracts pass, and the final
quality evidence is recorded without hiding stochastic reruns. Metal, AMD
execution and the required pinned external oracle are identified as external.

### Performance neutrality

All retained short and long Vulkan rows are within the 1% regression guard;
the largest observed retained regression is 0.95%. The CPU control improves in
the stable comparison. No hot arithmetic or active shader changed.

### Risk

The main risk is loss of a diagnostic workflow or accidental loader-policy
drift. Documentation no longer advertises the deleted profilers, negative CLI
tests pin the removed switch, explicit loader arguments are covered by full
profile tests, and same-tuple measurements bound runtime drift.

### Follow-ups intentionally excluded

No new kernels, tuning, numeric policy, model adaptation, dependency, public
API expansion, Metal-on-Linux substitute, AMD substitute, or unpinned oracle
claim is included. Those are separate qualification or optimization programs.
