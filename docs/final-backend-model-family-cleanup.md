<!--
This report owns the final backend/model-family cleanup evidence: baseline
identity, architecture and necessity inventories, cleanup decisions, validation,
performance comparison, and cross-hardware qualification. Historical campaign
detail remains in the backend-specific reports.
-->

# Final Backend and Model-Family Cleanup

## Baseline

```text
CLEANUP_BASELINE_SHA = 6e514c1eaba02b8dcb6d7202bfd13fcee0317f36

branch: cleanup/final-backend-model-family
HEAD: 6e514c1eaba02b8dcb6d7202bfd13fcee0317f36
merge-base with main: 6e514c1eaba02b8dcb6d7202bfd13fcee0317f36
worktree status: clean
```

At mission start, local `main`, `origin/main`, and `HEAD` were identical. The
three latest integration merges were NVIDIA long-context prefill (`8884465`),
AMD long-context prefill (`8e2f8cc`), and Metal long-context prefill
(`6e514c1`). The earlier NVIDIA, AMD, and Metal decode and specialization work
is also in this ancestry. This is the one authoritative cleanup branch; it will
not be pushed without explicit authorization.

The cleanup host is Linux x86_64 with an Intel i5-9600K and NVIDIA RTX 3060
12 GiB, driver 595.84, Vulkan 1.4.329. AMD and Apple Metal execution are
external qualification requirements for the final candidate.

### Exact baseline performance

These measurements were taken from the exact baseline SHA before any source
edit. All rows use release mode, F16 KV, greedy sampling, 32 requested tokens,
one warm-up, and three measured repetitions. The prompt is `N - 3` copies of
`" a"`, which produces exactly `N` Ministral chat tokens. Vulkan rows use
hybrid all-GPU placement with 100% weights and context 4096.

| Model | Backend | Prompt | TTFT mean | Prompt tok/s | Model decode tok/s | TTFT CV |
|---|---|---:|---:|---:|---:|---:|
| 3B Instruct | Vulkan-hybrid | 128 | 80.48 ms | 1590.55 | 75.35 | 0.73% |
| 8B Instruct | Vulkan-hybrid | 128 | 157.52 ms | 812.66 | 35.48 | 0.94% |
| 14B Instruct | Vulkan-hybrid | 128 | 252.40 ms | 507.13 | 24.37 | 0.56% |
| 8B Instruct | Vulkan-hybrid | 2048 | 2167.76 ms | 944.76 | 34.85 | 0.15% |
| 3B Instruct | CPU | 128 | 5151.13 ms | 24.85 | 7.38 | 0.53% |

The final comparison must repeat these exact tuples in the same session when
possible. Backend-specific reports retain the longer 15K/28K reference rows;
the 2K row here cheaply exercises context-dependent attention and decode on the
cleanup host.

## Initial Complexity

These are descriptive physical-source counts, not deletion targets.

| Metric | Baseline |
|---|---:|
| Engine Rust files | 171 |
| Engine Rust physical lines | 30,831 |
| Shader files | 54 (44 Vulkan, 10 Metal) |
| Shader physical lines | 4,664 |
| Rust structs / enums / traits | 109 / 28 / 7 |
| Rust functions, including tests | 1,279 |
| Rust constants | 117 |
| Engine backend features | 5 |
| Vulkan pipeline variants | 48 |
| Metal pipeline variants | 17 in the full Metal profile |

The five mutually exclusive backend features are `cpu`, `vulkan`,
`vulkan-hybrid`, `metal`, and `metal-hybrid`. No diagnostic or experimental
Cargo feature remains.

## Initial Architecture Inventory

### Model and family

| Area | Owner | Classification | Production reason |
|---|---|---|---|
| File opening and family entry | `family/mod.rs` | essential | Normalizes untrusted file failures and enters the sole supported family. |
| Architecture/profile detection | `family/mistral/detect.rs` | model-family semantics | Requires `mistral3` and the Q4_K_M GGUF profile before allocation. |
| Canonical semantic metadata | `family/mistral/config.rs` | model-family semantics | Validates and stores dimensions, heads, GQA, RoPE/YaRN, vocabulary, and special IDs once. |
| Release policy | `family/mistral/version.rs` | compatibility | Owns default context and the exact Reasoning system prompt; reference size rows are test-only. |
| Tensor mapping | `family/mistral/tensors.rs` | model-family semantics | Maps exact global/layer names, validates shapes/formats, and preserves tied versus dedicated output. |
| Tokenizer/chat profile | `family/mistral/tokenizer/`, `template.rs`, `decode.rs` | model-family semantics | Implements Tekken/BPE, Instruct/Reasoning classification, safe template construction, and UTF-8 output. |
| Dense graph | `family/mistral/graph/` | shared infrastructure | Records one common block/MLP/tail operation order for decode and bounded prefill. |
| Model owner | `family/mistral/mod.rs` | shared infrastructure | Binds validated family state to the statically selected runtime and immutable memory report. |

Model identity is established only by the `general.architecture = mistral3`
contract and chat-profile metadata. Execution backends do not inspect artifact
filenames, parameter-count strings, model names, or product names.

### Supported family artifacts

All six catalogued local files match the byte counts and SHA-256 identities in
`support/models.tsv`. Instruct and Reasoning share tensor semantics; the chat
profile changes template/default-sampling behavior. The current release record
still marks the historical 8B Reasoning campaign unsupported, so cleanup must
not silently broaden that claim.

| Scale/profile | Blocks | Hidden | FFN | Q heads | KV heads | GQA | Head dim | Q/K/V widths | Output |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| 3B Instruct/Reasoning | 26 | 3072 | 9216 | 32 | 8 | 4:1 | 128 | 4096/1024/1024 | tied embedding |
| 8B Instruct/Reasoning | 34 | 4096 | 14336 | 32 | 8 | 4:1 | 128 | 4096/1024/1024 | dedicated |
| 14B Instruct/Reasoning | 40 | 5120 | 16384 | 32 | 8 | 4:1 | 128 | 4096/1024/1024 | dedicated |

Common authenticated metadata is context 262,144, vocabulary 131,072, BOS 1,
EOS 2, full 128-dimension adjacent-pair YaRN RoPE, and Q4_K_M file type. The
artifacts contain F32 auxiliary tensors and Q4_K/Q6_K matrices. Q5_K remains a
required compatible internal tensor format with CPU/Vulkan/Metal consumers;
Q8_0 is parser-visible only so the public profile gate can reject it precisely.
KV supports F16 and per-vector INT8 on every backend contract.

### Runtime and request lifecycle

| Area | Owner | Classification | Production reason |
|---|---|---|---|
| Public engine/config | `api/engine.rs` | essential | Owns five explicit settings, family load, public memory/placement, and request entry points. |
| Request events/cancellation | `api/event.rs`, `api/request.rs` | essential | Provides bounded event, sampling, cancellation, and timing contracts. |
| Homogeneous session | `runtime/homogeneous.rs` | shared infrastructure | Owns one-backend KV/session lifecycle and serialized reusable KV state. |
| Partitioned session | `runtime/partitioned/` | required fallback | Owns immutable CPU-prefix/device-suffix execution and exactly one residual crossing. |
| Generation | `family/mistral/generation/` | model-family semantics | Renders, tokenizes, prefill/decode loops, samples, emits, and conditionally reuses an identical keyed prefix. |
| Sampling | `sampling.rs` | essential | Implements the public greedy and qualified stochastic policy. |
| KV layout | `kv_cache/` | shared infrastructure | Owns checked allocation, token-major payload/metadata layout, append, and exact-once free. |

Requests are serialized by the application runtime. A normal request owns a
fresh KV cache. Pure Vulkan/Metal may transfer it into one model cache keyed by
an opaque caller ID; an identical token prefix reuses it, otherwise reset
semantics produce fresh tail logits. CPU and hybrid builds intentionally use
the ordinary fresh-session path.

### Backend selection

```text
compile-time feature (exactly one)
  -> backend/selection.rs
  -> CPU | pure Vulkan | Vulkan hybrid | pure Metal | Metal hybrid
  -> pure device availability is an explicit error
  -> hybrid probes the device before placement
  -> immutable capability and memory plan
  -> generic implementation plus eligible specialization
  -> operation-local fallback
```

`backend/contract.rs` is the one graph/backend language boundary. It contains
only buffer, encoder, graph operation, attention, KV, readback, and lifecycle
methods. `backend/source.rs` exposes one canonical grouped tensor inventory and
one contiguous `WeightSelection` for hybrid ownership.

### Backend implementations

| Backend area | Classification | Required behavior |
|---|---|---|
| CPU scalar | required fallback | Portable F16/Q4_K/Q5_K/Q6_K projection, attention, RoPE, KV, reductions, and exact reference behavior. |
| CPU AVX2/F16C | qualified optimization | Automatic single-token/batched matmul and 4:1 GQA attention paths; scalar remains valid. |
| Generic Vulkan | required fallback | Base 256-thread pipelines, canonical uploaded weights, F16/INT8 KV, portable matmul and attention. |
| Portable optimized Vulkan | qualified optimization | Tiled/wide/cooperative attention and matmul selected by immutable limits and shape. |
| NVIDIA Vulkan | hardware specialization | Matrix2 Q4/Q6 prefill, Q64 attention, larger prefill batches, DP4A Q4 decode where available. |
| AMD Vulkan | hardware specialization | DP4A Q4 MMQ, bounded submissions, split-history GQA prefill, required-wave32 Q6/GQA decode. |
| Generic Metal | required fallback | Canonical weights, serial/segmented attention, per-row projection, F16/INT8 KV, and reductions. |
| Optimized Metal | qualified optimization | Cooperative quantized projection and tiled/parallel attention. |
| Apple-family Metal | hardware specialization | Apple9/unified-memory admission and Metal4 tensor projection when available. |
| Vulkan/Metal hybrid | required fallback | Checked all-device, mixed, and CPU-only placement with no post-load retry. |

### Weight representations

| Representation | Creator/owner | Consumer/selection | Lifetime and fallback |
|---|---|---|---|
| Memory-mapped canonical GGUF | `GgufFile` / model load | Tensor contract and all loaders | Model load; invalid spans fail before backend allocation. |
| CPU canonical buffers | CPU weight loader / `CpuBackend` | CPU embedding, matmul, logits, norms | Model lifetime; quantized bytes stay canonical, F32 auxiliaries narrow to F16. |
| Vulkan canonical uploads | Vulkan weight loader / `VulkanBackend` | Format-specific pipelines | Model lifetime; no alternate persistent weight packing. |
| Metal canonical uploads | Metal weight loader / `MetalBackend` | Format-specific Metal functions | Model lifetime; no alternate persistent weight packing. |
| Vulkan Q8 activation scratch | Vulkan runtime buffers | DP4A Q4 decode/MMQ and GQA partial storage | Persistent scratch reused per dispatch; exact float Q4/generic attention fallback. |
| Prefill row views | graph prefill buffers | Batched graph operations | Borrowed aliases into persistent scratch; no duplicate ownership. |
| Hybrid selected range | immutable placement plan | CPU prefix and device suffix loaders | Model lifetime; tied output identity and contiguous layers preserved. |

There is no native-FP16 predecoded weight state, transformed-weight cache, or
temporary conversion path without a current consumer.

## Production Necessity Graph

```text
Q4_K_M model load
  -> mistral3/profile detection
  -> canonical config + exact tensor map
  -> grouped WeightSource / optional contiguous WeightSelection
  -> CPU mapped buffer or Vulkan/Metal canonical upload
  -> format-specific embedding/projection/logits
  -> scalar/generic fallback
  -> loader, shape, format, and real-artifact tests

Q4_K prefill
  -> packed graph rows
  -> Backend::matmul_batched
  -> format + dimensions + registered capability
  -> NVIDIA Matrix2 | cooperative | AMD MMQ | portable batch
  -> portable Q4 batch fallback
  -> CPU/Vulkan numeric oracles + 3B/8B/14B quality/performance guards

Q6_K prefill/logits
  -> Q6 tensor mapping
  -> Backend::matmul_batched / logits
  -> Matrix2 | cooperative | portable Q6; AMD wave32 for decode/logits
  -> portable Q6 fallback
  -> Q6 numeric oracles + model-family teacher rows

GQA long-context attention
  -> canonical 32 query / 8 KV head metadata
  -> independent K/V KV layout
  -> decode or prefill routing by datatype, head_dim, ratio, rows, context
  -> NVIDIA Q64 / AMD split-history / wave32 / wave64 / Metal grouped path
  -> tiled, wide, segmented, serial, or portable exact fallback
  -> routing boundaries + numeric + long-context quality/performance guards

F16 or INT8 KV
  -> EngineConfig::kv_quant
  -> backend-independent payload/metadata offsets
  -> scheme-specific KV write
  -> scheme-specific attention route
  -> generic backend fallback of the same scheme
  -> layout, bit-parity, endpoint, and model validation

Hybrid execution
  -> explicit percentage/reserve plus authenticated model dimensions
  -> checked weight/runtime byte accounting
  -> all-device | mixed | CPU-only immutable plan
  -> selected global/layer ownership
  -> zero or one residual crossing per pass
  -> placement, exact-fit, overflow, endpoint, and parity tests

Keyed request reuse
  -> public opaque cache key
  -> identical rendered token prefix
  -> serialized pure-GPU model cache
  -> one-token prefill of the changed tail
  -> ordinary fresh KV on miss/non-pure backend
  -> sequential-request and prefix-invalidation tests
```

## Capability-State Audit

### Vulkan

| Fact | Established | Consumer | Necessity |
|---|---|---|---|
| `vendor_id` | physical-device properties | AMD/NVIDIA family gates and row policy | Isolates measured family behavior without product-name routing. |
| memory properties/budget support | bootstrap | allocations and hybrid available-memory plan | Required safe ownership/capacity. |
| `coopmat` shape | cooperative-matrix query | optional Q4/Q6 cooperative pipelines | Exact capability and shape gate. |
| `coopmat2` plus Q64 WG128 | NVIDIA extension query | Matrix2 matmul/attention registry and hybrid rows | Qualified NVIDIA route; absence falls back. |
| `dp4a` | Vulkan 1.3 feature query | Q4 MMVQ/MMQ pipeline registration | Qualified integer-dot path; absence uses float Q4. |
| subgroup size/operations | Vulkan 1.1 properties | wide/tiled/GQA/Q4 metadata pipelines | Required shader legality. |
| `wave32_control` | subgroup-size-control query | AMD wave32 pipelines | Required explicit subgroup contract. |
| workgroup/shared/push limits | device limits | optional pipeline registry | Prevents illegal pipeline creation. |
| storage-buffer offset alignment | device limits | graph prefill view checks | Prevents undefined bindings. |
| barrier-elision flag | backend command recorder | independent Q/K/V and gate/up runs | Retained synchronization optimization; default is correctness-safe. |

Each `PipelineCaps` field is destructured exactly once by registry construction;
each resulting optional pipeline is queried by an operation route. No orphan
capability field was found.

### Metal

| Fact | Established | Consumer | Necessity |
|---|---|---|---|
| unified memory + Apple9 family | device admission | pure/hybrid Metal availability | Defines the qualified Apple-family floor. |
| thread/threadgroup limits | device and pipeline load | wide, grouped, matrix routes | Prevents unsupported dispatch geometry. |
| per-pipeline execution width/memory | compiled pipeline state | operation-local eligibility | Preserves generic fallback when an optimized shape is unavailable. |
| Metal4 family | device query | tensor batched matmul pipeline | Optional Apple specialization; wide/row fallback remains. |
| recommended/current allocation | device query | unified-memory placement | Required checked plan and reserve behavior. |

CPU uses runtime AVX2/F16C detection at the kernel boundary; unsupported hosts
remain on scalar code. There is no user override or stale ISA flag.

## Representative Routing Table

| Operation/tuple | Expected and actual route | Fallback |
|---|---|---|
| CPU Q4/Q6 decode, AVX2/F16C | single-token SIMD | scalar per-format matmul |
| CPU 4:1 GQA | four-query reuse | pair/serial attention |
| Vulkan Q4 decode + DP4A + scratch | Q8 activation then MMVQ | float Q4 tiled GEMV |
| Vulkan AMD Q4 prefill | DP4A MMQ within row/scratch bounds | portable Q4 batch |
| Vulkan NVIDIA qualified Q4/Q6 dimensions | Matrix2 prefill | cooperative then portable batch |
| Vulkan F16 prefill, 4:1 GQA, complete 64-row NVIDIA tiles | Matrix2 Q64 | Matrix2 Q32, cooperative-QK, tiled, wide, portable |
| Vulkan F16 prefill, AMD wave32, rows <=64, end >=2K | split-history GQA | Matrix2/tiled/wide/portable |
| Vulkan F16 4:1 GQA decode | wave32, required-wave32, or exact wave64 by capability | wide/1024/generic decode |
| Vulkan INT8 decode | GQA INT8 split when eligible | generic INT8 decode |
| Metal quantized prefill rows 2..64, unmixed | cooperative batched, Metal4 tensor at >32 | wide or per-row projection |
| Metal F16 4:1 GQA prefill, rows 32/64 | tiled/matrix qualified routes | segmented or serial attention |
| Metal F16 4:1 GQA decode at >=1K | split/reduce grouped route in pure Metal | parallel/segmented/generic attention |
| Any mixed Metal suffix | stable mixed-placement route | CPU prefix plus generic Metal suffix |
| Unsupported shape/datatype/capability | never enters specialization | same-operation generic implementation or explicit unsupported-format error |

The exact dimension allow-list in Vulkan Matrix2 policy is a
shape/quantization qualification boundary, not model-name routing. It covers
the currently supported 3B/8B/14B projection shapes and deliberately sends
unmeasured dimensions to the portable kernels.

## Resource Ownership

| Resource | Creator and owner | Lifetime/reset/destruction | Consumer/fallback |
|---|---|---|---|
| Weights | backend loader -> concrete backend | Model lifetime; transactional load and backend drop | Graph operations; hybrid selected owner only. |
| Scratch/logits/reduction | concrete backend load | Model lifetime; views borrow, owners destroy once | Every request; allocation failure rolls back. |
| KV cache | homogeneous/partitioned session | Request lifetime or serialized pure-GPU cache; resize means a new validated model context | KV write and attention; miss creates fresh state. |
| Vulkan device/registry | init and pipeline build | Model lifetime; device idle, pipeline/layout/cache destruction before device | Vulkan operations; missing optional pipeline falls back. |
| Vulkan command buffer | `begin` encoder | Per submission; submit/wait then free | Recorded graph; failures are explicit. |
| Metal device/registry | device/pipeline load | Model lifetime; retained Objective-C resources | Metal operations; unsupported optimized route uses generic pipeline. |
| Metal encoder | command queue | Per submission; completion awaited before readback/reuse | Recorded graph. |
| Upload staging | Vulkan temporary allocation / Metal retained staging owner | Upload scope or backend lifetime as required by API ownership | Loader/crossing; no compute representation. |
| Hybrid crossing | partitioned session | Persistent bounded buffer, one copy per mixed pass | Only CPU/device boundary; all-device/CPU-only have zero crossings. |

No duplicate owner or cache layer without a reachable consumer was found in
the initial pass.

## Kernel and Shader Reachability

The Vulkan registry contains 48 variants: 26 mandatory base pipelines and 22
capability-gated variants. Repository-wide reference counts show every enum
variant has a specification, a construction path, and at least one operation
consumer. The groups are:

| Vulkan family | Selection | Fallback/test evidence |
|---|---|---|
| F16/Q4/Q5/Q6 matmul, embedding, logits | format dispatch | CPU numeric oracle and format-route tests |
| RMSNorm, RoPE, residual, SiLU | graph operation | CPU/reference graph shape tests |
| F16/INT8 KV write | KV scheme | layout and bit-parity tests |
| F16/INT8 decode attention | scheme, limits, GQA capability | generic scheme kernel and attention oracles |
| portable/wide/tiled/cooperative/Matrix2 prefill attention | limits, head shape, rows, context | ordered route-boundary tests and generic portable kernel |
| Q4/Q6 batch/cooperative/Matrix2 matmul | format, registered capability, exact qualified dimensions | portable batch plus numeric/model oracles |
| Q8 activation + Q4 MMVQ/MMQ | DP4A, format, scratch, vendor for MMQ | float Q4 path and numeric oracle |
| argmax/top-k | sampling request | CPU comparison and reduction tests |

Metal embeds ten source files and resolves 17 full-profile entry points. Every
entry appears in the fixed registry and operation dispatch. Metal4 tensor
matmul is the sole optionally omitted entry; the wide/standard matmul fallback
remains. The grouped decode and matrix prefill entries are pure-Metal
specializations; hybrid keeps its established generic routes. CPU kernels are
selected by operation, format, row count, and ISA, with scalar variants retained
for every supported operation.

No consumerless shader, obsolete tile variant, or unreachable pipeline was
proven in this pass. Hardware-specific static reachability is retained even
when the cleanup host cannot execute it.

## Model-Specific Conditional Audit

| Conditional class | Finding | Decision |
|---|---|---|
| Architecture/profile string | `mistral3`, Tekken, and exact Reasoning metadata are true family semantics. | retain |
| Artifact filenames | Present only in validation catalog/scripts, never runtime routing. | retain as evidence |
| Parameter labels (`3B/8B/14B`) | Tests/docs only. | retain as coverage labels |
| Exact projection dimensions | Vulkan Matrix2 qualification policy. | retain as shape-derived measured eligibility |
| `block_count >= 32` plus long context | AMD submission-duration guard. | retain as graph-depth/context safety policy, not a model-name check |
| Head dim 128 and GQA 4:1 | Vulkan/Metal attention eligibility. | retain as structural shape contract |
| Vendor IDs | AMD/NVIDIA architecture-family capability isolation. | retain; no product/device IDs |
| Apple family values | Metal capability admission and Metal4 availability. | retain; no product names |

Family identity is canonical in `MistralConfig` and the tokenizer chat profile.
Shape decisions consume that state through graph arguments; backend routing
begins only in concrete backend modules.

## Investigation and Dead-Code Audit

Initial warning-denied Clippy passes for CPU, Vulkan, and Vulkan-hybrid. A
repository-wide suspicious-pattern review found no production environment
switch, profiling counter, candidate-selection state, alternate weight
representation, or diagnostic buffer. Test-only failure injection/probe counts
remain valid lifecycle and fallback coverage.

Two cleanup candidates are proven before production editing:

1. `examples/decode.rs` is a steady-state campaign benchmark referenced only by
   the historical AMD decode report. The current public benchmark contract says
   `examples/bench.rs` is the sole iterative executable. Removing it changes no
   library/API/backend route and deletes its benchmark-only retry/statistics
   machinery.
2. Several comments still cite obsolete investigation labels (`m1`, `m2`,
   `m3`, `I1`, `I2`, `I3`, `D4`, `D6`, `D8`, `P10`) or a rejected Vulkan
   transfer-queue/`streaming.rs` design that is absent from the current `Device`.
   The associated invariants are real; the historical labels and nonexistent
   ownership description are not. Rewrite them as direct present-tense
   invariants without changing code.
3. The public throughput report still exposes first/last-half decode rates and
   the harness still computes them, although the maintained benchmark and its
   documented output use the later begin/middle/end thirds plus interval
   dispersion. Repository-wide consumer searches find the half fields only at
   their definitions and aggregation sites. History confirms the thirds
   superseded the half measurements. Remove this accidental benchmark-research
   surface rather than preserving two segment schemes; no engine request,
   backend, model, or documented benchmark contract changes.

The smallest first change is deletion of the one unconsumed example and repair
of stale comments in existing files. Main risk: accidentally removing a tool
used by a current validation script; repository-wide consumer searches show
none, and all-target builds plus support-script searches will recheck it.

### Approved initial structure

No new production file or abstraction is warranted. The first cleanup changes
only existing files and removes one:

```text
examples/
  decode.rs                         (remove; 199 physical lines)
crates/graph_horizon_engine/src/
  backend/cpu/...                   (comments only; no productive-line growth)
  backend/vulkan/...                (comments only; no productive-line growth)
  kv_cache/...                      (comments only; no productive-line growth)
  harness/throughput.rs             (~130 productive lines before, reduced)
  harness/stats.rs                  (~120 productive lines before, reduced)
src/app/engine/config.rs            (comment only; no productive-line growth)
docs/
  final-backend-model-family-cleanup.md
```

Further production edits require a separately recorded finding and the same
necessity/consumer proof. No orchestration file is planned to exceed 200
productive lines as a result of this change; category-K/I files retain their
declared exemptions.

## Cleanup Decisions

The accepted cleanup keeps every runtime route, capability gate, fallback,
weight representation, resource owner, kernel, and shader unchanged. It makes
three bounded reductions:

1. Removed the 199-line campaign-only `examples/decode.rs`. No support script,
   test, current benchmark document, or production caller consumed it. The AMD
   campaign report now marks its invocation as historical and points to the
   maintained `examples/bench.rs` interface.
2. Removed the superseded first/last-half decode calculation: two public
   `ThroughputReport` fields, two internal sample fields, their aggregation and
   arithmetic, and one obsolete unit test. This is the only public API change;
   it deliberately removes accidental benchmark-research surface with no
   repository consumer or documented output contract. The maintained thirds,
   interval dispersion, prompt/decode metrics, and benchmark output are intact.
3. Replaced stale campaign identifiers and nonexistent Vulkan transfer-queue
   ownership prose with direct present-tense invariants. This is comment-only
   outside the two removals above.

The main invariants remain: one compile-time backend, one canonical model
configuration, immutable capability-based routing, exact Q4_K_M tensor identity,
operation-local fallback, and exact-once resource release. The change reduces
conceptual complexity: one executable and one redundant measurement scheme are
gone, and no abstraction, dependency, resource state, dispatch branch, or
production file was added.

After these edits, formatting and warning-denied Clippy pass for CPU, Vulkan,
and Vulkan-hybrid. Workspace all-target tests pass in all three profiles; the
Vulkan runs execute the available RTX 3060 numeric/lifecycle kernel coverage.
Real-artifact, quality, performance, and cross-hardware qualification are
recorded separately below after the candidate is frozen.

## Final Complexity and Validation

Pending cleanup implementation, final same-session performance comparison,
line-by-line diff audit, and identical-candidate hardware qualification.

## Physical Qualification

| Backend | Candidate SHA | Device | Status |
|---|---|---|---|
| AMD Vulkan | pending | qualified AMD device required | pending external run |
| NVIDIA Vulkan | pending | RTX 3060 available locally | pending final candidate |
| Apple Metal | pending | Apple M4/macOS host required | pending external run |

Global completion remains prohibited until all three materially supported
physical backends pass on the identical final candidate SHA, or the project
explicitly accepts a documented external blocker.
