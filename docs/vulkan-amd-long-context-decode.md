<!--
This report preserves the AMD Vulkan long-context decode measurements,
candidate decisions, qualification, and later integration. It does not define
the current backend support or release contract.
-->

# AMD Vulkan long-context decode checkpoint

This report is the persistent checkpoint for the 2026-08-21 AMD Vulkan
long-context investigation. It distinguishes model tokens measured by the
engine from public text deltas, which do not have a one-to-one relationship.

## State and artifacts

- Baseline: `87e3e4b1bb17e811caac030f69261bd0498284ec` (`main`).
- Implementation and retained harness HEAD before this report: `fafabd0`.
- Branch: `perf/amd-long-context-decode`.
- Later integration: branch tip `8158d96`; merged by PR #37 (`e3d23f2`).
- Device: AMD Radeon RX 6750 XT, Navi 22 / RDNA2 / `gfx1031`, 40 CUs.
- Driver: RADV, Mesa 26.0.3, Vulkan 1.4.335.
- Memory: 12,868,124,672 device-local bytes reported by Vulkan.
- Native wave: wave32. Vulkan reports subgroup 64 and supports required
  subgroup-size control, so the retained AMD pipelines request wave32.
- Other relevant capabilities: integer dot product and FP16 available, 64 KiB
  LDS, 3 MiB L2, 96 MiB Infinity Cache, no Vulkan cooperative matrix.
- Primary artifact: `Ministral-3-3B-Instruct-2512-Q4_K_M.gguf`.
- Artifact size: 2,147,023,008 bytes.
- SHA-256:
  `9ed150d4367e68df0ac8e1540f6ddc65b42d0ee26378329d1ecbca60f93fc5f8`.
- Shape: 26 blocks, hidden 3072, Q 4096, K/V 1024, FFN 9216,
  32 query heads, 8 KV heads, head dimension 128, GQA 4:1, vocabulary 131072.
- Weights: Q4_K_M profile with 156 Q4_K and 27 Q6_K tensors.
- Primary context allocation: 32,768 tokens. The tested cache depths are exact
  positions inside the same allocation and do not change placement.
- Evidence logs are in the ignored local directory
  `target/amd-long-decode-20260821/`.

The 3B artifact is the primary control because every 128/2K/8K/15K/28K row is
fully resident on Vulkan with one placement. The available 14B artifact at a
32K F16 context instead selects a mixed plan (33 GPU, 7 CPU layers), so it is a
capacity regression case rather than a comparable all-GPU control.

## Reproduction commands

The historical exact-depth harness was built and run with:

```bash
cargo build --release --features vulkan --example decode
target/release/examples/decode MODEL 15434 32768 int8 32 5
target/release/examples/decode MODEL 15434 32768 f16 32 5
```

The campaign-only `decode` executable was removed by the final production
cleanup after these measurements were recorded. This command remains historical
evidence; `examples/bench.rs` is the maintained benchmark interface.

Arguments are `model depth context f16|int8 completion-tokens repetitions`.
The harness primes one caller-keyed prefix, requires exact prefix reuse, rejects
short completions, and reports engine model-token time independently from wall
time. The exact synthetic prompt is repeated `" a"`; the Ministral template
adds the remaining three tokens.

The real-generation command uses the public event-stream benchmark:

```bash
cargo build --release --features vulkan --example bench
target/release/examples/bench MODEL --context 32768 --kv int8 \
  --prompt "<15431 repetitions of ' a'>" --max-tokens 256 --warmup 0 --reps 1
```

Correctness and complete local qualification:

```bash
cargo fmt --all -- --check
cargo test --workspace --all-targets --features vulkan-hybrid
cargo test -p graph_horizon_engine --release --features vulkan-hybrid \
  vulkan_prefill_attention_long_context_numeric_qualification -- --ignored --nocapture
```

## Symptom reproduction and root cause

The cited 72.3 prompt tok/s and 4.5 text deltas/s did not identify a model, KV
format, or placement and therefore cannot be treated as a qualified tuple.
Fresh local results isolate the matching failure mode:

| tuple | prompt/decode result | conclusion |
|---|---:|---|
| 3B, F16, all GPU, KV=15,434 | 16.019 ms/model-token, 62.43 tok/s | healthy retained control |
| 3B, INT8, all GPU, KV=15,434 | 151.115 ms/model-token, 6.62 tok/s | reproduced collapse |
| 14B, F16, mixed 33/7, KV=15,434 | 10.10 prompt tok/s, 6.19 text deltas/s | different, much slower prefill tuple |

The old INT8 decode shader launched one workgroup per query head. With 4:1 GQA,
four workgroups independently loaded and dequantized the same K/V vector. Its
linear scaling coefficient was 9.011 microseconds per cached token, versus
0.272 for the exact-reuse F16 path. This was a routing/architecture defect, not
memory migration, CPU fallback, sampling, or a general Vulkan launch problem.

The retained INT8 split kernel assigns one history split to one KV head, loads
each packed K/V word and metadata pair once, reuses it for four query heads,
and feeds the existing FP32 online-softmax reducer. It is built only when the
device can execute the required subgroup/resource shape. Eligibility also
requires INT8, head dimension 128, exact 4:1 GQA, and sufficient existing
scratch. The portable per-query-head shader remains the fallback.

## Final decode matrices

### Retained INT8 specialization

These are strict same-source A/B binaries. The fallback binary differs only by
disabling the new eligibility branch. CV is below 0.34% for every completed row.

| KV | baseline tok/s | final tok/s | baseline ms/token | final ms/token | latency delta |
|---:|---:|---:|---:|---:|---:|
| 128 | 76.52 | 83.64 | 13.069 | 11.956 | -8.51% |
| 2,048 | 32.63 | 80.52 | 30.650 | 12.419 | -59.48% |
| 8,192 | 11.65 | 71.85 | 85.819 | 13.919 | -83.78% |
| 15,434 | 6.62 | 63.87 | 151.115 | 15.656 | -89.64% |
| 28,000 | reset during prime | not engine-measurable | reset | not engine-measurable | see boundary below |

At 28K the unchanged portable INT8 *prefill* command buffer triggers a RADV
guilty-context hard recovery before the harness can report its prime. The
direct long-context numeric test independently executes and validates the new
INT8 decode kernel at bases 27,936, 27,968 and 27,984. This is an existing
prefill/watchdog boundary, not a decode-kernel correctness failure; it is not
represented as a fabricated throughput number.

### Unaffected mature F16 path

| KV | baseline tok/s | final tok/s | baseline ms/token | final ms/token | latency delta |
|---:|---:|---:|---:|---:|---:|
| 128 | 83.60 | 83.99 | 11.963 | 11.906 | -0.47% |
| 2,048 | 79.56 | 80.27 | 12.569 | 12.458 | -0.88% |
| 8,192 | 71.02 | 71.01 | 14.081 | 14.083 | +0.01% |
| 15,434 | 62.43 | 62.46 | 16.019 | 16.010 | -0.05% |
| 28,000 | 51.02 | 51.59 | 19.600 | 19.385 | -1.10% |

The F16 dispatch and arithmetic are unchanged. The small shifts are session and
clock variance; there is no reproducible regression.

Fresh baseline dispersion (milliseconds/model-token) was:

| KV | mean | median | sample stddev | CV |
|---:|---:|---:|---:|---:|
| 128 | 11.9625 | 12.0000 | 0.0677 | 0.566% |
| 2,048 | 12.5687 | 12.5625 | 0.0342 | 0.272% |
| 8,192 | 14.0813 | 14.0625 | 0.0280 | 0.199% |
| 15,434 | 16.0188 | 16.0312 | 0.0356 | 0.222% |
| 28,000 | 19.6000 | 19.5938 | 0.0867 | 0.442% |

F16 prefill guards also remained within ordinary session noise:

| prompt tokens | baseline ms | final ms | delta |
|---:|---:|---:|---:|
| 128 | 247 | 248 | +0.4% |
| 2,048 | 4,232 | 4,242 | +0.2% |
| 8,192 | 24,795 | 24,789 | -0.0% |
| 28,000 | 192,856 | 191,239 | -0.8% |

## Decode scaling

Least-squares fits use model-token latency in milliseconds and cache depth `N`:

```text
F16 baseline:      T(N) = 11.920869 + 0.000271866*N
INT8 old fallback: T(N) = 12.036426 + 0.009011173*N  (through 15K)
INT8 final:        T(N) = 11.926834 + 0.000241941*N  (through 15K)
```

| fit | KV-dependent fraction at 15K | at 28K | predicted 28K ms/token |
|---|---:|---:|---:|
| F16 baseline | 26.0% | 39.0% | 19.53 |
| INT8 old fallback | 92.0% | 95.4% | 264.35 |
| INT8 final | 23.8% | 36.2% | 18.70 |

The final INT8 slope is slightly below F16 because INT8 stores 132 bytes per
K/V vector including metadata, versus 256 bytes for F16.

## Competitive reference

A compatible llama.cpp checkout was built at commit `9bebfcb4b` with Vulkan,
F16 K/V, all 3B layers on the same GPU, flash attention, greedy generation,
32 model tokens and five samples. The 2K llama.cpp anomaly reproduced within
its own samples and is retained rather than discarded.

| KV | ours F16 ms | llama.cpp ms | ours/llama | absolute gap ms |
|---:|---:|---:|---:|---:|
| 128 | 11.963 | 9.080 | 1.317 | +2.882 |
| 2,048 | 12.569 | 24.789 | 0.507 | -12.221 |
| 8,192 | 14.081 | 10.889 | 1.293 | +3.192 |
| 15,434 | 16.019 | 12.854 | 1.246 | +3.165 |
| 28,000 | 19.600 | 16.714 | 1.173 | +2.886 |

## Full F16 attribution by depth

Temporary Vulkan timestamps covered kernels and GPU gaps. Values are
milliseconds per output model token and include sampling. Every row sums to the
shown GPU critical path; the categories cover 100%, with no broad `other`
bucket. Profiling adds 1-5% overhead, so uninstrumented latency is authoritative.

| component | 128 | 2K | 8K | 15K | 28K |
|---|---:|---:|---:|---:|---:|
| normalization | 0.337 | 0.332 | 0.333 | 0.334 | 0.336 |
| Q + K + V + O projections | 2.126 | 2.126 | 2.120 | 2.124 | 2.118 |
| fused attention (K read, QK, softmax, V read, AV, reduce) | 0.156 | 0.672 | 2.313 | 4.258 | 7.613 |
| gate + up + down | 5.705 | 5.705 | 5.693 | 5.692 | 5.671 |
| Q8 activation quantization for Q4 MMVQ | 0.195 | 0.198 | 0.198 | 0.198 | 0.194 |
| RoPE + KV append + activation/residual + embedding | 0.077 | 0.078 | 0.079 | 0.081 | 0.080 |
| logits | 2.437 | 2.438 | 2.437 | 2.433 | 2.431 |
| barriers/inter-dispatch GPU gaps | 0.901 | 1.020 | 1.026 | 1.022 | 1.008 |
| sampling kernel + gap | 0.132 | 0.132 | 0.132 | 0.131 | 0.131 |
| **profiled GPU total** | **12.066** | **12.703** | **14.331** | **16.274** | **19.583** |

The final INT8 15K reprofile is 15.884 profiled GPU ms/token. Its attention is
3.828 ms, KV append/quantization is 0.042 ms, and every non-KV component matches
F16 within noise. The old INT8 attention cost is approximately 139.35 ms/token:
151.115 whole-token latency minus the measured 11.76 ms KV-independent path.

### Expanded final 15K budget

| component | ms/token | share of 15.884 profiled GPU time |
|---|---:|---:|
| block RMS normalization | 0.327 | 2.1% |
| final norm | 0.006 | <0.1% |
| Q projection | 0.572 | 3.6% |
| K projection | 0.248 | 1.6% |
| V projection | 0.625 | 3.9% |
| INT8 K/V read + QK + online softmax + AV + split reduction | 3.828 | 24.1% |
| RoPE | 0.012 | 0.1% |
| INT8 KV append/quantization | 0.042 | 0.3% |
| O projection | 0.680 | 4.3% |
| gate | 1.295 | 8.2% |
| up | 1.377 | 8.7% |
| SwiGLU activation (element-count allocation) | 0.029 | 0.2% |
| down | 3.028 | 19.1% |
| residual adds (element-count allocation) | 0.019 | 0.1% |
| activation Q8 quantization | 0.201 | 1.3% |
| embedding | 0.006 | <0.1% |
| logits | 2.433 | 15.3% |
| model-command barriers/inter-dispatch gaps | 1.025 | 6.5% |
| sampling reduction/readback GPU work and gap | 0.131 | 0.8% |
| **total** | **15.884** | **100%** |

QK, softmax and AV are fused with K/V reads and overlap on the critical path;
they cannot be assigned additive timestamp intervals without changing the
kernel. Their independent demands are: 6.58 GFLOP/token for QK+AV at 15K,
847.5 MB/token of exact INT8 K/V+metadata traffic, and online exponent/reduction
work for 32 query heads. This explicit roofline replaces a misleading invented
split: memory and arithmetic execute concurrently inside the same dispatch.

## Residency and memory capacity

All primary rows execute embedding, normalization, every projection, attention,
KV append, logits and sampling on Vulkan. There are no CPU layers, per-token
weight copies, placement changes, or host/device activation crossings as depth
grows. The 3B all-GPU plan reports:

| owner | bytes |
|---|---:|
| weights | 2,138,290,176 |
| configured F16 KV | 3,489,660,928 |
| configured INT8 KV | 1,799,356,416 |
| persistent scratch | 3,514,368 |
| fixed Vulkan buffers | 17,563,648 |
| placement reserve | about 588,000,000 |
| planned F16 GPU total | 6,237,092,044 |
| observed F16 process peak | 6.54-6.59 GB |

Active logical cache bytes by exact depth:

| KV | F16 | INT8 including metadata |
|---:|---:|---:|
| 2,048 | 218,103,808 | 112,459,776 |
| 8,192 | 872,415,232 | 449,839,104 |
| 15,434 | 1,643,659,264 | 847,511,808 |
| 28,000 | 2,981,888,000 | 1,537,536,000 |

The allocation is full-context and does not resize with live depth. Telemetry
shows stable residency, about 99% GPU busy at long F16 depths, stable clocks,
and no migration or allocation churn. At 15K the measured board power was about
190 W, junction about 97 C, and VRAM about 53 C.

| KV | samples | busy mean | peak VRAM bytes | SCLK MHz | power W | edge/junction/memory C |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 4 | 74.00% | 6,486,429,696 | 2,098 | 116.0 | 54.5 / 68.3 / 53.5 |
| 2,048 | 8 | 96.12% | 6,542,159,872 | 2,630 | 164.6 | 58.8 / 83.6 / 55.0 |
| 8,192 | 29 | 93.62% | 6,570,381,312 | 2,478 | 191.8 | 62.1 / 98.1 / 55.2 |
| 15,434 | 73 | 98.79% | 6,567,194,624 | 2,625 | 189.9 | 58.7 / 97.1 / 53.9 |
| 28,000 | 198 | 98.81% | 6,585,610,240 | 2,630 | 182.2 | 55.5 / 93.1 / 51.0 |

The lower 128/8K utilization means include load/idle samples because the sysfs
sampler spans the complete prime. Long steady-state samples saturate the GPU.

Resource lifetimes are: pipelines/weights/full KV/persistent scratch per model;
KV contents per caller session; command buffer and fence per submission; pushed
descriptors and small host vectors per dispatch; one four-byte sampling copy,
map and unmap per output token. No Vulkan tensor buffer, pipeline, KV structure,
or temporary-storage allocation occurs per token.

## Routing and M=1 matrix inventory

The shape and format routes do not depend on context length.

| operation | M/N/K | format | canonical weight bytes/token | actual best path | measured ms/token at 15K |
|---|---|---|---:|---|---:|
| Q | 1/4096/3072 | Q4_K | 184,025,088 | DP4A Q4 MMVQ | 0.572 |
| K | 1/1024/3072 | Q4_K | 46,006,272 | DP4A Q4 MMVQ | 0.248 |
| V | 1/1024/3072 | Q6_K | 67,092,480 | required-wave32 Q6 GEMV | 0.625 |
| O | 1/3072/4096 | Q4_K | 184,025,088 | DP4A Q4 MMVQ | 0.680 |
| gate | 1/9216/3072 | Q4_K | 414,056,448 | DP4A Q4 MMVQ | 1.295 |
| up | 1/9216/3072 | Q4_K | 414,056,448 | DP4A Q4 MMVQ | 1.377 |
| down | 1/3072/9216 | Q6_K | 603,832,320 | required-wave32 Q6 GEMV | 3.028 |
| logits | 1/131072/3072 | Q6_K | 330,301,440 | required-wave32 FP32 Q6 GEMV | 2.433 |

Q4_K uses 144 bytes per 256 weights (128 packed quant bytes plus 16 bytes of
scales/minima). Q6_K uses 210 bytes (192 packed quant bytes plus 18 metadata
bytes). Activation/output traffic is only tens of KiB per operation and is
negligible beside weights. Each Q4 projection records activation quantization
plus MMVQ; each Q6 projection is one GEMV dispatch. Across the model, the
profile records 26 projection dispatches for each layer-local operation and one
logits dispatch, plus the Q4 activation quantizers.

At 15K, canonical effective bandwidth is approximately 323/185/107/270 GB/s
for Q/K/V/O, 321/302/199 GB/s for gate/up/down and 136 GB/s for logits. The Q6
four-lane shader logically rereads paired nibble payloads (about 408 addressed
bytes per 210-byte block), but cache serves much of that reuse. It is primarily
bandwidth plus unpack/dequant/reduction limited; required wave32 is material to
parallelism. No register spill counter was available. Q6 uses about 2 KiB LDS
and eight wave32 waves per workgroup. Q4 MMVQ is integer-dot and weight-traffic
limited; packed metadata/unpack remain inside the measured projection interval.

Eligibility predicates:

- Q4 MMVQ: integer dot product, Q4_K, `K % 256 == 0`, persistent scratch fit.
- AMD batched Q4 MMQ: the same plus AMD vendor capability and batch greater
  than one; it affects prefill, not M=1 decode.
- Q6 decode/logits: Q6_K format; AMD plus subgroup-size control builds the same
  portable shader with required wave32.
- F16 GQA decode: head dimension 128, exact 4:1 GQA, scratch fit, subgroup32
  pipeline (native or AMD required-size control); wave64 has its qualified
  exact fallback.
- INT8 GQA decode: INT8 scheme plus the same 128/4:1/scratch conditions and the
  new subgroup32 split pipeline; otherwise portable INT8 is selected.
- Context length never changes any of these decode routes.

## Attention traffic and roofline

F16 exact GQA traffic is 4,096 bytes/history-position/layer: one 128-element K
and V vector for each of eight KV heads. At 15,434 this is 1.644 GB/token and at
28K 2.982 GB/token. Measured attention is 4.258 and 7.613 ms, or 386 and 392
GB/s. The 432 GB/s board-bandwidth floor is 3.805 and 6.902 ms before QK,
softmax, AV and reduction, so exact F16 attention has little removable time.

Final INT8 traffic is 2,112 bytes/history-position/layer: 2,048 code bytes plus
64 metadata bytes. At 15,434 this is 847.5 MB/token. The new kernel has traffic
amplification 1.0; the old per-query route had amplification 4.0. Final measured
attention is 3.828 ms (221 GB/s based on minimum bytes), with the gap from the
1.962 ms physical byte floor explained by u8 extraction, FP16 metadata decode,
dequantization, FP32 QK/AV, exponentials and reductions. Arithmetic intensity is
about 7.8 FLOP/byte before online-softmax operations. A practical exact INT8
floor is 3.5 ms on this architecture.

## Dispatch, synchronization and idle audit

At 15K, divided by output model tokens, timestamp instrumentation recorded:

- about 582 model dispatches plus one sampling dispatch per token;
- about 481 model compute barriers per token;
- 1.97 command buffers, queue submissions, fences and waits per token;
- one sampling transfer, map and unmap per token;
- CPU model recording 0.590 ms/token, including 0.173 ms descriptor pushes;
- CPU queue submit 0.022 ms/token;
- about 1.025 ms/token of GPU inter-dispatch/barrier gaps.

CPU wall does not materially exceed the GPU critical path. Fence/command reuse
can remove at most the roughly 0.1 ms/request difference outside measured model
and prefill time. Most barriers protect true producer/consumer dependencies;
Q/K, gate/up and Q/K RoPE already use explicit one-shot barrier elision. Control
redesign therefore fails the 5% economic gate.

## Amdahl ranking, floors and closed frontiers

The accepted candidate used baseline `T=151.115 ms`, affected attention
fraction `f=0.922`, and conservative `S_local=4` from GQA traffic reuse:

```text
predicted removable = 151.115 * 0.922 * (1 - 1/4) = 104.5 ms
predicted final     = 46.6 ms/token = 21.5 tok/s
measured final      = 15.656 ms/token = 63.87 tok/s
measured removal    = 135.46 ms/token
```

The extra gain comes from bounding the history into eight wave32 splits and
restoring parallelism, not traffic reuse alone.

Every current non-trivial prototype was screened with the same F16 `T=16.019`
ms and Q6 affected fraction `f=0.380` before implementation:

| candidate | realistic local speedup | predicted removable ms | predicted final ms / tok/s | measured KV128 verdict |
|---|---:|---:|---:|---|
| Q6 two-lane ownership | 1.50x | 2.03 | 13.99 / 71.5 | +2.77% latency, reject |
| Q6 subgroup-shuffle sharing | 1.20x | 1.02 | 15.00 / 66.7 | +1.99%, reject |
| Q6 per-row LDS staging | 1.30x | 1.41 | 14.61 / 68.4 | +1.88%, reject |

Q+K+V or gate+up fusion cannot eliminate their weight streams. All Q4
activation quantizers together cost only 0.201 ms/token, so even eliminating
that entire bucket caps whole-token gain at 1.3%; activation/intermediate bytes
are tens of KiB against gigabytes of weights. Fusion is below the 5% gate before
accounting for added register state and routing complexity.

After reprofile, remaining decode candidates rank as follows:

| component | current ms | physical floor | practical floor | removable ms | status |
|---|---:|---:|---:|---:|---|
| Q6 V/down/logits | 6.086 | canonical bytes / 432 GB/s | about 4.8 | at most 1.29 | exact load-sharing variants regress; expanded representation is <10% whole-token with memory/API complexity |
| Q4 projections + activation quantization | 4.372 | 2.88 ms canonical bytes only | about 3.9 | about 0.47 | near traffic floor; fusion removes little weight traffic |
| final INT8 attention | 3.828 | 1.962 byte-only | about 3.5 | about 0.33 | exact traffic amplification is already 1.0 |
| barriers/GPU gaps | 1.025 | 0 | about 0.8 | about 0.23 | dependencies already elided where independent |
| norms/RoPE/KV/elementwise/embedding | 0.770 | about 0.5 | about 0.6 | about 0.17 | individually below noise/economic gate |
| sampling | 0.131 | about 0.08 | about 0.10 | about 0.03 | not material |

No remaining decode design-space level has an economically meaningful,
production-grade removal under the task gates. The largest whole-request cost
is now INT8 prefill, not decode.

### Rejected hypotheses

- Q6 two-lane ownership: 12.294 versus 11.963 ms at KV128, +2.77% latency;
  dependency depth and reduced parallelism outweighed fewer paired loads.
- Q6 subgroup-shuffle sharing: 12.200 ms, +1.99%; shuffle cost exceeded cache
  hits avoided.
- Q6 per-row LDS block staging: 12.188 ms, +1.88%; extra LDS traffic and lower
  occupancy outweighed global-address reuse.
- Historical exact F16 GQA two-split, wave64 workgroup, larger-tile and alternate
  ownership variants did not beat the retained split architecture.
- Historical Q6 Q8/dot-product conversion was locally faster but exceeded
  quality tolerances (roughly 0.06-0.07 tensor differences).
- Historical exact Q6 packed staging, 64-row Q6 tiles, larger Q4 tiles, and
  gate/up activation sharing were below the economic gate or regressed.
- The 14B hybrid reproduction hypothesis was rejected: its 25.5-minute 15K
  request produced 10.10 prompt tok/s and 6.19 decode deltas/s, not the cited
  phase behavior.

## Correctness, quality and request impact

The standard hybrid workspace suite passed: 178 root tests, 231 engine tests,
the decode harness test, six family-contract tests and twelve semantic tests;
only explicitly ignored authenticated/external rows were skipped. The explicit
long-context GPU oracle passed F16 and INT8 at 2K, 8K and 28K. At 28K the new
INT8 attention maximum absolute delta is 1.526e-5 under unchanged tolerances.
It covers causal attention, GQA, F16, INT8 coding, Q4_K/Q6_K projections,
logits, prefill/decode equivalence and long-context KV behavior.

Post-change multi-model smoke rows also completed on the local 8B and 14B
Instruct Q4_K_M artifacts with INT8 KV at depth 128/context 2K: 24.719 and
41.781 ms/model-token respectively. The 8B route exercises the supported GQA
shape; shapes outside the exact 4:1 predicate retain the portable fallback.

The final 15,434-prompt/256-output INT8 public request measured:

| metric | result |
|---|---:|
| TTFT / prefill | 118.59 s |
| prompt throughput | 130.15 tok/s |
| average text-delta decode | 61.68/s |
| beginning third | 61.61/s |
| middle third | 61.77/s |
| ending third | 61.66/s |
| total wall | 123.78 s |

For the same 256 model-token count, the strict fallback measurements imply
about 38.7 s of decode versus about 4.0 s final, while prefill is unchanged at
about 118.5 s. That changes a modeled 157.2 s request to the measured 123.8 s.
For the cited 1,156 generated-model-token scale, the same steady-state rates
project decode from 174.7 to 18.1 s and total from about 293 to 137 s. This is a
model-token projection, not a claim that the unspecified cited request used the
same tokenizer/delta count.

## Retained work and next candidate

Retained as evidence or production behavior:

- recorded exact-depth F16/INT8 steady-state decode measurements;
- non-overlapping beginning/middle/end generation reporting;
- capability- and shape-gated wave32 INT8 4:1 GQA split decode kernel;
- portable INT8 fallback and unchanged numerical tolerances.

All timestamp instrumentation and rejected shader variants were removed before
qualification. No dependency, public inference API, model format, or lossy KV
policy was added.

Decode priority is closed because remaining removable decode cost is below the
economic gates. The next global candidate is an INT8 GQA-aware prefill
architecture: the portable INT8 prefill rereads K/V per query head, takes 85.2
of 118.4 profiled prefill seconds at 15K, and crosses the RADV command watchdog
near 28K. It requires its own baseline, Amdahl estimate, bounded-command design,
and prefill qualification; it must not be smuggled into this completed decode
change.
