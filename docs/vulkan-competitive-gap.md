<!--
This report is the persistent checkpoint for the current portable Vulkan
competitive-gap investigation. It owns the fresh baseline, attribution,
experiment registry, retained decisions, qualification, and stop evidence.
-->

# Vulkan Competitive Gap

## Status

`COMPLETE — QUANTITATIVE GLOBAL STOP`: the fresh baseline, current-source
attribution, candidate cycle, cleanup, and qualification are complete. No
production candidate satisfies both the performance and numerical gates.

## Reproducible baseline

| Item | Value |
|---|---|
| Date | 2026-08-20 Europe/Rome |
| Baseline branch | `main` |
| Baseline SHA | `4bc6c7f07684ebc8f6bf7080a4ad77db6a9354a4` |
| Investigation branch | `perf/vulkan-prefill-long-context-gap` |
| Initial worktree | clean |
| GPU | NVIDIA GeForce RTX 3060 12 GiB (`0x2487`) |
| Driver / Vulkan | NVIDIA 595.84 / device API 1.4.329, loader 1.4.341 |
| Graph Horizon build | locked Cargo release, pure Vulkan, full offload |
| llama.cpp | clean `9bebfcb4bc8b12a316e96ae03f33671eac1e72fd`, build 9826 |
| Common runtime | Vulkan0, F16 KV, exact causal attention, greedy |
| Graph Horizon repetitions | one warm-up and three measured repetitions |
| llama.cpp repetitions | internal warm-up and three measured repetitions |

The Graph Horizon input is `" a"` repeated `N - 3` times, which renders to
exactly `N` prompt tokens. llama.cpp generates `N` benchmark token IDs. Both
programs therefore execute the same shapes and causal work on the same GGUF
bytes, but token values differ. Graph Horizon TTFT additionally includes prompt
rendering/tokenization, first logits sampling, and first public text emission;
llama.cpp pure prompt processing excludes that host-side tail. This difference
is small relative to the long-context gaps and is not subtracted.

All models are the catalogued Ministral 3 Instruct Q4_K_M artifacts. The mixed
profile contains the production Q4_K and Q6_K matrices; there is no separately
supported pure-Q6 artifact or public weight-profile selector.

## Fresh prefill matrix

Times are mean milliseconds. For 3B/28K and 8B/128, 2K, and 28K, `ours` is the
mean of the two same-session bookends. Other Graph Horizon rows are the initial
three-repetition record. Every reported CV is below 0.5% except the first
clock-ramp-sensitive 3B/128 record at 4.79%; its later bookend CV is 0.45%.

| Model | Context | Ours | llama.cpp | Ratio | Absolute gap |
|---|---:|---:|---:|---:|---:|
| 3B | 128 | 76.29 | 42.02 | 1.82x | 34.27 |
| 3B | 512 | 240.13 | 128.79 | 1.86x | 111.34 |
| 3B | 2K | 1,014.54 | 531.70 | 1.91x | 482.84 |
| 3B | 8K | 5,078.92 | 2,555.15 | 1.99x | 2,523.77 |
| 3B | 16K | 12,951.63 | 6,304.26 | 2.05x | 6,647.37 |
| 3B | 28K | 28,972.89 | 13,731.69 | 2.11x | 15,241.20 |
| 8B | 128 | 150.18 | 98.43 | 1.53x | 51.75 |
| 8B | 512 | 566.24 | 289.22 | 1.96x | 277.02 |
| 8B | 2K | 2,347.76 | 1,180.25 | 1.99x | 1,167.51 |
| 8B | 8K | 10,749.68 | 5,303.49 | 2.03x | 5,446.19 |
| 8B | 16K | 25,156.75 | 12,201.25 | 2.06x | 12,955.50 |
| 8B | 28K | 52,001.88 | 24,743.20 | 2.10x | **27,258.68** |
| 14B | 128 | 245.70 | 150.90 | 1.63x | 94.80 |
| 14B | 512 | 948.16 | 439.48 | 2.16x | 508.68 |
| 14B | 2K | 3,879.71 | 1,784.33 | 2.17x | 2,095.38 |
| 14B | 8K | 17,207.64 | 7,846.25 | 2.19x | 9,361.39 |
| 14B | 16K | capacity failure | not compared | n/a | n/a |
| 14B | 28K | capacity failure | not compared | n/a | n/a |

The largest absolute and realistically actionable gap is 8B/28K. The 14B
16K/28K rows fail Graph Horizon initialization with exact F16-KV capacities on
this 12 GiB device and are not replaced by hybrid offload or smaller contexts.

## Decode regression baseline

Graph Horizon uses 32 requested tokens after the exact prompt length. llama.cpp
decode throughput is derived from paired prompt-only and prompt-plus-32 means in
one process: `32 / (T_pg - T_pp)`. Ratios compare per-token latency, equivalently
`llama tok/s / ours tok/s`.

| Model | KV | Ours tok/s | llama.cpp tok/s | Latency ratio |
|---|---:|---:|---:|---:|
| 3B | 128 | 71.03 | 104.87 | 1.48x |
| 3B | 2K | 69.76 | 97.40 | 1.40x |
| 3B | 28K | 42.36 | 47.68 | 1.13x |
| 8B | 128 | 34.63 | 49.75 | 1.44x |
| 8B | 2K | 33.42 | 47.67 | 1.43x |
| 8B | 28K | 23.72 | 30.17 | 1.27x |

Decode is a mandatory guard. Long decode is already inside the historical
1.3--1.4x envelope; short decode remains slightly outside it but contains much
less absolute request time than 8B/28K prefill.

## Current-source attribution

Temporary Vulkan timestamps cover 100.00% of stable 8B/8K kernel time and
99.96% of 8B/28K kernel time. A cold run was subtracted from an aggregate of one
warm-up plus three repetitions. The resulting category proportions are scaled
to the diagnostics-free public time; profiler time is not substituted into the
competitive baseline.

| Component | 8B/8K public-normalized | Share | 8B/28K public-normalized | Share |
|---|---:|---:|---:|---:|
| Attention | 1.733 s | 16.12% | 20.439 s | 39.30% |
| Q | 0.723 s | 6.72% | 2.531 s | 4.87% |
| K | 0.358 s | 3.33% | 1.263 s | 2.43% |
| V | 0.349 s | 3.24% | 1.226 s | 2.36% |
| O | 0.723 s | 6.73% | 2.532 s | 4.87% |
| gate | 1.502 s | 13.98% | 5.294 s | 10.18% |
| up | 1.505 s | 14.00% | 5.272 s | 10.14% |
| down | 3.501 s | 32.57% | 12.211 s | 23.48% |
| normalization | 0.059 s | 0.55% | 0.204 s | 0.39% |
| RoPE | 0.027 s | 0.25% | 0.091 s | 0.18% |
| KV append | 0.010 s | 0.10% | 0.036 s | 0.07% |
| residual / elementwise | 0.182 s | 1.70% | 0.636 s | 1.22% |
| embedding | 0.076 s | 0.71% | 0.264 s | 0.51% |
| logits | 0.003 s | 0.02% | 0.003 s | <0.01% |

The 28K format split is 21.915 s direct Q4, 8.414 s staged Q6, 20.439 s
attention, and 1.234 s all other work. The same current llama.cpp executable's
timestamp logger attributes its public-normalized 24.743 s to 11.828 s Q4,
2.455 s Q6, 9.186 s attention, and 1.274 s other. The competitive gap therefore
reconciles as follows:

| Rank | Component | Ours | llama.cpp | Absolute gap | Gap share |
|---:|---|---:|---:|---:|---:|
| 1 | dense attention | 20.439 s | 9.186 s | 11.252 s | 41.28% |
| 2 | Q4_K prefill | 21.915 s | 11.828 s | 10.088 s | 37.01% |
| 3 | Q6_K prefill | 8.414 s | 2.455 s | 5.959 s | 21.86% |
| 4 | all other work | 1.234 s | 1.274 s | -0.040 s | -0.15% |

The components account for the entire 27.259 s public gap within rounding.

## Routing, ownership, and traffic

The production routes are explicit and do not fall back under memory pressure.
The 109 complete 256-row 28K batches use Q64 attention; only the final 96-row
batch uses Q32. Every qualified recurrent Q4/Q6 shape uses Matrix2.

| Path | Current ownership | Declared/acquired resources | Comparator ownership |
|---|---|---|---|
| Q4 Matrix2 | 256 outputs x 128 tokens x K64, WG256, FP16 accumulator | 16,512 B scale/min shared + 8,192 B driver reserve | 128 outputs x 256 tokens x K64, direct canonical callback |
| Q6 Matrix2 | 32 outputs x 64 tokens x K128, WG256, FP32 accumulator | 16,384 B weight/output shared + 8,192 B driver reserve | 128 outputs x 256 tokens x K64, direct canonical callback with default FP16 accumulator |
| attention | Q64/KV64 WG128 for full batches; Q32 tail | acquired 255 registers/thread, 48,640 B driver shared, two WG/SM, zero stack | Matrix2 flash attention over the same dense causal pairs |

No physical DRAM/L2 transaction counter is exposed through the available
Vulkan profiler. The following is therefore a static logical-byte model, not a
claim about cache-miss traffic. It separates canonical payload, metadata,
tile-amplified weight reads, tile-amplified activation reads, and output stores.
Q4 payload/metadata are 128/16 bytes per 256 weights; Q6 are 192/18 bytes.

| Operation (all 34 layers, mixed rows split 17/17) | Canonical payload | Metadata | Logical weight reads | Logical activation reads | Outputs |
|---|---:|---:|---:|---:|---:|
| Q Q4 | 0.266 GiB | 0.033 GiB | 65.4 GiB | 116.2 GiB | 7.3 GiB |
| K Q4 | 0.066 GiB | 0.008 GiB | 16.4 GiB | 29.1 GiB | 1.8 GiB |
| V Q4 / Q6 | 0.083 GiB | 0.009 GiB | 32.1 GiB | 130.7 GiB | 1.8 GiB |
| O Q4 | 0.266 GiB | 0.033 GiB | 65.4 GiB | 116.2 GiB | 7.3 GiB |
| gate Q4 | 0.930 GiB | 0.116 GiB | 229.1 GiB | 406.7 GiB | 25.4 GiB |
| up Q4 | 0.930 GiB | 0.116 GiB | 229.1 GiB | 406.7 GiB | 25.4 GiB |
| down Q4 / Q6 | 1.162 GiB | 0.123 GiB | 448.5 GiB | 1,830.4 GiB | 7.2 GiB |
| **Total** | **3.702 GiB** | **0.439 GiB** | **1,086.0 GiB** | **3,036.0 GiB** | **76.3 GiB** |

Q4 canonical weights are logically traversed 219 times at 28K versus 110 in
llama.cpp; Q6 is traversed 438 times versus 110. Wider output ownership reduces
Graph Horizon's activation-read multiplicity, so these figures are explanatory
bounds, not a bandwidth-only diagnosis. The measured Q4/Q6 gaps also include
unpack arithmetic, barriers, accumulator pressure, shared staging, and Matrix2
callback resource costs.

## Global Amdahl ranking

| Rank | Target | Current share | Practical removable | Whole-request gain | Decision |
|---:|---|---:|---:|---:|---|
| 1 | exact attention architecture | 39.30% | <=5.134 s | <=9.87% | closed unless a new resource mechanism appears |
| 2 | lossless execution-native Q6 | 16.18% | comparator upper bound 5.959 s | <=11.46% | closed by measured exact and numeric variants |
| 3 | direct canonical Q4 neighbors | 42.14% | no economic local neighbor | 0 measured | closed by ownership/K endpoints |
| 4 | other kernels/host | 2.37% | <=1.234 s impossible deletion | <=2.37% | below gate |

The acquired attention stage isolation remains valid because both retained
attention shader binaries are byte-identical to the measured source. Against
the fresh 21.354 s raw attention total it leaves an exact practical floor of
15.990 s raw, or 15.304 s public-normalized. That is 5.134 s removable. Even
combining this optimistic attention floor with impossible deletion of all other
work yields 45.634 s, still 3.571 s above the current llama.cpp 1.7x ceiling of
42.063 s. Closing the broad target therefore also requires a material Q4 or Q6
change.

## Frontier checkpoints

### Q4_K prefill

Current time is 21.915 s at 8B/28K. The retained direct canonical callback
already reduced the component from 38.126 to about 22.7 s. Adjacent M128/N256,
M256/N256, and K128 variants regressed 132.9%, 147.6%, and 93.0%; integer and
staged alternatives regressed more. Practical current-architecture floor:
21.915 s. Reopen only for a new representation or hardware capability.

### Q6_K prefill

Current time is 8.414 s at 8B/28K. A direct canonical callback previously saved
only 1.17% at its screen. Fresh FP16 accumulation reduces local Q6 time by 8.0%
and diagnostics-free 8B/2K TTFT by 2.02%, below the 5% whole-request gate.

The recurrent Q6 set contains 1,069,547,520 values. Canonical storage is
0.817 GiB; aligned lossless int8 values plus the original sixteen signed scales
and FP16 multiplier is 1.121 GiB, a 0.304 GiB replacement premium. The screening
prototype retained canonical bytes for decode and therefore added the full
1.121 GiB. It preserved the exact `float(d) * float(scale) * float(q)` order
while removing 4/2-bit extraction.

With FP32 accumulation and 128-token ownership the native callback improved
instrumented 2K total time by only about 1.5%, below the gate. FP16 accumulation
raised the diagnostics-free A/B/A gain to 7.03% (2,322.49 -> 2,159.26 ms), but
failed the existing focused oracle: output 53 was -14.3203 versus -14.2452,
outside both 0.05 absolute and 0.5% relative tolerances. Exact FP32 with
256-token ownership passed the oracle (`max_abs=0.00733`,
`max_relative=0.00225`) but regressed instrumented TTFT to 2,702.65 ms. The
larger accumulator recreates the canonical direct-Q6 resource pathology, so the
representation frontier is closed without weakening the numerical contract.

### Long-context attention

Attention grows from 1.733 s / 16.12% at 8K to 20.439 s / 39.30% at 28K. The
same-shader stage isolation attributes roughly 9.07 s raw to QK MMA, 5.42 s to
Q/K load and addressing, and 6.86 s to live V/AV/online-softmax/state. Q64
already removes explicit score/probability materialization and per-KV-tile
barriers. Split ownership, persistent state, hierarchical reduction, score
staging, Q reuse, pre-scaled Q, and Q/KV tile families are closed. Practical
public floor: 15.304 s; no new exact resource mechanism is currently available.

## Experiment registry

| ID | Frontier | Target | Hypothesis | Amdahl removable | Result | Decision |
|---|---|---|---|---:|---|---|
| B0 | global | 8B/28K | establish current production and llama.cpp gap | 27.259 s raw gap | stable baseline | evidence |
| A0 | global | current profiler | attribute >=95% and verify routes | n/a | 99.96%; no lost route | evidence |
| Q6-01 | Q6 | FP16 accumulator/output tile | lower accumulator/shared pressure | 0.67 s extrapolated | 2K total -2.02%, Q6 local -8.0% | REJECT below 5% gate |
| Q6-02 | Q6 | lossless native bytes, FP32, 128 tokens | remove unpack and halve weight traversals | <=5.959 s | about 1.5% instrumented whole gain | REJECT below gate |
| Q6-03 | Q6 | native bytes plus FP16 accumulator | match comparator arithmetic/resources | >=5% | diagnostics-free 2K -7.03%; focused oracle failed | REJECT numeric contract |
| Q6-04 | Q6 | native bytes, exact FP32, 256 tokens | reach 110 weight traversals at 28K | >=5% | oracle passed; instrumented 2K 2,702.65 ms regression | REJECT performance |

### Q6-02 structure record

Before the execution-native prototype, the necessary local structure is:

```text
crates/graph_horizon_engine/src/backend/vulkan/
├── mem/
│   ├── buffers.rs (about 190 productive lines; carry one optional native offset)
│   ├── repack.rs (about 70 productive lines; lossless canonical-Q6 conversion)
│   └── weights.rs (about 155 productive lines; select/upload the extra layout)
├── kernels/
│   ├── coopmat.rs (about 105 productive lines; bind/dispatch the native tile)
│   └── matmul/prefill/mod.rs (about 145 productive lines excluding tests; select it)
├── pipeline/
│   ├── kernel.rs (about 160 productive lines; one private kernel ABI)
│   └── mod.rs (about 145 productive lines; capability-gated registration)
└── shaders/matmul/
    └── matmul_q6_k_native_matrix2_f16out.comp (about 170 productive lines, category K)
```

`repack.rs` was a distinct representation domain, so no one-file folder was
introduced. The prototype respected the proposed boundaries and line limits.
All files in this tree were rejected experiment code and have been removed.

## Retained changes

No production source change. This report is the only retained branch delta.

## Rejected directions

Q6-01 through Q6-04 are removed. Historical negative results remain acquired
evidence and will not be repeated without a materially new measured cause.

## Qualification and final state

The final production tree is byte-identical to baseline commit `4bc6c7f`.
Temporary timestamp instrumentation, native-weight data, shaders, routes, and
tests are absent. Current-source qualification passes:

- formatting and `git diff --check`;
- pure Vulkan: 166 application tests, 156 engine tests with five intentional
  ignores, five integration tests with one ignore, and 12 semantic tests with
  one ignore;
- Vulkan hybrid: 166 application tests, 229 engine tests with four intentional
  ignores, six integration tests with one ignore, and 12 semantic tests with
  one ignore;
- pure Vulkan and Vulkan-hybrid all-target warning-denied Clippy.

The raw build, benchmark, profiler, and rejected-candidate evidence remains
under `target/vulkan-gap` and is intentionally not committed.

## Final decision

The measured 8B/28K request is 52.002 s versus llama.cpp's 24.743 s. Attention,
Q4, and Q6 explain 41.28%, 37.01%, and 21.86% of the gap; other work is already
at parity. Exact attention has at most 5.134 s practical headroom, Q4's adjacent
ownership/K geometries regress, and the complete Q6 representation cycle finds
only a below-gate exact path or a faster path that violates the established
numeric oracle.

Even the optimistic attention floor plus impossible deletion of every other
kernel misses the 1.7x ceiling by 3.571 s. Reopen only for a new NVIDIA
capability, a new exact resource mechanism not represented in the registry, or
an explicitly approved model/numerical contract change. The current production
source is the smallest qualified result.
