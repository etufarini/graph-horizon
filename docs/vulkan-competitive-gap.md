<!--
This report is the persistent checkpoint for the current portable Vulkan
competitive-gap investigation. It owns the fresh baseline, attribution,
experiment registry, retained decisions, qualification, and stop evidence.
-->

# Vulkan Competitive Gap

## Status

`CONTINUE`: the fresh baseline is complete and the first detailed attribution
target is 8B/28K prefill. No production candidate has been implemented yet.

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

## Global ranking before attribution

| Rank | Workload | Whole-request gap | Initial decision |
|---:|---|---:|---|
| 1 | 8B/28K prefill | 27.259 s | detailed current-source attribution |
| 2 | 3B/28K prefill | 15.241 s | guard and transfer check |
| 3 | 8B/16K prefill | 12.956 s | guard and scaling check |
| 4 | 14B/8K prefill | 9.361 s | capacity-qualified transfer check |

The ranking is only a workload selection. Candidate ranking awaits current
Q4/Q6/attention timestamps, practical floors, traffic/resource models, and
Amdahl calculations.

## Frontier checkpoints

### Q4_K prefill

Current attribution, floor, and next candidate: pending fresh timestamps and
routing audit.

### Q6_K prefill

Current attribution, floor, and next candidate: pending fresh timestamps and
routing audit. Q6 will be evaluated independently from Q4.

### Long-context attention

Current attribution, floor, and next candidate: pending fresh timestamps and
traffic/resource reconstruction at 8K and 28K.

## Experiment registry

| ID | Frontier | Target | Hypothesis | Amdahl removable | Result | Decision |
|---|---|---|---|---:|---|---|
| B0 | global | 8B/28K | establish current production and llama.cpp gap | 27.259 s raw gap | stable baseline | evidence |

## Retained changes

None yet.

## Rejected directions

None in this current-source cycle. Historical negative results remain acquired
evidence and will not be repeated without a materially new measured cause.

## Next action

Reintroduce the previously qualified Vulkan timestamp profiler as temporary
instrumentation, attribute at least 95% of 8B/8K and 8B/28K execution, audit
actual Q4/Q6/attention routing, then calculate practical floors and Amdahl-rank
the first production candidate.
