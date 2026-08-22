<!--
This report is the persistent checkpoint for the Apple Metal long-context
prefill campaign. It owns the baseline, attribution, audits, experiment ledger,
qualification, and stop record; it defines no runtime API or support contract.
-->

# Apple Metal long-context prefill optimization

## Result and identity

The reported 15,435-token symptom of 213.24 seconds was not reproducible on the
qualified current source. Fresh diagnostics-free Graph Horizon measured
103.184 seconds (149.59 tok/s). The largest removable component was exact F16
causal attention: 53.295 of 94.345 profiled GPU seconds, or 56.49%.

The retained M13 change pipelines two adjacent Q/K SIMD-matrix loads before
their MMAs. It changes no operation, arithmetic order, precision, cache layout,
memory capacity, dispatch geometry, or eligibility. At 15,435 tokens it reduces
profiled attention to 29.621 seconds (-44.42%) and complete GPU time to 70.675
seconds (-25.13%). Diagnostics-free mean TTFT falls from 103.184 to 84.298
seconds (-18.30%, 183.10 tok/s). A separate cold real request prefills 15,435
tokens in 70.942 seconds and generates all 256 requested tokens at 18.72 tok/s.

| Field | Value |
|---|---|
| Date | 2026-08-21, Europe/Rome |
| Baseline source | `e3d23f2` |
| Baseline ordinary binary | SHA-256 prefix `6648c841` |
| Retained implementation | `0665976` |
| Qualified source before docs | `305a848` |
| Final ordinary binary | SHA-256 `15ec669193e5be0876fba60e277c7da735d1e7fdcb963a1340c4826db40d3df6` |
| Branch | `perf/metal-long-context-prefill` |
| Device | Apple M4, 10 GPU cores, 24 GB unified memory |
| Host | fanless MacBook Air, macOS 26.3, Metal 4 |
| Model | `Ministral-3-3B-Instruct-2512-Q4_K_M.gguf` |
| Model identity | 2,147,023,008 bytes; SHA-256 `9ed150d4367e68df0ac8e1540f6ddc65b42d0ee26378329d1ecbca60f93fc5f8` |
| Graph Horizon | pure Metal, all layers Metal, context 32,768, F16 KV, rows 64 |
| llama.cpp | `2b63e0610bbc2be990ae1360d5256efcdc3f9efb`, build 10237 |
| llama.cpp tuple | Metal, full offload, F16 K/V, flash attention, batch/microbatch 64 |
| Push state | nothing pushed |

The machine was on AC power. `pmset` reported no thermal or performance
warning. `powermetrics` requires root, so clocks, power, and numeric temperature
were unavailable. Live `ioreg` during 15K reported 98% device utilization, 84%
renderer/tiler utilization, zero recoveries, about 7.58 GB driver allocation,
and about 6.38 GB GPU in-use system memory. There were no page faults or swaps.
Metal reports SIMD width 32, 1,024 maximum threads/threadgroup, 32,768 bytes
threadgroup memory, a 19.07 GB recommended working set, and a 14.30 GB maximum
buffer length.

## Measurement contract and commands

Graph Horizon inputs are `"a "` repeated `N - 4` times. Each matrix point uses
one warm-up and three measured requests, 32 requested tokens, greedy sampling,
one diagnostics-free release binary, and a 32,768-token allocation. Long rows
run serially. CV above 1% is reported rather than filtered.

Graph Horizon TTFT includes tokenization, first sampling, and the first public
text delta. llama.cpp's benchmark prompt timer covers prompt evaluation only.
This disadvantages Graph Horizon slightly and makes close ratios descriptive.

```sh
cargo build --locked --release --no-default-features --features metal \
  --example bench

target/release/examples/bench \
  "<model.gguf>" \
  --context 32768 --kv f16 --prompt "$(printf 'a %.0s' {1..2044})" \
  --max-tokens 32 --warmup 1 --reps 3

"<llama-bench>" \
  -m "<model.gguf>" \
  -p 128,512,2048,8192,15435,28000 -n 0 -b 64 -ub 64 \
  -ctk f16 -ctv f16 -ngl 99 -fa on -r 3
```

## Fresh baseline and final matrix

| Context | Baseline mean / median / SD / CV | Final mean / median / SD / CV | Baseline tok/s | Final tok/s | Delta |
|---:|---:|---:|---:|---:|---:|
| 128 | 334.86 / 335.00 / 1.29 ms / 0.38% | 331.65 / 331.56 / 0.17 ms / 0.05% | 382.25 | 385.95 | -0.96% |
| 512 | 1,367.75 / 1,367.02 / 1.41 ms / 0.10% | 1,336.47 / 1,336.74 / 0.58 ms / 0.04% | 374.34 | 383.10 | -2.29% |
| 2,048 | 6,122.21 / 6,122.36 / 3.53 ms / 0.06% | 5,694.01 / 5,691.09 / 5.11 ms / 0.09% | 334.52 | 359.68 | -6.99% |
| 8,192 | 35,219.96 / 35,140.22 / 199.55 ms / 0.57% | 30,343.79 / 30,043.27 / 1,075.49 ms / 3.54% | 232.60 | 270.20 | -13.84% |
| 15,435 | 103,184.15 / 103,575.79 / 865.96 ms / 0.84% | 84,297.78 / 84,350.72 / 322.29 ms / 0.38% | 149.59 | 183.10 | -18.30% |
| 28,000 | 280,057.75 / 279,254.26 / 4,689.83 ms / 1.67% | 214,704.39 / 215,732.69 / 1,967.94 ms / 0.92% | 100.00 | 130.42 | -23.34% |

The gain grows with history length, as an attention-only change should. Stable
2K A-to-B and B-to-A comparisons reproduce -7.0% and -6.4%. Decode guards are
30.11, 35.91, 25.31, 19.04, and 12.27 tok/s at history 128, 2K, 8K, 15K, and
28K. Baseline is 29.97, 35.70, 25.73, 19.04, and 12.84. M13 is unreachable
for row-one decode; movements are thermal noise. The hottest 28K guard moves
-4.4%, below the 5% limit.

## Competitive matrix and scaling

| Context | Final ours | llama.cpp | Ratio | Gap |
|---:|---:|---:|---:|---:|
| 128 | 331.65 ms | 299.18 ms | 1.109x | +32.47 ms |
| 512 | 1.336 s | 1.357 s | 0.985x | -20.38 ms |
| 2,048 | 5.694 s | 5.649 s | 1.008x | +45.02 ms |
| 8,192 | 30.344 s | 31.229 s | 0.972x | -0.885 s |
| 15,435 | 84.298 s | 72.853 s | 1.157x | +11.445 s |
| 28,000 | 214.704 s | 176.111 s | 1.219x | +38.593 s |

These are matched same-device, same-session references, subject to the event
boundary and fanless thermal caveats above. A fixed + linear + quadratic fit is:

```text
baseline = -0.190560 + 2.3571456e-3 N + 2.7372093e-7 N^2
final    = -0.331553 + 2.5033228e-3 N + 1.8534061e-7 N^2
llama    = -0.237621 + 2.8109862e-3 N + 1.2455922e-7 N^2
```

At 15K the final linear term is 38.639 seconds versus llama.cpp's 43.388;
the quadratic term is 44.155 versus 29.675. At 28K those terms are 70.093 +
145.307 versus 78.708 + 97.654 seconds. The remaining gap is quadratic
attention, not matrix projection or setup.

## Baseline attribution and the 213-second budget

The temporary `metal-profile` feature timestamps adjacent operation categories
inside the encoder and was removed before qualification. The fresh 15K profile
accounts for the GPU interval exactly:

| Baseline component | Seconds | Share |
|---|---:|---:|
| fused causal attention (QK + online softmax + AV) | 53.295 | 56.49% |
| Q/K/V/O projections | 11.522 | 12.21% |
| gate/up/down projections | 26.591 | 28.19% |
| norm, RoPE, KV write, activation, residual, embedding, logits | 2.937 | 3.11% |
| complete GPU interval | 94.345 | 100.00% |

Projection seconds are Q 4.181, K 1.638, V 1.696, O 4.008, gate 8.477, up
8.494, and down 9.620. Norm is 0.123, RoPE 1.136, KV write 0.137, activation
0.427, residual 1.088, embedding 0.022, and logits 0.004 seconds. There is no
hidden broad bucket. QK, softmax, and AV cannot be timestamped separately
without changing the fused kernel; the traffic/work model below keeps them
explicit rather than inventing boundaries.

At 8K, 15K, and 28K the baseline GPU intervals are 35.518, 94.345, and
278.209 seconds. Attention uses 14.180, 53.295, and 193.876 seconds. Accounted
kernel time is 98.39%, 98.85%, and 99.17%. CPU recording is only 270, 343, and
641 ms; submission is 2.2, 1.7, and 3.8 ms; waits are 35.611, 94.531, and
278.569 seconds. The workload is GPU-critical. The old 213-second observation
is therefore explained as a stale source/session symptom, not a hidden current
component: fresh current-source TTFT is 103.184 seconds.

## Final attribution and practical floors

Post-gain profiling measures a 70.675-second GPU interval:

| Component | Current | Practical floor | Removable | Closure |
|---|---:|---:|---:|---|
| fused exact attention | 29.621 s | 29.621 s qualified | 0 s identified | M13 wins; M12/M14/M15 and acquired ownership variants regress |
| Q/K/V/O projections | 11.534 s | about 11.53 s | about 0 s | Metal 4 tensor path is at comparator-derived linear floor |
| gate/up/down projections | 26.634 s | about 26.63 s | about 0 s | same canonical tensor path and 64-row reuse |
| all other GPU work | 2.886 s | at least 2.35 s | below 0.54 s | no dominant stage; complete tail/record envelope is below 1% |
| complete GPU interval | 70.675 s | about 70.14 s | below 0.54 s identified | no candidate clears the economic gate |

The comparator fit exposes a theoretical 14.48-second quadratic gap at 15K
and 47.65 seconds at 28K. It is not called removable: every concrete exact
attention dataflow identified in the allowed space is tested, resource
infeasible, or has the wrong sign. A theoretical envelope without a viable
mechanism is not a practical floor.

## Routing, residency, and resource lifetime

| Operation | Shape / format | Actual best route |
|---|---|---|
| Q | M64 N4096 K3072, Q4_K | Metal 4 cooperative tensor projection |
| K/V | M64 N1024 K3072, Q4_K | Metal 4 cooperative tensor projection |
| O | M64 N3072 K4096, Q4_K | Metal 4 cooperative tensor projection |
| gate/up | M64 N9216 K3072, Q4_K | Metal 4 cooperative tensor projection |
| down | M64 N3072 K9216, Q6_K | Metal 4 cooperative tensor projection |
| prefill attention | rows64, heads32/8, dim128, F16, 4:1 GQA | `metal_attention_prefill_matrix` |
| fallback | other row/shape/placement/scheme | established tiled/segmented/serial route |
| decode | row1, F16 or INT8 | established grouped/split/generic route; M13 unreachable |

All 26 layers and dominant matrices are Metal resident. Shared storage is used
on unified memory; there is no CPU fallback, staging copy, migration, or
context route change. One command buffer/encoder owns each 64-row graph chunk,
then submits and waits: 128, 242, and 438 commands at 8K, 15K, and 28K. CPU
encoding is not critical, so command fusion cannot meet the 5% gate.

Weights and pipelines are per model/backend. Scratch, logits, reduction, and
staging persist. KV is request/session state under the existing serialized
reuse contract. M13 creates no allocation or lifetime.

## Memory audit

The exact all-Metal plan at context 32,768 is 2,138,290,176 weight bytes,
3,514,368 scratch bytes, 540,672 fixed bytes, 16,384 staging bytes, and
3,489,660,928 F16 KV bytes. F16 KV costs 106,496 bytes/context token.

| Prompt extent | Represented KV | Planned live buffers before reserve |
|---:|---:|---:|
| 2,048 | 218,103,808 | 2,360,465,408 |
| 8,192 | 872,415,232 | 3,014,776,832 |
| 15,435 | 1,643,765,760 | 3,786,127,360 |
| 28,000 | 2,981,888,000 | 5,124,249,600 |

The benchmark allocates full 32,768 capacity once, so process maximum RSS is
about 4.371 GB and peak footprint about 5.758 GB at every matrix point. No
resize, churn, fault, swap, eviction, fallback, or route crossover appears.

## Projection and weight traffic inventory

Canonical Q4_K uses 144 bytes/256 weights and Q6_K 210; metadata is inside the
blocks. The retained projection kernel dequantizes one 64-by-K32 tile to about
4 KiB threadgroup half storage and feeds Metal cooperative tensors. Q4 and Q6
keep separate block decoding, 16-value packed dequantization, FP16 graph
boundaries, and unchanged scalar oracle tolerances. There is no expanded
persistent weight representation.

At 15,435 tokens there are 242 chunks and 6,292 dispatches per projection role.
Required stream assumes each compressed block is read once per chunk. Effective
rate includes dequantization and compute; it is not a DRAM counter.

| Role | Unique bytes / 26 layers | Stream/request | Seconds | Rate |
|---|---:|---:|---:|---:|
| Q | 184.025 MB | 44.534 GB | 4.186 | 10.64 GB/s |
| K | 46.006 MB | 11.134 GB | 1.636 | 6.81 GB/s |
| V | 46.006 MB | 11.134 GB | 1.696 | 6.57 GB/s |
| O | 184.025 MB | 44.534 GB | 4.016 | 11.09 GB/s |
| gate | 414.056 MB | 100.202 GB | 8.492 | 11.80 GB/s |
| up | 414.056 MB | 100.202 GB | 8.516 | 11.77 GB/s |
| down | 603.832 MB | 146.127 GB | 9.626 | 15.18 GB/s |

The minimum compressed stream is 457.866 GB/request and input traffic
amplification is 1x at 64-row ownership. Each block is dequantized once and
reused across 64 rows. Larger ownership, swizzled staging, K64, FP32 scratch,
and alternate Q4/Q6 variants are acquired measured evidence. A new execution
representation would add conversion and/or persistent bytes without an
identified saving, so its request break-even is not finite for this campaign.

## Exact-attention traffic and state

One workgroup owns eight query rows and one query head; four query heads map to
each KV head. It processes 64 history columns per tile, fuses QK, causal mask,
online softmax, and AV, and writes only the final F16 output. Threadgroup state
is about 9.25 KiB: Q 2 KiB, probabilities 1 KiB, scores 2 KiB, scale 0.25 KiB,
and FP32 result 4 KiB. No global score, softmax, dequantized KV, or copy-only
intermediate exists.

Logical combined K/V traffic is 1.801 TB at 8K, 6.360 TB at 15K, and 20.873 TB
at 28K. Baseline attention rates are 127.0, 119.3, and 107.7 GB/s logical.
These are cache-oblivious, not physical DRAM bytes. Perfect four-head GQA reuse
would divide them by four, but the grouped path loses parallelism and regresses
attention 36.75% at 12K. At 15K useful QK+AV work is about 25.4 trillion FMAs
across 26 layers, plus exponentials and reductions. Q vectors account for
about 3.29 GB logical traffic. Online state is max/sum and FP32 output only.

The kernel has three threadgroup-memory synchronizations per C64 tile around
score publication, probability publication, and PV completion. M13 adds only
`mem_none` SIMD-group scheduling barriers around paired Q/K loads; they impose
no memory dependency. Pairing probability or V loads is slower, so the final
compiler schedule is intentionally asymmetric.

## Amdahl decision

```text
T_gpu = 94.345 s
attention = 53.295 s
f = 0.5649
realistic S_local = 1.80x
removable = 23.69 s
predicted GPU = 70.66 s
predicted diagnostics-free TTFT = 79.50 s
predicted rate = 194.15 tok/s
predicted whole-TTFT gain = 22.95%
```

Measured is 29.621 seconds attention (1.799x), 70.675 seconds GPU, 84.298
seconds TTFT, 183.10 tok/s, and 18.30% whole-TTFT gain. The GPU prediction
materializes almost exactly; the endpoint difference is broader boundaries and
thermal/session state.

## Candidate ledger

| Candidate | Mechanism | Result | Decision |
|---|---|---|---|
| M12 block-major PV probability reuse | fewer repeated TGM probability loads | attention +0.83%, GPU +0.50% at 8K | REJECT |
| M13 paired Q/K loads | hide SIMD-matrix load/issue latency | attention -46.0% at 8K, -44.4% at 15K | KEEP `0665976` |
| M14 pair probability and V loads | pipeline two PV MMAs | attention +12.6%, GPU +5.5% at 8K | REJECT |
| M15 pair only V loads | isolate value scheduling | attention +7.7%, GPU +4.1% at 8K | REJECT |
| M16 head-major Metal F16 KV | contiguous fixed-head history | 2K +1.6%; 8K B-to-A -10.5%, reverse +1.1% | REJECT |

M16 passed the serial oracle but required a backend-specific layout. Its cold
gain disappeared in reverse order and it lost at stable 2K, so token-major
remains. Experiment/revert history is `7f4f516`, `3499751`, `6e53288`, and
`305a848`; no M16 source survives. Acquired rejected directions were not
repeated: split query-head history, grouped long GQA, eight groups on one tile,
K64 projection, FP32 scratch, and wider/split ownership variants.

## Qualification

- The 64-row serial Metal attention oracle passes at base 0 and 512.
- Pure Metal passes 144 unit, 5 family, and 12 semantic tests; expected external
  tests remain ignored.
- Metal-hybrid passes 205 unit tests and the same integration counts. It retains
  two pre-existing unused-`simd` CPU warnings.
- Pure-Metal Clippy passes with `-D warnings`; formatting and diff checks pass.
- Authenticated 3B teacher rows pass F16 and INT8 KV. All 16 IDs exactly match
  pinned llama.cpp `13f2b28b0`, including top-two, lifecycle, sequential, and
  cancellation gates.
- Real 8B and 14B F16 smoke requests each complete 16 tokens through the same
  qualified shape. Their cold 128-token TTFTs are 849.83 and 2,019.17 ms;
  these are behavior guards, not speed claims.
- Q4_K/Q6_K projection oracles are unchanged. INT8, mixed placement,
  unsupported dimensions, non-4:1 GQA, non-64 rows, and unsupported devices
  retain generic fallbacks.

M13 does not change numerical policy. Q/K loads and MMAs retain their original
sequence; only the compiler-visible load schedule is paired. No tolerance was
changed.

## Real request and global stop

| 15,435 prompt + 256 generation | Result |
|---|---:|
| TTFT | 70.942 s |
| Prompt throughput | 217.57 tok/s |
| Generated tokens | 256 |
| Decode throughput | 18.72 tok/s |
| Decode interval CV | 1.00% |
| Approximate decode / total | 13.68 / 84.62 s |

Baseline 15K matrix prefill plus 256 tokens at 19.04 tok/s is about 116.63
seconds; final matrix prefill plus that rate is about 97.74 seconds (-16.2%).
The cold final request is faster but is not mixed into the A/B claim. Decode
remains much smaller than prefill, so global priority does not switch.

After M13, projections total 38.168 seconds (54.0% GPU) and attention 29.621
seconds (41.9%). Projections are the largest component but at their measured
floor. Attention retains the largest theoretical gap but no economically
meaningful identified candidate. Norm/RoPE/KV/elementwise/tail, encoding,
residency, layout, canonical representation, and numerical-policy spaces are
below gate or measured negative.

Literal llama.cpp parity is not universal: long-point ratios are 0.972--1.219x.
The second global-stop condition applies. No remaining candidate across
routing, lifetime, command structure, Q4/Q6 representation, projection reuse,
exact-attention ownership, GQA reuse, KV layout, intermediates, precision, or
Apple-family specialization contains at least 5% realistically removable
whole-prefill time with positive evidence.

The largest theoretical frontier is exact attention's quadratic gap. It is
closed economically, not declared physically impossible: grouped GQA loses
occupancy, wider/split variants regress or hit resources, PV pipelining
regresses, head-major KV does not reproduce, and M13 captures the profitable
schedule. Reopen only with new counters or a genuinely different exact
algorithm with a precomputed >=10% whole-prefill bound, not another tile sweep.

No profiler, rejected path, diagnostic feature, dependency, or API change
remains, and nothing was pushed.
