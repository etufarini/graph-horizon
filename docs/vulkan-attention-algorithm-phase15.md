<!--
This report owns Phase 15 evidence for Vulkan long-context attention algorithm
and representation screening: fresh baselines, the physical/current-floor
model, exact and numerical architecture gates, and the final stop decision.
It defines no new runtime behavior, approximation contract, or model format.
-->

# Vulkan Attention Algorithm Phase 15

## Outcome

Phase 15 retains no production change. Fresh diagnostics-free 28K TTFT is
45,892.68 ms before investigation and 45,861.16 ms in the source-identical
median recheck. Fresh stable GPU prefill is 46,092.240 ms, of which exact dense
attention is 26,435.802 ms, or 57.35%.

The priority exact split-K candidate fails before implementation. A full
attention dispatch already exposes 256 workgroups to 28 SMs with two resident
workgroups per SM. Splitting does not change the 124-register, 38,272-byte
resource limit, so it creates more waves without increasing resident
workgroups or active warps. Even split 8 has only a 7.5% ideal attention
wave-granularity ceiling, 4.32% of the diagnostics-free request before any
merge. Its exact partial states add 34.079 MB of temporary storage and 193.823
GB of request-level global reads and writes. At a favorable 300 GB/s, traffic
alone reduces the ideal ceiling to at most 2.91% of the request before merge
compute, dispatches, barriers, and rounding-order effects.

Materially different exact candidates also fail their pre-code gates.
In-workgroup hierarchy crosses both the two-workgroup register and shared-memory
limits; persistent ownership has no positive linear setup term in the measured
8K--28K curve; head scheduling is bounded by the acquired K-hot ceiling;
exact state packing changes neither limiting resource; and two-pass exact
attention must store a global probability matrix or repeat QK. The first
allowed numerical screen, FP16 score staging with FP32 softmax and AV state,
has only about 2.62% optimistic whole-request headroom and therefore is not
implemented or quality-tested.

This satisfies the mission's pre-code stop conditions. Dense exact attention
is closed on this hardware under the current model representation. The final
branch is `perf/vulkan-attention-algorithm-phase15`, based on Phase 14 evidence
commit `42ebd51`; effective production source remains Phase 12 commit
`6cb170f`. Nothing is pushed.

## Baseline, method, and invariants

- Required baseline branch: `perf/vulkan-native-lowprecision-weights-phase14`.
- Phase 14 evidence: `42ebd51`; capability audit: `456e2ac`.
- Effective production source: `6cb170f`.
- Initial worktree: clean; Phase 10 and Phase 12--14 reports were read in full.
  Phase 9 and Phase 11 evidence was then consulted to avoid repeating retained
  accumulator persistence, GQA reuse, tile, and global-control experiments.
- Model: `Ministral-3-3B-Instruct-2512-Q4_K_M.gguf`, 2,147,023,008 bytes,
  SHA-256 `9ed150d4367e68df0ac8e1540f6ddc65b42d0ee26378329d1ecbca60f93fc5f8`.
- Device: RTX 3060 12 GiB (`0x2487`), driver 595.84, Vulkan device API 1.4.329,
  28 SMs, 7,501 MHz memory clock, and 170 W board limit.
- Runtime: release pure Vulkan, context 32,768, F16 KV, greedy sampling,
  `GRAPH_HORIZON_PREFILL_PREDECODE_MLP=1`, and exact prompts made from `" a"`
  repeated `tokens - 3` times.
- Diagnostics-free short rows use two warm-ups; long rows use one. Repetitions
  are 15/10/5/3/3/3 for 128/512/2K/8K/16K/28K.
- The profiler uses one independent cold request plus a separate warm-up and
  three measured requests. Stable time is `(warm-up+3 total - cold) / 3`.

The invariant is Phase 12 FP16-native gate/up, canonical compressed decode,
exact causal dense attention, F16 Q/K/V and output, FP32 QK, online-softmax and
AV accumulation, current GQA mapping, unchanged weights, unchanged tolerances,
and no global score matrix. A candidate may proceed only with at least 5%
credible whole-request gain after its own traffic and synchronization.

## Fresh diagnostics-free baseline

The first pre-code 28K denominator is 45,892.68 ms with 20.98 ms SD and 0.05%
CV. A qualifying architecture must therefore remove at least 2,294.63 ms.
After all screening, a temporary post-measurement sample logger supplied the
missing medians; it was removed completely before final verification.

| Context | Mean TTFT | Median | SD | CV | Prompt tok/s |
|---:|---:|---:|---:|---:|---:|
| 128 | 97.02 ms | 97.015 ms | 0.32 ms | 0.33% | 1,319.29 |
| 512 | 368.09 ms | 368.157 ms | 0.47 ms | 0.13% | 1,390.96 |
| 2K | 1,559.76 ms | 1,558.545 ms | 3.52 ms | 0.23% | 1,313.02 |
| 8K | 7,912.28 ms | 7,908.608 ms | 20.40 ms | 0.26% | 1,035.36 |
| 16K | 20,405.28 ms | 20,410.598 ms | 36.12 ms | 0.18% | 802.93 |
| 28K | **45,861.16 ms** | **45,870.049 ms** | **31.43 ms** | **0.07%** | **610.54** |

The final 28K mean differs from the first pre-code mean by -31.52 ms / -0.07%,
well inside session drift. No production source changed between them.

## Fresh profiled baseline

Diagnostics-free TTFT and profiler builds are separate evidence streams and
are not subtracted from one another.

| Context | GPU prefill | CPU fence wall | Attention | Attention share |
|---:|---:|---:|---:|---:|
| 128 | 108.008 ms | 108.125 ms | 1.384 ms | 1.28% |
| 512 | 368.474 ms | 368.725 ms | 9.589 ms | 2.60% |
| 2K | 1,572.228 ms | 1,573.206 ms | 144.681 ms | 9.20% |
| 8K | 7,983.480 ms | 7,988.352 ms | 2,257.036 ms | 28.27% |
| 16K | 20,508.888 ms | 20,516.728 ms | 9,045.433 ms | 44.10% |
| 28K | **46,092.240 ms** | **46,106.894 ms** | **26,435.802 ms** | **57.35%** |

At 28K, CPU record/submit/fence-wait/descriptor times are approximately
81.340/5.785/46,106.894/6.165 ms per stable request. GPU gaps above 50 us are
1.406 ms/request. CPU control and GPU idle are not architectural opportunities.

Active telemetry is:

| Context | Samples | GPU util. | Core | Memory | Power | Temperature | Sampled device VRAM used |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 8K | 67 | 98.69% | 1,925 MHz | 7,501 MHz | 166.16 W | 65.54 C | 8,536 MiB |
| 16K | 165 | 99.41% | 1,914 MHz | 7,501 MHz | 168.25 W | 70.66 C | 8,610 MiB |
| 28K | 372 | 99.54% | 1,908 MHz | 7,482 MHz | 168.73 W | 72.33 C | 8,614 MiB |

The device-level memory samples include the profiling build, display, and other
process overhead and are not substituted for Phase 12's 8,260 MiB
diagnostics-free application result. No row is idle, thermally throttled, or
capacity limited.

## 28K macro attribution

| Component | GPU time | GPU prefill |
|---|---:|---:|
| Exact attention | 26,435.802 ms | 57.35% |
| Seven recurrent matmuls | 18,713.161 ms | 40.60% |
| Other | 943.277 ms | 2.05% |
| **Total** | **46,092.240 ms** | **100%** |

The matmul rows are Q/K/V/O 2,451.453/679.347/720.256/2,542.282 ms,
gate/up/down 3,271.489/3,267.431/5,780.903 ms. Phase 12 FP16-native gate/up is
therefore present and unchanged.

## Updated attention attribution

The device exposes timestamps for the fused kernel but no physical-byte,
instruction-sampling, cache, or in-kernel timestamp counters. The Phase 10
downstream-live causal chain remains source-identical, so its measured shares
are normalized to the fresh direct attention total. This attributes 99.99%; the
2.644 ms remainder is share-rounding and is included in `other`.

| Causal region | Seconds | Attention | Whole GPU request |
|---|---:|---:|---:|
| Q/K tensor load, staging, and QK Matrix2 | 9.789 | 37.03% | 21.24% |
| Online max/exp/sum and state rescale | 3.542 | 13.40% | 7.69% |
| Probability/V load, staging, and AV Matrix2 | 9.253 | 35.00% | 20.07% |
| Accumulator rescale, address/shared traffic, synchronization, epilogue | 3.852 | 14.57% | 8.36% |
| **Attention** | **26.436** | **100%** | **57.35%** |

The fused causal regions cannot truthfully be decomposed into additive seconds
for Q load versus K load versus MMA, or max versus exponential versus sum:
those operations overlap in one timestamp and an ablation changes downstream
issue. Their separate physical work is nevertheless exact:

| Subregion | Static/dynamic work | Standalone seconds |
|---|---:|---:|
| Q tensor loading | 1.308 TB logical, eight tensor loads/visit | fused; not separately observable |
| K tensor loading | 2.615 TB logical, eight transposed tensor loads/visit | fused; not separately observable |
| QK Matrix2 | 83.684 executed TFLOP, 1,276,913,664 MMA operations | fused in 9.789 s region |
| Tile/running maximum | 326.890 billion score lanes plus 5.108 billion tile states | fused in 3.542 s region |
| Exponential | one per score plus one alpha per query/tile | fused in 3.542 s region |
| Sum | one probability accumulation per score | fused in 3.542 s region |
| Online state/rescale | FP32 `(m,l,alpha)` and output rescale per tile | fused across softmax/other |
| V tensor loading | 2.615 TB logical, one tensor load/visit | fused; not separately observable |
| AV Matrix2 | 83.684 executed TFLOP, 159,614,208 MMA operations | fused in 9.253 s region |
| Shared score/probability traffic | 5.230 TB/request | fused across all regions |
| Final FP32 accumulator round-trip | 23.855 GB/request | fused in other |
| Required internal barriers | 320,684,416 dynamic executions | 431.863 ms acquired normalized causal bucket |
| Geometry/address/conversion/epilogue | remaining fused work | 3,419.834 ms causal remainder |

The current optimized SPIR-V has three tensor loads, one shared cooperative
load, two cooperative stores, two Matrix2 multiply-add sites, two cooperative
per-element sites, four static barriers, 35 loads, 72 stores, 30 access chains,
and two exponential sites. The attribution does not fabricate physical DRAM or
per-operation seconds that the hardware cannot expose.

## Physical model

At 28K there are 875 Q tiles per head, 728,000 Q-tile/head/layer workgroups,
and 159,614,208 workgroup-KV visits. A workgroup processes 219.25 KV64 tiles on
average and up to 438. The exact mathematical causal pair count is lower than
the tiled execution count only by 0.225%.

| Quantity | Mathematical minimum | Current exact algorithm | Decomposition cost |
|---|---:|---:|---:|
| QK FLOPs | 83.496 TFLOP | 83.684 TFLOP | 0.188 TFLOP causal padding |
| AV FLOPs | 83.496 TFLOP | 83.684 TFLOP | 0.188 TFLOP causal padding |
| Q bytes | 5.964 GB unique | 1.308 TB logical | per-KV-tile Q reload |
| K bytes | 1.491 GB unique | 2.615 TB logical | per-Q-tile traversal, cacheable |
| V bytes | 1.491 GB unique | 2.615 TB logical | per-Q-tile traversal, cacheable |
| Output bytes | 5.964 GB | 5.964 GB | none |
| Global intermediates | 0 | 0 | none in retained one-pass algorithm |
| Shared traffic | nonzero implementation need | 5.254 TB logical | scores, probabilities, final accumulator |
| Internal barriers | nonzero implementation need | 320.684 million | two/visit plus Q-tile setup/finalization |
| QK Matrix2 operations | implementation-dependent | 1,276,913,664 | eight per KV visit |
| AV Matrix2 operations | implementation-dependent | 159,614,208 | one per KV visit |

The 167.0 TFLOP useful QK+AV work is mathematically unavoidable for dense exact
attention. The 167.368 executed TFLOP is unavoidable in the current Q32/KV64
tiling. Repeated logical loads, shared traffic, barriers, and serial `(m,l,O)`
ownership are consequences of the current decomposition, but Phase 10's cache
probes prove that logical byte removal is not equal to elapsed-time removal.

## Resource and parallelism model

| Resource | Current | Next useful threshold |
|---|---:|---:|
| Threads / subgroups | 256 / 8 | unchanged by split-K |
| Registers | 124/thread | at most 85.33 for three WG/SM |
| Driver shared | 38,272 B/WG | at most 34,133 B for three WG/SM |
| Stack | 0 reported | — |
| Resident workgroups | 2/SM | 3/SM requires both thresholds |
| Resident warps / modeled occupancy | 16 / 33.3% | split-K remains 16 / 33.3% |
| Ready workgroups / full dispatch | 256 | 56 resident slots across 28 SMs |

An isolated workgroup has a long online-softmax dependency chain. The request
does not: a full dispatch already has 4.57 resident waves ready, and the GPU
scheduler interleaves two resource-limited workgroups per SM. Split-K shortens
each chain while multiplying the number of equal-resource chains; total work
and resident parallelism stay constant.

The fused kernel does not expose an isolated serial-scan timestamp. Its direct
request-level scan cost is the 26.436 s attention timestamp; the owned chain is
219.25 KV tiles on average and 438 maximum. Ideal isolated chain lengths are
109.625/54.812/27.406 tiles for splits 2/4/8, while the request-level wave model
below accounts for the already-available parallel workgroups.

## Current exact floor

The practical floor uses acquired evidence rather than the unattainable 61.1
TFLOP/s dense peak: half of the K-hot ceiling for plausible reuse and half of
the valid-barrier bucket, with AV and softmax unchanged because V-hot and prior
softmax rewrites did not improve.

| Region | Current | Practical floor | Remaining local headroom |
|---|---:|---:|---:|
| QK/load | 9.789 s | 9.505 s | 0.284 s |
| Softmax/state | 3.542 s | 3.542 s | 0 |
| AV/load | 9.253 s | 9.253 s | 0 |
| Other | 3.852 s | 3.636 s | 0.216 s |
| **Exact current architecture** | **26.436 s** | **25.936 s** | **about 0.500 s** |

`current / floor` is about 1.019. Remaining local exact headroom is only 1.09%
of the whole GPU request, consistent with Phase 10/11. Phase 15 does not reopen
that local tuning.

## Architectural opportunity table

This table was completed before any prototype code.

| Candidate | Exact? | Affected/removable | Added traffic/state | Predicted 28K TTFT | Numeric risk | Complexity | Decision |
|---|---|---:|---|---:|---|---|---|
| Cross-WG split-K + exact merge | yes | at most 1.337 s after traffic at split 8, before merge compute | 193.823 GB/request, 34.079 MB temporary | at most -2.91% | low reduction-order drift | very high | reject below 5% |
| In-WG hierarchical partitions | yes | no work removed; possible dependency overlap | +16,640 B state per extra partition/WG | negative after 2->1 WG residency | low | high | reject resource cliff |
| Persistent Q/sequence ownership | yes | no positive O(N) setup term; dense O(N²) remains | more live Q state or same traversal | below 0.5% | none | high | reject |
| GQA/head scheduling only | yes | K-hot ceiling 0.574 s; V has no positive ceiling | none if scheduling-only | at most -1.25% | none | medium | reject |
| Explicit multi-head K/V sharing | yes | same byte ceiling as above | multiple FP32 `(m,l,O)` states; prior one-WG cliff | negative/under 5% | none | very high | reject acquired architecture |
| Exact state packing/layout | yes | 384 B scalar shared state; no padding | none | no residency change, immaterial | none | low | reject |
| Two-pass exact QK then AV | yes | no dense work removed | >=653.8 GB probabilities or repeated 83.7 TFLOP QK | negative | order drift only | very high | reject |
| FP16 score shared staging, FP32 state/AV | no | optimistic 1.204 s attention | removes 1.961 TB shared traffic | at most -2.62% | medium at long context | medium | reject below 5%, no quality run |
| INT8-like probability staging | no | optimistic <=0.402 s attention | removes <=0.654 TB shared traffic plus quantization | at most -0.88% | high | high | reject |
| Narrow FP32 AV accumulator | no | potentially larger but violates first numerical-iteration guard | lower register state, unknown throughput | not eligible | high cumulative error | medium | do not implement |

The first exact candidates fail on algorithm/dataflow economics, not cosmetic
tile choices. The numerical candidates keep running max, running sum, and AV
accumulation FP32 as required; none clears the coding gate.

## Exact split-K model

For one partition and query row, exact merge state is
`m_i` (FP32), `l_i` (FP32), and 128 FP32 weighted-output values. One Q32
workgroup partition therefore writes 16,640 B. The mathematically correct merge
uses `m=max(m_i)`, rescales each `l_i` and `o_i` by `exp(m_i-m)`, sums them,
and normalizes the output. No global score matrix is needed.

| Split | Avg KV tiles/partition | Full-dispatch wave units | Temporary per active dispatch | Request state write+read | Favorable traffic floor | Best attention | Best TTFT gain |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 219.250 | 5.000 | 0 | 0 | 0 | 26.436 s | 0% |
| 2 | 109.625 | 5.000 | 8.520 MB | 48.456 GB | 0.162 s | >=26.598 s | regression |
| 4 | 54.812 | 4.750 | 17.039 MB | 96.911 GB | 0.323 s | >=25.437 s | <=2.18% |
| 8 | 27.406 | 4.625 | 34.079 MB | 193.823 GB | 0.646 s | >=25.099 s | <=2.91% |

Wave units are `ceil(256*split/56)/split`, before the small 96-workgroup final
dispatch. The traffic floor assumes a favorable 300 GB/s and zero merge
compute, zero extra barriers, zero dispatch cost, and perfect partition
balance. It is therefore an upper bound, not a prediction that split 8 would
actually save 2.91%. The ideal pre-merge split-8 ceiling is 7.5% attention or
4.32% whole TTFT, already below the mandatory 5% gate.

The split would keep QK/AV FLOPs and Matrix2 utilization, but add 2,860 merge
dispatches/request, global partial-state writes and reads, final max/sum/output
rescaling, and new graph barriers. Atomics are unnecessary and were not
considered because they would add contention without removing state traffic.

## Prototype A

Architecture: exact cross-WG split-K / sequence-parallel online attention.

Decision: **REJECT BEFORE CODE**. The ideal whole-request ceiling is below 5%
before merge, and the favorable traffic-only model reduces it below 3%.
Consequently there is no shader, buffer, dispatch, threshold, register/shared
result, performance result, or correctness delta to report. This is the
explicit split-K upper-bound stop rule, not a missing implementation.

## Prototype B

Architecture: materially different in-WG hierarchical or persistent attention.

Decision: **REJECT BEFORE CODE**. An additional exact FP32 Q32x128 output state
adds at least 16 values/thread and crosses the 128-register two-WG threshold;
spilling it to shared adds 16 KiB/WG and also crosses to one WG. Persistent
ownership has no positive setup bound in the measured quadratic curve. No
second implementation is economically authorized.

## Performance and final state

No candidate is retained, so baseline and final production are byte-identical.

| Context | Baseline/final TTFT | Delta |
|---:|---:|---:|
| 128 | 97.02 ms | 0% by construction |
| 512 | 368.09 ms | 0% by construction |
| 2K | 1,559.76 ms | 0% by construction |
| 8K | 7,912.28 ms | 0% by construction |
| 16K | 20,405.28 ms | 0% by construction |
| 28K | 45,861.16 ms | 0% by construction |

| Context | Baseline/final attention | Delta |
|---:|---:|---:|
| 2K | 144.681 ms | 0% by construction |
| 8K | 2,257.036 ms | 0% by construction |
| 16K | 9,045.433 ms | 0% by construction |
| 28K | 26,435.802 ms | 0% by construction |

Final 28K breakdown is QK/load 9.789 s, softmax/state 3.542 s, AV/load
9.253 s, attention other 3.852 s, attention total 26.436 s, recurrent matmul
18.713 s, and other GPU work 0.943 s. There is no merge category because no
split candidate passed the pre-code gate.

## Decode, short, correctness, and memory guards

The diagnostics-free 32-token source-identical decode guard is:

| Prompt history | Decode throughput | SD | CV | Final delta |
|---:|---:|---:|---:|---:|
| 128 | 47.25 tok/s | 0.04 | 0.08% | 0% by construction |
| 2K | 46.49 tok/s | 0.03 | 0.07% | 0% by construction |
| 28K | 29.23 tok/s | 0.02 | 0.07% | 0% by construction |

Short prefill is 97.02 ms at 128 and unchanged. Boundary-context and split-merge
correctness tests are not applicable because no routing threshold, split path,
partial state, or merge exists in the final source. The established Phase 12
attention, layer, logits, F16/INT8 KV, CPU-oracle, GQA, odd/tail, and greedy
sequence `2757,10637,2479,1636` evidence remains authoritative because all
production and test source is byte-identical to `6cb170f`. No tolerance or
precision contract changed.

Final weight memory remains 1.991741 GiB canonical plus 2.742188 GiB FP16-native
gate/up, 4.733929 GiB total resident weights. No partial-state buffer or new
allocation exists. Phase 12's diagnostics-free 28K application result remains
8,260 MiB; profiling telemetry is reported separately above.

## Experiment registry

| ID | Algorithm | Exact/numeric | Attention delta | TTFT delta | Quality | Decision |
|---|---|---|---:|---:|---|---|
| P15-SPLIT2 | cross-WG KV split 2 + exact merge | exact | modeled regression | modeled regression | not run; exact formula | reject before code |
| P15-SPLIT4 | cross-WG KV split 4 + exact merge | exact | <=-3.78% after traffic-only floor | <=-2.18% | not run; exact formula | reject below gate |
| P15-SPLIT8 | cross-WG KV split 8 + exact merge | exact | <=-5.06% after traffic-only floor | <=-2.91% | not run; exact formula | reject below gate |
| P15-HIER | multiple local FP32 states | exact | negative resource model | negative | not run | reject occupancy cliff |
| P15-PERSIST | persistent Q/sequence blocks | exact | no positive measured setup bound | <0.5% | unchanged arithmetic | reject |
| P15-GQA-SCHED | reorder heads for cache locality | exact | <=-2.17% attention | <=-1.25% | unchanged arithmetic | reject |
| P15-STATE-PACK | compact exact scalar state | exact | immaterial | immaterial | exact | reject |
| P15-SCORE-F16 | FP16 shared scores, FP32 softmax/AV | numeric | optimistic <=-4.56% attention | <=-2.62% | not run | reject below coding gate |
| P15-PROB-I8 | compact probability staging | numeric | optimistic <=-1.52% attention | <=-0.88% | not run | reject below coding gate |

No prototype or diagnostic source survives, and no numerical quality gate is
claimed for candidates that were correctly rejected before code.

## Sparse and approximate Phase 16 analysis

Dense exact attention is closed, so the following is analysis only. Window
rows use exact causal-pair arithmetic and optimistically scale all attention
time with retained pairs; implementation overhead and non-scaling work make
actual gains smaller.

| Algorithm | Attention work reduction | Memory reduction | Predicted 28K TTFT | Quality/model risk | Complexity |
|---|---:|---:|---:|---|---|
| Exact 16K sliding window semantics | 17.21% pairs | KV can be capped near 16K | about 41.34 s / -9.91% ideal | changes Ministral dense-attention semantics | medium |
| Exact 8K sliding window semantics | 50.05% pairs | KV can be capped near 8K | about 32.66 s / -28.83% ideal | high long-range recall risk | medium |
| Exact 4K sliding window semantics | 72.88% pairs | KV can be capped near 4K | about 26.63 s / -41.98% ideal | very high quality risk | medium |
| Hybrid local 8K + sparse global tokens | roughly 45--50% | KV retained unless eviction is defined | roughly 33--34 s ideal | global-token policy is model-semantic | high |
| Block-sparse learned/static pattern | pattern-dependent, plausibly 50--75% | metadata plus optional KV reduction | roughly 26--34 s ideal | model was not trained/qualified for pattern | very high |
| KV/token reduction outside a recent window | roughly 25--50% | proportional compressed history | roughly 33--39 s ideal | selection errors accumulate with context | very high |
| Approximate low-rank/kernel attention | potentially asymptotic | potentially large | unproven | incompatible algorithm and retraining risk | extreme |

These rows do not authorize semantic changes. A Phase 16 mission must choose a
model-compatible approximation contract and quality suite before implementation.

## Final decision

**STOP — no Phase 15 production candidate is retained.**

Stop conditions reached:

1. split-K's ideal upper bound is below 5% whole request;
2. partial-state traffic and merge reduce its favorable ceiling below 3%;
3. hierarchical, persistent, GQA scheduling, state packing, and two-pass exact
   decompositions are materially different and also fail their gates;
4. no exact algorithmic candidate clears 5%;
5. allowed first-iteration numerical representations also fail the performance
   gate before quality risk can be justified;
6. the next credible jump is sparse/windowed/approximate or model-semantic.

### Final bottleneck

Exact dense attention remains 26.436 s / 57.35% of 28K GPU prefill. Recurrent
matmul is 18.713 s / 40.60%; everything else is 0.943 s / 2.05%.

### Dense exact-attention status

Closed on the qualified RTX 3060 under the current Ministral representation.
The remaining gap to dense hardware peak is real, but no supported decomposition
raises residency, removes work, or amortizes a new merge sufficiently to reach
5% end-to-end. This conclusion does not claim mathematical or hardware
optimality on other GPUs.

### Recommended Phase 16

Choose semantics explicitly before code. The lowest-risk next question is
whether the model can tolerate a 16K long-context window or a hybrid 8K-local
plus sparse-global policy under multi-prompt 28K logits, top-k, greedy-sequence,
retrieval, and long-range instruction tests. Do not implement approximation
until that quality contract and model compatibility are approved.

## Verification and repository state

- production, workspace, support, examples, manifest, and build-script source
  are byte-identical to `6cb170f`;
- the temporary median diagnostic is absent;
- no split shader, partial buffer, merge pipeline, routing threshold, flag,
  test hook, dependency, or numerical candidate exists;
- `cargo fmt --all -- --check` passes;
- workspace/all-target Clippy with warnings denied passes for pure Vulkan and
  Vulkan-hybrid;
- the release Vulkan-profile engine library reports 154 passed, zero failed,
  and four intentional authenticated/long-context ignores;
- the family-agnostic integration suite reports four passed and one intentional
  authenticated ignore; `docs_contract` has the pre-existing checkout failure
  because root `VALIDATION.md` is absent, before it reaches link validation;
- the new documentation-index target exists and `git diff --check` passes;
- the final worktree is clean and nothing is pushed.

There is no production implementation commit because no candidate passed the
pre-code gate. The only Phase 15 commit is evidence/documentation.
