# CUDA Amdahl campaign 2026-09-05

## Recovery state

- Campaign ID: `cuda-amdahl-20260905` (new campaign; historical campaign completed).
- Branch: `perf/cuda-amdahl-20260905`.
- Immutable start and current retained runtime: `62384895acfde94cf28fded1294ad860daabf2ff`.
- State: running, baseline and correctness complete; timeline acquisition; no production changes.
- Attempts: 0/10 minimum. No overall deadline; each A/B comparison has a two-hour limit.
- Persistent private evidence: `target/cuda-amdahl-20260905/`.
- Current-campaign retained commits: none.

Recovery checkpoint: baseline tool session `6964` completed successfully; all
three raw records and metadata are present. Correctness runner session `66222`
completed local tests; separate parity session `89689` passed after correcting
the private model link (details below). Collect one diagnostic
32-token, zero-warmup, one-repetition trace for each regime using the copied
baseline executable, `LD_PRELOAD` set to the absolute private `timeline.so`,
stdout as a diagnostic benchmark record and stderr as private CSV. Use
`analysis.py` to summarize it; reject dropped records and measure overhead
against matching uninstrumented diagnostic rows if the baseline aggregates do
not bound it. No production candidate has begun.

## Invariants and procedure

Keep artifact bytes, backend, placement, context, KV, sampling, build settings,
prompts and generation counts fixed. Select the objective from measured fixed-work
critical paths across short/medium/long. Preserve public interfaces, formats,
validation and lifecycle contracts. Main risks are numeric drift and moving cost
between prefill and decode; isolated candidates must pass correctness before A/B.
No machine settings or dependencies change. One GPU measurement at a time.

Keep requires >=5% objective improvement, objective CV <=5% in both records,
and <=5% regression in every control. A 3–5% gain is interesting evidence only;
less than 3% rejects. Unstable objective permits one complete A/B rerun, then
`not_verified`. Both other regimes are controls before retaining a candidate.
Numeric candidates use existing local CUDA error bounds and pinned teacher-forced
top-two parity; exact candidates additionally need a declared identity gate.
KV changes require payload/layout/quality tests and f16/int8 parity.

Completion requires ten distinct implemented attempts, a terminal pool, fresh
discovery, and evidence closing every material remaining hotspot. Historical
attempts and unchanged parameter retries never count.

## Tuple fixed before measurement

| Field | Value |
|---|---|
| Host | Linux x86_64; compute capability 8.6; 12 GiB VRAM; ordinal 0 |
| Driver / compiler | 580.173.02 / CUDA Toolkit 12.4.131 |
| Rust / Cargo | 1.97.1 / 1.97.1 |
| Backend / placement | `cuda`, all GPU |
| Build | `--locked --profile fast --no-default-features --features cuda` |
| Model | `3b-instruct`, Q4_K_M, 2147023008 bytes |
| SHA-256 | `9ed150d4367e68df0ac8e1540f6ddc65b42d0ee26378329d1ecbca60f93fc5f8` |
| Context / KV / sampling | 4096 / f16 / greedy |
| Workloads | `benchmark` repeated 124 / 1020 / 3580 times, separated by spaces, final period |
| Calibration | expected 128 / 1024 / 3584; recheck with current engine tokenizer |
| Requested completion / repetitions | 32 / one warm-up plus three measured |
| Oracle | CPU llama.cpp 9973 (`13f2b28b0`); local library path required |

Authentication passed `stat -c %s` and `sha256sum` on the local 3B artifact.
Desktop graphics clients remain present; capture private device identity and
telemetry and reject incomparable/contended measurements. Nsight Systems is
absent; installed CUPTI permits an equivalent temporary activity timeline.
Instrumentation is diagnostic and must have its overhead measured.

## Historical evidence (inherited, not current attempts)

Read `cuda-amdahl-optimization.md` and reachable CUDA history, including initial
implementation, typed-view/lifecycle corrections and hybrid qualification.
Inherited commits: `5d52db5` parallel matmul; `a9f6466` 128-thread matmul;
`34c9525` decode-isolated four-token tiling; `0056aa3` parallel prefill attention.
Historical rejected premises: shared tiled decode (decode regression), Q4
metadata shuffle (regression), warp-first reduction (insufficient gain), and
eight-token tiling (+1.71%). Do not retry without a changed measured premise.
The old long row timed out before the retained improvements; measure it anew.
Old release-profile results are not the current fast-profile A/B baseline.

## Active candidate pool

Initial pool from the current unprofiled screen and short/medium CUPTI timelines:

| ID | Premise and measured owner | Regime / objective | State | Cost / risk |
|---|---|---|---|---|
| C01 | Spread decode attention across head dimensions and blocks; medium decode attention owns 8352.759 ms and launches only one block | Medium model decode | ready, selected for implementation | Small: reuse existing parallel attention body; score reduction reorders |
| C02 | Parallel RMSNorm width; short prefill+decode normalization owns 406.330 ms with one block and serial width loops | Short model decode | ready | Small kernel and launch change; numeric reduction risk |
| C03 | Specialize matmul weight format outside K loops; short batched+single+logits own 3327.535 ms | Short prompt throughput, decode control | deferred until C01 | Moderate; eliminate per-element format branching without new formats; exact gate required |
| C04 | Make four batched accumulators statically indexed and unroll valid-token work; dominant batched matmul retains runtime inner bounds | Short prompt throughput | deferred until C03 | Small; same four-token tile and accumulation order, exact gate |
| C05 | Warp-level score reduction for prefill attention; medium owns 2302.472 ms with block barriers at every context token | Long prompt throughput | deferred until long timeline | Moderate numeric risk; covers a different operation from historical matmul shuffle rejection |
| C06 | Multiple output rows per matmul block, each owned by a warp, reusing input across outputs and reducing barriers | Short prompt throughput | deferred until current matmul specialization measured | Moderate numeric risk and occupancy uncertainty; different mapping, not a historical block-width rerun |
| C07 | Parallelize exact total-order argmax; short decode sampling owns 107.860 ms in one thread | Short model decode | deferred until C01/C02 refresh | Small; current full-request ideal ceiling only 2.16%, may become material after larger removals |

The pool is intentionally not ten guesses. Replenish from each decision/profile
until ten distinct implemented attempts or an explicit incomplete stop. C01/C02
remove serial work and expose parallelism; C03/C04 reduce repeated work; C06
changes reuse/mapping; C05 reduces synchronization. Transfers and host submission
are not initial candidates: measured GPU gaps are 0.1–0.4% of the phase, and
device copies inside generation are small. Allocating a new weight/KV layout
without evidence of a capacity or traffic limit is deferred discovery, not an
assumed optimization. Existing formats and all-GPU placement remain fixed.

Preliminary Amdahl estimates use fractions of total fixed-work GPU timeline
through the final sample (prefill plus decode, no overlapping CPU time added).
Phase-local fractions are recorded separately to evaluate the named objective.

| ID | Full-request p | Phase p | Credible s | Added o | Predicted global gain | Ideal full-request ceiling | Expected saved time | Uncertainty |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| C01 medium | 0.2885 | 0.8181 | 5 | 0.001 | 29.8% | 40.5% | 6.65 s | Launch is one block; 32 head blocks expose substantial parallelism, but per-token barriers remain |
| C02 short | 0.0796 | 0.0992 decode | 8 | 0.001 | 7.4% | 8.65% | 0.35 s | Width-parallel reductions remove serial work; finite gain discounted for launch floor |
| C03 short | 0.6515 | 0.9117 prefill | 1.2 | 0.001 | 12.1% | 187.0% | 0.55 s | Uniform compiler branching may already be cheap; inspect PTX before implementation |
| C04 short | 0.3733 | 0.9117 prefill | 1.25 | 0.001 | 7.95% | 59.6% | 0.38 s | Requires proof the compiler retained dynamic accumulator/bound work |
| C05 long | 0.2477 | 0.3357 prefill | 2 | 0.002 | 13.9% | 32.9% | 14.23 s | Per-context block barriers dominate this serial score schedule; check register pressure |
| C06 short | 0.6515 | 0.9117 prefill | 1.25 | 0.003 | 14.6% | 187.0% | 0.65 s | High implementation uncertainty; no hardware-counter bandwidth claim |
| C07 short | 0.0211 | 0.0358 decode | 8 | 0.0002 | 1.86% | 2.16% | 0.09 s | Not viable at this checkpoint; awaits measured bottleneck shift |

C01 is selected by the largest credible whole-request effect,
not by comparing token rates with milliseconds. Its local objective is model
decode throughput on medium; prompt throughput, TTFT, public-delta throughput,
and short/long are controls. Keep/stability/regression thresholds stay unchanged.
Gate: all baseline commands, the new long-context f16/int8 numeric test and
pinned f16 real-model parity. The candidate does not alter KV storage; score
reduction is covered by the established local numeric bound, not claimed exact.
Planned change: `shaders/attention.cuh` (category K, ~100–160 productive lines)
and `kernels/attention/decode.rs` (~40 productive lines). No new runtime file.
Preserve one output per query head, cache bounds, causal online-softmax order,
buffer formats and dispatch ABI. Main risk: changed dot reduction rounding.

## Measurements and decisions

Calibration at the immutable runtime produced exactly 128 / 1024 / 3584 tokens.
Prompt SHA-256 values, respectively:
`24bbd008b03e025caf3b6aece0a67c380d27303a33e9d1282fedc1ff1c27a2b8`,
`0bdbe2964b9c8129d215ab61bfbf447b1e7f111cd95167096fe2b3f8568343f8`,
`9bd7645ed8e09684e7b84fc90e735e106595c414cf8ff1e0cc9e5bf14a7be5fc`.

| Baseline regime | Prompt tok/s (CV) | TTFT ms (CV) | Model decode tok/s (CV) | Public deltas/s (CV) | Complete wall s |
|---|---:|---:|---:|---:|---:|
| Short | 60.40 (0.0152) | 2119.50 (0.0153) | 10.31 (0.0433) | 10.00 (0.0433) | 23.03 |
| Medium | 54.79 (0.0010) | 18690.88 (0.0010) | 3.14 (0.0016) | 2.94 (0.0016) | 118.89 |
| Long | 41.71 (0.0001) | 85919.58 (0.0001) | 1.04 (0.0008) | 0.98 (0.0008) | 467.59 |

Short emits 32 completion tokens / 32 deltas, medium 32 / 31. Fixed-work
approximation from aggregate means is TTFT + (decoded_tokens-1)/decode_tps:
5.2195 s short (40.6% TTFT), 28.8950 s medium (64.7% TTFT). This measures
through the last public delta without counting first sampling twice. These inverse means
are descriptive approximations, not an exact mean elapsed time or isolated
prefill attribution. Long emits 32 completion tokens and 31 public deltas:
approximately 116.532 s through the final delta, of which 73.7% is TTFT.
The complete long acquisition passed without timeout, so it is a runnable
control for every candidate. Timeline attribution must settle priorities.

Exact baseline commands (repository root; model symlink points to authenticated
bytes, and the copied executable preserves baseline A across later builds):

```sh
cargo build --locked --profile fast --no-default-features --features cuda --example bench --example tokenize
python3 target/cuda-amdahl-20260905/run.py calibrate baseline
cp target/fast/examples/bench target/cuda-amdahl-20260905/baseline-bench
python3 target/cuda-amdahl-20260905/run.py bench baseline target/cuda-amdahl-20260905/baseline-bench short medium long
```

The private runner records exact expanded commands, elapsed seconds, exit codes,
raw benchmark stdout, full pre/post device state and one-second telemetry.
Temporary tools are `run.py` (~80 productive lines, fixed workload orchestration),
`timeline.c` (~70, CUPTI activity callbacks) and `analysis.py` (~60, interval
union and phase summary). They introduce no production dependency or interface.
The trace uses installed CUDA 12.4 structs and concurrent-kernel activity,
correlation IDs and dropped-record checks following the current official
[CUPTI activity documentation](https://docs.nvidia.com/cupti/main/main.html).
GPU intervals are unioned; API time is not added to overlapping GPU time.

Local correctness preflight passed; timeline acquisition is running.
No speedup or largest hotspot is yet claimed.
RUSTFLAGS, CARGO_ENCODED_RUSTFLAGS, CUDA_VISIBLE_DEVICES, CUDA_LAUNCH_BLOCKING,
OMP_NUM_THREADS and RAYON_NUM_THREADS were unset. Short and medium pre/post
device counters show no increment in software power capping or thermal slowdown.

Before numeric experiments, added one focused existing-file CUDA test for dense
attention scores with f16/int8 KV at `(head_dim, context)` values `(3,17)`,
`(128,3584)`, `(256,129)`, two query heads sharing one KV head, and three causal
queries near the end of cache. It compares stored-value scalar references and
prefill/decode using the unchanged canonical local numeric bound. This is
test-only gate support, no productive runtime lines or new production file.

Baseline gate results: formatting and CUDA workspace check passed; CPU workspace
170 root and 164 engine unit tests plus integration suites passed; 11 selected
error-matrix tests and all 33 CUDA tests passed. The new long-history test passes
on the untouched production runtime. Pinned real-model f16 parity passed with
all 16 recorded local top-1 token IDs equal to the oracle.

Initial parity invocation exited before inference with a byte-count mismatch:
`artifact_size` uses `stat` without dereferencing symbolic links. The original
artifact itself was already authenticated. Replaced only the campaign-owned
symlink with a hard link to the same file, then reran the unchanged canonical
parity command with the required local `LD_LIBRARY_PATH`. Log:
`baseline-parity-hardlink.log`. No oracle, threshold, runtime or artifact bytes
changed. This setup correction is not an optimization attempt.

### Initial timeline and limiter analysis

`python3 target/cuda-amdahl-20260905/run.py trace baseline-timeline target/cuda-amdahl-20260905/baseline-bench short medium long`
completed with zero dropped records for all three rows. Before this, two private
collector attempts failed at teardown: an uninitialized CUPTI record iterator
caused `cuptiActivityGetNextRecord` to fault. A small GDB diagnostic identified
the cursor; initialized it to NULL as required by the installed API. A trial
driver pin did not help and was removed. These failures neither changed
production nor count as optimization attempts; their timings are not evidence.

| Regime | Prefill through sampling ms | Batched matmul ms | Prefill attention ms | Decode ms | Decode attention ms | GPU gaps across both phases ms |
|---|---:|---:|---:|---:|---:|---:|
| Short | 2091.465 | 1906.845 | 35.481 | 3016.031 | 1157.333 | 18.880 |
| Medium | 18741.404 | 15309.371 | 2302.472 | 10209.920 | 8352.759 | 68.149 |
| Long | 86179.803 | 53369.426 | 28932.070 | 30631.278 | 28776.008 | 186.142 |

Instrumented public TTFT versus unprofiled means: short 2093.41 ms (-1.23%),
medium 18743.64 ms (+0.28%), long about 86182 ms (+0.31%). These diagnostic
single repetitions bound perturbation near baseline variation; they are not
A/B performance verdicts or confidence intervals. GPU activity timestamps use
one stream and nonoverlapping interval unions. Cold load/copies precede the
first generation kernel and are excluded, matching the public generation target.

Decode attention launches `(grid=1, block=256)` while only 32 threads own query
heads. The PTX declares a 1024-byte local accumulator per thread; CUPTI reports
40 registers and zero local bytes, so do not interpret the latter as proof of
no local accesses. Source and PTX show serial dimension/context loops and local
loads/stores. This is a parallelism/latency limiter, not demonstrated bandwidth
saturation. C01 replaces that serial ownership with 32 blocks of 128 threads,
one dimension per thread, using the existing bounded parallel reduction. It
removes the local accumulator array and exposes independent head work without
changing cache traffic, layout, allocations, or inter-token softmax order.
No replaying hardware-counter profile was needed to establish this limiter.
Long's C01 full-request fraction is 0.2463 (phase fraction 0.9394), so medium
remains the largest conservative whole-request gain at the chosen local s=5.
