# CUDA Amdahl campaign 2026-09-05

## Recovery state

- Campaign ID: `cuda-amdahl-20260905` (new campaign; historical campaign completed).
- Branch: `perf/cuda-amdahl-20260905`.
- Immutable start: `62384895acfde94cf28fded1294ad860daabf2ff`.
- Current retained runtime: `eebe52a` (C12), building on `7682cd2` (C16), `b948f67` (C14), `b01470c` (C11), `14b669d` (C02), `2547df3` (C03), `f251074` (C05), and `61a540a` (C01).
- State: running, C12 profiles complete; C18 selected, exact baseline snapshot running.
- Attempts: 12 reached correctness (minimum 10); kept 8, code rejected/restored 4 (C06/C08/C13 rejected, C07 interesting), not_verified 0, closed-untried 2 (C04/C10). No overall deadline; each A/B comparison has a two-hour limit.
- Persistent private evidence: `target/cuda-amdahl-20260905/`.
- Current-campaign retained commits: `61a540a` (C01), `f251074` (C05), `2547df3` (C03), `14b669d` (C02), `b01470c` (C11), `b948f67` (C14), `7682cd2` (C16), `eebe52a` (C12).

Recovery checkpoint: baseline, all three `baseline-timeline` acquisitions and
C01 correctness/A/B/controls completed. Sessions `65351`, `84583` and `82983`
are terminal. C01 is accepted. Session `3733` completed all three refreshed
timelines with no dropped records. C05 is accepted after exact equality,
canonical gates, f16/int8 parity and all three A/B rows. Sessions `57416` and
`39069` completed. Temporary bit printing was removed, preserving the permanent
long-history test. Preserve `c05-exact-baseline.log` and `c05-exact-candidate.log`
as evidence. Session `56032` completed all refreshed C05 timelines with zero
dropped records. C06's wide-dot test passed on accepted C05 in session `63196`;
all canonical C06 gates passed in session `73467`. Its objective rejected it:
both production files are restored exactly to HEAD. Diagnostic session `80709`
completed successfully. C03 was kept against C05, and all refreshed profiles
completed in session `36470`. No C06 production code remains. The normalization
test is committed in `031aba2`; C02 canonical gates passed in session `15081`.
C02's short objective passed in session `61650`; both controls completed in
session `76898`. All regimes pass against C03; C02 is accepted in this commit.
All `c02-timeline` rows completed in session `87311`; current ranking selects
C11. Its baseline tail test passed and is committed in `3fffeb9`. Canonical
gates passed in session `51190`, extra int8 parity in `89187`, objective in
`19830`, and both controls in `98968`. C11 is accepted against C02 in this
commit. All `c11-timeline` rows completed in session `38969`; new discovery
added C13/C14. C14 is accepted in `b948f67` after gates and all controls; its
profiles completed in `53915`. C13's boundary test passed and is committed in
`36c9912`; production now modifies only attention.cuh and is not yet accepted.
Canonical gates passed in `34634`; extra int8 parity precedes long objective.
C04/C10 are closed-untried with zero current target owner. Continue this pool.
Baseline and accepted immutable executables, plus all A/B records, remain under
the same private campaign directory. No other production candidate is applied.

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
| C01 | Spread decode attention across head dimensions and blocks; medium decode attention owns 8352.759 ms and launches only one block | Medium model decode | kept | Small: reuse existing parallel attention body; score reduction reorders |
| C02 | Parallel RMSNorm width; short prefill+decode normalization owns 406.330 ms with one block and serial width loops | Short model decode | kept | Numeric gates and all controls passed; objective +16.492% |
| C03 | Specialize matmul weight format outside K loops; short batched+single+logits own 3327.535 ms at baseline | Short prompt throughput, decode control | kept | Exact gate and all controls passed; objective +6.613% |
| C04 | Move full-tile token bounds outside K loops; PTX already scalarizes four accumulators but repeats four token-bound branches at every K step | Short prompt throughput | closed-untried | C14 routes every declared prefill batch away from this legacy kernel; target owner p=0 |
| C05 | Warp-level score reduction for attention; long prefill owns 28887.801 ms after C01 with block barriers at every context token | Long prompt throughput | kept | Same reduction tree, exact f16/int8 output and complete controls passed |
| C06 | Multiple output rows per matmul block, each owned by a warp, reusing input across outputs and reducing barriers | Short prompt throughput | rejected, restored | Correctness passed; objective -0.404% |
| C07 | Parallelize exact total-order argmax; short decode sampling owns 107.860 ms in one thread | Short model decode | restored, interesting objective | Exact gates passed; +4.595% below keep threshold |
| C08 | Encode the existing 128-thread matmul geometry as compile-time loop/reduction bounds; PTX retains runtime block width and reduction loop | Short model decode | rejected, restored | Exact gates passed, objective +1.097% below 3% |
| C09 | Read aligned packed f16 metadata with one 16-bit load instead of two byte loads plus recombination; PTX confirms repeated byte loads in hot matmul | Short prompt throughput | ready | Small; prove two-byte alignment for every packed format and exact output |
| C10 | Warp-first attention reduction to remove remaining inter-warp shared-tree stages; unlike C05 this changes pairing | Long prompt throughput | closed-untried | C11 removes that shared tree completely: current owner p=0 and ideal gain 0% |
| C11 | Four context scores per attention block iteration, one warp per score; merge their online softmax together | Long prompt throughput | kept | Numeric and f16/int8 parity gates plus all controls passed; objective +22.881% |
| C12 | Pair the two 128-wide K slices belonging to one quantized block, reusing block address and half metadata within each thread | Short model decode | kept | Exact/canonical gates and controls pass; objective +31.712% |
| C13 | Buffer attention scores in block-shared memory, parallelize stable softmax, then scan V without per-context barriers | Long prompt throughput | rejected, restored | Correctness passed; prefill -0.045% fails its objective despite decode +26.978% |
| C14 | Factorized quantized prefill via warp matrix instructions, applying float scale/bias outside exact integer-quant products | Short prompt throughput | kept | All numeric/parity gates and controls passed; objective +119.104% |
| C15 | Isolate buffered attention to decode, preserving online prefill without its shared score allocation | Long model decode | ready | Direct C13 phase evidence; same numeric gates, separate prefill ownership |
| C16 | Apply Q4/Q5 float correction once per format-defined K32 coefficient group, retaining Q6 K16 | Long prompt throughput | kept | Numeric/parity gates and all controls pass; objective +5.246% |
| C17 | Re-evaluate exact argmax after C12's retained decode reduction materially raises its removable fraction | Short model decode | ready | Re-evaluation of C07, not a new countable mechanism; never a selective rerun |
| C18 | Compute WMMA scale/bias only for the coefficient-owner lanes, retaining integer quant work for every lane | Long prompt throughput | ready, awaiting current C12 comparison | PTX executes discarded coefficient loads/conversions; exact gate required |

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
| C08 short | 0.6515 | 0.9117 prefill | 1.1 | 0.0005 | 6.24% | 187.0% | 0.30 s | Attribution upper bound is all matmul; actual savings depend on generated code and synchronization share |
| C09 short | 0.6515 upper bound | 0.9117 upper bound | 1.05 | 0 | 3.20% | 187.0% upper bound | 0.16 s predicted | Actual removable instruction/dependency share uncertain; cheap diagnostic candidate, not a predicted keep |

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

Exact baseline commands (repository root; model hard link points to authenticated
bytes, and the copied executable preserves baseline A across later builds):

```sh
cargo build --locked --profile fast --no-default-features --features cuda --example bench --example tokenize
python3 target/cuda-amdahl-20260905/run.py calibrate baseline
cp target/fast/examples/bench target/cuda-amdahl-20260905/baseline-bench
python3 target/cuda-amdahl-20260905/run.py bench baseline target/cuda-amdahl-20260905/baseline-bench short medium long
```

Public benchmark reproduction without the private runner (build each desired
revision with the command above, authenticate the supplied model against the
size/SHA table, then replace the model path):

```sh
python3 - <<'PY'
import subprocess
model = '/absolute/path/Ministral-3-3B-Instruct-2512-Q4_K_M.gguf'
for words in (124, 1020, 3580):
    prompt = ' '.join(['benchmark'] * words) + '.'
    subprocess.run(['target/fast/examples/bench', model,
                    '--context', '4096', '--kv', 'f16', '--prompt', prompt,
                    '--max-tokens', '32', '--warmup', '1', '--reps', '3'],
                   check=True)
PY
```

Use the pinned reference server's library directory in `LD_LIBRARY_PATH` when
running parity. Canonical gate commands, independent of the private runner:

```sh
cargo fmt --all -- --check
cargo check --workspace --locked --no-default-features --features cuda
cargo test --workspace --locked --no-default-features --features cpu
cargo test -p graph_horizon_engine --locked --no-default-features --features cuda error_matrix -- --nocapture --test-threads=1
cargo test -p graph_horizon_engine --locked --no-default-features --features cuda backend::cuda:: -- --nocapture --test-threads=1
support/testing/parity-check.sh --models-dir /absolute/path/models --model-id 3b-instruct --backend cuda --kv f16 --reference-server /absolute/path/pinned/llama-server
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

Local correctness preflight and all baseline timelines passed.
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

### C01 kept

Implemented one block per query head for decode and reused the existing prefill
attention body. Removed the now-unused serial duplicate; net production change
is 16 insertions / 47 deletions. Conceptual complexity decreases: both phases
share the same ownership and math. No ABI, KV layout, allocation, or interface
change. Correctness: formatting, CUDA workspace check, CPU workspace, 11 selected
error-matrix tests, all 33 CUDA tests including the declared long-history bound,
and pinned f16 parity passed before any candidate performance. All 16 local
top-1 IDs match the fixed oracle. Also run int8 real-model parity before A/B
because the same body serves both supported cache formats. The final decision
below follows the complete objective and both controls.

While C01 ran, inspected current matmul PTX: format comparisons occur inside
the K loop, confirming C03's premise. The compiler already scalarizes all four
accumulators, so C04 is narrowed to the still-present per-step full-tile bounds
(four compare/branch pairs); it must not claim a register-array improvement.
Runtime block-width/reduction bounds and repeated two-byte half reconstruction
support new C08 and C09 respectively. These are queued, not counted attempts.
Their broad matmul fractions are ownership upper bounds; instruction-cost
uncertainty lowers ranking and must be resolved by isolated measured experiments.

C01's extra pinned int8 real-model parity passed, again with all 16 top-1 IDs
equal to the oracle. The candidate fast build completed and is preserved as
`target/cuda-amdahl-20260905/c01-bench`. Objective and first control results:

| C01 regime | Prompt tok/s (CV) | TTFT ms (CV) | Model decode tok/s (CV) | Public deltas/s (CV) | Complete wall s |
|---|---:|---:|---:|---:|---:|
| Medium objective | 54.81 (0.0004) | 18682.26 (0.0004) | 11.60 (0.0002) | 10.88 (0.0002) | 86.66 |
| Short control | 61.66 (0.0020) | 2075.90 (0.0020) | 16.31 (0.0018) | 15.82 (0.0016) | 17.13 |
| Long control | 41.65 (0.0002) | 86043.27 (0.0002) | 6.43 (0.0002) | 6.03 (0.0002) | 364.69 |

Medium model decode improves 269.43%, public-delta throughput 270.07%; prompt
throughput improves 0.04% and TTFT falls 0.05%. Short model decode improves
58.20%, prompt throughput 2.09% and TTFT falls 2.06%. Both rows preserve the
baseline completion/delta counts and pass the predeclared gates. No
stability rerun applies. Medium has zero thermal-slowdown increment and about
0.894 s of software power-cap counter increase over 86.66 s at unchanged stock
settings; classify this as brief power-limited operation, not an unthrottled
record. The effect and controls must still hold across the complete tuple.

```sh
python3 target/cuda-amdahl-20260905/run.py gate c01
# Additional canonical parity command: same f16 command with --kv int8.
cargo build --locked --profile fast --no-default-features --features cuda --example bench
cp target/fast/examples/bench target/cuda-amdahl-20260905/c01-bench
python3 target/cuda-amdahl-20260905/run.py bench c01 target/cuda-amdahl-20260905/c01-bench medium
python3 target/cuda-amdahl-20260905/compare.py baseline c01 medium model_decode_tps medium
python3 target/cuda-amdahl-20260905/run.py bench c01 target/cuda-amdahl-20260905/c01-bench short long
```

After long finishes, compare all three rows, inspect telemetry and elapsed time,
and retain C01 only if every gate passes. This step completed: long model
decode improves 518.27%, public-delta throughput 515.31%; long prompt throughput
regresses 0.144% and TTFT increases 0.144%, both below the 5% control limit.
All objective CVs in A and B are <=0.16%. All three candidate rows preserve
prompt/completion/delta counts. Long has no increment in power-capping or thermal
slowdown; short has about 0.654 s of stock power capping and no thermal slowdown.
Environment settings and device remain identical. Comparison finished well
inside two hours; no rerun used. Terminal state: **keep**.

Whole-request elapsed approximations through the last public delta fall from
5.2195 to 4.0354 s short (-22.7%), 28.8950 to 21.4396 s medium (-25.8%), and
116.5318 to 91.0184 s long (-21.9%). These aggregate inversions are descriptive,
not confidence intervals. The retained source is 99 physical lines for the
category-K attention shader and 41 for checked decode dispatch, comfortably
within the orchestration limit. Next refresh phase fractions on the accepted
runtime and continue the pool; this first keep does not complete the campaign.

### Refresh after C01

`python3 target/cuda-amdahl-20260905/run.py trace c01-timeline target/cuda-amdahl-20260905/c01-bench short medium long`
completed on all three regimes with zero dropped records. Short measures 2063.187 ms before the first sample and 1960.483 ms
after it. Decode attention falls from 1157.333 to 120.986 ms (about 9.57x local),
consistent with the measured public gain. Short's remaining largest owner is
matmul: 1878.154 ms batched + 1243.279 ms single + 155.715 ms logits, about 81.4%
of the 4023.669 ms total timeline. RMSNorm is 410.318 ms (10.2% overall, 15.45%
of decode), argmax 108.148 ms (2.69% overall, 5.52% of decode). GPU gaps remain
17.661 ms overall, so host-side submission still has a small ideal ceiling.

C03 is now ready. C05 can first preserve the current binary reduction tree:
keep the shared stages above a warp and replace the final five shared/barrier
stages with ordered warp shuffles. This removes synchronization without changing
paired additions. Before making that exact candidate, capture the synthetic
long-history f16/int8 outputs on accepted C01 and require byte equality afterward,
in addition to all canonical gates. A reordered warp-first variant would be a
separate later premise only if the remaining barrier cost warrants it. No C05
production edit has been made.

Medium's refreshed timeline is 18446.651 ms before sampling and 2740.845 ms
decode; batched matmul owns 15049.655 ms, prefill attention 2286.728 ms and
decode attention 893.949 ms. Long is 85923.358 ms before sampling and 4986.238 ms
decode; batched matmul owns 53163.622 ms, prefill attention 28887.801 ms and
decode attention 3121.867 ms. Long's public instrumented TTFT is 85925.56 ms
(-0.137% versus C01 unprofiled mean); model decode 6.41 versus 6.43 tok/s.
The largest remaining overall owner is matmul; the largest ready candidate's
credible effect is C05 on the measured shared attention body.

Refreshed priority estimates (`p` of the whole request; local objective/control
fractions remain separate). These replace the initial estimates for selection:

| Priority / ID | Regime | p | s | o | Predicted global gain | Ideal global gain ceiling | Predicted saved ms | State |
|---|---|---:|---:|---:|---:|---:|---:|---|
| 1 / C05 | Long | 0.3521 | 2 | 0.002 | 21.07% | 54.35% | 15823.0 | ready, selected |
| 2 / C06 | Short | 0.8145 | 1.25 | 0.003 | 19.03% | 438.99% | 643.4 | deferred until C03 |
| 3 / C03 | Short | 0.8145 | 1.2 | 0.001 | 15.57% | 438.99% | 542.2 | ready |
| 4 / C04 | Short | 0.4668 | 1.25 | 0.001 | 10.18% | 87.54% | 371.6 | deferred until C03 |
| 5 / C02 | Short | 0.1020 | 8 | 0.001 | 9.68% | 11.36% | 355.0 | ready |
| 6 / C08 | Short | 0.8145 | 1.1 | 0.0005 | 7.94% | 438.99% | 295.9 | deferred until C03 |
| 7 / C09 | Short | 0.8145 | 1.05 | 0 | 4.03% | 438.99% | 156.1 | deferred until C03 |
| 8 / C07 | Short | 0.0269 | 8 | 0.0002 | 2.39% | 2.76% | 93.8 | deferred until C02 |

Uncertainty remains as in the premise table; these estimates are hypotheses,
not measured improvements. C05 is a smaller, directly attributable change than
the deferred output-mapping experiment. For C07, decode phase fraction is now
0.0552, so its local objective ceiling is 5.84%; its full-request ceiling stays
small and it remains lower priority until larger work has been removed.

C05 exact gate preparation added temporary integer-bit output to the existing
long-history test, then ran that one test on C01. It passed, capturing f16/int8
prefill and decode for all three declared dimension/context pairs. Capture:
`c05-exact-baseline.log`; the first marker follows the test-name prefix, so
extract with `rg -o 'cuda-attention-bits.*'` rather than anchoring to line start.
No expected values are updated after candidate implementation. This supports
an exact reduction-order claim and does not count as an optimization attempt.
Planned C05 structure is the existing category-K `shaders/attention.cuh`
(~110 productive lines), no new production file; dispatch geometry stays fixed.

### C05 kept

The shader retains shared-memory tree stages at strides >=32, then uses warp
shuffles for the same paired additions below 32. An active mask and a reduced
starting stride preserve blocks smaller than a warp and padded dimensions.
Only lane zero consumes the final score; the existing shared-softmax barrier
still publishes its state to the block. No dispatch, format, buffer, softmax
order or allocation changes. Production diff is confined to the attention
category-K shader (+14/-2 lines); temporary bit output is test-only.

Exact test command on accepted C01 and on C05:

```sh
cargo test -p graph_horizon_engine --locked --no-default-features --features cuda attention_long_history_and_dimension_tails_match_reference -- --nocapture --test-threads=1
diff <(rg -o 'cuda-attention-bits.*' target/cuda-amdahl-20260905/c05-exact-baseline.log) <(rg -o 'cuda-attention-bits.*' target/cuda-amdahl-20260905/c05-exact-candidate.log)
python3 target/cuda-amdahl-20260905/run.py gate c05
```

All six captured cases match bit-for-bit; `c05-exact.diff` is empty and the
reference test passes. Immutable baseline log SHA-256:
`439fd42565146815b575423f43a475daf03ea753a36788faf0ea4f6fd2852246`.
Formatting, CUDA check, CPU workspace, 11 error-matrix and all 33 CUDA tests
passed. Pinned f16 real-model parity also passed. Additional int8 real-model
parity must print `pass:` before building and running the candidate objective.
Session `57416` performs that sequence and preserves `c05-bench`.

```sh
python3 target/cuda-amdahl-20260905/run.py bench c05 target/cuda-amdahl-20260905/c05-bench long
# After the running objective completes:
python3 target/cuda-amdahl-20260905/compare.py c01 c05 long prompt_tps long
# Only after a provisional passing objective:
python3 target/cuda-amdahl-20260905/run.py bench c05 target/cuda-amdahl-20260905/c05-bench short medium
python3 target/cuda-amdahl-20260905/compare.py c01 c05 long prompt_tps short medium long
```

Retain only after all controls pass. Remove temporary bit-output instrumentation
before retaining C05 (the permanent long-history test remains), then update the
pool and rerank. A rejected candidate still counts as attempt two after this
correctness gate; a performance failure is not an excuse to stop the campaign.

During C05 measurement, verified C09's alignment prerequisite: CUDA allocations
are aligned and `CudaBuffer::view` rejects every odd byte offset. All packed
block strides (144, 176, 210) and the half-factor offsets (0, 2, 208) are even.
Thus a two-byte metadata read needs no new layout, padding or public constraint.
This establishes a bounded implementation prerequisite only; C09 remains queued
and has not been implemented or counted.

C03 implementation constraint from static review: hoist format dispatch outside
the reduction loops but pass one shared scratch array into the specialized
device bodies. Declaring a separate shared array per template instance could
multiply per-block shared storage even though only one format executes, changing
occupancy and confounding the premise. Preserve the current kernel entry ABIs,
128-thread geometry, four-token tile and shared storage sizes. The bounded plan
fits only `shaders/matmul.cuh` (~140 productive category-K lines); exact patterned
multi-block and full/tail batch output must be captured before that edit.

C05 long objective completed in 346.04 s, one warm-up and three measured reps:

```text
prompt_tokens=3584 completion_tokens=32 decoded_tokens=31 prompt_tps_mean=44.00 prompt_tps_median=44.00 prompt_tps_stddev=0.01 prompt_tps_cv=0.0002 ttft_ms_mean=81448.65 ttft_ms_median=81450.76 ttft_ms_stddev=17.57 ttft_cv=0.0002 model_decode_tps_mean=6.45 model_decode_tps_median=6.45 model_decode_tps_stddev=0.00 model_decode_tps_cv=0.0007 decode_tps_mean=6.05 decode_tps_median=6.05 decode_tps_stddev=0.00 decode_tps_cv=0.0007 delta_interval_ms_mean=165.24 delta_interval_ms_median=159.78 delta_interval_ms_stddev=28.97 delta_interval_ms_cv=0.1753 decode_begin_tps=6.27 decode_middle_tps=6.25 decode_end_tps=5.67
```

Against C01: objective +5.642%, TTFT -5.340%, model decode +0.311%, public
decode +0.332%. The objective CV is 0.02% in both A and B, so no rerun applies.
Pinned int8 parity also passed with all 16 top-1 IDs matching the oracle.
The full-request gain is substantially below the initial 21.07% prediction;
the assumed 2x local speedup was optimistic. Do not propagate that local speedup
to related synchronization candidates. Refresh actual attention times after the
controls before ranking a bounded variant. C05 remains unaccepted until both
short and medium controls complete without a >5% regression.

C05 controls completed and passed:

| Regime | Prompt tok/s (CV) | TTFT ms (CV) | Model decode tok/s (CV) | Public deltas/s (CV) | Complete wall s |
|---|---:|---:|---:|---:|---:|
| Short | 61.85 (0.0017) | 2069.49 (0.0017) | 16.27 (0.0006) | 15.79 (0.0006) | 17.09 |
| Medium | 56.10 (0.0010) | 18252.01 (0.0010) | 11.63 (0.0009) | 10.91 (0.0009) | 84.79 |

Short prompt improves 0.31%, TTFT falls 0.31%, model decode regresses 0.245%
and public decode 0.190%. Medium prompt improves 2.354%, TTFT falls 2.303%,
model decode improves 0.259% and public decode 0.276%. No control exceeds the
5% regression limit; all counts match C01. Both objective CVs remain 0.02%,
correctness and exact-output gates passed, comparison finished within two hours,
and no rerun was required. Terminal state: **keep**. Long device counters show
0.846 s of stock software power capping over 346.04 s and zero thermal slowdown;
this is classified as brief power-limited operation at unchanged settings.

Removed the temporary exact-output statements before committing; the production
diff remains only the attention reduction (+14/-2), with no dispatch or storage
change. Conceptual complexity increases slightly for the explicit warp boundary
and short-block mask; both protect the existing reduction order and valid lanes.
The shader remains a single category-K operation and no new file is introduced.

### Refresh after C05 and C06 selection

C05 timelines completed with zero dropped records. Prefill attention versus C01
falls 35.747 -> 25.274 ms short, 2286.728 -> 1872.478 ms medium, and
28887.801 -> 24567.576 ms long: local speedups 1.414x, 1.221x, 1.176x. Decode
attention is essentially unchanged (long 3121.867 -> 3113.900 ms). The effect
depends on available blocks; do not use the short local gain to predict long.
Long's instrumented public outcome is close to the unprofiled row and is not
used as the retention record.

Current short timeline: 2051.322 ms prefill/first sample + 1956.775 ms decode,
4008.097 ms total. Batched matmul is 1880.592 ms, single matmul 1240.370 ms,
logits 158.121 ms (81.81% combined); RMSNorm 399.394 ms (9.96% combined).
Medium totals 20878.144 ms; long totals 86717.248 ms, including 53288.460 ms
batched matmul and 27681.475 ms attention across both phases. Gaps remain below
0.5% overall, so a host allocation/submission experiment still has a small ceiling.

Dependency review shows C06 does not require C03 specialization: its warp
ownership can be measured independently with unchanged format decoding. Remove
that unnecessary dependency and choose C06, whose credible whole-request gain
is largest among the remaining ready entries. This corrects the earlier queue
ordering rather than silently preferring an easier lower-ranked edit.

| Remaining priority | ID / state | Full-request p | s | o | Predicted global gain | Ideal gain ceiling | Saved ms |
|---|---|---:|---:|---:|---:|---:|---:|
| 1 | C06 ready, selected | 0.8181 short | 1.25 | 0.003 | 19.13% | 449.68% | 643.8 |
| 2 | C03 ready | 0.8181 short | 1.2 | 0.001 | 15.65% | 449.68% | 542.5 |
| 3 | C04 deferred until C03 | 0.4692 short | 1.25 | 0.001 | 10.24% | 88.42% | 372.1 |
| 4 | C02 ready | 0.0996 short | 8 | 0.001 | 9.44% | 11.06% | 345.5 |
| 5 | C08 deferred until C03 | 0.8181 short | 1.1 | 0.0005 | 7.98% | 449.68% | 296.1 |
| 6 | C10 deferred until C06 | 0.3192 long | 1.15 | 0.002 | 4.13% | 46.89% | 3437.8 |
| 7 | C09 deferred until C03 | 0.8181 short | 1.05 | 0 | 4.05% | 449.68% | 156.1 |
| 8 | C07 deferred until C02 | 0.0270 short | 8 | 0.0002 | 2.39% | 2.77% | 93.7 |

These are rounded ranking estimates with the prior uncertainty caveats. C10 is
a distinct bounded variant: C05 removed only within-warp stages while preserving
pairing; remaining stages cross warps and still use shared writes and barriers.
Use conservative s=1.15 after C05's negative prediction calibration. It is not
an attempted or accepted optimization, and waits for C06's bottleneck shift.

C06 predeclared gate: canonical CPU workspace, CUDA check/error tests, full CUDA
suite, existing bounded packed-format numeric tests, authenticated real-model
top-two parity, and the new wide-dot scalar-reference test. Reduction order
changes, so no bit-exact claim. The new test covers widths 3 (F16), 768, 3072,
9216, one/nine input rows, three output rows, and every supported weight format;
checks every batched output against stored-weight scalar reference with the
unchanged bound, exact batch/single agreement and float logits. All 26 cases
passed on C05 before C06 edits; raw log `matmul-wide-c05-baseline.log` contains
temporary bit snapshots, whose print statements were removed before checkpoint.

C06 structure: existing `shaders/matmul.cuh` (~80 productive category-K lines)
and `kernels/matmul.rs` (~115 productive orchestration lines), no new file.
Keep 128 threads/block, assign four independent output rows to its four warps,
iterate K by lane with stride 32, reduce each output inside that warp, and use
ceiling(output_width/4) blocks. Invalid output rows return as an entire warp,
so full-warp shuffle masks remain valid. Keep the four-token prefill tile,
all layouts, formats, spans, placement and interfaces. Main risk is reordered
floating-point reduction and altered latency/occupancy. Objective: short prompt
throughput; decode, TTFT and both other regimes are controls. Canonical 5% keep,
5% CV and 5% control limits remain fixed before measurement.

### C06 rejected

Implemented four warp-owned output rows per 128-thread block. K loops use
lane/stride 32 and warp reductions; the whole invalid output warp returns
before shuffles. Checked Rust launches use `output_width.div_ceil(4)` without
addition overflow. Batched matmul retains four-token weight reuse and tail
guards; float logits share the same warp dot. Removed shared reduction arrays
and all cross-warp barriers from this operation. Net production diff is
22 insertions / 29 deletions across the two predeclared files; no new abstraction,
allocation, public interface or dependency.

`python3 target/cuda-amdahl-20260905/run.py gate c06` passed formatting, CUDA
workspace check, CPU workspace suites, all 11 error-matrix tests and all 34 CUDA
tests including the new 26-case wide-dot bound and exact batched/single checks.
Pinned real-model f16 parity passed before performance. This is a numeric
candidate under the existing bound, not a claim of unchanged output bits.
The fast executable is preserved as `c06-bench` for immutable comparison.

```sh
python3 target/cuda-amdahl-20260905/run.py bench c06 target/cuda-amdahl-20260905/c06-bench short
# After the active objective completes:
python3 target/cuda-amdahl-20260905/compare.py c05 c06 short prompt_tps short
# Only after provisional keep:
python3 target/cuda-amdahl-20260905/run.py bench c06 target/cuda-amdahl-20260905/c06-bench medium long
python3 target/cuda-amdahl-20260905/compare.py c05 c06 short prompt_tps short medium long
```

Short A/B: prompt 61.85 -> 61.60 token/s (-0.404%, CV 0.17% -> 0.23%);
TTFT 2069.49 -> 2077.86 ms (+0.404%); model decode 16.27 -> 16.38 (+0.676%,
CV 0.06% -> 0.22%); public decode 15.79 -> 15.89 (+0.633%). Counts remain
128/32/32. Stable objective below 3%: **rejected**, no rerun or other controls
needed. Restored both production files; retained the negative patch privately.

The immutable rejected binary's short diagnostic has zero dropped records and
confirms four output rows/block. Batched matmul costs 1906.634 ms versus C05's
1880.592 ms, single matmul 1246.889 versus 1240.370 ms, logits 142.514 versus
158.121 ms. Eliminating block barriers and reducing block count did not improve
the two dominant operations. This causal ablation lowers the synchronization/
launch hypothesis; it does not establish a memory-bandwidth roof. Format tests
and decode instructions remain inside the K loop in accepted PTX. C03 tests
that instruction-cost hypothesis without changing parallel ownership. Its local
speedup estimate remains uncertain; no occupancy or bandwidth counter claim.

Remaining rank: C03 ready/selected, C04 deferred until C03, C02 ready, C08 and
C09 deferred until C03, C10 ready (same conservative C05-based estimate), C07
deferred until C02. Fractions and estimates above remain applicable because C06
was restored; no new bounded variant of its disproven ownership premise is
supported by this trace.

C03 predeclared structure: existing `shaders/matmul.cuh` (~140 productive lines,
category K), no new file. Specialize device bodies on format outside K loops;
each entry retains one shared scratch array passed to the specialized body,
avoiding per-specialization shared allocations. Preserve 128 threads, four-token
tile, arithmetic order, rounding, all ABIs and bounds. Risk: compiler contraction
or increased register pressure. Gate: exact equality against the frozen 26-case
`matmul-wide-c05-baseline.log` snapshots, all canonical gates and f16 parity.
Objective short prompt throughput; standard thresholds and other-regime controls.

### C03 kept

Implemented format-specialized device bodies with one shared allocation per
entry. Production +35/-9 lines in the existing matmul shader; no launch, format,
layout or numeric order changes. The 26 wide-dot bit snapshots exactly match
accepted C05 (`diff -u` over `rg -o 'cuda-matmul-bits.*'` is empty). Frozen
baseline log SHA-256 is `b14fe6001c1d366f4762371fa9c06e1232643929c1dd7136c8c045888dfac1c8`;
candidate is `864e63c4d44f2812574c26aad2d19229369278977ee5ffd3ee6a9d7a12453608`.
Temporary print statements removed. `run.py gate c03` passed formatting, CUDA
workspace check, CPU suites, 11 error tests, 34 CUDA tests and pinned f16 parity
with all 16 top-one IDs unchanged. Session `83493` builds/copies the immutable
binary and runs the short objective using the same command as C06 with `c03`.
Compare `compare.py c05 c03 short prompt_tps short`; only a provisional keep
authorizes the medium/long control measurements within this candidate.

Objective and both controls completed successfully, no rerun. Values are means
with fractional CV in parentheses; compare to the complete C05 table above.

| C03 regime | Prompt tok/s (CV) | TTFT ms (CV) | Model decode tok/s (CV) | Public deltas/s (CV) | Complete wall s |
|---|---:|---:|---:|---:|---:|
| Short objective | 65.94 (0.0012) | 1941.27 (0.0012) | 17.22 (0.0027) | 16.71 (0.0026) | 16.12 |
| Medium control | 59.81 (0.0013) | 17120.49 (0.0013) | 12.17 (0.0002) | 11.42 (0.0002) | 79.84 |
| Long control | 46.35 (0.0002) | 77320.95 (0.0002) | 6.65 (0.0001) | 6.24 (0.0001) | 329.36 |

Short objective +6.613%, objective CV 0.17% -> 0.12%, model decode +5.839%.
Medium prompt +6.613%, model decode +4.643%; long prompt +5.341%, model decode
+3.101%. All token counts match the reference; no control regresses. Medium's
stock software power-cap counter increases 0.954 s over 79.84 s; long has no
increment. Neither has thermal slowdown. This is a classified brief stock
power-limited medium row, not a changed setting. Comparison within two hours.
Terminal state **keep** after complete correctness, exact-output and A/B gates.

PTX confirms a single 4096-byte shared allocation for batched matmul and format
branches before specialized K loops. The +35/-9 productive-line change adds
only the specialized bodies necessary to move the decision out of hot loops;
no independent abstraction, resource or API. The measured effect is smaller
than the 15.65% full-request estimate, so future instruction-removal estimates
must retain substantial uncertainty rather than claiming proportional gains.
Next: refreshed `c03-timeline` short/medium/long, then rerank the persistent pool.

C02 preparatory gate structure: add one focused test to existing
`kernels/tests.rs` (~55 test lines, excluded from productive limit). Before
production edits, check RMSNorm widths 1, 3, 129, 3072, 9216, rows 1/3, epsilon
0/1e-5, mixed signs and magnitudes against the serial stored-weight reference.
Keep the existing absolute/relative bound; then repeat with the candidate and
run all canonical gates and pinned parity before A/B. Production structure:
`shaders/normalization.cuh` (~35 productive category-K lines),
`kernels/normalization.rs` (~45 orchestration lines). One block owns one row,
128 threads stride through width and use a shared binary sum; output remains
f16. Numeric pairing changes, but storage, epsilon, spans and public ABI do not.
The risk is rounding drift or small-width overhead. Objective remains short
model decode throughput; prefill, TTFT and other regimes are controls.

### Refresh after C03; C02 selected

All three C03 timelines completed in session `36470`, zero dropped records.
Short prefill/first sample 1934.663 ms + decode 1855.778 ms: batched matmul
1761.962 ms, single 1158.231 ms, logits 149.708 ms total. Matmul local reductions
are only about 6–8%, supporting instruction cost but not the originally assumed
20%. Calibrate the other matmul instruction-removal estimates by approximately
0.4 of their proposed incremental local gains (C04 1.25 -> 1.10, C08 1.10 ->
1.04, C09 1.05 -> 1.02). These remain uncertain bounded experiments, not closed.
RMSNorm totals 397.981 ms, 10.50% of the short request and 15.79% of decode;
one active thread for a decode row establishes serial execution/underutilization,
not a bandwidth ceiling. A block-width reduction directly removes that limiter.
Argmax totals 108.688 ms and owns 5.67% of short decode, so its phase-specific
ideal ceiling can still reach 5% despite its smaller whole-request ceiling.

Long totals 82213.420 ms: batched matmul 49419.047 ms, attention 27219.966 ms.
Short instrumentation TTFT differs -0.24%, medium -0.76%, long +0.09%; long
decode -0.30% versus the unprofiled rows. These single traces are diagnostic,
not retention A/B, and remain sufficiently close for attribution.

| Priority / state | p full request | s | o | Predicted gain | Ideal ceiling | Saved ms |
|---|---:|---:|---:|---:|---:|---:|
| C02 ready, selected | 0.10500 short | 8 | 0.001 | 9.995% | 11.731% | 344.443 |
| C04 ready after C03 | 0.46484 short | 1.10 | 0.001 | 4.303% | 86.861% | 156.388 |
| C10 ready | 0.33109 long | 1.15 | 0.002 | 4.295% | 49.497% | 3386.004 |
| C08 ready after C03 | 0.80991 short | 1.04 | 0.0005 | 3.162% | 426.056% | 116.178 |
| C07 deferred until C02 | 0.02867 short | 8 | 0.0002 | 2.553% | 2.952% | 94.344 |
| C09 ready after C03 | 0.80991 short | 1.02 | 0 | 1.614% | 426.056% | 60.194 |

C02's 20-case baseline test passed on accepted production in
`c02-baseline-test-corrected.log`. The initial test-only build had two API/type
mistakes, corrected before any production edit; it is not an attempted numeric
candidate or a changed tolerance. C02 now uses the predeclared numeric gate.

### C02 kept

Implemented one 128-thread block per row; inactive width lanes contribute zero
to a 128-entry shared binary sum, then each thread writes its strided f16 output.
Checked dispatch retains all validation and documents the matching scratch
geometry. Production diff +17/-7 across the two predeclared files, no new file
or interface. This replaces serial width work with a locally owned reduction;
the explicit synchronization is necessary for that ownership invariant.

`run.py gate c02` passed fmt, CUDA workspace check, CPU suites, all 11 error
tests, 35 CUDA tests (including the 20-case normalization gate), and pinned f16
parity with all 16 top-one IDs unchanged. The immutable `c02-bench` now runs
`run.py bench c02 target/cuda-amdahl-20260905/c02-bench short`; compare
`compare.py c03 c02 short model_decode_tps short`. Correctness was completed
before performance; numeric reordering is bounded, not bit-exact.

Short objective completed: model decode 17.22 -> 20.06 token/s (+16.492%,
CV 0.27% -> 0.19%); prompt 65.94 -> 69.21 (+4.959%, CV 0.12% -> 0.28%);
TTFT 1941.27 -> 1849.45 ms (-4.730%); public decode 16.71 -> 19.47 (+16.517%,
CV 0.26% -> 0.20%). Counts 128/32/32 unchanged, complete wall 14.72 s.
Provisional keep only: medium/long controls now run serially against C03.

### Pool replenishment during C02 controls

C03's remaining long attention cost is 27219.966 ms. Static review of that
measured owner finds another distinct mechanism, **C11**: batch four consecutive
context tokens per block iteration, assigning each warp an independent score,
then merge those four scores into one online softmax update. Unlike C10's
replacement of a single score's reduction tree, this amortizes block barriers
across context tokens and reduces exponential evaluations from eight per four
tokens to five. It is not the historical matmul tile experiment. It requires
neither a second launch nor global scratch or cache-layout changes.

C11 starts deferred until the active C02 decision and profile complete; do not
bundle it with normalization. On C03's long owner fraction p=0.33109, use a
discounted local s=1.4 and o=0.004: predicted full-request gain about 9.97%,
ideal ceiling 49.50%, approximately 7.45 s saved. Phase-specific prefill owner
p=0.31203 has a 45.35% ideal ceiling. Moderate implementation cost and numeric
risk, higher uncertainty than C02: more live accumulators and changed softmax
pairing may offset synchronization savings. Next profile must rerank it.

Bounded structure if selected: existing `shaders/attention.cuh` (~150 productive
category-K lines), `kernels/attention/{decode,prefill}.rs` (~45 each); no new
file or interface. Keep one block per query head with 128 threads; each thread
owns up to two output dimensions (validated head dimension <=256), and four
warps score four context tokens with causal tail guards. Preserve both KV
formats. Gate is the existing dense long-history CPU-reference bound, full
canonical suite, and pinned f16/int8 model parity before any performance.
This replenishment is planning only and does not increment attempts.

### C02 complete decision

Both control acquisitions passed in session `76898`, with identical token
counts. The complete C02 records (means, fractional CV):

| C02 regime | Prompt tok/s (CV) | TTFT ms (CV) | Model decode tok/s (CV) | Public deltas/s (CV) | Complete wall s |
|---|---:|---:|---:|---:|---:|
| Short objective | 69.21 (0.0028) | 1849.45 (0.0028) | 20.06 (0.0019) | 19.47 (0.0020) | 14.72 |
| Medium control | 62.46 (0.0027) | 16394.36 (0.0027) | 13.59 (0.0027) | 12.75 (0.0027) | 75.84 |
| Long control | 47.90 (0.0002) | 74816.16 (0.0002) | 7.04 (0.0001) | 6.60 (0.0001) | 318.26 |

Against C03, medium model decode +11.668%, prompt +4.431%; long model decode
+5.865%, prompt +3.344%. No throughput or TTFT control regresses. Short objective
+16.492%, both objective CVs below 5%, correctness and numeric gates passed;
terminal **keep**, within two hours and without rerun. Stock software power-cap
counter increments are 0.974 s short and 0.614 s medium; zero long. All thermal
slowdown counters remain unchanged. Ordinary dynamic clocks are recorded in the
private telemetry, with no machine-setting changes.

Retain only the predeclared +17/-7 production lines in normalization shader and
dispatch. The permanent test was committed separately before the experiment.
The shader is 26 physical category-K lines; dispatch remains below 45 productive lines.
Next: `run.py trace c02-timeline target/cuda-amdahl-20260905/c02-bench short medium long`
and refresh all non-terminal entries, including newly discovered C11.

### Refresh after C02; C11 selected

Session `87311` completed all timelines, zero dropped records. Short total
3420.348 ms: 1833.833 ms prefill/first sample and 1586.515 ms decode; RMSNorm
2.385 + 19.713 = 22.098 ms versus 397.981 ms at C03, local speedup 18.0x.
Its remaining short phase ceiling is 1.26% decode and 0.13% prefill: no further
normalization-only candidate can meet the 5% threshold on these rows. Long
RMSNorm is 66.770 + 20.960 ms, likewise immaterial. Instrumented long TTFT differs
-0.034% and model decode -0.43% from unprofiled C02; traces remain diagnostic.

Long total 79350.087 ms: batched matmul 49573.221 ms; attention 24218.398 ms
prefill + 3074.500 ms decode. Host/transfer gaps remain below 0.5%; no launch-gap
or allocation-only premise reaches the retention ceiling. The following ranks
use current full-request owners, calibrated estimates and unchanged thresholds:

| Priority / state | p | s | o | Predicted gain | Ideal ceiling | Saved ms |
|---|---:|---:|---:|---:|---:|---:|
| C11 ready, selected | 0.34396 long | 1.4 | 0.004 | 10.409% | 52.429% | 7480.571 |
| C04 ready | 0.51615 short | 1.1 | 0.001 | 4.813% | 106.677% | 157.073 |
| C10 ready, reconsider after C11 | 0.34396 long | 1.15 | 0.002 | 4.478% | 52.429% | 3401.243 |
| C08 ready | 0.90081 short | 1.04 | 0.0005 | 3.535% | 908.116% | 116.792 |
| C07 ready after C02 | 0.03187 short | 8 | 0.0002 | 2.847% | 3.292% | 94.692 |
| C09 ready | 0.90081 short | 1.02 | 0 | 1.798% | 908.116% | 60.413 |

C07's declared decode phase owns 6.65%, ideal ceiling 7.13%, so it remains
eligible despite its smaller whole-request ceiling. C11 has the largest
credible end-to-end gain and is selected under its predeclared structure/gate.
Add `(head_dim, context)=(129,33)` to the existing attention reference test
before production edits: it exercises the second output dimension's tail under
128-thread ownership. Three adjacent causal queries plus the other cases cover
all four context-tile tail lengths in both KV formats. No tolerance changes.
Session `29975` tests this baseline on accepted C02. Use tile indices bounded
by `position/4`, not an overflowing `token += 4` loop, in the candidate.

### C11 kept

The extended baseline attention test passed on C02 before production changes.
C11 implements four warp-owned scores per tile, a shared four-score softmax
update and two bounded output accumulators per thread. Invalid context lanes
write negative infinity and never load K/V; output lanes beyond dim never write.
The final barrier protects shared weight readers before the next tile. Tile
index arithmetic preserves the existing u32 position boundary without wrapping.
Production +51/-57 lines across the predeclared shader and two dispatch files;
the shader is 123 physical category-K lines. No new file, allocation or ABI.

Initial formatting preflight requested collapsing two launches; applied rustfmt
before running the substantive gates. `run.py gate c11-checked` passed formatting,
CUDA workspace check, CPU suites, 11 error tests, all 35 CUDA tests (including
eight dense attention reference cases), and pinned f16 parity with all 16 top-one
IDs unchanged. Additional int8 parity is running before the long prompt objective.
The predeclared numeric tolerance is unchanged; this is not a bit-exact candidate.

Extra pinned int8 parity passed (`c11-int8-parity.log`), all 16 local top-one IDs
equal to the oracle. The immutable fast executable now runs the objective:

```sh
python3 target/cuda-amdahl-20260905/run.py bench c11 target/cuda-amdahl-20260905/c11-bench long
python3 target/cuda-amdahl-20260905/compare.py c02 c11 long prompt_tps long
# Only if provisionally passing:
python3 target/cuda-amdahl-20260905/run.py bench c11 target/cuda-amdahl-20260905/c11-bench short medium
python3 target/cuda-amdahl-20260905/compare.py c02 c11 long prompt_tps short medium long
```

Continued static discovery adds **C12**, distinct from the rejected historical
warp-metadata broadcast: the 128-thread dot loop visits each 256-value quant
block in two successive iterations, reconstructing the same half scale(s) and
block address twice. Pair those contributions inside one thread, retaining its
original accumulation order, so inlining can reuse metadata without shuffles.
First verify the compiled loads actually disappear; source-level pairing alone
does not establish savings. F16 and packed-width tails keep their validation.

C12 waits for C08's cheaper compile-time-geometry experiment and the refreshed
profile; fixed 128-thread ownership bounds its pairing invariant. On current
short p=0.90081, conservatively use s=1.04 and o=0.001: predicted gain 3.481%,
ideal ceiling 908.116%, about 115.082 ms saved. Metadata may be cache-hot or
compiler reuse may fail, so confidence is low but its ceiling warrants a bounded
experiment. Existing `shaders/matmul.cuh` (~160 productive category-K lines), no
new file, dependency or format. Require exact 26-case matmul snapshots, canonical
gates and pinned parity before the objective; other regimes remain controls.
This is planning-only, not a countable attempt or permission to bundle C08/C09.

C11 long objective passed provisionally in session `19830`: prompt 47.90 ->
58.86 token/s (+22.881%, CV 0.02% -> 0.22%); TTFT 74816.16 -> 60892.86 ms
(-18.610%); model decode 7.04 -> 12.00 (+70.455%, CV 0.01% -> 0.24%); public
decode 6.60 -> 11.26 (+70.606%, CV 0.01% -> 0.25%). Counts remain 3584/32/31,
complete wall 254.98 s. Stock power-cap counter increases 0.694 s; no thermal
slowdown. No rerun needed. Short and medium controls are now acquiring serially;
do not retain until both pass against C02 under the predeclared limits.

### C11 complete decision

Both controls completed in session `98968`; all counts match C02. Full records:

| C11 regime | Prompt tok/s (CV) | TTFT ms (CV) | Model decode tok/s (CV) | Public deltas/s (CV) | Complete wall s |
|---|---:|---:|---:|---:|---:|
| Long objective | 58.86 (0.0022) | 60892.86 (0.0022) | 12.00 (0.0024) | 11.26 (0.0025) | 254.98 |
| Short control | 70.30 (0.0026) | 1820.77 (0.0026) | 21.13 (0.0037) | 20.51 (0.0038) | 14.30 |
| Medium control | 66.64 (0.0011) | 15367.12 (0.0011) | 17.63 (0.0003) | 16.56 (0.0002) | 69.52 |

Short prompt +1.575%, model decode +5.334%; medium prompt +6.692%, model decode
+29.728%. No control regresses. Long objective +22.881% passes the fixed 5%
threshold, with both objective CVs below 5%; complete numeric, f16/int8 parity
and A/B gates pass, within two hours and without rerun. Terminal **keep**.

The new local ownership needs four score slots and up to two output accumulators,
but removes the former cross-warp shared reduction tree and amortizes softmax
updates across four context tokens. Net production -6 lines, no storage/API
expansion. C05 remains a successful earlier campaign decision; C11 supersedes
its exact-tree implementation with the separately gated numeric tile. C10's
original shared-tree premise must be reviewed against this accepted code, not
retried against an obsolete runtime. Next: all three `c11-timeline` profiles.

### Refresh after C11; new structural candidates

Session `38969` completed all three traces, zero dropped records. Short totals
3340.989 ms (1818.174 prefill/first sample + 1522.814 decode), with 1762.887 ms
batched matmul, 1171.311 ms single matmul and 153.090 ms logits. Short attention
is only 11.013 + 49.771 ms; argmax still owns 106.132 ms of decode (6.97%,
phase ideal ceiling 7.49%). Long totals 63640.769 ms, with 50224.132 ms batched
matmul and attention 9717.201 + 1198.329 ms. Attention's local long speedups
versus C02 are 2.492x prefill and 2.566x decode. Instrumented TTFT is within
0.13% of unprofiled short/long; diagnostic decode within 0.6%. C11 controls had
0.779 s stock power capping short, zero medium, and zero thermal slowdown.

C10's cross-warp shared-score tree no longer exists, so its current removable
owner and ideal gain are zero: **closed-untried**, not an attempt. C11's remaining
context scan and scalar softmax updates support C13; scalar matmul and repeated
dequantization/FMA work support C14. The model-family tensor contract requires
Q4_K/Q6_K matrices, so the batched-matmul owner is within C14's packed scope.

| Priority / state | p full request | s | o | Predicted gain | Ideal ceiling | Saved ms |
|---|---:|---:|---:|---:|---:|---:|
| C14 ready, selected | 0.52765 short | 1.5 | 0.02 | 18.467% | 111.709% | 520.809 |
| C13 ready | 0.17152 long | 2 | 0.004 | 8.904% | 20.703% | 5203.202 |
| C04 ready | 0.52765 short | 1.1 | 0.001 | 4.928% | 111.709% | 156.922 |
| C08 ready | 0.92406 short | 1.04 | 0.0005 | 3.631% | 1216.906% | 117.071 |
| C12 deferred until C08 | 0.92406 short | 1.04 | 0.001 | 3.578% | 1216.906% | 115.401 |
| C07 ready | 0.03279 short | 8 | 0.0002 | 2.933% | 3.390% | 95.193 |
| C09 ready | 0.92406 short | 1.02 | 0 | 1.845% | 1216.906% | 60.535 |

C13 design, not implemented: existing `shaders/attention.cuh` (~230 productive
category-K lines, cohesive attention variants), no host ABI changes. For
positions <4096, score four contexts per warp group into a 4096-float shared
array, reduce a stable maximum, compute exponentials and denominator in parallel,
then accumulate V with no per-context barriers. Reuse shared reduction storage
only after all readers finish. Keep C11's online fallback at position >=4096;
its extra shared allocation/occupancy cost must be measured, not assumed free.
Gate adds `(dim,context)=(3,4097)` to the existing CPU-reference test so adjacent
queries cross 4095/4096, plus canonical gates and f16/int8 parity. No larger
performance tuple is requested; the extra context is validation-only. Risk is
changed summation order and shared-memory occupancy; s=2 is uncertain, supported
by C11's measured synchronization/softmax removal but discounted for storage.

C14 design, selected before implementation: the installed CUDA headers and
existing capability floor (compute >=7.5, warp32, shared >=48 KiB) support the
standard warp matrix API without a new dependency or architecture route. Follow
the current [CUDA warp matrix contract](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html#warp-matrix-functions):
all lanes participate uniformly, shared matrix pointers are 32-byte aligned,
half leading dimensions are multiples of eight, and float dimensions multiples
of four. Store fragments to shared memory before indexed access; never assume
their internal element mapping. Do not use saturate-to-finite mode.

Use only packed formats and rows >=16; F16 and smaller batches retain the
accepted path. A 128-thread block owns 16 input rows and 64 output rows; each
warp computes a 16x16x16 product. Quantized integers fit half exactly (Q4/Q5
nonnegative small integers; Q6 signed -32..31), while the input is already half.
Within each aligned K16 group, packed scale/minimum are constant. Compute
`factor * dot(input, quant) + bias * sum(input)` in float, then accumulate those
group contributions in float. This avoids narrowing reconstructed quantized
weights to half, which could overflow valid large scales. F32 factorization
and reduction order still change and require the declared numeric gate.

One shared staging set per entry: A 16x16 half, B 64x16 half in column-major
form, C 16x64 float, row sums and per-output float factors/biases; <8 KiB total.
Zero-pad output/token tails, keep every lane through barriers, and retain eight
float output accumulators per thread. New checked launch grid is ceiling(N/64)
by ceiling(M/16), with the same fixed ABI and 128 threads. Reuse the trusted
module's static symbol inventory; no runtime strings or global allocations.

Predeclared C14 structure (no new files): `shaders/matmul.cuh` (~260 productive
category-K lines), `shaders/quant.cuh` (~125 category-K lines),
`kernels/matmul.rs` (~130 orchestration lines), `module.rs` (~150 orchestration
lines), `kernels/tests.rs` (~110 added test lines, excluded from the limit).
The K exemption is the rule for these single-operation format families; do not
fragment them solely for line count. Main risks: numeric factorization drift,
register pressure, shared staging/barriers, and fragment alignment/tail bugs.
The conservative local s=1.5 and o=0.02 include substantial staging uncertainty;
this is an instruction-work/reuse hypothesis, not a hardware-counter claim.

C14 gate before production: add packed Q4/Q5/Q6 reference cases at rows 16/17,
outputs 17/65, widths 256/768/3072. Repeat a five-block `patterned_weight` fixture
instead of exceeding that helper's signed-byte scale range. Compare every output
to stored-value scalar reference at the unchanged 0.02+2% bound; keep all old
tests unchanged. Include high-scale cancellation: d=65504, uniform quant=2,
alternating +/-1 input, expected zero, so weights exceed half range but valid
output does not. Run the new gate on C11 first, then canonical CPU/CUDA checks,
all local tests and pinned model parity on C14 before performance. Objective
short prompt throughput, C11 as baseline, TTFT/decode and medium/long controls.

### C14 kept

Baseline gate passed in `c14-baseline-test.log` before production edits and was
committed as `b13f6ad`: 36 dense shape/format cases plus three large-scale
cancellations. Existing tests remain unchanged. Implemented the predeclared
factorized packed WMMA path, selected only for rows >=16 and non-F16 weights.
The trusted function inventory gains one fixed entry; its existing failpoint
test automatically covers the additional resolution boundary. Zero-padded tails
never exit around barriers. The scratch matrices have explicit 32-byte alignment;
all fragment access uses the supported load/store operations. Product accumulators
and group factors/biases remain float; no reconstructed weight is rounded to half.

Production +134/-3 across the four predeclared files. Added one per-format
dequantization-parts function to expose the narrow K16 invariant and one cohesive
matmul variant; no global allocation, dependency, device setting or public API.
`run.py gate c14` passed fmt, CUDA workspace check, CPU suites, 11 error tests,
all 36 CUDA tests, and pinned f16 parity with all 16 top-one IDs unchanged.
Correctness completed before the immutable `c14-bench` short objective:

```sh
python3 target/cuda-amdahl-20260905/run.py bench c14 target/cuda-amdahl-20260905/c14-bench short
python3 target/cuda-amdahl-20260905/compare.py c11 c14 short prompt_tps short
# Only after provisional keep:
python3 target/cuda-amdahl-20260905/run.py bench c14 target/cuda-amdahl-20260905/c14-bench medium long
python3 target/cuda-amdahl-20260905/compare.py c11 c14 short prompt_tps short medium long
```

Short objective provisionally passes: prompt 70.30 -> 154.03 token/s
(+119.104%, CV 0.26% -> 0.09%), TTFT 1820.77 -> 831.01 ms (-54.359%). Model
decode 21.13 -> 21.11 (-0.095%, CV 0.37% -> 0.07%), public decode 20.51 ->
20.49 (-0.098%, CV 0.38% -> 0.07%). Counts 128/32/32 unchanged, wall 10.31 s.
PTX contains the expected aligned WMMA loads, float-accumulating matrix products,
and shared stores. No stability rerun needed; medium/long controls now run.

Verified parity reachability: `family/mistral/parity.rs` passes the complete
16-token prompt to `session.prefill`; CUDA reserves 32 prefill rows and graph
prefill forwards each chunk's actual length to batched matmul. Thus pinned
parity exercises the rows>=16 C14 route, not just the unchanged decode kernel.

C14 controls completed in session `48208`; all counts match C11. Full records
(means and fractional CV) and final decision:

| C14 regime | Prompt tok/s (CV) | TTFT ms (CV) | Model decode tok/s (CV) | Public deltas/s (CV) | Complete wall s |
|---|---:|---:|---:|---:|---:|
| Short objective | 154.03 (0.0009) | 831.01 (0.0009) | 21.11 (0.0007) | 20.49 (0.0007) | 10.31 |
| Medium control | 141.76 (0.0010) | 7223.61 (0.0010) | 17.80 (0.0035) | 16.71 (0.0034) | 37.03 |
| Long control | 110.38 (0.0002) | 32468.30 (0.0002) | 12.01 (0.0004) | 11.27 (0.0005) | 141.28 |

Medium prompt +112.725%, model decode +0.964%; long prompt +87.530%, model
decode +0.083%. Worst control is short public decode -0.098%, well inside the
5% bound. Short objective +119.104%, both objective CVs <5%, all correctness
gates pass; **keep**, no rerun, complete comparison within two hours. Stock
power-cap increments: 0.772 s short, 0.618 s medium, zero long; no thermal
slowdown. No system settings changed.

Keep the isolated +134/-3 production lines; the separate baseline test remains
committed. Complexity increases for an explicit factorized matrix path and
its shared-memory ownership, justified by the measured reduction in the
dominant prefill cost. It remains one matmul family with no global lifetime
or format/API expansion. Next acquire `c14-timeline` short/medium/long and rerank;
do not reuse obsolete broad matmul fractions for the SIMT-only candidates.

### Refresh after C14; C13 selected

All traces in session `53915` completed with zero dropped records. Short total
2345.908 ms: prefill/first sample 829.179 ms, decode 1516.730 ms. Tensor matmul
775.150 ms versus C11's legacy batched 1762.887 ms, local speedup 2.274x;
long 21804.171 versus 50224.132 ms, 2.303x. Tensor entry uses 128 threads,
48 registers and zero local-memory spill in the activity records. Long total
34974.345 ms; attention owns 9512.614 + 1194.614 ms. Trace long TTFT is 0.5%
below the unprofiled mean; traces remain diagnostic, not extra A/B attempts.

No legacy batched matmul occurs in any declared public row: **C04 closed-untried**,
owner p=0, ideal gain zero on its target. Its smaller-batch implementation is
preserved, but testing a different short prompt just to manufacture an effect
would change the campaign tuple. C08/C12 also lose their prefill owner; before
implementation, retarget their still-material single-matmul/logits mechanism to
short **model decode**, keeping the same exact-output gates and thresholds.
C09 affects both legacy and tensor half metadata. Current ranking:

| Priority / state | p full request | s | o | Predicted gain | Ideal ceiling | Saved ms |
|---|---:|---:|---:|---:|---:|---:|
| C13 ready, selected | 0.30615 long | 2 | 0.004 | 17.5% | 44.1% | 5213.7 |
| C07 ready | 0.04757 short | 8 | 0.0002 | 4.32% | 5.00% | 97.177 |
| C08 ready, decode objective | 0.56095 short | 1.04 | 0.0005 | 2.15% | 127.8% | 49.44 |
| C12 deferred until C08, decode objective | 0.56095 short | 1.04 | 0.001 | 2.10% | 127.8% | 48.27 |
| C09 ready | 0.89137 short | 1.02 | 0 | 1.78% | 820.5% | 41.00 |

Estimates are rounded; C07's decode-phase owner is 7.13%, and C08/C12 own 86.46%
of decode, so their phase-specific ceilings exceed 5%. C13 has the largest
remaining predicted effect. Added the predeclared `(3,4097)` boundary case to
the permanent attention test; all ten format/shape cases passed on accepted
C14 in `c13-baseline-test.log`, before its production changes.

### C13 rejected

Implemented the bounded buffered-score path with the unchanged online function
as fallback. Each warp writes disjoint scores; block barriers separate score,
maximum, exponential/sum and value phases. An explicit barrier after reading
the maximum protects the reuse of partial[0]. Both paths retain 128-thread
ownership and the existing validated cache bounds. Only the attention shader
changes (+77/-1); no new ABI, allocation, dependency or machine setting. Shared
storage is finite and below the already-required per-block capability; its
occupancy cost, including on fallback kernels, remains a measurement risk.

`run.py gate c13` passed formatting, CUDA workspace check, CPU suites, 11 error
tests, all 36 CUDA tests (ten dense attention boundary/format cases), and pinned
f16 parity with all 16 top-one IDs unchanged. Extra pinned int8 parity runs next,
then the long prompt objective against immutable C14. Correctness gates and
tolerances are unchanged; the shared softmax is a numeric, not exact-bit, candidate.

Extra int8 parity passed (`c13-int8-parity.log`), all 16 top-one IDs unchanged.
Build/copy immutable `c13-bench`, then acquire `run.py bench c13
target/cuda-amdahl-20260905/c13-bench long`. Compare with
`compare.py c14 c13 long prompt_tps long`; only a provisional keep proceeds to
short/medium controls. No C13 performance was observed before both parity gates.

C13 objective completed: prompt 110.38 -> 110.33 token/s (-0.045%, CV 0.02% ->
0.03%); TTFT 32468.30 -> 32485.24 ms (+0.052%). Model decode 12.01 -> 15.25
(+26.978%, candidate CV rounded to 0.0000), public decode 11.27 -> 14.32
(+27.063%, CV 0.02%). Counts 3584/32/31 unchanged, wall 139.11 s. **Rejected**:
the fixed prefill objective is below 3%; do not redefine it after observing
decode. No rerun or other controls needed. Saved `c13.patch` privately and
restored attention.cuh exactly to accepted HEAD with apply_patch.

Rejected-binary diagnostic session `35856` completed, zero dropped records:
prefill attention 9612.613 ms versus C14's 9512.614 ms, decode attention 637.976
versus 1194.614 ms. Tensor matmul remains 21811.752 versus 21804.171 ms. The
effect is phase-specific; the larger shared storage can constrain prefill
occupancy, but these timings alone do not prove that limiter. They do support
**C15**, a bounded variant: invoke the online body directly from prefill entries,
and the buffered body only from decode entries. Direct entry wiring must prevent
the buffered allocation from remaining reachable in prefill code.

C15 objective is declared now, before implementation: long model decode; prompt,
TTFT and short/medium controls unchanged. Full-request p=0.03416, conservative
local s=1.8, o=0.0005, predicted gain ~1.49%, ideal ceiling 3.54%, ~513 ms saved;
decode-phase p=0.44748 has an 80.99% ideal ceiling, making it eligible. Small
shader-only variant (~210 productive category-K lines), no new host interface
or global storage. Reuse C13's ten-case reference gate and pinned f16/int8 parity.
It is ready but ranks below C07/C08/C12/C09 by current whole-request estimates.

After C13 restoration, current fractions remain the C14 table. Select **C07**,
largest remaining ready prediction (4.321% whole request; decode ceiling 7.68%).
Before production, extend exact argmax coverage to lengths 1/3/255/256/257/
131072/131075, mixed raw float bit patterns, tied maxima across lanes, and all
negative-NaN minimum keys. Expected IDs use Rust total_cmp with first-index ties.
Structure: existing `shaders/argmax.cuh` (~45 productive category-K lines),
`kernels/argmax.rs` (~45 orchestration lines), one new existing-file test (~35
test lines). One 256-thread block scans strided inputs and reduces key/index
pairs; use a u64 scan index to avoid wrapping at the u32 length boundary.
Keep the exact total order and lowest-index tie invariant. Objective short model
decode, C14 reference, all canonical gates and standard controls/thresholds.

### C07 interesting, restored

The 21-case exact baseline test passed in `c07-baseline-test.log` and was
committed before production as `c8abb5f`. Implemented a 256-thread strided scan
and shared reduction of `(total_key, index)` pairs. A minimum-key negative NaN
still beats an inactive lane by its valid lower index; uint64 scan arithmetic
cannot wrap at the u32 length boundary. Dispatch changes only the block width.
Production +25/-8 in the two predeclared files, no new resource or interface.

`run.py gate c07` passed formatting, CUDA workspace check, CPU suites, 11 error
tests, all 37 CUDA tests and pinned f16 parity (all 16 top-one IDs unchanged).
The exact integer-order gate passed before `c07-bench` was built for the short
model-decode objective against C14. Other metrics and regimes remain controls.

Short result: model decode 21.11 -> 22.08 (+4.595%, CV 0.07% -> 0.18%), public
decode 20.49 -> 21.39 (+4.392%, CV 0.07% -> 0.19%). Prompt 154.03 -> 152.84
(-0.773%, CV 0.09% -> 0.05%); TTFT 831.01 -> 837.48 ms (+0.779%). Counts
128/32/32 unchanged, wall 10.11 s. **Interesting objective, not kept**: stable
gain is between 3% and 5%. Restored both production files exactly to HEAD,
retaining `c07.patch` privately. No selective rerun, no keep/control campaign.
Pool accounting includes C07 among restored-code outcomes and explicitly
distinguishes its interesting classification from the two below-3% rejections.
An immutable-binary short diagnostic runs next to bound residual sampling cost
before considering any variant. Do not lower the threshold to retain this code.

Diagnostic `c07-timeline-short` completed (status 0, zero dropped records):
decode argmax 108.200 -> 1.499 ms, decode matmul 1165.133 -> 1183.689 ms,
total decode 1516.730 -> 1429.319 ms. Attribution is diagnostic, not a replacement
for the unprofiled 4.595% result. Remaining argmax is only 0.105% of candidate
decode; even eliminating it cannot bridge the retention gap. No bounded
argmax-only variant is supported. C07 is terminal interesting/restored.

### C16 predeclared: format-granular WMMA correction

Fresh attribution using temporary `matrix.py` parses the authenticated GGUF
tensor index with bounded counts/strings, verifies the 26-layer graph launch
order, launch dimensions, dtype and dropped-record count, then partitions the
already captured C14 tensor durations. Q4_K owns 654.987/775.150 ms short and
18450.654/21804.171 ms long; Q6_K owns the rest. This is measured launch
attribution, not an estimate from weight bytes. All widths remain multiples of
256 and no public workload or format changes.

Q4/Q5 coefficients are constant over K32, whereas C14 corrects each K16 result.
C16 accumulates two WMMA products before one float correction and shared-store
cycle for Q4/Q5, retaining K16 for Q6's coefficient granularity. It halves
Q4/Q5 correction/barrier cycles without reconstructing large weights in half.
The M16/N64 geometry and dispatch are unchanged. More shared staging and changed
float association are the principal risks; no exact-bit claim is made.

Select long `prompt_tps` against immutable C14; fixed TTFT/decode and both other
regimes are controls. Whole-request p=0.527548, local s=1.4, added o=0.004,
predicted gain 17.196%, ideal ceiling 111.662%, estimated saving 5131.718 ms.
The speedup estimate is uncertain: only correction/barrier work is halved,
not tensor arithmetic or packed loads. Moderate-cost bounded shader experiment.
C16 ranks before C08 (2.153%), C12 (2.101%, deferred until C08), C09 (1.779%),
and C15 (1.49%); their earlier evidence and state remain unchanged.

Structure is the existing `cuda/shaders/matmul.cuh` (~210 productive category-K
lines), no new files or interfaces. Staging grows from 7232 to 9792 bytes,
within existing capability admission. K32 half leading dimensions and K16
slice offsets preserve WMMA alignment; no tail lane exits. Reuse unchanged
39-case packed-prefill range/reference gate, all 37 CUDA tests, CPU workspace,
CUDA check/error matrix and pinned f16/int8 parity before performance. All
thresholds and the complete-comparison two-hour limit remain fixed.

Implemented only the predeclared shader: constexpr GROUP, widened staging, and
two K16 fragment products before one K32 correction. Both matrix load offsets
remain 32-byte aligned and the same three barriers protect each complete group.
The change removes repeated work without adding a new path or abstraction.
Gate `run.py gate c16` is running in session `98184`; production is unaccepted.

Session `98184` completed: formatting, CUDA check, CPU workspace, 11 error
tests, all 37 CUDA tests and pinned f16 parity passed. Extra int8 parity also
passed in `25139`, all 16 local top-one IDs equal the fixed oracle in both.
Built/copied immutable `c16-bench`; command `run.py bench c16
target/cuda-amdahl-20260905/c16-bench long` acquires the predeclared objective.
C16 is attempt ten, but the pool is not terminal and the campaign must continue.

Long objective session `18659` completed: prompt 110.38 -> 116.17 (+5.2455%,
CV 0.02% -> 0.19%), TTFT 32468.30 -> 30850.71 ms (-4.9821%), model decode
12.01 -> 12.02 (+0.0833%, CV 0.04%), public 11.27 -> 11.28 (+0.0887%, CV
0.04%). Counts 3584/32/31 unchanged; wall 134.82 s. Provisional keep only:
`run.py bench c16 target/cuda-amdahl-20260905/c16-bench short medium` runs both
controls before retention. No stability rerun is permitted or needed.

### C16 kept

Both controls completed in session `64663`. Complete comparison command:
`compare.py c14 c16 long prompt_tps short medium long`. All correctness gates,
stability and controls pass; **keep**. Candidate means (CV fraction in brackets):

| Regime | Prompt token/s | TTFT ms | Model decode token/s | Public delta/s | Wall s |
|---|---:|---:|---:|---:|---:|
| short | 164.69 [.0037] | 777.24 [.0036] | 20.97 [.0010] | 20.35 [.0010] | 10.16 |
| medium | 150.44 [.0002] | 6806.50 [.0002] | 17.71 [.0006] | 16.63 [.0006] | 35.29 |
| long | 116.17 [.0019] | 30850.71 [.0019] | 12.02 [.0004] | 11.28 [.0004] | 134.82 |

Prompt gains versus C14: +6.921/+6.123/+5.246%; worst control is short public
decode -0.683%. Counts unchanged in every regime. Stock power-cap counter
increments short .693843 s, medium 0, long .913890 s; thermal counters remain
zero and no device settings changed. These sub-second transients are disclosed
with the same capture protocol as baseline, not selectively rerun. This bounded
three-repetition result clears the declared rule; CV is not a confidence interval.
Production is +23/-17 in one category-K shader, no new files or dependencies.
All other production candidates are absent. Refresh all three timelines before
reranking; current retained runtime includes C16 in this commit.

All `c16-timeline` rows completed in `14836`, zero dropped records. Short
prefill/decode 772.125/1517.904 ms, long 30724.563/2661.221 ms. Tensor durations
short 718.841, long 20235.622 ms; Q4 alone 599.080/16878.633 ms versus C14
654.987/18450.654. Q6 119.760/3356.989 ms is effectively unchanged. Register
count remains 48, no local memory; PTX confirms the declared 9792-byte staging.
This supports the mechanism but not the optimistic local 1.4 speedup estimate.
Long prefill attention remains 9508.299 ms, decode attention 1196.927 ms.

Recomputed ranking (full-request p, with selected-phase eligibility unchanged):

| ID | p | s | o | Predicted gain | Ideal gain | Saved ms | State |
|---|---:|---:|---:|---:|---:|---:|---|
| C08 | .57594 short | 1.04 | .0005 | 2.213% | 135.813% | 49.58 | selected |
| C12 | .57594 short | 1.04 | .001 | 2.161% | 135.813% | 48.44 | deferred until C08 |
| C09 | .88984 short | 1.02 | 0 | 1.776% | 807.732% | 39.96 | ready |
| C15 | .03585 long | 1.8 | .0005 | 1.567% | 3.718% | 515.3 | ready |

Short decode matmul+logits own 86.59%; C15 long decode attention owns 44.98%.
These are ownership ceilings, not measured removable instruction fractions.
No new argmax premise follows from C16, which does not materially change decode.

### C08 predeclared implementation

Objective remains short model decode, now against accepted immutable C16. Host
`kernels/matmul.rs` is the only dispatcher of single/batched matmul and logits,
and always launches 128 threads; current PTX still reads `%ntid.x` inside these
loops and emits runtime reduction bounds. Replace those shader loop bounds with
128 and reduction start 64, plus a local invariant comment; leave all launch
geometry, scratch size, tree pairing and float operation order unchanged.
This is not the old block-size experiment. Existing `shaders/matmul.cuh`
(~210 productive category-K lines), no new file/function. The main risk is a
future dispatch mismatch, documented at the kernel boundary.

Before performance, require exact equality of all 26 frozen wide-dot bit
snapshots (`matmul-wide-c05-baseline.log`; rows 1/9 remain on unchanged SIMT),
then remove temporary printing and run all canonical gates and pinned parity.
Both other regimes are controls after a provisional keep. PTX must demonstrate
constant bounds; correctness or exact-gate failure restores the candidate.

Implemented C08 in one shader (+6/-4): fixed iteration step and reduction start
in both SIMT functions. Exact gate `c08-exact.log` passed in `53241`; all 26
`cuda-matmul-bits` lines equal the frozen baseline (`diff -u` empty). Removed
the temporary eprintln before canonical `run.py gate c08`. No test edits remain.

Canonical gate `72973` completed: formatting, CUDA check, CPU workspace, 11
error tests, all 37 CUDA tests and pinned f16 parity passed. All 16 top-one IDs
are unchanged. Saved generated `c08.ptx`: no `%ntid.x` remains in the three
SIMT entry bodies; K increments are constant 128 and reduction is unrolled.
K-loop packed metadata loads are still present, so C12's pairing premise is not
closed by compilation alone. Built/copied `c08-bench`; acquire short objective
with `run.py bench c08 target/cuda-amdahl-20260905/c08-bench short` against C16.

### C08 rejected

Objective `15242` completed: model decode 20.97 -> 21.20 (+1.0968%, CV 0.10%
-> 0.23%), public 20.35 -> 20.58 (+1.1302%, CV 0.20%), prompt 164.69 ->
165.37 (+0.4129%, CV 0.09%), TTFT 777.24 -> 774.04 ms (-0.4117%, CV 0.09%).
Counts 128/32/32 unchanged, wall 10.11 s. **Reject** below 3%, no rerun or
control campaign. Saved `c08.patch` and restored the shader exactly to HEAD.
An immutable-binary diagnostic (`14855`) measures the remaining dot owner.
Constant bounds did not remove enough work; no unchanged geometry variant is
supported. C12 remains independently supported by repeated packed metadata
loads in the K loop, not by C08's insufficient loop-control effect.

### C12 predeclared implementation

C16 remains the accepted baseline and fractions. Select C12 (2.161% conservative
whole-request prediction, 135.813% ideal ceiling, ~48.44 ms saved), then C09
(1.776%) and C15 (1.567%). C12 is no longer deferred. Objective short model
decode; same exact 26-case gate, canonical gates, fixed controls and thresholds.

In existing `shaders/matmul.cuh` (~235 productive category-K lines), change only
the packed single-dot loop: iterate 256-value blocks, compute one block address,
call existing q4/q6 accessors for thread index and index+128, then add the two
products in their original per-thread sequence. All callers already launch 128
threads and packed widths are validated multiples of 256. Keep F16 loop and
shared reduction identical to accepted C16. The fixed pair is necessary to
express metadata reuse, not a bundled C08 reduction optimization. No new
function, layout, allocation, shuffle or interface. Main risk is preserving
dequant arithmetic and exact sum order. Require generated-code evidence that
the block's half metadata loads are reused; all numerical expressions retain
the original accessor definitions. Temporary bit prints are removed after the
exact gate, before the canonical gate and performance.

C08 diagnostic completed with zero dropped records: decode matmul 1168.382 ms
versus C16 1167.725, logits 148.880 versus 146.607; registers rise 28 -> 32,
local bytes stay zero. No material kernel gain is visible in this single trace;
it does not establish occupancy as the cause. C08 remains rejected/restored.

C12 exact gate `25012` passed and all 26 snapshots match frozen baseline.
Temporary prints removed. Generated PTX proves reuse: within each Q5 block loop
the bytes at block+0/+1 and +2/+3 load once into two half registers, reused by
both accessor conversions; two successive `fma.rn.f32` instructions preserve
the per-thread sum sequence. Q4/Q6 likewise retain original accessor math.
Canonical `run.py gate c12` runs in `5144`; fast build in `17360` is not a
performance acquisition. No rejected C08 reduction changes are present.

C12 canonical gates completed: formatting, CUDA workspace check, CPU suites,
11 error tests, all 37 CUDA tests and pinned f16 parity passed, all 16 top-one
IDs unchanged. Production +20/-2, one shader. Saved `c12.ptx` and immutable
`c12-bench`; objective command `run.py bench c12
target/cuda-amdahl-20260905/c12-bench short`, reference C16. Attempt 12 is pending
public outcome; other controls run only after provisional keep.

C12 short objective `8848` passed provisionally: model 20.97 -> 27.62
(+31.712%, CV 0.10% in both), public 20.35 -> 26.82 (+31.794%, CV 0.11%),
prompt 164.69 -> 165.48 (+0.480%, CV 0.06%), TTFT 777.24 -> 773.52 ms
(-0.479%, CV 0.06%). Counts 128/32/32 unchanged, wall 8.70 s. Run
`run.py bench c12 target/cuda-amdahl-20260905/c12-bench medium long` before keep.
The conservative local 1.04 estimate understated the mechanism's effect;
refreshed attribution, not the source-level load count, must explain the gain.

Continued read-only discovery of current largest prefill owner adds **C18**.
In `c12.ptx` (same C16 WMMA body), Q5 loop `$L__BB4_40` loads block half bytes,
converts and multiplies scale/bias for every quant; only later `$L__BB4_41`
tests the local K index and skips coefficient stores for 31/32 lanes. Source
confirms Q4/Q5 need coefficients only at K32 boundaries, Q6 only at K16. This
is directly observed discarded instruction work, not inferred bandwidth saturation.

C18 gates coefficient calculations in the existing `cuda_weight_parts` helper
at those format-defined boundaries. Its only caller uses factor/bias solely
for the corresponding owner lane. All integer quants and consumed coefficients
remain identical; no shared storage, barriers, geometry, ABI or float operation
order changes. Existing `shaders/quant.cuh` (~120 productive category-K lines),
same matmul caller; no new function/file. Risk: preserve the private helper's
owner-only result invariant and all Q4/Q5/Q6 alignment boundaries explicitly.
Declare exact SIMT 26-case and tensor 39-case snapshots on accepted runtime,
existing bounded tests, all canonical gates and pinned f16/int8 parity before A/B.
Long prompt objective, both other regimes and TTFT/decode controls unchanged.
Current C16 whole-request p=.6061, conservative s=1.2, o=.001 predicts ~11.1%,
ideal ceiling ~154%, ~3339 ms saved; lower instruction counts do not guarantee
that gain because branch divergence and cache-hot loads can limit it. Small
bounded experiment. Rerank after C12's accepted-or-restored state is known.

Also reserve **C17** as a conditional re-evaluation of C07: only an accepted
C12 and refreshed decode fraction can change the premise behind C07's 4.595%
result. This would be the same exact argmax implementation on a materially
changed accepted runtime, **not an additional countable attempt**, a duplicate
mechanism, or a stability rerun. Until that evidence exists it stays deferred.

### C12 kept

Controls completed in `49897`; `compare.py c16 c12 short model_decode_tps
short medium long` passes every fixed threshold. **Keep** after exact snapshots,
canonical correctness and all controls; no rerun. Candidate means [CV fraction]:

| Regime | Prompt token/s | TTFT ms | Model decode token/s | Public delta/s | Wall s |
|---|---:|---:|---:|---:|---:|
| short | 165.48 [.0006] | 773.52 [.0006] | 27.62 [.0010] | 26.82 [.0011] | 8.70 |
| medium | 151.54 [.0025] | 6757.31 [.0025] | 22.21 [.0012] | 20.86 [.0015] | 33.76 |
| long | 116.39 [.0009] | 30792.26 [.0009] | 13.89 [.0025] | 13.04 [.0024] | 133.16 |

Model decode gains +31.712/+25.409/+15.557%; prompt +.480/+.731/+.189%.
Counts unchanged in all regimes; no control regression. Stock power-cap counter
increments short .973722 s, medium .733843 s, long 0; thermal counters zero,
no setting change or competing inference. Added branch/loop structure is local
and necessary to make one packed-block address own both contributions; no new
abstraction, interface, dependency, format or resource ownership. Production
one shader +20/-2. C12 is kept in this commit; refresh affected timelines and
recompute fractions before selecting C18/C17/C09/C15.

Refreshed `c12-timeline` completed in `29045`, zero dropped records. Short
prefill/decode 768.673/1157.970 ms; decode matmul 861.412 versus C16 1167.725,
logits 92.126 versus 146.607. Registers 40 versus 28, no local storage. Despite
more registers, shared metadata/address reuse produces a material kernel gain;
do not infer occupancy or bandwidth saturation from timing alone. Long
prefill/decode 30746.261/2302.753 ms; tensor 20244.451 ms and prefill attention
9512.112 ms remain unchanged within diagnostic variability.

Current ranking from these profiles:

| ID | Whole-request p | s | o | Predicted gain | Ideal gain | Saved ms | State |
|---|---:|---:|---:|---:|---:|---:|---|
| C18 long | .612558 | 1.2 | .001 | 11.246% | 158.103% | 3341.03 | selected |
| C17 short | .057774 | 8 | .0002 | 5.302% | 6.132% | 97.01 | ready, non-counting re-evaluation |
| C09 short | .868774 | 1.02 | 0 | 1.733% | 662.042% | 32.82 | ready |
| C15 long | .035945 | 1.8 | .0005 | 1.572% | 3.729% | 511.45 | ready |

C17 now has a changed premise: decode argmax remains 107.926 ms while decode
falls from 1517.904 to 1157.970 ms; owner rises from 7.06% to 9.32%, ideal
phase gain 10.28%. Its original implementation may therefore be re-evaluated,
but the campaign's distinct-mechanism count will not increase for it.

### C18 exact baseline

Temporary prints snapshot all 39 tensor cases on accepted C12 using
`cargo test -p graph_horizon_engine --locked --no-default-features --features
cuda packed_prefill_tiles_match_reference_and_preserve_weight_range --
--nocapture --test-threads=1`, output `c18-tensor-baseline.log`, session `48860`.
The unchanged 26 SIMT snapshots remain the original frozen baseline (C12 exact
gate already matched them). Only test instrumentation is present at this stage.
After this baseline passes, implement the predeclared owner-lane guard in
`quant.cuh`; select long prompt against C12 with all predeclared gates/controls.
