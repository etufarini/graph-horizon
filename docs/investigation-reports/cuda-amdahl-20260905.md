# CUDA Amdahl campaign 2026-09-05

## Recovery state

- Campaign ID: `cuda-amdahl-20260905` (new campaign; historical campaign completed).
- Branch: `perf/cuda-amdahl-20260905`.
- Immutable start: `62384895acfde94cf28fded1294ad860daabf2ff`.
- Current retained runtime: `c15f222` (C30), above `4d9e718` (C31), `1a32047` (C29) and the earlier accepted commits listed below; S01 remains a separate correctness fix.
- State: running, C30 accepted and all profiles refreshed. C28 rejected and fully restored. C32 deferred for a Q6-prefill-only cache design. Not complete.
- Attempts: 25 distinct reached correctness (minimum 10): 15 kept, 9 rejected (C06/C08/C13/C18/C20/C23/C25/C26/C28), 1 interesting/restored (C07); plus 4 non-counting re-evaluations kept (C17/C22/C27/C29). Total retained optimization commits19 plus separate S01 correctness commit, not_verified0, closed-untried2 (C04/C10). No overall deadline; each A/B comparison has a two-hour limit.
- Persistent private evidence: `target/cuda-amdahl-20260905/`.
- Current-campaign retained optimization commits: `61a540a` (C01), `f251074` (C05), `2547df3` (C03), `14b669d` (C02), `b01470c` (C11), `b948f67` (C14), `7682cd2` (C16), `eebe52a` (C12), `8ab48ff` (C19), `65ef6ed` (C17 re-evaluation), `56afc25` (C15), `6a77c87` (C09), `cd6993b` (C21), `e7790cd` (C22 non-counting bounded re-evaluation), `99f4330` (C24), `2d77b2f` (C27 non-counting re-evaluation), `1a32047` (C29 non-counting bounded re-evaluation). Separate correctness support: S01 `6d37587`.

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
`36c9912`; C13 was subsequently rejected/restored. C07 is interesting/restored;
C08 and C18 are rejected/restored. C16, C12 and C19 were subsequently kept with
complete controls and refreshed profiles; the latest states are in the pool
and chronological decision sections below. C04/C10 are closed-untried with zero
current target owner. Continue this pool.
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
| C09 | Read aligned packed f16 metadata with one 16-bit load instead of two byte loads plus recombination; PTX confirms repeated byte loads in hot matmul | Short model decode (retargeted before first attempt after C19) | kept | Exact/canonical/f16/int8 gates and all controls pass; objective +9.439% |
| C10 | Warp-first attention reduction to remove remaining inter-warp shared-tree stages; unlike C05 this changes pairing | Long prompt throughput | closed-untried | C11 removes that shared tree completely: current owner p=0 and ideal gain 0% |
| C11 | Four context scores per attention block iteration, one warp per score; merge their online softmax together | Long prompt throughput | kept | Numeric and f16/int8 parity gates plus all controls passed; objective +22.881% |
| C12 | Pair the two 128-wide K slices belonging to one quantized block, reusing block address and half metadata within each thread | Short model decode | kept | Exact/canonical gates and controls pass; objective +31.712% |
| C13 | Buffer attention scores in block-shared memory, parallelize stable softmax, then scan V without per-context barriers | Long prompt throughput | rejected, restored | Correctness passed; prefill -0.045% fails its objective despite decode +26.978% |
| C14 | Factorized quantized prefill via warp matrix instructions, applying float scale/bias outside exact integer-quant products | Short prompt throughput | kept | All numeric/parity gates and controls passed; objective +119.104% |
| C15 | Isolate buffered attention to decode, preserving online prefill without its shared score allocation | Long model decode | kept | Strengthened numeric/canonical/f16/int8 gates and controls pass; objective +34.066% |
| C16 | Apply Q4/Q5 float correction once per format-defined K32 coefficient group, retaining Q6 K16 | Long prompt throughput | kept | Numeric/parity gates and all controls pass; objective +5.246% |
| C17 | Re-evaluate exact argmax after C12's retained decode reduction materially raises its removable fraction | Short model decode | kept, non-counting re-evaluation | Exact/canonical gates and all controls pass; objective +8.529% |
| C18 | Compute WMMA scale/bias only for the coefficient-owner lanes, retaining integer quant work for every lane | Long prompt throughput | rejected, restored | Exact/canonical gates pass; prompt -13.747%, TTFT +15.946% |
| C19 | Pack 64 coefficient owners into two active warps before the quant-staging loop | Long prompt throughput | kept | Exact/canonical gates and all controls pass; objective +31.205% |
| C20 | Reuse WMMA weight/coefficient staging across 32 tokens instead of two independent 16-token blocks | Medium prompt throughput | rejected, restored | All exact/canonical gates pass; objective +1.613% below 3% |
| C21 | Give each physical thread two logical reduction leaves sharing packed nibble bytes, preserving the 128-leaf tree | Short model decode | kept | Exact/canonical/f16/int8 gates and all controls pass; objective +20.095% |
| C22 | Restrict C20 weight reuse to large output grids that remove a resource wave; preserve M16 elsewhere | Medium prompt throughput | kept, non-counting bounded re-evaluation | Exact/canonical/extra parity and all controls pass; objective +7.240% |
| C23 | Keep online softmax coefficients warp-local to remove their block-wide publication barrier | Long prompt throughput | rejected, restored | Exact/canonical gates pass; prompt -14.059%, TTFT +16.358% |
| C24 | Give each prefill query head one independent warp, packing four exact score trees into eight-lane subgroups | Long prompt throughput | kept | Exact gates pass; long prompt +13.665%, all controls within 5% |
| C25 | Stage two adjacent Q4/Q5 coefficient groups together, reusing packed low/high bytes and synchronization | Medium prompt throughput | rejected, restored | Exact/canonical/frozen/sanitizer gates pass; medium prompt -3.214% |
| C26 | Keep C25 byte/barrier reuse but store each stage with the original K32 shared row pitch | Medium prompt throughput | rejected, restored | All exact/canonical/frozen/sanitizer gates pass; medium prompt +2.580%, below3% |
| C27 | Consolidate C21's exact logical leaves in one warp per packed output, preserving four partial sums | Short model decode | kept, conservative non-counting C06 re-evaluation | Short decode +46.265%, all controls pass; exact107/canonical/frozen/sanitizer gates pass |
| C28 | Cache decoded integer quants and f32 coefficients losslessly at load time | Short model decode, prefill controls | rejected, restored | All exact/canonical/frozen/hybrid/sanitizer gates pass; objective -11.700%, public control -11.776% |
| C29 | Restrict C26 stage-major pairing to Q4/Q5 output grids<=3072, preserving ordinary larger grids | Medium prompt throughput | kept, non-counting bounded re-evaluation | Medium prompt +5.120%; all controls pass; exact/canonical/frozen/sanitizer gates pass |
| C30 | Parallelize long-history buffered decode within512-thread blocks, merging four f32 V partial sums | Medium model decode, long control | kept | Medium model decode +19.457%, long +61.511%; all numeric/canonical/frozen/sanitizer gates and controls pass |
| C31 | Share F16 K/V tiles across four prefill queries and use existing half-input/f32-accumulator WMMA for QK | Long prompt throughput | kept | Long prompt +21.603%; all numeric/canonical/frozen/sanitizer gates and controls pass |
| C32 | Cache only Q6 projection prefill while retaining raw decode/embedding/logits weights | Medium prompt throughput | kept | +12.063%, all controls pass; commit10b186c; non-counting C28 re-evaluation |
| C33 | Lane-owned MMA accumulators remove shared C round trip and one block barrier | Medium prompt throughput | rejected, restored | All gates pass, medium prompt-15.461%; direct loads have concrete shared-bank collisions; keep C32 |
| C34 | Share the existing padded M16 attention tile across eight real queries | Long prompt throughput | kept | Long prompt+7.706%; worst control-2.816%; exact/canonical/frozen/hybrid/sanitizer gates pass; non-counting C31 re-evaluation |
| C35 | Pad C33 MMA shared rows to remove the concrete scalar-load bank collisions | Medium prompt throughput | rejected, restored | Exact/canonical/frozen/hybrid/sanitizer gates pass, prompt-11.466%; non-counting C33 re-evaluation |
| C36 | Preserve direct accumulator ownership while using optional K16 MMA on capable targets | Medium prompt throughput | deferred behind C34 | Compiled C32 uses HMMA.16816 but C33 forces HMMA.1688; needs separate target-compatible PTX with unchanged75 fallback |

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

Baseline `48860` passed with 39 snapshots. Implemented owner-lane guards only
in `cuda_weight_parts`, whose sole caller is the WMMA staging function. All 26
SIMT and all 39 tensor snapshots match bit-for-bit in `c18-exact.log` (`86450`,
65 markers, both diffs empty). Restored the test file exactly to HEAD to remove
all temporary prints before canonical `run.py gate c18`.

The lane guard preserves results, but SIMD execution is an important uncertainty:
with K-contiguous staging a warp can still execute coefficient instructions for
its one active owner lane. Thus 31/32 discarded lane results do not imply 31/32
fewer warp instructions. The original s=1.2 estimate remains only a hypothesis;
the unprofiled outcome must decide. A compact coefficient-owner phase would be
a distinct causal variant if the guard alone does not reduce warp work.

Canonical C18 gate `69639` completed: formatting, CUDA check, CPU workspace,
11 error tests, all 37 CUDA tests and pinned f16 parity passed, all 16 top-one
IDs unchanged. Fast build `86864` completed; immutable `c18-bench` and `c18.ptx`
saved. Extra int8 parity precedes the fixed long prompt objective against C12.
Only `quant.cuh` is changed in production (+15/-10); tests restored exactly.

Extra int8 parity passed in `23517`, all 16 local IDs unchanged. Acquire
`run.py bench c18 target/cuda-amdahl-20260905/c18-bench long`, then
`compare.py c12 c18 long prompt_tps long`; both controls only after a
provisional keep. Baseline C12 and candidate remain within the two-hour window.

Read-only C18 PTX confirms the lane guard branches before coefficient loads,
but every K32 staging warp still contains an owner. This supports a distinct
bounded **C19**: move the 64 coefficient owners into one `threadIdx.x < 64`
phase before quant staging. Each owner calls the existing helper at group base
and stores factor/bias; the B loop consumes only the quant result, allowing
unused coefficient calculations to disappear after inlining. Two full owner
warps replace coefficient work issued by every staging warp. Preserve the same
barrier, staging, WMMA layout, consumed arithmetic and output bits. This changes
ownership grouping, not merely the guard condition or a tuning parameter.

C19 is deferred until C18's objective decision, then must be reranked. Existing
`matmul.cuh` (~240 productive category-K lines), no helper/file/ABI or extra
storage. Main risks are compiler dead-code elimination and strided metadata
loads; require PTX evidence for both compact owners and metadata-free quant
staging, exact 26+39 snapshots and all canonical/f16/int8 gates. Long prompt
objective, same controls. On C12 p=.612558, s=1.25, o=.001: predicted ~13.83%,
ideal 158.10%, saved ~4015.84 ms; estimates depend on the currently unmeasured
coefficient instruction share and must not be interpreted as promised speedup.

### C18 rejected

Long objective `46038` completed: prompt 116.39 -> 100.39 (-13.7469%, CV 0.09%
-> 0.01%), TTFT 30792.26 -> 35702.32 ms (+15.9458%, CV 0.01%), model decode
13.89 -> 13.92 (+0.2160%, CV 0.03%), public 13.04 -> 13.06 (+0.1534%, CV
0.03%). Counts 3584/32/31 unchanged, wall 152.98 s. **Reject** for objective
regression and TTFT control regression. No rerun or other controls. Saved
`c18.patch` and restored quant.cuh exactly to accepted HEAD. Immutable-binary
diagnostic `run.py trace c18-timeline .../c18-bench long` runs in `77729`.

C18 validates no runtime change; its guard does not reduce the number of
coefficient-executing warps. Source/PTX and the measured regression support
the already-declared C19 compact-owner variant. C12 remains baseline, so the
current profile fractions remain valid: select C19 (~13.83% conservative whole
request), then C17 (5.302%), C09 (1.733%), C15 (1.572%). No deferred entry is
silently closed. C19 changes only matmul.cuh, with original quant.cuh restored;
all 65 exact snapshots and canonical/f16/int8 gates precede long objective.

C18 diagnostic `77729` completed with zero dropped records: tensor prefill
25163.176 ms versus C12 20244.451, prefill attention 9503.026 versus 9512.112;
decode remains 2304.728 versus 2302.753 ms. Tensor registers stay 48, no local
memory. This localizes the regression to the guarded staging kernel and does
not establish bandwidth/occupancy as the limiter.

Implemented C19 only in matmul.cuh: compact coefficient phase before A/B staging,
removed interleaved coefficient stores from B loop. Shared bytes, barriers,
formats and numeric expressions unchanged. Exact `64832` passed: all 26 SIMT
and 39 tensor snapshots match frozen accepted output, both diffs empty.
Temporary test instrumentation restored exactly to HEAD before `run.py gate c19`.
Fast build `59682` completed. PTX now branches at thread index >63 around the
coefficient-only phase; its body contains half scale/min loads but no quant
loads. B staging contains only packed quant loads/bit extraction and integer-to-
half conversion, with no discarded coefficient loads or float scaling. Compiler
dead-code elimination therefore realizes the predeclared mechanism.

C19 canonical gates `42948` passed: formatting, CUDA workspace check, CPU
workspace, 11 error tests, all 37 CUDA tests and pinned f16 parity with all
16 top-one IDs unchanged. Saved `c19.ptx` and `c19-bench`; extra int8 parity
runs before the long objective. Production +11/-4 in the existing shader,
original helper contract preserved; conceptual complexity stays local and the
ownership invariant is explicit. This is countable attempt 14, not accepted yet.

Extra int8 parity `24452` passed, all 16 IDs unchanged. Acquire
`run.py bench c19 target/cuda-amdahl-20260905/c19-bench long` against C12,
then `compare.py c12 c19 long prompt_tps long`; no performance was measured
before all declared correctness gates passed.

C19 long objective `85361` passed provisionally: prompt 116.39 -> 152.71
(+31.2054%, CV .09% -> .01%), TTFT 30792.26 -> 23469.56 ms (-23.7810%, CV
.01%), model decode 13.89 -> 13.90 (+.0720%, CV .05%), public 13.04 -> 13.05
(+.0767%, CV .05%). Counts 3584/32/31 unchanged; wall 104.01 s. Run
`run.py bench c19 target/cuda-amdahl-20260905/c19-bench short medium` before
retaining the candidate. No rerun; the prediction understated the measured gain.

### C19 kept

Both controls completed in `2936`; complete comparison `compare.py c12 c19
long prompt_tps short medium long` passes all declared gates. **Keep** after
exact output, canonical/f16/int8 correctness, objective and both controls.
Candidate means [CV fraction]:

| Regime | Prompt token/s | TTFT ms | Model decode token/s | Public delta/s | Wall s |
|---|---:|---:|---:|---:|---:|
| short | 250.15 [.0004] | 511.69 [.0004] | 27.67 [.0028] | 26.88 [.0029] | 7.62 |
| medium | 217.98 [.0002] | 4697.63 [.0002] | 22.10 [.0004] | 20.77 [.0008] | 25.40 |
| long | 152.71 [.0001] | 23469.56 [.0001] | 13.90 [.0005] | 13.05 [.0005] | 104.01 |

Prompt gains +51.166/+43.843/+31.205%; worst control medium model decode
-.4953%. Counts unchanged in all regimes; no rerun. Stock power-cap increments
short .823520 s, medium 0, long .593948 s; thermal counters remain zero. No
setting change or competing inference. This is a bounded three-repetition
retention result, not an inferential significance claim. One shader +11/-4,
no helper changes or new resource/interface. C19 is retained in this commit;
refresh all affected profiles before C17/C09/C15 selection.

C19 refreshed profiles `11982` completed, zero dropped records. Short
prefill/decode 513.612/1162.282 ms, tensor 461.953 ms versus C12 717.319;
long 23384.748/2297.645 ms, tensor 12909.205 versus 20244.451 ms. Tensor
registers rise 48 -> 56, local bytes stay zero; the measured gain survives this
resource tradeoff. Long prefill attention 9485.067 ms is now the second-largest
owner; decode attention remains 1191.679 ms. Recompute and rank ready entries:

| ID | Whole-request p | s | o | Predicted gain | Ideal gain | Saved ms | State |
|---|---:|---:|---:|---:|---:|---:|---|
| C17 short | .065310 | 8 | .0002 | 6.039% | 6.987% | 95.44 | selected re-evaluation |
| C15 long | .046401 | 1.8 | .0005 | 2.054% | 4.866% | 516.79 | ready |
| C09 short | .848306 | 1.02 | 0 | 1.691% | 559.221% | 27.88 | ready |

C15 is phase-eligible: long decode owner is 51.87%, ideal phase gain >100%.
C17 short decode argmax 106.056/1162.282 ms =9.125%, ideal phase gain 10.04%.
C12, not a favorable rerun, changed that premise; C19 further reduced the
whole-request denominator. `git diff c8abb5f HEAD --` both argmax files is empty.

### C17 re-evaluation predeclared

Reapply exactly private `c07.patch` via apply_patch: `shaders/argmax.cuh`
(~45 productive category-K lines), `kernels/argmax.rs` (~45 orchestration lines),
no new files/tests/interfaces. Same 256-thread total-key/first-index reduction,
u64 strided scan and four-byte output invariant. Existing 21-case raw-bit/tie
test is the exact gate; all canonical gates and pinned f16 parity precede short
model-decode objective against immutable C19, then both other controls before
keep. All thresholds unchanged. Preserve C07's original interesting result;
C17 is one re-evaluation comparison, not attempt 15 or a duplicate new premise.

Applied exactly C07's production patch: `cmp c07.patch c17.patch` succeeds.
Canonical `55425` passed formatting, CUDA check, CPU workspace, 11 error tests,
all 37 CUDA tests including the 21-case total-order gate, and pinned f16 parity
with all 16 top-one IDs unchanged. Fast build `64074` completed; copied immutable
`c17-bench`. Acquire `run.py bench c17 .../c17-bench short`, then
`compare.py c19 c17 short model_decode_tps short`; both other controls follow
only provisional keep. Distinct attempt count remains 14.

C17 short objective `19504` passes provisionally: model decode 27.67 -> 30.03
(+8.5291%, CV .28% -> .29%), public 26.88 -> 29.08 (+8.1845%, CV .29%), prompt
250.15 -> 250.25 (+.0400%, CV .41%), TTFT 511.69 -> 511.50 ms (-.0371%, CV
.41%). Counts 128/32/32 unchanged; wall 7.28 s. The changed accepted runtime
now supports a gain above the same fixed threshold; original C07 evidence is
not rewritten. Run `run.py bench c17 .../c17-bench medium long` before keep.

Read-only C15 gate audit: the long-history test already checks the last decode
row against prefill using the bounded numeric gate, but for context 4097 that
last row is the online fallback (position 4096). Before C15 production, extend
the existing test to decode all three rows, including buffered positions
4094/4095 and fallback 4096, and compare each against both prefill and its scalar
reference. Keep the existing tiny-history exact equality test unchanged. This
strengthens the selected phase-specific gate; it does not relax any tolerance.
Existing test file only (~25 additional test lines), no new helper or test file.

### C17 kept (non-counting re-evaluation)

Controls `88848` completed. `compare.py c19 c17 short model_decode_tps short
medium long` passes correctness, stability and every control: **keep**. Means
[CV fraction] on the same accepted runtime plus the exact C07 patch:

| Regime | Prompt token/s | TTFT ms | Model decode token/s | Public delta/s | Wall s |
|---|---:|---:|---:|---:|---:|
| short | 250.25 [.0041] | 511.50 [.0041] | 30.03 [.0029] | 29.08 [.0029] | 7.28 |
| medium | 218.67 [.0003] | 4682.92 [.0003] | 23.98 [.0004] | 22.47 [.0005] | 25.07 |
| long | 152.51 [.0015] | 23500.22 [.0015] | 14.56 [.0003] | 13.65 [.0001] | 103.55 |

Decode gains +8.529/+8.507/+4.748%; worst control long TTFT +.1306% (prompt
-.1310%). Counts unchanged in every regime. No rerun. Stock power-cap increments
short .554069 s, medium .794137 s, long 0; thermal counters zero. All tuple and
machine-setting constraints unchanged. The 14 distinct attempts remain 9 kept,
4 rejected and 1 originally interesting; C17 adds a separately reported retained
re-evaluation, not a fifteenth distinct attempt. This preserves original C07
history while retaining its now-qualified implementation. C17 is kept in this
commit, production +25/-8 across the two predeclared files.

C15's strengthened baseline test was compiled with `--no-run`, then executed
after C17 controls with `cargo test -p graph_horizon_engine --locked
--no-default-features --features cuda attention_long_history_and_dimension_tails_match_reference
-- --nocapture --test-threads=1` (`c15-baseline-test.log`): pass. All thirty
decode rows now satisfy both the scalar reference and existing prefill bound;
the tiny-history exact assertion remains unchanged. This separate test-only
checkpoint precedes any C15 production edit. Refresh C17 profiles first.

Refreshed C17 profiles `82850` completed, zero dropped records. Short
prefill/decode 510.930/1058.093 ms; long 23336.684/2183.685 ms. Remaining long
decode attention is 1189.308 ms (54.46% of decode), whereas prefill attention
9474.054 ms and tensor 12879.070 ms are unchanged within diagnostic variability.
Ranking: C15 p=.046602 long, s=1.8, o=.0005, predicted 2.063% whole request,
ideal 4.888%, saved 515.82 ms (phase ideal >119%); then C09 p=.905362 short,
s=1.02, o=0, predicted 1.807%, ideal 956.656%, saved 27.85 ms.

### C15 implementation boundary

Select C15 against immutable C17, long model decode objective. The thirty-row
baseline test is committed in `bd87ce9` and passes before production. Structure:
existing `shaders/attention.cuh` (~210 productive category-K lines); no new host
interface, buffer or function beyond the predeclared buffered attention body.
Apply the already-gated C13 numeric body, call it only from decode entries,
and call the unchanged online body directly from both prefill entries. Name
the buffered helper explicitly and document why prefill must not reference it.
All numerical expressions, 4096-score bound, fallback and barriers remain C13's.
Require both the original tiny-history exact test and strengthened scalar/bounded
long tests, canonical gates and f16/int8 parity before the objective. Verify
generated prefill entries do not reference buffered shared storage; the installed
PTX assembler can provide an independent static resource check. Other regimes
and TTFT/prompt stay controls; no threshold or tuple changes.

C15 implemented only the declared shader wiring/body (+82/-5). Canonical gate
`61491` passed formatting, CUDA check, CPU workspace, 11 error tests, all 37
CUDA tests (including thirty decode rows and tiny exact equality), and pinned
f16 parity with all 16 top-one IDs unchanged. Fast build `89180` completed.
Extra int8 parity precedes performance; saved `c15.ptx` and `c15-bench`.

Independent static resource check (installed tool, no GPU profiling/replay):
`ptxas --gpu-name sm_86 --verbose target/fast/build/graph_horizon_engine-60f4ad72f87b54df/out/cuda_kernels.ptx
-o target/cuda-amdahl-20260905/c15.cubin`, log `c15-resources.log`. Both prefill
entries use 40 registers and **40 shared bytes**; both decode entries use 40
registers and **16936 shared bytes**. All have zero stack/spill bytes. Thus
prefill does not own buffered score storage; this verifies C15's distinct
resource-reachability premise, not a hardware occupancy measurement.

Extra int8 parity `13900` passed, all 16 local IDs unchanged. Acquire the fixed
objective with `run.py bench c15 target/cuda-amdahl-20260905/c15-bench long`,
then `compare.py c17 c15 long model_decode_tps long`. Both other regimes are
controls before retention. No C15 performance was acquired before all gates.

C09 queue audit before its first implementation: C19 removed coefficient work
from every quant-staging lane, so its old broad prefill-owner estimate is no
longer causal. For a model Q4 matrix O x K in the fixed short trace, 31 decode
steps issue 31*O*(K/256)*4 half-metadata warp groups, whereas four prefill batches
issue 4*(O/64)*2*(K/32)*2 groups in C19's compact coefficient phase. The ratio
is 62; Q6's K16 coefficient granularity gives 31. Model dimensions are multiples
of 64/256, verified by the existing inventory/launch mapping. Logits further
increase decode ownership. These are source/trace-derived instruction-work
ratios, not measured bandwidth or a promised speedup.

Therefore retarget still-unattempted C09 to **short model decode**, before any
C09 production code or performance. On current C17, decode matmul+logits
957.889/1569.023 ms gives p=.61050 (phase .90530), s=1.02, o=0, predicted whole
gain ~1.21%, ideal ~156.74%, saved ~18.78 ms. This lowers its conservative rank
below C15 but retains a material phase ceiling. Same exact 26+39 snapshots,
canonical/f16/int8 gates, TTFT/prompt and both other regimes as controls.
Alignment proof remains checked even-offset views plus even block strides
144/176/210 and metadata offsets 0/2/208. No algorithm or input precision change.

C15 long objective `87173` passed provisionally: model decode 14.56 -> 19.52
(+34.0659%, CV .03% -> .04%), public 13.65 -> 18.30 (+34.0659%, CV .01%),
prompt 152.51 -> 152.30 (-.1377%), TTFT 23500.22 -> 23533.10 ms (+.1399%).
Prompt/TTFT candidate CV prints .0000 but standard deviations are .01 token/s
and 1.09 ms, not zero uncertainty. Counts 3584/32/31 unchanged; wall 101.67 s.
Run `run.py bench c15 .../c15-bench short medium` before retention, then
`compare.py c17 c15 long model_decode_tps short medium long`. No rerun needed.

### C15 kept

Both controls completed in `98896`. Complete comparison passes every declared
gate: **keep** after numeric/reference and tiny exact gates, canonical/f16/int8
parity, stable objective and both controls. Candidate means [CV fraction]:

| Regime | Prompt token/s | TTFT ms | Model decode token/s | Public delta/s | Wall s |
|---|---:|---:|---:|---:|---:|
| short | 252.26 [.0017] | 507.42 [.0017] | 30.83 [.0010] | 29.85 [.0011] | 7.15 |
| medium | 217.74 [.0022] | 4702.96 [.0022] | 26.70 [.0005] | 25.02 [.0006] | 24.45 |
| long | 152.30 [.0000 rounded] | 23533.10 [.0000 rounded] | 19.52 [.0004] | 18.30 [.0001] | 101.67 |

Model decode gains +2.664/+11.343/+34.066%; worst control medium TTFT +.4279%
(prompt -.4253%). Counts unchanged in all regimes. No rerun. Stock power-cap
increments short .818714 s, medium 0, long .671703 s; thermal counters zero,
no machine settings changed. C15 adds a second cohesive attention algorithm
only because phase-specific resource ownership demonstrably matters; explicit
entry wiring preserves online prefill. No host interface, global allocation,
cache layout or format changes. Production +82/-5 in one category-K shader.
C15 is retained in this commit; refresh all affected profiles before C09.

C15 profiles `7490` completed with zero dropped records. Short prefill/decode
511.251/1043.708 ms; decode attention 27.592 versus C17 50.037 ms. Long
23491.784/1640.801 ms; decode attention 637.308 versus 1189.308 ms. Tensor and
prefill attention remain 12976.517/9534.437 ms, consistent with unchanged
prefill resource ownership. Decode attention keeps 40 registers and no local
bytes. C09's current conservative decode-matmul owner is 964.373 ms /1554.959
ms: p=.620192, phase p=.923988, s=1.02, o=0, predicted 1.231%, ideal ownership
ceiling 163.291%, saved 18.91 ms. Actual removable instruction share remains
uncertain; this is a cheap bounded eligible experiment, not a predicted keep.

### C09 predeclared implementation

Select short model decode against immutable C15. In existing `shaders/quant.cuh`
(~105 productive category-K lines), replace only `cuda_half_at`'s two-byte
reconstruction with one aligned `__half` load and the identical conversion.
Document the alignment invariant: device allocation alignment, even checked
views, even packed block strides and all six audited call-site offsets. Existing
F16 direct loads and all arithmetic/dequant expressions remain unchanged.
No new file/function/format or public interface. Main risk is alignment, covered
by the typed-view boundary and call-site proof, not a new per-load branch.

Require all 26 original SIMT and 39 accepted tensor bit snapshots, canonical
gates (including packed embedding reference coverage) and f16/int8 parity before
objective performance. The frozen tensor snapshot remains valid transitively:
C19 matched it exactly, and C17/C15 do not affect independent matmul inputs.
Remove temporary prints before canonical gates. Both other regimes plus
prompt/TTFT/public decode remain controls under the same thresholds.

C09 exact gate `68152` passed: all 65 bit snapshots match, both diffs empty.
Restored the test file exactly to HEAD, preserving the committed C15 coverage.
Production is only the predeclared half-load replacement (+2/-2). Fast build
`76499` completed; saved `c09.ptx` and `c09-bench`. Generated metadata accesses
now use `ld.global.u16` at the original aligned offsets, without byte assembly;
the following half-to-float conversions are unchanged. Canonical gates run next.

C09 canonical `2780` completed: formatting, CUDA check, CPU workspace, 11 error
tests, all 37 CUDA tests and pinned f16 parity passed, all 16 top-one IDs
unchanged. Extra int8 parity precedes short model-decode A/B. This is countable
attempt 16; only the half-load helper differs from accepted C15.

Extra int8 parity `76226` passed, all 16 local IDs unchanged. Acquire
`run.py bench c09 target/cuda-amdahl-20260905/c09-bench short`, then
`compare.py c15 c09 short model_decode_tps short`. Both other regimes only
after provisional keep; all correctness gates precede performance.

C09 short objective `90212` passed provisionally: model decode 30.83 -> 33.74
(+9.4389%, CV .10% -> .22%), public 29.85 -> 32.68 (+9.4807%, CV .18%),
prompt 252.26 -> 264.69 (+4.9275%, CV .19%), TTFT 507.42 -> 483.59 ms
(-4.6963%, CV .19%). Counts 128/32/32 unchanged; wall 6.70 s. The target was
retargeted from instruction-work evidence before production/performance, not
selected after seeing this near-5% prompt result. Run `run.py bench c09
.../c09-bench medium long` before keep; same controls and thresholds, no rerun.

### C09 kept

Both controls `43368` completed. `compare.py c15 c09 short model_decode_tps
short medium long` passes every gate: **keep**. Means [CV fraction]:

| Regime | Prompt token/s | TTFT ms | Model decode token/s | Public delta/s | Wall s |
|---|---:|---:|---:|---:|---:|
| short | 264.69 [.0019] | 483.59 [.0019] | 33.74 [.0022] | 32.68 [.0018] | 6.70 |
| medium | 228.81 [.0017] | 4475.30 [.0017] | 29.21 [.0021] | 27.37 [.0023] | 23.22 |
| long | 157.63 [.0001] | 22737.39 [.0001] | 20.84 [.0004] | 19.53 [.0003] | 97.89 |

Model decode gains +9.439/+9.401/+6.762%; prompt +4.927/+5.084/+3.500%.
No control regression and counts unchanged throughout. No rerun. Stock
power-cap increments short .873784 s, medium .573858 s, long 0; thermal counters
zero, no setting changes. Production +2/-2 in one helper, alignment invariant
documented; conceptual complexity decreases by removing byte reconstruction.
C09 is retained in this commit. The current pool is terminal, but that alone
does not complete the campaign: refresh profiles and perform the required fresh
discovery across all three regimes and every material remaining mechanism.

Read-only device attributes, enum IDs verified in installed cuda.h, are recorded
privately in `device-attributes.log`: 28 multiprocessors, 1536 resident threads,
102400 shared bytes and 65536 registers per multiprocessor. These are static
resource ceilings, not achieved occupancy. The query used only cuInit and
cuDeviceGetAttribute; it created no inference context or configuration change.

### Fresh discovery after C09: token and packed-field reuse

All three `c09-timeline` profiles completed, zero dropped records. Diagnostic
prefill/decode milliseconds: short 486.322/968.840, medium 4527.184/1122.827,
long 22817.936/1547.425. Tensor prefill owns 437.030/3503.161/12256.917 ms;
decode dot plus logits 890.903/882.919/851.911 ms. These non-overlapping owners
rank two new bounded mechanisms; they are not hardware bandwidth/occupancy
measurements. Whole-request Amdahl estimates use the contemporaneous three rows:

| Candidate / regime | p | s | o | Predicted gain | Ideal owner ceiling | Saved ms |
|---|---:|---:|---:|---:|---:|---:|
| C20 short | .300331 | 1.15 | .002 | 3.861% | 42.925% | 54.09 |
| C20 medium | .620027 | 1.15 | .002 | 8.563% | 163.177% | 445.63 |
| C20 long | .503047 | 1.15 | .002 | 6.794% | 101.226% | 1550.00 |
| C21 short | .612236 | 1.10 | .001 | 5.782% | 157.889% | 79.54 |
| C21 medium | .156269 | 1.10 | .001 | 1.338% | 18.521% | 74.62 |
| C21 long | .034964 | 1.10 | .001 | .218% | 3.623% | 53.08 |

C20 ranks first on medium prompt throughput, not long merely for absolute work.
The fixed 32-token batch currently launches two M16 blocks per output tile;
both stage the same B and coefficients. Reuse these across M32 while retaining
each cell's K order and float corrections. Local s=1.15 discounts duplicated
work removal for unchanged A/output work and fewer independent blocks. Added
overhead .002 allows larger register/shared state. Small output grids can
underfill 28 multiprocessors; this is a risk to measure, not an occupancy claim.
C21's separate premise pairs low/high nibbles currently read by different
logical lanes; C12 only paired the two halves of a 256-value block within one
lane. Physical ownership changes, the original 128-leaf floating tree does not.
Its s=1.10 is uncertain until generated byte loads and public decode are checked.
Each bounded experiment is expected below one hour, with the same two-hour
comparison cap, exact gate, >=5% keep/CV<=5% and all controls.

### C20 structure and correctness declared before production

Existing files only:

```text
crates/graph_horizon_engine/src/backend/cuda/
  shaders/matmul.cuh (~245 productive lines, category K: one matmul family)
  kernels/matmul.rs (~135 productive lines, checked shape/launch dispatch)
  module.rs (~150 productive lines excluding tests, fixed module ownership)
  kernels/tests.rs (tests excluded from productive-line counts)
```

Generalize the tensor body over TOKENS=16/32. One resource-template helper is
beneficial specifically to keep two external entry points' shared storage
separate: M16 remains 9792 bytes, M32 estimates 14976. Packed rows>=32 select
M32; rows16..31 stay M16, smaller/F16 paths unchanged. Preserve all arithmetic
per cell, bounds, barriers and K grouping. No public API/dependency/new file.

Before production, extend the existing tensor test to rows16/17/32/33 and
high-range cancellation at16/32: 78 cases, plus the original26 SIMT cases.
Freeze full output bits on C09, then require all104 identical after C20.
The unchanged canonical 16-token prompt cannot exercise M32, so add a separate
predeclared real-model oracle case using the already-calibrated short128 prompt.
Use a private copy of the authenticated parity runner and a temporary fixed
prompt fixture in the parity module, on both baseline and candidate. Freeze
baseline prompt/oracle/local IDs and require candidate identity plus top-two
parity for f16/int8. This supplements, never replaces, the canonical gate.
Remove the temporary fixture and bit prints before running unchanged canonical
gates and performance. Do not regenerate a favorable oracle after a failure.
Objective medium prompt throughput against C09; TTFT/model/public decode and
short/long regimes are controls. Test-only checkpoint precedes production.

C20 baseline test checkpoint `c58ac6d` passed (session26494); temporary snapshots
in session12538 captured all26+78 records. The first additional oracle command
failed before inference because its relative model path was interpreted from
Cargo's integration-test working directory (E01). This is an invocation error,
not a numerical failure or candidate attempt. Resolve the same authenticated
file to its absolute path and run the extra baseline once with corrected paths;
preserve the failed raw log, unchanged prompt/oracle protocol and all snapshots.

Resolved extra baseline session43001 passed f16 and int8. Both freeze oracle
IDs `2757,10637,2479,1636,6483,13578,1278,3541,50147,115799,58987,3325,5213,1033,9246,1639`;
local top one differs only at step13 (1294), already accepted by the unchanged
top-two baseline criterion. Candidate must retain those exact local IDs too.
Private `frozen-parity.py c20 c20-resolved` replays these frozen IDs without
calling the oracle again. Baseline production remains C09; only committed
test expansion and declared temporary fixtures precede C20 implementation.

C20 exact session36300 passed all104 snapshots with empty diffs. The extra
replay's preliminary token-count assertion caught a metadata mistake before
candidate model inference: the canonical parity conversation includes an
explicit empty System message, so the unchanged short text renders **130** IDs,
not the benchmark's128. Correct only the private replay count to130, preserving
the already-frozen prompt bytes, all130 IDs and all16 oracle/local IDs exactly.
No numeric threshold, reference or workload is changed to accept the candidate;
this additional row exercises four full32 batches plus a two-token tail.

Extra replay session59471 passed f16/int8 against frozen130-token references,
with identical local top-one IDs. Removed both temporary tracked fixtures,
restoring tests/parity exactly to test checkpoint `c58ac6d`. Static `ptxas`
resource check reports M16 63 registers/9792 shared bytes and M32
72 registers/14976 shared bytes, zero stack/spill bytes. This verifies resource
isolation, not achieved occupancy. Canonical gates and fast build precede
medium objective performance; all three production files match the declaration.

C20 canonical session13158 passed formatting, CUDA workspace check, CPU
workspace tests, 11 error tests, all37 CUDA tests and f16 parity. Extra canonical
int8 session61935 also passed; both preserve all16 original local IDs. Fast
build31408 completed and immutable `c20-bench` is saved. Production +47/-23
across the three declared files; physical lengths226/129/168 including tests,
within productive estimates. Attempt17 now reaches its exact and canonical
gates. Acquire `run.py bench c20 .../c20-bench medium`, then
`compare.py c09 c20 medium prompt_tps medium`; both controls before retention.

### C20 rejected and restored

Objective session25307 completed: medium prompt228.81 ->232.50 (+1.6127%,
CV .17% -> .08%), TTFT4475.30 ->4404.34 ms (-1.5856%, CV .08%), model
decode29.21 ->29.25 (+.1369%, CV .24%), public27.37 ->27.42 (+.1827%, CV .24%).
Wall22.99 s; counts1024/32/31 unchanged. No rerun warranted. Correctness passes,
but objective <3%: **reject**, no controls needed for retention. Saved `c20.patch`
and restored only its three production files exactly to HEAD/C09; the useful
expanded test remains. Diagnostic immutable-binary medium trace next attributes
whether lost block parallelism offsets duplicate B removal. C21 stays ready.
Attempt count17: 11 distinct kept, 5 rejected, 1 interesting/restored, plus
the kept non-counting C17 re-evaluation (12 accepted runtime commits).

C20 diagnostic session42820 completed with zero dropped records. Tensor time
3503.161 ->3409.696 ms; its larger resource footprint offsets most B reuse.
The authenticated inventory/ordered launch mapping gives these paired costs:

| Format / K / N | C09 ms | C20 ms |
|---|---:|---:|
| Q4 /3072/4096 | 277.847 | 293.385 |
| Q4 /3072/1024 | 263.009 | 339.941 |
| Q6 /3072/1024 | 218.378 | 246.398 |
| Q4 /4096/3072 | 344.380 | 353.084 |
| Q4 /3072/9216 | 1259.165 | 947.174 |
| Q6 /9216/3072 | 768.950 | 837.340 |
| Q4 /9216/3072 | 371.432 | 392.373 |

Only the large output grid wins. Static resource ceilings permit at most8 M16
blocks or6 M32 blocks per multiprocessor (register/shared limits, not achieved
occupancy). Across28 multiprocessors, N9216 has288 M16 blocks versus144 M32:
it can reduce two resource waves to one, unlike the smaller grids. This supports
a bounded dispatch re-evaluation C22, not repeating universal M32. Treat it as
non-counting, preserving C20's rejection. Its causal owner on medium is
1259.165/5650.010 ms, p=.222861, conservative s=1.2, o=.001, predicted3.749%,
ideal28.677%, saved204.21 ms; below C21 short's5.782%, but phase ceiling remains
material. Declare a simple large-grid shape gate before implementing C22, not
after its A/B; no device-settings changes or broad performance claim.

### C21 predeclared implementation

Select short model decode against restored C09: current p=.612236 whole request,
phase .919556, s=1.1, o=.001, predicted5.782%, ideal157.889%, saved79.54 ms.
One physical64-thread packed dot retains the original128 logical leaves. Each
thread owns two leaves sharing low/high nibble bytes: Q4/Q5 logical i and i+32,
Q6 i and i+64. Each leaf still accumulates its original i then i+128 products
in block order; shared slots restore the identical128-leaf reduction tree.
F16 remains128 physical threads, batched SIMT/tensor paths unchanged.

Existing `shaders/quant.cuh` (~155 productive category-K lines) gains one cohesive
paired-dequant helper that protects this packed-field relationship;
`shaders/matmul.cuh` (~240 category-K lines) owns the two logical sums;
`kernels/matmul.rs` (~130 productive orchestration lines) selects64 only for
packed single/logit dots. No new files/dependencies/public APIs. Main risks are
leaf mapping, changed compiler contraction and higher per-thread registers.
Require all104 frozen C20-baseline snapshots, unchanged canonical gates and
f16/int8 parity before short objective A/B; TTFT/prompt/public decode and both
other regimes remain controls under the same thresholds. C22 stays queued.

C21 exact session80002 passed all104 frozen snapshots, both diffs empty.
Temporary test prints removed, restoring the expanded tests exactly to HEAD.
Static `ptxas` reports packed single/logit entries40 registers,1024 shared bytes,
zero stack/spills; paired ownership did not increase their register allocation.
The Q4/Q5 helper explicitly loads one packed byte for both nibbles and one Q5
high-bit byte for both leaves; Q6 shares its low/high bytes across two leaves.
Canonical gates and immutable fast build run before any C21 performance.

C21 canonical session83966 passed fmt/check/CPU workspace/11 error/all37 CUDA
tests/f16 parity; int8 session84833 also passed with all16 canonical local IDs
unchanged. Fast build88017 completed. PTX shows independent two-FMA leaf chains,
two shared stores at the original logical offsets, and the same64..1 reduction
levels; extraction consumes the loaded byte for both nibbles. Production
+50/-10 in the three declared files. Acquire `run.py bench c21 .../c21-bench
short`, then `compare.py c09 c21 short model_decode_tps short`. Both controls
must pass before keep; no candidate performance preceded correctness.

C21 objective session88291 is provisional keep: short model decode33.74 ->40.52
(+20.0948%, CV .22% -> .15%), public32.68 ->39.23 (+20.0428%, CV .11%),
prompt264.69 ->265.78 (+.4118%, CV .16%), TTFT483.59 ->481.61 ms (-.4094%,
CV .16%). Counts128/32/32 unchanged; wall6.06 s. Run both remaining regimes
with the same immutable binary before retaining; no rerun needed.

### C21 kept

Both controls completed in session71465. Full `compare.py c09 c21 short
model_decode_tps short medium long` passes every declared gate: **keep**.
Candidate means [CV fraction]:

| Regime | Prompt token/s | TTFT ms | Model decode token/s | Public delta/s | Wall s |
|---|---:|---:|---:|---:|---:|
| short | 265.78 [.0016] | 481.61 [.0016] | 40.52 [.0015] | 39.23 [.0011] | 6.06 |
| medium | 229.57 [.0003] | 4460.57 [.0003] | 34.13 [.0006] | 31.98 [.0006] | 22.54 |
| long | 157.81 [.0019] | 22710.32 [.0019] | 22.98 [.0004] | 21.54 [.0002] | 97.16 |

Model decode gains +20.095/+16.844/+10.269%; prompt +.412/+.332/+.114%.
No control regression, unchanged128/32/32,1024/32/31,3584/32/31 counts. No rerun.
Exact104 snapshots and canonical gates passed before performance. The additional
paired helper is a narrow format invariant, not a generic abstraction: logical
leaf order is explicit despite two leaves per physical thread. Production
+50/-10 across the declared three files, no public/format/KV ownership changes.
C21 is retained in this commit; refresh all three affected profiles and rerank.

Environment readback: C20 medium stock power-cap counter increment .993741 s;
C21 short .833801 s, medium .654170 s, long0. Thermal counters remain0 in all
paired records; sampled clocks remain about1980 MHz with stock desktop clients.
No machine setting changed. These bounded stock-cap intervals are disclosed,
not suppressed or used to select a favorable rerun.

C21 profiles session66552 completed, zero dropped records. Prefill/decode ms:
short483.230/800.940, medium4510.395/953.096, long22656.215/1394.574. Decode
dot+logits now723.242/717.549/712.121 ms, versus890.903/882.919/851.911 after
C09. Remaining prefill tensor435.661/3490.989/12178.945 ms and attention
11.218/733.644/9494.469 ms are unaffected owners within diagnostic variability.

Fresh source review adds C23: online attention currently publishes four softmax
weights and a rescaling factor from thread0 through shared memory, requiring
three block barriers per four-context tile. Each warp can compute the identical
scalar coefficients in lane0 and broadcast them locally, leaving only score-ready
and score-consumed barriers. Preserve the same four-score maximum, exponential,
denominator and V arithmetic order; no cache/layout/precision change or4096-score
allocation. Risk: four lane-zero computations instead of one and more registers
may outweigh one removed barrier. This is not C13 global score buffering or
C05's score-reduction tree. Exact long/tail/tiny-history outputs and canonical
f16/int8 gates are required before its performance. C23 remains a new distinct
ready premise, not yet implemented or counted.

Current C23 whole-request fractions short/medium/long .008735/.134281/.394767;
with s=1.10, o=.002, predicted -.120/1.031/3.508%, ideal .881/15.511/65.226%,
saved -1.55/55.77/815.03 ms. Local s discounts duplicate scalar work; actual
barrier share is uncertain, no hardware stall-counter claim. Its long phase
ceiling is material despite conservative prediction below5%.

### C22 predeclared bounded dispatch and tests

Select medium prompt throughput: its observed C20 shape effect has lower
uncertainty than C23, with a slightly higher predicted whole-request gain
(about3.9%). Both benefit multiple regimes; medium is also the representative
tie-break. This is a non-counting C20 re-evaluation, never a new attempt quota.
Reuse precisely C20's gated M32 kernel only for packed rows>=32 and
**output_width>=8192** (at least128 full output tiles); keep M16/SIMT elsewhere.
This fixed shape threshold precedes all C22 performance. It bounds the
large-grid evidence in this validation environment, not a universal speed claim.
The old universal M32 result remains rejected; no retry of its smaller grids.

Structure remains C20's declared tree: matmul.cuh (~255 category-K productive
lines with retained C21), kernels/matmul.rs (~135 productive), module.rs (~150
excluding tests); no new files/API/dependency. Separate entries still protect
small-batch resources. The dispatch condition is the only change to C20's body.
Extend the tensor test with N8193 at K256/M33 for all three packed formats,
covering the threshold path plus both tails, while retaining all104 prior cases.
Freeze the resulting107 full-output snapshots on accepted C21 before production.
Replay the additional frozen130-token oracle on C21 and C22, f16/int8, then
restore all temporary fixtures and run unchanged canonical gates before A/B.
Objective medium prompt, all prior controls and thresholds unchanged.

Exact refreshed matrix attribution (`c21-matrix-{short,medium,long}.log`) gives
C22 owners155.151/1254.776/4395.647 ms. Fractions .120818/.229666/.182765,
s=1.2/o=.001, predicted1.951/3.872/3.036%, ideal13.742/29.814/22.364%,
saved24.57/203.67/708.56 ms. These replace the preliminary scaled estimates;
ranking and objective remain unchanged before C22 production/performance.

C22 baseline session72466 passed all107 matrix snapshots and the frozen extra
130-token f16/int8 top-two plus local-ID identity gates on C21. Thus C21 also
preserves the already-frozen longer model fixture. Remove temporary prints and
prompt fixture, retaining only the three useful large-grid test cases before
the C22 production edit. This test-only checkpoint does not count as an attempt.

C22 test checkpoint `0ec0a87` precedes production. Candidate session9866 passed
all107 full-output snapshots with empty diffs and both frozen130-token parity
replays, preserving exact local IDs. Removed temporary fixtures; test/parity
files now match HEAD exactly. Run canonical gates and both canonical KV rows
before the medium objective; the only dispatch difference from C20 is the
predeclared output-width threshold, and C21 decode remains intact.

C22 canonical session76178 passed formatting, CUDA check, CPU workspace,11 error
tests, all37 CUDA tests and both canonical f16/int8 parity rows, all16 original
local IDs unchanged. Build93421 completed; `c22-bench`/`c22.ptx` immutable.
Static resource check remains M16 63 registers/9792 shared bytes, M32
72/14976, zero stack/spills. Acquire medium prompt objective against C21 using
`run.py bench c22 .../c22-bench medium` then `compare.py c21 c22 medium
prompt_tps medium`; both other regimes before retention, no quota increment.

C22 medium objective session61427 passes provisionally: prompt229.57 ->246.19
(+7.2396%, CV .03% -> .13%), TTFT4460.57 ->4159.38 ms (-6.7523%, CV .13%),
model decode34.13 ->33.97 (-.4688%, CV .38%), public31.98 ->31.85 (-.4065%,
CV .33%). Counts1024/32/31 unchanged; wall21.34 s. No rerun needed. Acquire
short and long controls with the same binary before deciding keep. The universal
C20 rejection is not overwritten; only this predeclared bounded dispatch is
eligible for retention after all controls.

### C22 kept as non-counting bounded re-evaluation

Both controls session46526 completed; full `compare.py c21 c22 medium
prompt_tps short medium long` passes every declared gate: **keep**.
Candidate means [CV fraction]:

| Regime | Prompt token/s | TTFT ms | Model decode token/s | Public delta/s | Wall s |
|---|---:|---:|---:|---:|---:|
| short | 286.35 [.0046] | 447.02 [.0046] | 40.03 [.0019] | 38.75 [.0022] | 5.97 |
| medium | 246.19 [.0013] | 4159.38 [.0013] | 33.97 [.0038] | 31.85 [.0033] | 21.34 |
| long | 165.38 [.0000 rounded] | 21671.24 [.0000 rounded] | 22.98 [.0004] | 21.54 [.0005] | 93.09 |

Prompt gains +7.739/+7.240/+4.797%; the objective was medium before production.
Worst control: short model decode -1.209% and public -1.224%, within the declared
5% limit, not claimed unchanged. Long TTFT sample deviation .04 ms explains
the rounded-zero CV; uncertainty is not zero. All token/delta counts unchanged.
No rerun. Stock power-cap increments short .953916 s, medium .673849 s, long0;
thermal counters0, sampled clocks about1980 MHz and no setting changes.

The shape exception adds one fixed entry and one explicit condition, justified
by the isolated negative result on smaller output grids; no tuning surface or
public option is added. M16 resource isolation and107 exact snapshots pass,
including N8193/M33 tails. Canonical and frozen extra parity passed for both KV
schemes. Production +47/-23 across the three declared files. C22 is retained
in this commit, but does not increment distinct attempts; C20 remains rejected.
Refresh all three affected profiles before selecting C23.

C22 retained in `e7790cd`; profiles session26237 completed with zero dropped
records. Prefill/decode ms short444.963/797.266, medium4168.487/945.646,
long21597.927/1399.507. Combined tensor owner397.449/3157.474/11111.011 ms;
prefill attention10.931/728.146/9511.781 ms. The bounded M32 dispatch reduces
its intended prefill owner without the universal small-grid penalty.

### C23 selected: warp-private online coefficients

C23 is the remaining ready entry. Refreshed whole-request p values
.008799/.142379/.413602 (short/medium/long), s=1.1, o=.002, predict
-.120/1.106/3.691%, ideal .888/16.602/70.533%, saved -1.49/55.97/818.71 ms.
Select **long prompt throughput** against C22; its phase owner44.04% makes a
bounded experiment eligible even below the conservative whole-request5% signal.
No counter-derived barrier fraction is claimed; duplicated scalar work is the
main uncertainty. Same >=5% objective/CV<=5%, all controls and two-hour cap.

Change only existing `shaders/attention.cuh` (~220 productive category-K lines,
one attention family); temporary prints stay in existing kernel tests. No new
file, function, buffer, dependency or public interface. Keep shared scores[4],
make previous/weights/denominator warp-local, compute them in each lane0 using
the identical expression/iteration order, then shuffle from lane0. Remove only
the coefficient-publication block barrier; retain score-ready and final-reader
barriers. Broadcast the final denominator uniformly before dimension predicates.
The same online body is also the decode fallback for position>=4096: preserve
that behavior and inspect compiled decode resources, with all decode controls
still mandatory. Do not add a duplicate algorithm merely to avoid a speculative
resource risk; measurements decide whether this smallest change is acceptable.

Freeze40 full output vectors on C22 from the existing long/tail test (10 prefill,
30 decode across f16/int8, including4094/4095/4096). Require exact candidate
identity, original scalar bounds and tiny-history equality. The accepted C22
already passes the frozen130-token model fixture; require C23 to replay it
unchanged for both KV schemes, then remove temporary fixtures and pass all
canonical gates before performance. No numeric tolerance is relaxed.

C23 baseline session42627 passed and froze all40 attention vectors in
`c23-attention-baseline.log`; only temporary bit printing differs from C22.
Proceed with the declared single-body coefficient ownership change.

C23 exact session22903 passed all40 frozen vectors and both unchanged130-token
model parity replays with identical local IDs. Removed temporary prints/prompt,
restoring tests/parity exactly to HEAD. Static `ptxas` reports all four attention
entries still40 registers with zero stack/spills: prefill shared bytes40 ->16,
decode16936 ->16912. The fallback resource risk did not require a duplicate body.
Production is one shader +11/-7; canonical gates and fast build precede A/B.

C23 canonical session31162 passed formatting/check/CPU workspace/11 error/all37
CUDA tests and both canonical f16/int8 parity rows, original16 local IDs intact.
Fast build56614 completed. The generated prefill entry has exactly two static
`bar.sync` sites versus C22's three in the same dynamic tile loop, confirming
the intended work removal without claiming a measured stall fraction. Acquire
`run.py bench c23 .../c23-bench long`, then `compare.py c22 c23 long prompt_tps
long`; no candidate performance preceded gates, both controls before keep.

### C23 rejected and restored

Objective session79729 completed: long prompt165.38 ->142.13 (-14.0585%,
candidate CV .01%), TTFT21671.24 ->25216.21 ms (+16.3579%, CV .01%), model
decode22.98 ->22.75 (-1.0009%, CV .04%), public21.54 ->21.32 (-1.0214%,
CV .04%). Baseline prompt/TTFT CV printed .0000 with TTFT deviation .04 ms,
not zero uncertainty. Counts3584/32/31 unchanged; wall107.50 s. No rerun.
Correctness passed, but objective regresses and TTFT control exceeds5%:
**reject**. Save `c23.patch`, restore its sole production shader exactly to
C22/HEAD; no temporary tracked fixtures remain. No other controls are needed
for retention. An immutable-binary diagnostic long trace follows to locate the
added cost; fewer barriers did not suffice when four warps duplicate coefficients.
Any next variant must remove that measured/source-backed cause, not repeat C23.

C23 diagnostic session76759 completed, zero dropped records: attention
9511.781 ->13088.841 ms while combined tensor11111.011 ->11110.219 ms.
Prefill25178.315/decode1412.197 ms; attention still40 registers and no local
bytes. This localizes the regression to the changed attention schedule, not
tensor work or increased register allocation. Stock power-cap increment
.533972 s, thermal counters0; no setting changes.

Fresh discovery adds C24: one warp per prefill query head avoids both block
barriers and fourfold replicated coefficient computation. Within that warp,
four eight-lane subgroups score four context tokens. Each lane accumulates four
of the original32 logical dot leaves, combines the original16/8 tree levels
locally, then the4/2/1 levels within its eight-lane subgroup. Thus score and
four-context online arithmetic order can be preserved exactly, unlike a plain
eight-lane reduction. Each lane owns up to8 output dimensions. Four independent
query-head warps share a128-thread launch, with no shared scratch or block
barriers; warp-uniform tail returns are safe. Keep the existing online body
only for decode's large-context fallback; prefill gets a cohesive dedicated
numeric helper. Main risks: register pressure, less-coalesced subgroup loads,
leaf mapping and partially occupied final blocks. Declare exact40-vector,
tiny-history and frozen130-token plus canonical gates before implementation.

Current C24 long owner remains C22 attention9511.781/22997.434 ms:
p=.413602, credible s=1.3, o=.005, predicted about9.96%, ideal70.533%,
saved about2079.03 ms. This discounts elimination of all barriers for extra
logical-leaf/output registers and different load grouping; it is not an
occupancy or bandwidth measurement. C24 ranks above C25, whose prefill reuse
premise needs a resource-isolated bounded design before it is ready.

C25 source evidence: adjacent Q4/Q5 K32 groups read the same packed quant byte
for its low/high nibbles in separate iterations. A two-group staging tile can
share that byte and one ready/consumer barrier pair while preserving the two
separate WMMA/f32 corrections in their original order. Unlike C21 this targets
prefill, not logical decode leaves. Estimated shared staging doubles
9792/14976 ->19584/29952 bytes for M16/M32, so Q6 and ordinary entries must
remain resource-isolated. No performance or count is attributed yet; inspect
current Q4 shape ownership and predeclare s/overhead/exact gate before selection.

### S01: pre-existing single-row prefill ABI prerequisite

While inspecting C24's checked launch boundary, source review found that
`attention::validate` includes the rows scalar only when rows>1, although the
prefill kernel ABI always requires it. `graph/prefill/encode.rs` can emit a
one-row tail and calls prefill unchanged; there is no upper-layer decode bypass.
This is a correctness prerequisite in the exact path to be changed, not a new
performance candidate or a claimed gain. Resolve it separately before C24.

First add a test-only prefill argument-count assertion at the unsafe dispatch
boundary (11 f16,13 int8) so the regression is caught before an invalid driver
call. Extend the existing tiny causal test with a one-row prefill compared to
decode at the same position for both KV schemes; capture the baseline failure.
Smallest fix: common validation returns the common decode-shaped prefix/tail;
prefill explicitly inserts its mandatory rows scalar at ABI index8, irrespective
of its value. Decode and every rows>1 argument vector remain identical.
No public interface/dependency or numeric change.

Existing structure: attention/mod.rs (~95 productive validation/ABI lines),
attention/prefill.rs (~45 dispatch lines), exec/dispatch.rs (~130 productive
lines excluding the new test-only guard), kernels/tests.rs (tests excluded).
Require the new test, all canonical gates and f16/int8 parity; then refresh the
public baseline before C24 so the support fix is not bundled into its A/B.
S01 is a separate correctness-support commit, excluded from optimization counts.

S01 baseline session16080 fails exactly at the new test-only guard: f16 prefill
has10 arguments, kernel requires11. The guard fires before the unsafe driver
call; no invalid launch is attempted. Apply the declared explicit rows insertion
in prefill and remove shape-dependent ABI selection from common validation.
The same fixed operation distinction covers int8's12 versus13 arguments.

Before S01 retention, add a separate authenticated real-model gate with the
fixed text `benchmark` repeated123 times plus a final period and the canonical
empty System message: expected129 prompt IDs, four32-token batches plus one
row. This directly exercises the repaired public graph tail, unlike the existing
130-token fixture. Use the unchanged pinned16-step top-two protocol for f16/int8,
freeze its first oracle result, and never change the prompt/reference after a
failure. Temporary fixed prompt fixture/private runner only; restore canonical
source before retention and public baseline. This is correctness verification,
not a changed performance workload or a new optimization attempt.

S01 canonical session23493 passed formatting/CUDA check/CPU workspace/11 error/
all37 CUDA tests (including both one-row cases) and f16/int8 canonical parity,
all16 original local IDs unchanged. Fast build64802 completed before the
temporary one-row fixture. Extra session71148 passed both authenticated
129-token real-model rows, directly exercising the one-row graph tail. Restore
the fixed canonical prompt exactly; the immutable `s01-bench` was built from
that canonical source, with no print instrumentation. S01 removes implicit
shape-based ABI selection; test-only arity checking protects the unsafe boundary
without adding runtime work. Capture the three-row public baseline before C24.

S01 committed separately as `6d37587` (correctness support, not an optimization
attempt/keep). Public baseline session11030 completed, one warm-up and three
measured repetitions with unchanged token/delta counts. Means [CV fraction]:

| Regime | Prompt token/s | TTFT ms | Model decode token/s | Public delta/s | Wall s |
|---|---:|---:|---:|---:|---:|
| short | 290.32 [.0016] | 440.89 [.0016] | 40.82 [.0026] | 39.51 [.0023] | 5.89 |
| medium | 246.10 [.0025] | 4160.96 [.0025] | 33.98 [.0034] | 31.85 [.0032] | 21.26 |
| long | 165.46 [.0001] | 21660.89 [.0001] | 22.96 [.0004] | 21.52 [.0006] | 93.00 |

These are the next A/B baseline, not a claimed S01 speedup. Source reasoning
shows identical scalar vectors on the fixed32-row performance batches; the
corrected one-row path is covered separately by the129-token oracle. Refresh
the three diagnostic timelines before C24, preserving the same campaign/pool.

S01 stock power-cap increments short .813473 s, medium0, long0; thermal counters
remain0. Diagnostic session70479 completed, zero dropped records. Prefill/decode
ms short445.260/798.015, medium4171.709/947.805, long21574.094/1395.962.
C24 attention owners11.901/727.154/9510.303 ms yield p .009573/.142036/.414030,
s=1.3, o=.005: predicted -.278/2.857/9.956%, ideal .967/16.555/70.657%, saved
-3.47/142.21/2079.84 ms. The long prompt objective remains selected before code.

### C24 implementation boundary

Existing `shaders/attention.cuh` (~305 productive category-K lines) gains one
cohesive warp-owned prefill helper; existing `kernels/attention/prefill.rs`
(~50 productive orchestration lines) launches ceil(rows*q_heads/4) blocks,
128 threads each. Keep the fixed ABI and S01 explicit rows argument. Existing
prefill entry names call the new helper; decode and its online fallback remain
unchanged and do not reference it. No new file, dependency, public API or cache
layout. A separate numeric helper is beneficial here to preserve decode's
different parallelism/resource contract, not a generic scheduling abstraction.

Within each warp, subgroup lane0..7 owns logical leaves i/i+8/i+16/i+24 for one
of four context tokens. Accumulate each leaf in its old stride32 order, combine
(leaf0+leaf16)+(leaf8+leaf24), then subgroup4/2/1 shuffles. Only after the
four raw scores are available may their register slots become weights. Lane0
computes the unchanged maximum/denominator/weight sequence once per query head;
all lanes receive coefficients before V updates. Each output dimension keeps
the original four-token accumulation order. Whole-warp tail returns are legal
because no shared storage or block barrier exists. Use64-bit query IDs/product
bounds in the shader, keeping the checked host u32 launch conversion.

Freeze40 current S01 attention vectors before editing, require exact candidate
identity plus all scalar/tiny/one-row tests. Also replay both predeclared frozen
model fixtures (130 tokens and129 one-row-tail tokens), f16/int8, with exact
local IDs and unchanged top-two criterion. Restore fixtures before canonical
gates and performance. Long prompt keep>=5%, CV<=5%, TTFT/model/public decode
and both other regimes remain controls; two-hour cap, no changed workload.

C24 baseline session77060 captured all40 vectors successfully on S01; no numeric
production change preceded it. Current C25 ownership was separately mapped from
the authenticated tensor sequence, retaining it below C24 in the queue.

C25 current Q4 prefill owners273.238/2187.046/7646.772 ms give whole-request
p .219773/.427198/.332902. With conservative s=1.1, o=.005, predicted
1.521/3.502/2.592%, ideal28.168/74.580/49.903%, saved18.62/173.22/580.31 ms.
This is medium-first and below C24; estimated cost45–60 minutes with high shared
resource uncertainty. Exact107 matrix vectors and both frozen model fixtures
will be required. The doubled shared footprint is not treated as free.

C24 exact session70580 passed all40 S01 vectors, empty diff, plus the updated
tiny/one-row test. Static resource check: prefill f16 uses48 registers, int8 uses
80, both zero shared/stack/spill bytes; decode remains40 registers/16936 shared
bytes with zero spills. No claim of achieved occupancy. Temporary vector prints
removed; run frozen130-token then129-token f16/int8 gates before canonical tests.

C24 frozen130-token session17265 passed both KV schemes with identical local
IDs. The private replay now additionally accepts only the explicitly named
`one-row` fixture (129 IDs from S01), leaving the original130-token fixture and
its strict count unchanged. Run it under a separate label after changing only
the temporary fixed prompt; never regenerate either frozen oracle.

C24 one-row replay session29837 passed both KV schemes with exact frozen129-token
local IDs. Restored the canonical prompt. The installed Compute Sanitizer is
available: before performance, additionally run memcheck and synccheck on the
focused long/tail test, using `--error-exitcode 99`. Consult installed `--help`
and [official documentation](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html).
These are diagnostic correctness runs, never performance records; no tool
installation, replay timing comparison or machine-setting change is involved.

C24 canonical session49465 passed all gates and both canonical KV parity rows,
original16 IDs intact; fast build53539 completed. Generated prefill PTX has zero
block-barrier sites. Run the two sanitizer tools serially on the exact focused
test binary `target/debug/deps/graph_horizon_engine-b116abb1c981e6b9`, full test
name `backend::cuda::kernels::tests::attention_long_history_and_dimension_tails_match_reference`
with `--exact --nocapture --test-threads=1`; logs `c24-memcheck.log` and
`c24-synccheck.log`. No performance acquisition until these checks finish.

Sanitizer session94421 completed: memcheck and synccheck each run the selected
test successfully and report **0 errors** (test wall1.61/.71 s, diagnostic only).
C24 now passes all predeclared exact, bounded, canonical, frozen-model and
memory/synchronization gates. Production +87/-4 across the two declared files;
tests/parity source match S01 exactly. Acquire `run.py bench c24 .../c24-bench
long`, then `compare.py s01 c24 long prompt_tps long`. Both other regimes remain
mandatory controls before keep; no candidate performance preceded correctness.

C24 long objective session32050 passes provisionally: prompt165.46 ->188.07
(+13.6649%, candidate CV rounded .0000), TTFT21660.89 ->19056.70 ms (-12.0225%,
CV rounded .0000), model decode22.96 ->22.98 (+.0871%, CV .04%), public
21.52 ->21.54 (+.0929%, CV .06%). Counts3584/32/31 unchanged; wall82.73 s.
No rerun warranted. Run short and medium controls against S01 before retention;
rounding of CV is not zero uncertainty.

C24 controls session30395 completed successfully. Means [CV fraction]:

| Regime | Prompt token/s | TTFT ms | Model decode token/s | Public delta/s | Wall s |
|---|---:|---:|---:|---:|---:|
| short | 289.00 [.0032] | 442.91 [.0032] | 40.29 [.0019] | 39.02 [.0019] | 5.94 |
| medium | 254.53 [.0003] | 4023.04 [.0003] | 33.76 [.0011] | 31.63 [.0008] | 20.72 |
| long | 188.07 [.0000 rounded] | 19056.70 [.0000 rounded] | 22.98 [.0004] | 21.54 [.0006] | 82.73 |

Versus S01, medium prompt +3.4254%; short prompt -.4547%, TTFT +.4582%,
model decode -1.2984%, public -1.2402%. Every control passes the predeclared
5% regression limit; long TTFT standard deviation .13 ms, not zero uncertainty.
Stock power-cap increments short .913940 s, medium0, long .277578 s; thermal0.
No selective rerun. Counts stay128/32/32,1024/32/31,3584/32/31. Keep C24.
Its separate numeric helper increases source size but protects distinct prefill
and decode resource contracts; it removes shared ownership and block barriers
from prefill without changing the fixed score arithmetic.

Diagnostic launch initially used the occupied public label `c24`; exclusive
log creation rejected it before inference and preserved every public artifact.
Corrected diagnostic label to `c24-timeline`, session59440, same immutable
binary and tuple. Refresh all three rows before reranking C25.

C24 committed `99f4330`; diagnostic59440 completed all rows, zero dropped
records. Prefill/decode ms442.571/802.694,4035.248/957.606,19071.319/1398.475.
Prefill attention9.498/574.412/6938.256 ms; tensor396.899/3174.015/11139.958.
Decode dot/logits725.027/720.029/715.604; long decode attention634.327 ms.
Current largest measured owners remain tensor prefill and packed decode;
long attention remains material. No saturation claim from static resources.

### C25 bounded design selected before implementation

Restrict paired K32 staging to Q4/Q5 M16 dispatch with outputs<8192. Preserve
C22 M32 large grids and Q6 exactly through separate ordinary entries. This is
a pre-performance resource decision: doubling M32 shared14976->29952 would
reduce its static ceiling6->3 blocks/SM, forcing the144-block measured grid
into two waves. M16 shared9792->19584 permits5 blocks/SM, still enough for the
measured32/96/128-block small grids at32 rows. Registers may lower that ceiling;
verify the generated resources, never claim achieved occupancy from this bound.

Narrowed authenticated Q4 owners155.418/1244.686/4359.191 ms (exclude N9216),
whole-request p .124807/.249293/.212957. With s=1.1,o=.005: predicted
.639/1.798/1.457%, ideal14.261/33.208/27.058%, saved7.90/88.19/293.94 ms.
Medium remains objective. Cheap bounded diagnostic value justifies testing
below the conservative prediction threshold; the ideal ceiling exceeds5%.
Cost45–60 minutes, main uncertainty shared/register pressure and compiler reuse.

Structure: existing shaders/matmul.cuh (~275 productive category-K lines),
kernels/matmul.rs (~155 orchestration lines), module.rs (~145 excluding tests).
Add a compile-time stage count1/2 to the cohesive tensor helper and one fixed
paired entry. Ordinary entries instantiate stage1; paired entry only stage2
Q4/Q5 M16. Two adjacent K32 groups share packed-byte loads and three block
barriers instead of six, but retain two WMMA accumulators and two f32 correction
steps in their original order. No weight/cache layout or public API changes.

Freeze current107 matrix vectors before editing, then require exact identity,
all canonical gates and both frozen129/130-token fixtures for f16/int8. Keep
the2% scalar bound and all references unchanged. Medium prompt>=5%, CV<=5%,
all other metrics/regimes controls<=5% regression; two-hour comparison cap.
Run focused memcheck/synccheck if the new shared layout reaches exact correctness.

C25 baseline51876 and candidate79779 each pass all15 kernel tests and emit107
frozen vectors; exact diff is empty. Temporary prints removed. Paired entry
uses56 registers/19584 shared bytes, zero stack/spills; its static shared limit
is5 blocks/SM as estimated. Generic stride-loop adaptation changes ordinary
compiler register counts too (M16 63->56, M32 72->71), while their shared
footprints and all numeric vectors remain identical. This is not byte-identical
ordinary machine code; controls and subsequent per-shape attribution must cover
it, and any performance conclusion must disclose that compilation effect.
Frozen130-token replay91738 passes f16/int8 with exact local IDs;129-token
one-row replay is next, before canonical gates and performance.

One-row56285 also passes both KV schemes. Before performance, PTX inspection
shows the compiler did NOT merge low/high quant loads from two independent
`cuda_weight_parts` calls (Q4 separate u8 registers/addresses; Q5 duplicates
the high-bit byte too). Complete the already-declared reuse premise explicitly:
add a narrow paired-integer quant decoder in existing quant.cuh (~180 productive
category-K lines), called only by stage2. It centralizes the packed K64 invariant;
stage1 retains the existing helper. Also retain the original group-index loop
form for stage1 rather than introducing incidental induction-variable changes.
These are pre-performance implementation corrections within C25, not additional
attempts or relaxed gates. Repeat exact107 and both frozen fixtures after them.

C25 completed reuse passes exact session70801 (107 vectors, empty diff).
PTX now explicitly reuses one loaded quant byte for `and ...,15` and `shr ...,4`
and shares Q5 high bits. Paired resources64 registers/19584 shared, zero spills;
ordinary M16/M32 resource counts restored to63/72 registers and9792/14976 shared.
Frozen130 replay98803 passes both schemes. No candidate performance yet.

C25 final one-row replay26344 passes both schemes. Canonical session34101 passes
fmt/CUDA check/CPU workspace/11 error/all37 CUDA tests and canonical f16/int8
parity, unchanged local IDs. Fast build5685 completed; tests and canonical
prompt match HEAD exactly. Focused memcheck/synccheck session62661 runs serially
on the81-case tensor test with `--error-exitcode 99`, before objective acquisition.

### C25 rejected and restored

Sanitizer62661 completed: memcheck and synccheck both0 errors (22.91/22.58 s
diagnostic tests). Objective10787 then completed with unchanged1024/32/31 counts.
Medium prompt254.53 ->246.35 token/s (-3.2138%, CV .0003 ->.0001),
TTFT4023.04 ->4156.65 ms (+3.3211%, CV .0001), model decode33.76 ->33.83
(+.2073%, CV .0018), public31.63 ->31.70 (+.2213%, CV .0016), wall21.36 s.
Stable failed objective: reject, no rerun or control expansion. Power-cap
increment .993712 s, thermal0. Restore all four production files exactly to
C24 HEAD, preserve private c25.patch/immutable binary/logs. No rejected code or
temporary prints/prompts remain. Diagnostic63776 measures the rejected binary
to attribute the loss and inform bounded alternatives; it is not a public A/B.

C25 diagnostic63776 completed, zero dropped records. Q4 shape ms C24->C25:
K3072/N4096 275.948->425.588; K3072/N1024 261.098->226.531;
K4096/N3072 342.099->352.975; K9216/N3072 365.540->379.361.
Unchanged wide Q4 949.593->941.336 and Q6 total979.737->972.630.
The loss is localized to the paired entry, not its ordinary controls. Restricting
the unchanged candidate to N1024 would retain only34.568 ms savings, .86% of
medium prefill; that measured unchanged subcase does not justify a threshold
claim or a filler retry. C25's whole premise remains rejected.

Fresh discovery C26 instead changes a concrete new boundary: paired staging
changed WMMA row pitch32->64. Stage-major A/B tiles could restore the original
K32 pitch and aligned fragment addresses while still loading each low/high byte
once and sharing barriers. A bank/transaction penalty is a hypothesis, not a
measured counter; layout versus resource scheduling remains uncertain. Same
narrowed owner/p/s/o as C25, medium predicted1.798%, ideal33.208%, cost30–45 min.

C27 revisits warp output ownership only after C21 removed duplicated packed
loads and exposed a different 64-physical/128-logical leaf representation.
Keep four original partial sums per warp lane, pair the64 and32 tree levels
locally, then exact16/8/4/2/1 shuffles. Four output warps per128-thread block
avoid a one-warp-block occupancy cap; F16 and SIMT batches retain their current
paths. Unlike C06's stride32 numerical regrouping, all128 logical leaves keep
their original stride128 contribution order. Conservatively count this as a
non-counting re-evaluation, not another distinct attempt.

C27 packed dot/logits owners725.027/720.029/715.604 ms give whole-request
p .582227/.144212/.034959, s=1.1,o=.002: predicted5.366/1.123/.118%,
ideal139.365/16.851/3.623%, saved63.42/55.47/24.12 ms. Short model decode is
the objective; both other regimes and prompt/TTFT/public metrics are controls.
It ranks above C26. Cost30–45 minutes, uncertainty extra live registers versus
removed block barriers and grid scheduling. Existing matmul.cuh (~280 productive
category-K lines) and kernels/matmul.rs (~150 orchestration lines), no new file,
entry, dependency, public API or allocation. Require frozen107 exact matrices,
canonical gates and frozen129/130 f16/int8 local IDs before performance, plus
focused synccheck/memcheck. Keep>=5%, CV<=5%, controls<=5%, two-hour comparison.

C28 load-path inspection confirms CudaFormat is crate-private and standalone
preflight currently budgets retained packed bytes plus maximum upload staging.
A possible320-byte/256-weight representation stores integer quants and the
original f32 coefficients without half reconstruction; it increases retained
weight bytes and must never reduce context/placement or invalidate hybrid
planning. No implementation or performance claim yet. C09/C21 measured unpacking
benefits motivate it, but budget/fallback and exact CPU/GPU coefficient semantics
must be resolved before ranking it ready. It is not closed for mere complexity.

### C27 exact implementation

Clean C24 runtime after C25 restoration uses the already-frozen107 vectors in
c25-exact-baseline.log; the report-only HEAD9046dbb changes no runtime.
Implementation touches only the two declared matmul files. Each packed warp
keeps four stride128 sums; Q4/Q5 and Q6 differ only in which paired leaf is64
versus32. Restore those tree levels locally, then perform guarded warp shuffles.
Packed tail return is warp-uniform; F16 keeps the original block tree and grid.
Existing shared scratch remains in the combined entry for F16, but packed code
does not use it or execute its block barriers. No extra entry/allocation.

Exact80720 passes all15 kernel tests and107 vectors, empty diff against C24.
Generated dot/logits resources stay40 registers/1024 shared bytes, zero spills.
Removed temporary prints; frozen130-token replay underway before one-row and
canonical gates. This C06 re-evaluation remains excluded from distinct attempts.

C27 frozen130 session5312 and one-row129 session82098 pass f16/int8 with exact
frozen local IDs. Restored the canonical prompt before canonical64561 and fast
build97384. No candidate performance has been observed. After canonical tests,
run both sanitizers on dense_operations_cover_wide_dots_and_output_tails,
which exercises packed whole-warp output tails and F16 block barriers.

C27 canonical64561 and fast97384 completed successfully: all37 CUDA tests,
11 error tests, CPU workspace, formatting/CUDA check and both canonical parity
schemes pass, original16 local IDs unchanged. Sanitizer51039 passes memcheck
and synccheck with0 errors (.62/.44 s diagnostic test). All predeclared gates
precede short objective98133 against C24. Temporary source hooks are removed.

C27 short objective98133 provisionally passes: model decode40.29 ->58.93
token/s (+46.2646%, CV .0019 ->.0018), public39.02 ->57.04 (+46.1814%,
CV .0011), prompt289.00 ->289.07 (+.0242%, CV .0045), TTFT442.91 ->442.81
ms (-.0226%, CV .0045), wall4.92 s. Counts128/32/32 unchanged. No rerun.
Both medium and long controls are mandatory before acceptance; acquire them
with the same immutable executable, then refresh all three diagnostic profiles.

Read-only C28 capacity inventory from the authenticated bounded GGUF header:
retained raw2,138,290,176 bytes versus proposed cached4,286,380,032;
maximum upload staging330,301,440 ->503,316,480. Inventory53 F32 tensors
(already converted to half),27 Q6 and156 Q4. This is a size calculation, not
an allocation or performance result. Exact extra retained cost2,148,089,856
bytes requires a checked all-or-raw fallback against the same reserve, full KV
and scratch plan. Hybrid loading must retain its current raw-byte contract.

### C27 kept after all controls

Controls31845 completed, unchanged counts128/32/32,1024/32/31,3584/32/31.
Full candidate means [CV fraction]:

| Regime | Prompt token/s | TTFT ms | Model decode token/s | Public delta/s | Wall s |
|---|---:|---:|---:|---:|---:|
| short | 289.07 [.0045] | 442.81 [.0045] | 58.93 [.0018] | 57.04 [.0011] | 4.92 |
| medium | 253.92 [.0026] | 4032.78 [.0026] | 46.04 [.0000 rounded] | 43.14 [.0004] | 19.89 |
| long | 188.44 [.0001] | 19019.13 [.0001] | 28.20 [.0005] | 26.43 [.0003] | 81.45 |

Against C24, model decode +46.2646/+36.3744/+22.7154%; public
+46.1814/+36.3895/+22.7019%. Largest negative control is medium TTFT
+.2421% (prompt -.2397%), below5%. No rerun; rounded CV is not certainty.
Stock power-cap increments short .776646 s, medium .639663 s, long0;
thermal0. Retain the two-file +42/-22 change with its exact logical-tree
invariant. Four lane sums replace shared cross-warp ownership without adding
allocation or an interface. S01 remains a separate correctness support fix.
Refreshed session83911 completed all three diagnostic rows before next selection.

C27 retained commit `2d77b2f`. All83911 traces have zero dropped records.
Prefill/decode ms442.048/546.161,4016.676/690.386,18981.851/1139.901.
Packed decode dot/logits467.530/455.058/453.499 ms, compared with C24
725.027/720.029/715.604. Tensor prefill396.018/3163.151/11088.638;
prefill attention9.498/573.898/6923.414. Long decode attention637.803 ms
now exceeds packed decode; that shift must be considered in final discovery.

C26 refreshed narrowed Q4 owners156.528/1239.620/4347.121 ms yield whole-request
p .158396/.263353/.216041, s=1.1,o=.005, predicted .949/1.931/1.486%,
ideal18.821/35.750/27.558%, saved9.29/89.16/294.58 ms. Select this ready,
bounded medium-first layout experiment. C28 still lacks its exact coefficient
conversion and cold-load/capacity risk-specific gate; it remains deferred while
the ready layout diagnostic runs, not closed or silently discarded.

### C26 implementation boundary

Apply the isolated C25 tensor/quant/entry/dispatch patch on accepted C27, never
replace C27's dot/host changes. Then change only A/B shared addressing to
stage-major K32 tiles: A offset stage*TOKENS*GROUP, B offset stage*64*GROUP;
WMMA leading dimension stays GROUP=32, while outer work advances K64.
Coefficients, C tiles, two separate correction steps and barrier count remain
C25's. No padding sweep, numeric regrouping or broadening to M32/Q6.
Same existing files/estimates as corrected C25: matmul.cuh ~280 category-K,
quant.cuh ~180 category-K, kernels/matmul.rs ~165 orchestration, module.rs ~145
excluding tests. Additional risk is wrong stage offsets; full107 exact matrices,
both frozen model fixtures and memcheck/synccheck remain mandatory before the
medium prompt A/B against C27. Retention/CV/control/two-hour rules unchanged.
Freeze current107 vectors again because C27 altered the dot implementation,
even though its earlier exact gate proved equality transitively.

C26 baseline43772 passes15 kernel tests and freezes107 vectors on C27.
Patch reapplication first refused an outdated C25 context line in logits and
made no edits. Updated only that context to C27's accepted writer predicate,
then applied the isolated tensor patch; inspection confirms C27 dot/host changes
remain intact. No history rewrite or whole-file baseline replacement.
Candidate85178 passes the same107 vectors exactly and all15 kernel tests;
temporary prints removed. Static paired resources remain64 registers/19584
shared bytes, zero spills; M16/M32 remain63/72 and9792/14976; dot40/1024.
Proceed through the unchanged frozen model fixtures before canonical gates.

C28 read-only numeric design evidence: a finite binary16 coefficient times a
six-bit unsigned or eight-bit signed scale has at most18 significant bits,
so its product is exactly representable in binary32. Preserve signed zero,
Q4/Q5 subtraction/FMA ordering and Q6 signed integer range; never narrow the
reconstructed weight to half. Non-finite packed metadata must retain raw
decoding, avoiding assumptions about CPU/GPU NaN payload conversion. Existing
backend/f16.rs is the shared exact widening authority, but its current cfg
excludes standalone cuda and would need an in-scope cfg extension. No new
precision mode is implied. Capacity/fallback, hybrid isolation, malformed
blocks, exhaustive finite coefficients and cold-load overhead still need the
explicit C28 implementation/gate design after C26's decision.

C26 frozen130 replay35226 and one-row129 replay33831 pass f16/int8 with exact
local IDs. Canonical prompt restored; canonical gates and fast build started.
No C26 performance precedes the remaining canonical and sanitizer gates.

C26 canonical9744 passes every gate plus f16/int8 canonical parity; build49969
completed. Sanitizer98468 passes memcheck/synccheck with0 errors. Medium
objective starts only afterward, on the immutable c26-bench against C27.

C28 bounded-design refinement under read-only review: prefer a one-time GPU
conversion calling the existing quant helper after the raw upload, rather than
duplicating coefficient/quant arithmetic in Rust. The source allocation must
stay live until the conversion encoder submits/synchronizes, then be released;
preflight must cover the expanded destination plus the raw conversion temporary.
A cheap finite-metadata scan can retain raw decoding for special values.
This would avoid the proposed standalone f16 cfg extension and CPU arithmetic
duplication. Its new checked conversion boundary and cold-start cost still need
tests and explicit estimates before production edits; C28 remains deferred.

### C26 rejected and restored

Objective9024 completed: medium prompt253.92 ->260.47 token/s (+2.5796%,
CV .0026 ->.0029), TTFT4032.78 ->3931.32 ms (-2.5159%, CV .0029),
model decode46.04 ->45.24 (-1.7376%, CV .0008), public43.14 ->42.37
(-1.7849%, CV .0010). Counts1024/32/31 unchanged, wall19.57 s.
Stable objective below3%: reject, no rerun or extra controls. Power-cap
increment .554111 s, thermal0. Restore all four candidate production files to
accepted C27, preserve private c26.patch and immutable evidence. No rejected
C25/C26 production changes remain. Diagnostic73305 attributes the corrected
layout before closing unchanged variants or selecting C28; do not infer a
shared-bank counter result from the public improvement over C25.

C26 diagnostic73305 completed with zero dropped records. Q4 shape ms C27->C26:
K3072/N4096 276.452->328.802 (still loses); K3072/N1024 258.167->196.399;
K4096/N3072 340.234->278.278; K9216/N3072 364.767->298.418.
Unlike C25, three smaller-grid shapes now improve, saving190.072 ms combined;
the4096-output shape loses52.350 ms. Ordinary wide Q4 946.971->964.582,
Q6 total976.559->999.420 disclose diagnostic environmental/codegen variance;
do not subtract it to manufacture a public gain. The fixed public objective
remains rejected despite the successful exact layout correction.

Fresh bounded C29 directly removes the measured losing grid: use C26 only for
Q4/Q5 outputs<=3072 (at32 rows, grid<=96 blocks), preserving ordinary M16 for
larger small grids and C22 M32 for large grids. This is selected from shape
attribution before C29 performance, not a selective rerun of C26. Count it
conservatively as a non-counting bounded re-evaluation. Counterfactual savings
are near the threshold, not proof of a keep; run a complete objective A/B.
No widening of the chosen cutoff after observing results.

C28 remains deferred for its cross-layer capacity/transaction/cold-start gate;
C29 is a ready isolated dispatch experiment using already-understood layout
code. It is worthwhile even if the conservative global prediction is below5%,
because the ideal owner ceiling exceeds5% and three measured shape wins provide
new evidence absent in C25. Preserve every prior rejection and its own metric.

C29 current narrowed owners122.387/963.168/3374.643 ms give whole-request
p .123847/.204622/.167711, s=1.2,o=.002: predicted1.900/3.317/2.664%,
ideal14.135/25.726/20.151%, saved18.42/151.11/522.20 ms. Medium prompt
objective selected; short/long and all other metrics remain controls. Existing
four files and productive estimates match C26; only change its dispatch cutoff
8192-exclusive to3072-inclusive. Preserve C27. Reuse frozen107 current C27
vectors in c26-exact-baseline.log, require exact candidate identity, both frozen
129/130 f16/int8 fixtures, canonical gates and focused sanitizers. Keep>=5%,
CV<=5%, controls<=5%, one stability rerun only if CV fails, two-hour comparison.
No candidate performance before correctness; a low stable gain is not a retry.

C29 exact37308 passes all15 kernel tests and107 frozen C27 vectors with an
empty diff. Temporary prints removed; both frozen real-model fixtures follow.
The shader is exactly C26's validated stage-major implementation; only the
predeclared host output-grid cutoff differs. Counts remain22 distinct attempts.

C29 frozen130 session34014 passes both schemes. One-row129 replay follows;
canonical source will be restored before canonical gates and fast build.

Read-only discovery on C27's newly dominant long-decode attention identifies a
smaller alternative to global KV/cache restructuring: the buffered kernel uses
only32 query-head blocks, four warps each, and each V dimension serially scans
the full history. A512-thread block could distribute the existing K score work
across16 warps and the V history across four128-lane groups. Keep score trees
and softmax's128 logical leaves unchanged; only merge four f32 V partial sums.
Separate long-history entries would isolate additional ~4 KiB shared partials
from the short path, without new global scratch, memory-plan changes or launches.
This is not yet an implemented candidate. Before selection it needs a numeric
reordering gate, a frozen real-model prompt whose decode actually crosses512
positions (existing129/130 fixtures do not), and reranking after C29. Do not
claim achieved occupancy or bandwidth saturation from this parallelism bound.

Assign this distinct mechanism C30, deferred until its long-history fixture and
C29 decision. Current medium/long owners186.541/637.803 ms are27.02%/55.95%
of model decode; use a conservative local s=1.5,o=.002. Whole-request fractions
are .039630/.031697, yielding about1.13%/.87% predicted fixed-request gain.
Although whole-request ceilings are only4.13%/3.27%, the declared model-decode
target ceilings are37.02%/127.43%; keep those phase and whole-request denominators
explicit rather than incorrectly closing a material decode target by prefill
amortization. Short eligibility is zero below512 positions. It ranks below
C29's ready whole-request prediction; no candidate performance or count yet.

C29 one-row65123 passes both schemes. Restored canonical source, starting
canonical gates and fast build. No temporary test/parity hook remains in source.

C29 canonical77865 passes fmt/CUDA check/CPU workspace/11 error/all37 CUDA
tests and both canonical parity rows; fast42367 completed. Sanitizer37634
passes both focused tools with0 errors. Start the medium objective against C27
only after these gates; both other regimes are mandatory if provisional keep.

C29 medium objective91636 provisionally passes: prompt253.92 ->266.92
token/s (+5.1197%, CV .0026 ->.0032), TTFT4032.78 ->3836.37 ms
(-4.8703%, CV .0032), model decode46.04 ->46.11 (+.1520%, CV .0038),
public43.14 ->43.21 (+.1623%, CV .0036), wall19.11 s. Counts1024/32/31
unchanged. The margin above5% is small; the predeclared CV screen passes, but
this is not a statistical-significance claim. No rerun permitted or warranted.
Run short and long controls before retention, preserving the same executable.

### C29 kept after controls

Controls60354 completed successfully, unchanged token/delta counts. Means [CV]:

| Regime | Prompt token/s | TTFT ms | Model decode token/s | Public delta/s | Wall s |
|---|---:|---:|---:|---:|---:|
| short | 302.09 [.0017] | 423.72 [.0017] | 57.87 [.0018] | 56.00 [.0017] | 4.93 |
| medium | 266.92 [.0032] | 3836.37 [.0032] | 46.11 [.0038] | 43.21 [.0036] | 19.11 |
| long | 192.75 [.0002] | 18593.60 [.0002] | 27.72 [.0005] | 25.99 [.0006] | 79.89 |

Short/long prompt +4.5041/+2.2872%; TTFT -4.3111/-2.2374%. Negative controls
are short model -1.7987%, public -1.8233%; long model -1.7021%, public -1.6648%.
Every control remains within5%, so retain C29 under the predeclared threshold,
without asserting unchanged decode or statistical significance. Exact identity
and bounded model gates pass before every public row. The fixed entry and
compile-time stage count add complexity to protect separate shared-memory
contracts; the measured medium result justifies this narrow path, not universal
pairing. C25 and C26 retain their rejected history. Refresh all three profiles.

C29 stock power-cap increments short .913920 s, medium1.053986 s, long0;
thermal counters remain0. Preserve raw telemetry and these nonzero counters;
do not claim unthrottled hardware or selectively rerun a favorable row. The
same stock configuration/desktop-client conditions remain the recorded tuple.
Refreshed diagnostic session9593 runs all three rows, separate from public A/B.

C29 retained commit `1a32047`;9593 completed all three profiles, zero dropped
records. Prefill/decode ms419.236/547.300,3846.066/698.203,18533.195/1154.468.
Total tensor prefill373.995/2980.253/10507.219; attention prefill
9.484/578.163/7038.978. Decode dot/logits467.967/458.972/455.723;
decode attention27.953/186.916/646.091. Long prefill attention now exceeds
every individual tensor entry, while combined tensor work remains larger.

### C31 fresh discovery and bounded design

Current C24 prefill reloads each K/V tile for independent queries. Existing
matmul.cuh proves the supported half-input/f32-accumulator WMMA path is already
available in this exact build. C31 stages K/V once for four queries of one head,
computes QK with WMMA, then keeps softmax and PV in f32. No probability/weight
narrowing or int8-to-half approximation: select only F16 KV, rows>=4, and
head dimensions divisible by16; retain C24 for int8, smaller batches and tails.

Four real queries occupy a padded M16 tile. This deliberately retains the
current256-block geometry at32 rows/32 heads; a full16-query tile would cut
block parallelism by four. Only one warp performs the padded QK matrix work;
all128 threads consume f32 weights and update PV. Padding/idle matrix warps and
new barriers are explicitly costs, not free Tensor acceleration. Shared Q is
staged once; each16-context tile stages K/V, stores QK scores, publishes online
coefficients, and finishes PV reads before replacing shared storage.
Preserve causal masks before softmax; PV must skip future tokens, not multiply
their possibly non-finite V by zero. Output query tails never exit through a
block barrier. Reconstructed KV values are never narrowed beyond existing F16.

C31 owner p .009812/.127229/.357532, credible s=1.15,o=.005: roughly
-.37/1.17/4.34% whole-request predicted gain, with a long ideal ceiling>55%.
Long prompt objective ranks above ready C30's ~1.2% medium whole-request
prediction. C28 still needs cross-layer capacity/cold-start gates. C31 is a
bounded kernel/dispatch experiment (~45–60 minutes) despite prediction<5%:
query reuse and QK matrix work are distinct from C24's exact warp score tree.
Main uncertainty is reduced active matrix lanes versus K/V reuse, register
pressure, shared traffic and barriers. No achieved-occupancy/bandwidth claim.

Structure: existing attention.cuh (~450 productive category-K lines),
kernels/attention/prefill.rs (~70 orchestration lines), module.rs (~150 excluding
tests), exec/dispatch.rs (~130 excluding its test-only arity guard). Add one
fixed F16 tensor-prefill entry and include it in S01's11-argument test guard.
No new file, dependency, global scratch, cache layout, public API or capability.

Before production, extend the existing long/tail test with (dim,context,rows)
(16,17,4),(128,3584,4),(128,3584,5),(256,129,4),(128,33,32), preserving all
five original rows=3 cases and both KV schemes. These cover Q4 tiling, causal
K16 boundaries, query tails and the actual32-row workload. Baseline must pass
the unchanged scalar .02+.02*abs(expected) bound. C31 is numeric reordering,
not an exact-output candidate; retain the original bound and teacher-forced
16-step top-two gate, plus frozen local IDs for129/130 real-model fixtures.
Run all canonical gates, f16/int8 parity and memcheck/synccheck before public
long A/B; both other regimes and TTFT/decode controls<=5%, objective>=5%,
CV<=5%, canonical rerun rule/two-hour cap. Never weaken a failing gate.

C31 extended baseline83369 passes on C29. Before production, strengthen its
new(dim16,context17,rows4,F16) case: poison the second query's future K/V token
with NaNs and require the first query to remain exactly equal to its unpoisoned
output. Later queries are intentionally poisoned and not asserted finite.
This protects causal masking both before softmax and during PV, specifically
against `0 * NaN`; it is additional coverage, not a relaxed numeric gate.

The test's first draft failed formatting, then compilation before GPU work:
KV-write takes byte payload offsets, not a token position. After inspecting
that signature, set both F16 K/V offsets to(base+1)*dim*2 for this one-KV-head
fixture. Keep the failed compiler log; rerun the corrected baseline under a
new label. No production candidate or performance result is involved.

C31 corrected causal baseline26703 passes, including the NaN future-token check.
Pre-production resource design narrows eligibility to F16, dim==128, rows>=4,
base>=512. A generic maximum-dim256 allocation would reserve ~25.6 KiB shared
per block; fixed128 staging needs ~13.4 KiB. Preserve C24 for short histories,
other dimensions, int8 and tiny tails. This is a declared resource/overhead
boundary before any C31 performance, not a post-result workload change.

Trace attribution sorts prefill kernels by start time and excludes the first
16 batches *26 layers (base<512), verifying104/832/2912 total records.
Eligible owners are0/432.486/6893.835 ms; use these narrowed fractions instead
of all attention. Short is now an unchanged control. Add(dim128,context529,
rows4) to the baseline test and apply the existing NaN future-token check there
too, retaining its earlier dim16 coverage. Long rows4/5 cases already exercise
eligible full/tail query groups. No reference or bound changes.

Existing129/130 real-model fixtures no longer cover the narrowed new path.
Before production, declare a new immutable fixture: `benchmark` repeated543
times, spaces and final period, with the canonical empty System message.
Expected549 prompt IDs; batches at base512 and the five-row tail at544 exercise
C31, and its decode can later cover C30. Authenticate the same model and freeze
the first pinned16-step f16/int8 oracle results before candidate implementation.
This is correctness support, never a new performance workload. No choosing a
replacement prompt/reference after a failure. Restore canonical source before
public benchmarks. Structure/other gates remain as declared above.

C31 final narrowed baseline58852 passes all11 shape/history/row cases for both
KV schemes and both NaN-future checks. Eligible p is0/.095172/.350160,
s=1.15,o=.005: predicted-.498/.747/4.240%, ideal0/10.518/53.884%,
saved-4.83/33.69/800.76 ms. The uniform overhead estimate is conservative even
for the unchanged short GPU path. Long remains selected before code.

New549-token baseline63852 passes f16/int8 on retained C29, with all16 local
IDs identical to the pinned oracle:
2757,10637,2479,1636,6483,15566,1278,3541,50147,115799,58987,34796,1033,9246,1639,3508.
The private long-parity.sh verifies549 IDs before requesting the CPU oracle;
the frozen replay now accepts only an additional literal `long` fixture with
that strict count. Existing129/130 selection and IDs remain unchanged. Restore
the canonical prompt before the support checkpoint and C31 production edits.

C30 inspection additionally found the existing dispatch guard limits blocks
to256 threads. Its future512-thread variant therefore requires a narrowly
validated fixed-kernel exception and rejection tests, not a blanket weakening
of every kernel's launch guard. No C30 production change is made now.

Support tests committed separately in `92bab5a`. C31 production now implements
only its narrowed four-query F16 path. Numeric session39264 passes all11
shape/history cases, both schemes and both poisoned-future checks under the
unchanged bound. The new entry uses40 registers/13360 shared bytes, zero
stack/spills; original F16/int8 prefill remain48/80 registers, zero shared.
Formatting passes; S01 arity checking includes the new fixed11-argument entry.
Start frozen549 replay before129/130 and canonical gates. No performance yet.

C31 frozen549 session36710 passes f16/int8 with exact frozen local IDs. Replay
the unchanged130- and129-token fixtures next, then restore canonical source.

C31 frozen130 session23299 and frozen129 session47755 both pass f16/int8,
including exact frozen local IDs. Canonical source restored. Full gate31012
passes formatting, CUDA workspace check, CPU workspace tests,11 local error
tests,37 CUDA tests and canonical f16 parity. Int8 parity and sanitizers follow
before any performance decision.

C30 capability inspection: Device currently requires only256 block threads.
Preserve that support boundary; a512-thread candidate needs a cached optional
module capability and old-path fallback when unavailable, in addition to the
narrow fixed-kernel dispatch guard. Do not silently raise device requirements.

C31 canonical int8 session27662 passes with unchanged16 local IDs; fast59445
completed. Focused sanitizer41880 passes memcheck3.61s and synccheck1.63s,
both0 errors. Long objective5950 completes in66.75s: prompt192.75 ->234.39
token/s (+21.6031%, CV .0002 ->.0020), TTFT18593.60 ->15290.80ms
(-17.7631%, CV .0020), model decode27.72 ->27.80 (+.2886%, CV .0000
rounded), public25.99 ->26.06 (+.2693%, CV .0003). Counts3584/32/31
unchanged. Stock power-cap counter advances .733826s, thermal counters0;
observed SM1987/memory7301MHz and temperatures66–67C. No unthrottled or
statistical-significance claim, no selective rerun. Short/medium controls59974
are mandatory before acceptance; immutable candidate executable preserved.

### C31 kept after controls

Controls59974 complete, preserving128/32/32 and1024/32/31 counts. Means[CV]:

| Regime | Prompt token/s | TTFT ms | Model decode token/s | Public delta/s | Wall s |
|---|---:|---:|---:|---:|---:|
| Short | 302.14[.0019] | 423.64[.0019] | 58.11[.0010] | 56.26[.0005] | 4.92 |
| Medium | 280.21[.0023] | 3654.43[.0023] | 45.69[.0030] | 42.81[.0029] | 18.29 |
| Long | 234.39[.0020] | 15290.80[.0020] | 27.80[.0000 rounded] | 26.06[.0003] | 66.75 |

Versus C29, short/medium prompt+.0166/+4.9790%; largest negative control
medium public-.9257% (model-.9109%), within5%. Short power-cap increment
.654537s, medium0; thermal counters0 throughout. No rerun used. C31 is kept
in this commit after bounded numeric, all three frozen fixtures, canonical
f16/int8, CPU/CUDA/error and sanitizer gates. Added complexity is one fixed
numeric kernel with narrow shape dispatch; no ownership or public interface
changes. Record18 accepted optimization commits,14 distinct kept plus four
non-counting re-evaluations. Refresh all three profiles before reranking.

C31 committed as `4d9e718`; serial three-regime profile42594 running.
C30 now has its required frozen549-token oracle. Before production, add
(dim,context,rows)=(3,514,3),(129,1025,3),(256,4097,3) to the existing scalar
attention test, preserving all previous cases. These cover511/512 selection,
dimension tails, and4094/4095/4096 buffered/fallback boundaries in both schemes.
Run this strengthened baseline on C31, then commit support separately.

C30 structure, no new file: shaders/attention.cuh (~470 productive category-K
lines), kernels/attention/decode.rs (~45 orchestration), module.rs (~160),
exec/dispatch.rs (~145), kernels/tests.rs (tests excluded). Template the one
buffered attention operation by128/512 threads. Existing entries remain128;
new two fixed entries require512x1x1 and positions512..4095. Only128 logical
softmax leaves write the existing128-element shared reduction; all512 threads
participate in barriers. Four contiguous V history segments produce f32
partials merged in fixed order; no probability narrowing, no global allocation
or extra launch. Optional module capability is queried once; query failure or
<512 threads keeps the original path. Device admission requirements unchanged.
Dispatch rejects the new entries at every geometry except512x1x1 and rejects
>256 for all old entries. Add rejection tests before any unsafe launch, and
exercise forced capability fallback in numeric tests. Main risk: altered PV
summation and barrier/resource correctness. Gate remains scalar .02+.02*abs,
three frozen fixtures and canonical16-step f16/int8, CPU/CUDA/error suites,
memcheck/synccheck, followed by the predeclared medium model-decode objective
and short/long plus all non-target controls. Rerank with refreshed C31 profiles
before implementation; no C30 performance or attempt yet.

C31 profile42594 completes all three traces with0 dropped records. Prefill/
decode ms: short424.780266/553.847067; medium3680.644040/698.771460;
long15226.937087/1145.791485. Tensor-prefill totals378.184299/3019.297872/
10473.090268ms; attention-prefill totals9.572596/374.651622/3772.236116ms.
The new tensor-attention owner is228.423715/3628.092264ms medium/long versus
C29 eligible432.485632/6893.834752ms. This supports the public improvement,
not an independent unprofiled speed claim. Decode attention27.873064/
186.856116/639.175215ms, dot+logits474.817745/461.811309/458.069004ms.
Prefill gaps6.648871/52.549473/169.452288ms and decode gaps10.316113/
11.058561/9.920031ms remain small. Combined matrix work is largest remaining
prefill owner; long decode attention is larger than dot+logits.

C30 refreshed whole-request p0/.042666907/.039039016, s1.5,o.002:
predicted-.1996/1.2374/1.1136%, ideal0/4.4569/4.0625%, saved-1.96/
53.53/180.31ms. Decode-phase p0/.267406622/.557846016 gives predicted
-.1996/9.5453/22.5413%, ideal0/36.5014/126.1656%. Medium remains the
declared objective (highest whole-request effect); long is a mandatory control.
Short is ineligible; its uniform overhead estimate is conservative. C28's
cross-layer cache gate remains deferred, not falsified. C30 is the highest
ready bounded candidate. Baseline76918 passes all14 scalar cases in2.40s;
commit these extra support cases before production. No threshold/oracle change.

C30 support checkpoint `e772a0a`; clean worktree confirmed before production.
Implementation uses the planned512-thread specialization and optional cached
module capability, with exact geometry rejection. Numeric98207 passes all14
cases; added forced-capability fallback and guard tests43746 pass. Static
new F16/int8 entries use40/48 registers and20992 shared bytes, zero stack or
spills. Old entries remain40 registers/16936 shared bytes. These are resource
bounds, not measured occupancy. C30 reaches its correctness gate as the24th
distinct attempt; frozen/canonical/sanitizer and performance gates still pending.

C30 frozen549 session48511 and frozen130 session23634 pass both schemes and
exact frozen local IDs. Frozen129 follows, then restore canonical source.

C30 frozen129 session66608 passes both schemes. Canonical source restored;
gate56562 passes fmt, CUDA workspace check, CPU workspace,11 error tests,
all38 CUDA tests and f16 parity. Canonical int8 session18827 passes; fast19168
completed and immutable executable/PTX saved. Sanitizer69191 passes memcheck
6.15s and synccheck2.50s, both0 errors. Only now start medium model-decode
objective against C31; both controls required if provisional keep.

C30 medium2844 provisionally passes: model decode45.69 ->54.58 token/s
(+19.4572%, CV .0030 ->.0020), public42.81 ->51.11 (+19.3880%,CV .0016).
Prompt280.21 ->278.54(-.5960%,CV .0002), TTFT3654.43 ->3676.31ms
(+.5987%,CV .0002), within controls. Counts1024/32/31; wall18.11s.
Power-cap increment .554086s, thermal counters0. No rerun; short/long next.

### C30 kept after controls

Controls9284 complete with unchanged128/32/32 and3584/32/31 counts. Means[CV]:

| Regime | Prompt token/s | TTFT ms | Model decode token/s | Public delta/s | Wall s |
|---|---:|---:|---:|---:|---:|
| Short | 304.04[.0012] | 420.99[.0012] | 58.29[.0000 rounded] | 56.47[.0004] | 4.89 |
| Medium | 278.54[.0002] | 3676.31[.0002] | 54.58[.0020] | 51.11[.0016] | 18.11 |
| Long | 234.22[.0023] | 15302.20[.0023] | 44.90[.0045] | 42.07[.0045] | 64.86 |

Against C31, short/long model decode+.3098/+61.5108%; public+.3733/
61.4351%. Largest negative control is medium TTFT+.5987%, within5%.
Short power-cap increment .973690s, long0; thermal counters0; long observed
SM1980/memory7301MHz, temperature67C. No rerun or significance claim.
C30 kept after all predeclared gates. Complexity adds one specialization of
the existing operation and one cached capability bit to preserve old-device
admission. The fixed-kernel geometry guard protects the larger launch invariant;
no KV/global allocation/ownership changes. Count24 distinct attempts,15 kept
plus four non-counting retained re-evaluations=19 optimization commits.
Refresh all three profiles before C28 selection and fresh discovery.

C30 retained as `c15f222`. Profile1431 completes all three traces with0 drops.
Prefill/decode ms:422.984315/551.517230,3660.077159/577.968137,
15202.672669/708.805028. Prefill matrix totals376.236302/3003.636301/
10456.572495ms; decode dot+logits472.825795/468.909172/471.213855ms.
Decode attention28.955761/56.123217/187.264681ms; eligible medium/long
owners were186.856116/639.175215ms on C31. Prefill gaps6.584159/47.065504/
172.610374ms; decode gaps9.887959/13.304755/11.000824ms. Matrix work now
dominates every decode regime and combined prefill; attention remains material
in long prefill. No achieved occupancy/bandwidth counter is available.

### C28 bounded implementation and gate

Select the existing C28 lossless load-time weight cache next. Remove repeated
bit extraction and coefficient reconstruction, not quantization precision.
Current matrix owners (prefill plus decode, omitting small embedding/first
logits conservatively) give p .871278349/.819374318/.686786391. With credible
s1.15 and o.02 (extra steady-state traffic/dispatch uncertainty), predicted
whole-request gains10.3320/9.5140/7.4784%, ideal676.8701/453.6311/219.2709%,
saved91.26/368.18/1107.13ms. Short model decode remains selected; its owned
phase p=.857318265. The large byte expansion could instead lose performance:
these are hypotheses, not a bandwidth-saturation claim. One-time conversion,
temporary allocations, synchronization and startup costs are measured separately
below; they are not silently zero or hidden in steady-state public throughput.

Private cache layout per256 weights:320 bytes, first64 bytes f32 metadata,
then256 one-byte quants. Q4/Q5 store eight(factor,negative bias) pairs and
unsigned quants; Q6 stores16 factors and signed quants[-32,31]. New crate-private
format tags distinguish these three layouts; original raw tags/paths remain.
One GPU conversion uses the existing raw `cuda_weight_parts` arithmetic.
All finite half coefficients widen exactly, and coefficient/integer products
fit f32 without overflow. Preserve original logical reduction leaves, Q4/Q5
K32 versus Q6 K16 groups, paired-stage order and all dispatch geometry. Never
narrow reconstructed weights to half. Non-finite metadata keeps that tensor raw.
Cached views require four-byte alignment for f32 metadata (raw views stay even).

Standalone admission first validates the original raw plan, then chooses the
expanded cache only if the complete reserve/KV/scratch/weights/staging plan fits;
otherwise retain all raw weights, never change context or placement. Expanded
weights4,286,380,032 bytes include destinations; max cached staging503,316,480
conservatively covers the largest simultaneous raw conversion330,301,440.
Finite-metadata fallback may use less, never more. Hybrid always retains raw
loading and its existing byte accounting. Conversion source allocation stays
live through encoder submit even after a launch error; unpublished transactions
must release every allocation. No dependency, public API or device requirement.

Necessary structure, estimates exclude tests:

```text
cuda/
  loader.rs (~150 lines): choose admitted standalone cache, isolate hybrid
  module.rs (~165 lines): one additional fixed conversion entry
  mem/
    budget.rs (~175 lines): checked raw/cached plan arithmetic
    buffer.rs (~180 lines): private cache tags and aligned views
    weights/
      mod.rs (~155 lines): existing selected WeightSet transaction
      cache.rs (~95 lines): single-tensor conversion ownership and metadata gate
  kernels/
    mod.rs (~75 lines): checked format/span mapping
    matmul.rs (~150 lines): preserve paired dispatch for corresponding cache tags
    tests.rs (tests only): raw/cache identity and unchanged references
  shaders/
    quant.cuh (~250 lines, category K): lossless decode/cache variants
    matmul.cuh (~320 lines, category K): same operation on cached format variants
```

Split existing mem/weights.rs into its domain folder before implementation:
conversion ownership plus the existing transaction would exceed200 productive
lines together. No prefix-named sibling or public abstraction. Existing tests
and history remain with weights/mod.rs. No change to embedding dispatch is
needed beyond the shared private format mapping.

Pre-performance gates: exact raw/cache GPU embedding bytes across every finite
binary16 coefficient (63,488 patterns per format, with representative integer
scales/quants, not every Cartesian combination); preserve signed zeros and high
range. Extend all107 existing matrix vectors with raw/cache identity including
f32 logits and original references. Add malformed block, non-finite fallback,
cache view alignment, exact-fit/one-byte capacity fallback/overflow and upload
transaction failpoint checks, including release after converted weights.
Run canonical CPU/CUDA/error gates, all three frozen f16/int8 fixtures,
canonical f16/int8, CUDA-hybrid feature check and canonical25%-mixed f16/int8
parity to verify isolation, plus focused memcheck/synccheck. No oracle weakening.

Before production, capture three separate startup observations on C30 using a
temporary timer around `Engine::new` in examples/bench.rs. Same model/context/
KV/build, short prompt, max2, warmup0,reps1 for this diagnostic only. Freeze
the startup timer before candidate code and repeat identically on C28; report
raw times, dispersion and startup/steady-request break-even if applicable.
Remove timer before canonical public benchmarks and final commits. Public A/B
remains max32,warmup1,reps3 on loaded-model throughput, objective>=5%,CV<=5%,
both other regimes/all metrics controls<=5%, canonical rerun/two-hour rule.
Memory/startup tradeoff must be explicit in any retention, not a cold-speed
claim. C28 is ready after the separate startup baseline; no implementation or
countable attempt yet.

C28 startup baseline build25907 and serial session13264 complete on C30.
Engine initialization ms878.103766,701.841815,677.758611 (median701.841815);
walls1.50/1.32/1.30s. These are fresh-process initialization observations,
not a disk-cold experiment: no OS cache flush or machine setting changes.
Retain the slower first observation; diagnostic dispersion is not the public
objective CV and does not authorize selective repeats. Temporary example timer
removed exactly; preserve its immutable baseline executable and raw logs.
The public C30 tuple remains the max32 benchmark already recorded, not these
max2 diagnostic rates. C28 production follows this checkpoint.

Pre-production inspection finds one additional in-scope invariant: public
`Engine::memory()` currently uses the family's original representation estimate,
so enabling a cache without updating that immutable summary would underreport
weights. Preserve its API and32-byte per-tensor accounting convention. Compute
the CUDA summary from the retained immutable WeightSet spans, counting tied
embedding once, with checked alignment/sums. This is representation accounting,
not live allocator telemetry. The family homogeneous summary will accept its
already-loaded selected backend and use this value only under standalone CUDA;
every other backend and hybrid accounting stays unchanged. Move the existing
homogeneous call after backend load, adding only its backend argument.

Additional existing files: cuda/mod.rs (~170 productive orchestration lines)
owns the backend-local immutable byte summary; family/mistral/memory.rs (~125)
selects it; family/mistral/mod.rs stays<=200 productive lines excluding its
test-only imports/modules, with no extra responsibility. Add aligned/tied
summary coverage and record the selected weight bytes in the temporary startup
diagnostic for A/B. No public field/signature, placement or external effect.
Startup baseline mean752.568064ms, sample SD109.381943ms,CV .145344918;
the high startup dispersion is disclosed separately and does not trigger the
steady-state objective rerun rule. Engine::new synchronously calls family::load,
confirmed in api/engine.rs; timer placement measures actual initialization.

C28 production draft now includes the declared private formats,320-byte GPU
conversion, retained raw fallbacks, checked capacity choice, aligned views and
immutable CUDA memory summary. weights.rs moved with its tests to
weights/mod.rs; new cache.rs owns the single-tensor conversion transaction.
First compile43541 found only an incorrect relative Rust import after the
file move (the module namespace does not gain a level); corrected to the
original `super::buffer`. Preserve failed compile log. No GPU candidate
performance or relaxed gate. Initial raw-path tests and dedicated cache gates
are next; no cache performance has been observed.

C28 resolved check/raw session59261 passes all38 existing CUDA tests.
Dedicated coefficient gate49180 passes in .67s: bit-identical f32 embedding
outputs over all63,488 finite half metadata patterns for each of Q4/Q5/Q6,
including signed zeros and extrema. This is the25th distinct attempt reaching
its predeclared correctness gate; matrix/cache capacity/transaction/summary,
frozen/canonical/hybrid/sanitizer gates remain pending. No performance yet.

C28 cache gates79833 pass all42 CUDA tests in56.46s. Every107 matrix fixture
now runs both raw and cached layouts and asserts identical output bits, including
single projections/f32 logits, paired/ordinary/wide tensor tiles and high-range
cancellation. Original scalar references remain unchanged. Capacity exact-fit,
one-byte raw fallback, overflow, malformed blocks, special metadata, aligned
views and every selected cached-upload failpoint pass. Added immutable-summary
test checks two aligned raw spans, cached tied embedding counted once, and a
dedicated raw tail. Coefficient assertion diagnostics now report only the first
mismatching byte instead of dumping64 MiB vectors; the exact gate is unchanged.

C28 summary/hybrid-check42449 passes. Frozen549 session7396 passes both KV
schemes with exact frozen local IDs;129/130 and canonical gates follow. Static
resources remain unchanged for dot/logits40 registers/1024 shared, tensor
M16 63/9792, M32 72/14976 and paired64/19584, all0 spills. New conversion
entry uses20 registers and0 shared,0 spills. This is not achieved occupancy.

C28 frozen130 session2561 passes both KV schemes with exact frozen local IDs;
frozen129 session39982 running. Canonical source will be restored afterward.
Current accepted optimization commits additionally include C31 `4d9e718` and
C30 `c15f222`; C28 is not accepted and has no performance result.

C28 frozen129 session39982 passes both schemes; canonical source restored.
Canonical gate93584 passes fmt/CUDA workspace/CPU workspace/11 error tests,
all43 CUDA tests and f16 parity. Family boundary productive count198 after
excluding its test-only imports/model and test module; all modified orchestration
files remain below200 productive lines. Remaining serial parity session41015
covers standalone int8 and both25%-mixed hybrid rows before sanitizers.

C28 parity41015 passes standalone int8 and both25%-mixed hybrid rows, all16
canonical local IDs unchanged. Fast62513 completes; uninstrumented candidate
executable/PTX saved. Sanitizer23262 memcheck passes all16 kernel tests in58.05s
with0 errors; synccheck still running. Reintroduce the frozen startup timer
only for a separate executable, plus a post-timer immutable weight-byte record;
the public candidate binary is already preserved without instrumentation.

C28 synccheck23262 also passes all16 kernel tests in53.15s,0 errors.
Startup build65923 and serial diagnostic36795 complete after correctness:
initialization1032.296543,880.173383,875.195984ms (median880.173383), walls
1.69/1.53/1.58s. Each immutable API summary reports4,286,380,032 weight bytes,
matching the admitted cache inventory. Median startup overhead178.331568ms
versus C30; retain all first observations and report variability, not a cold
speedup. Both temporary example hooks removed exactly; canonical prompt also
restored. Start short model-decode objective with the previously saved
uninstrumented c28-bench; both other regimes required for provisional keep.

### C28 rejected and restored

Objective69774 completes in5.24s: short model decode58.29 ->51.47 token/s
(-11.7001%, CV .0000 rounded ->.0033), public56.47 ->49.82(-11.7762%,
CV .0032). Prompt304.04 ->329.22(+8.2818%,CV .0054), TTFT420.99 ->388.80ms
(-7.6463%,CV .0054). Counts128/32/32 unchanged. Reject the declared decode
objective and public control; do not retarget to the favorable prompt result.
No rerun or other public controls are needed after rejection. Power-cap
increment .668421s, thermal counters0. Startup candidate mean929.221970ms,
sample SD89.299884ms,CV .096101779; mean overhead176.653906ms and median
178.331568ms, with +2,148,089,856 retained weight bytes. No startup benefit.

Save complete candidate including new files in private c28.patch, c28-bench,
c28.ptx and startup binaries/logs. Restore all candidate production/tests to
accepted C30 and move weights/mod.rs back to weights.rs; remove only the new
cache.rs. These rejected files are recoverable from c28.patch. git status now
contains only this investigation record; C31/C30 and their permanent tests are
preserved. Count25 distinct:15 kept,9 rejected,1 interesting/restored, plus
four non-counting retained re-evaluations;19 optimization commits remain.

Diagnostic93105 completes with0 drops. The original summary's first-kernel
boundary incorrectly includes the new load-time conversion; retain that raw
summary but invalidate its835.503404ms prefill duration. Correct the private
analyzer to start at the first forward embedding, which is also the previous
baseline boundary. Separate corrected forward summary is preserved; conversion
kernels own61.933839ms before generation, not removable prefill time.

Authenticated shape attribution (same728 prefill/5642 decode projection calls,
checked grid and first-sampling boundary) isolates the failure. C30 ->C28:
prefill Q4 253.527743 ->265.776374ms (worse), Q6 122.708559 ->82.210950ms
(save40.497609ms). Decode Q4 371.514836 ->429.050395ms (worse), Q6
53.689473 ->61.546003ms (worse). Only Q6 K3072/N1024 decode saves1.146446ms;
its entire original8.373046ms owner is1.518% of551.517230ms decode, ideal
gain1.542%, below5%. Do not retry that unchanged subset as an objective.
Extra traffic is consistent with the losses, not proof
of bandwidth saturation. Both Q6 prefill shapes improve: K3072/N1024
26.994901 ->18.532899ms; K9216/N3072 95.713658 ->63.678051ms.

Fresh bounded variant C32: retain original weights for decode/embedding/logits
and add a lossless cache only for Q6 projection prefill. This directly removes
both observed losing mechanisms (Q4 cache and decode expansion), but requires
explicit dual-representation ownership/capacity and gates before implementation.
Assign deferred pending its bounded design; conservatively count it as a
non-counting C28 re-evaluation if attempted. Do not infer acceptance from C28's
favorable non-objective. No C32 production exists yet.

Corrected C28 forward prefill397.593704ms, gap6.803227ms; decode622.107581ms.
C32 authenticated inventory adds511,180,800 bytes for Q6 projection caches
while retaining original2,138,290,176 bytes:2,649,470,976 total. Original
330,301,440-byte staging already conservatively covers conversion destinations
when raw and cached representations are both included in retained weights.
This is a finite size estimate, not allocation/performance evidence. C32 still
needs narrow dual-owner/view/lifecycle/dispatch design before becoming ready.

### C32 selected design and predeclared gate

Fresh C30 attribution, before C32 implementation or performance, gives Q6
prefill owners122.708559/994.119299/3454.486942ms; p=.125919307/.234570239/
.217106607. Use s1.3 (below the observed C28 short local~1.49),o=.003:
predicted2.6756/5.3887/4.9430%, ideal14.4059/30.6456/27.7313%, saved25.39/
216.70/749.45ms. Medium has the largest credible whole-request effect, so it
replaces the provisional short label as C32's objective before any candidate
code/performance. This is not retargeting C28's failed objective; C28 stays
rejected. All three prompt/TTFT/decode metrics remain mandatory controls.

Keep every original raw weight. Only selected layer matrices of Q6 type and
two dimensions may own an additional320-byte-per-block cache; embedding,
tail/logits and norms never do. Standalone chooses this option only if original
weights plus511,180,800 additional bytes and full-context/reserve/scratch/
staging fit. Hybrid never creates it. Conversion consumes the already-retained
raw GPU allocation, so there is no redundant upload or raw temporary beyond
the original representation. Non-finite metadata retains raw alone.

CudaBuffer privately owns an optional Arc<CudaBuffer> prefill companion. A
checked one-time attachment accepts only raw Q6 plus a matching-size cached Q6
buffer with no companion of its own; cycles/nesting and mismatched formats
are impossible through that boundary. Full aliases retain the companion;
partial raw views discard the optimization and remain valid raw views. The
cached allocation itself requires f32 alignment. Matmul encode_batched chooses
the companion only for rows>=16; all single/logit/embedding/SIMT paths use raw.
Immutable public weight accounting includes both aligned representations,
counting tied global output once. No public API, dependency or placement change.

Structure (productive estimates, tests excluded): existing loader.rs~155,
module.rs~165, cuda/mod.rs~175, mem/budget.rs~180, kernels/mod.rs~70,
kernels/matmul.rs~150, family/mistral/memory.rs~125 and mod.rs198. Split
mem/weights.rs as declared for C28 into weights/{mod.rs~155,cache.rs~100}.
Dual representation would take buffer.rs beyond200, so split before editing:
mem/buffer/{mod.rs~175,transfer.rs~100}; mod owns formats/allocations/views and
companion invariants, transfer owns CPU/device upload/read/write. Existing
buffer tests stay with their responsibilities; no test/history is discarded.
Shaders quant.cuh~210 and matmul.cuh~320 remain category K, with only one
cached Q6 variant instead of C28's three cached formats.

Gate before performance: exact raw/cache Q6 coefficient/107 matrix cases and
unchanged scalar references; raw single/logit identity with attached companion;
non-finite metadata, malformed blocks, full/partial alias lifetime, alignment,
attachment format/size/nesting rejection, both-representation byte accounting,
exact-fit/raw fallback/overflow, conversion and selected-upload release.
All canonical CPU/CUDA/error and three frozen f16/int8 fixtures, standalone
f16/int8, hybrid check plus25%-mixed f16/int8 isolation, memcheck/synccheck.
Reuse immutable C30 public and fresh-process startup baselines (runtime unchanged
since capture); repeat the same three startup observations and report memory/
startup tradeoff. Public objective medium prompt>=5%, CV<=5%, every control
regression<=5%, canonical rerun and two-hour comparison limit. Preserve C28's
rejection and count C32 conservatively as a non-counting re-evaluation.

C32 draft reapplies C28 only as a reversible source starting point, then removes
Q4/Q5 caches and narrows selection/ownership as declared. buffer.rs split into
buffer/{mod.rs,transfer.rs}; weights.rs split into weights/{mod.rs,cache.rs}.
Both original test suites remain. Cached Q6 is now a companion of the raw
allocation; globals stay raw, only two-dimensional layer projections (excluding
norm slots0/5 explicitly) can attach it, and only rows>=16 select it. Initial
compile48445 passes. Preserve original source-representation validation in the
public memory summary and require the dual representation never to undercount
it; this also avoids making shared weight accounting unused in standalone CUDA.
Cache gates8093 pass, including exhaustive finite Q6 reconstruction and new
alias/ownership checks. Full matrix/canonical/frozen gates follow. C32 is a
pending non-counting re-evaluation; no performance or acceptance yet.

C32 full gate84828 passes all43 CUDA tests in57.56s, including107 raw/companion
matrix fixtures, finite Q6 reconstruction, raw decode/logit identity, alias
lifetime/attachment checks, norm/global exclusion, capacity and aligned dual
summary. Test helper GGUF Q6 tensors now carry a two-dimensional shape to
exercise eligible layer matrices; payloads/references/thresholds are unchanged.
Begin frozen549 before129/130 and canonical gates. No C32 performance yet.

C32 frozen549 (14334), frozen130 (completed process recovered from both
immutable logs after context rollover), and frozen129 (31705) pass for both
KV formats with all16 local IDs identical to their respective frozen baseline.
The canonical prompt is restored before gate57809; thresholds and references
are unchanged. Candidate remains pending.

Fresh source discovery, queued behind C32: tensor prefill still stores opaque
WMMA accumulators to shared C, then synchronizes and reloads them for the
per-group correction. Explicit lane-owned MMA accumulators could remove that
round trip; publishing row sums with A/B could reduce three block barriers to
two without changing coefficient-group order. This is not yet implemented or
accepted. The documented f16 A/B and f32 accumulator ownership in
[PTX ISA8.4, m16n8k8 fragments](https://docs.nvidia.com/cuda/archive/12.4.1/parallel-thread-execution/index.html)
provides a bounded mechanism to inspect after C32's decision. Register pressure,
shared-load layout, target admission and changed MMA arithmetic grouping need
explicit verification. Do not declare fresh discovery exhausted while this
material tensor-prefill owner has an untested source-supported mechanism.

C32 canonical57809 passes fmt, CUDA workspace check, CPU workspace tests,
11error_matrix, all43 CUDA tests (57.79s) and standalone f16 parity.
Standalone int8 and25%-mixed hybrid f16/int8 pass7251 with identical canonical
IDs. Hybrid check19008 exposed only an unused standalone accounting method;
gate that method under cfg(cuda or test), preserving test coverage, then the
resolved hybrid check passes without warnings. Fast/public binary29975 and
separate temporary startup-timer binary63946 build cleanly; the example source
is restored exactly before public measurements. Static resources retain the
C30 ordinary/paired/wide tensor counts63/64/72 registers and9792/19584/14976
shared bytes, raw dot40/1024, with no spills; conversion uses16 registers and
no shared storage. These are compiler resource counts, not achieved occupancy.
Serial memcheck/synccheck19500 is running before the public objective.

### C32 accepted — Q6 prefill companion, raw decode retained

Memcheck/synccheck19500 pass all16 kernel tests in55.43/51.12s, both0 errors.
Remove the test-only leftover one-format loop and unreachable Q4/Q5 metadata
branches without changing values or bounds; the exhaustive finite-coefficient
test passes again1105 in1.36s. Production code is unchanged by that cleanup.

Unprofiled objective1105 and controls13519, same authenticated fast CUDA tuple,
all-GPU context4096 f16, warmup1/reps3/max32:

| Regime | Prompt t/s [CV] | TTFT ms [CV] | Model decode t/s [CV] | Public delta t/s [CV] | Wall s |
|---|---:|---:|---:|---:|---:|
| short | 340.41 [.0044] | 376.02 [.0044] | 58.54 [.0028] | 56.69 [.0031] | 4.73 |
| medium objective | 312.14 [.0028] | 3280.57 [.0028] | 55.40 [.0036] | 51.90 [.0042] | 16.34 |
| long | 256.05 [.0023] | 13997.42 [.0023] | 45.07 [.0028] | 42.24 [.0031] | 59.65 |

Counts remain128/32/32,1024/32/31,3584/32/31. Against immutable C30, medium
prompt+12.062899%, TTFT-10.764598%, model+1.502382%, public+1.545686%.
Short prompt+11.962242%, long+9.320297%; every control improves, all objective
CVs<=5%. No stability rerun used. Comparison completes about76 minutes after
the C30 baseline, within the two-hour limit. Power-cap counter increments
short/medium/long .673863/0/0s, thermal counters0. Stock conditions unchanged;
medium starts cooler after test/idle intervals (48C) and rises to58C, with
observed core clocks around1995–2002MHz and a final1920MHz sample, versus the
earlier baseline's usual1980–1987MHz. Record this natural clock/temperature
variation rather than claim frequency-controlled or statistical isolation.

Fresh-process startup1105:709.151465,705.400658,721.369990ms; mean711.974038,
sample SD8.350454, CV.011728593, median709.151465ms. C30's preserved baseline
median701.841815ms gives+7.309650ms; its mean752.568064ms was skewed by the
first878.103766ms observation, which remains included. Do not claim faster
startup from the lower candidate mean. No OS cache flush: these are fresh
processes, not guaranteed disk-cold loads. The observed median extra cost is
less than one request's TTFT saving even on short (44.97ms), but startup noise
precludes a precise universal amortization claim. Public immutable weights are
exactly2,649,470,976 bytes in all three observations, +511,180,800 versus C30.

Keep C32 as a non-counting C28 re-evaluation; C28 remains rejected. The added
ownership/budget complexity protects a necessary invariant: cached prefill
must neither replace raw decode storage nor undercount physical retained
weights. Two small domain splits preserve original tests and keep orchestration
below200 productive lines; no dependency, public API or placement change.
Counts remain25 distinct attempts:15 kept,9 rejected,1 interesting/restored,
plus5 accepted non-counting re-evaluations (20 optimization commits total),
separate S01 correctness and2 closed/untried entries. New profiles35420 run
serially for all three regimes before selecting the next candidate.

C32 profiles35420 complete all three regimes with0 dropped events. Forward
boundaries exclude weight conversion. Short/medium/long prefill-first-sample
378.520838/3321.522521/14100.331428ms, decode555.739340/590.446123/716.350495ms;
prefill gaps5.939729/49.354619/181.362665ms, decode gaps11.146637/13.385617/
11.819491ms. Tensor-prefill totals331.837728/2660.634682/9293.730390ms;
decode matmul+logits476.540697/478.346683/475.708982ms. Authenticated182-call
projection order, grid and batch checks attribute Q6 prefill82.187524/634.392543/
2240.290969ms (C30:122.708559/994.119299/3454.486942), Q4
249.650204/2026.242139/7053.439421ms. Thus the selected Q6 owner improves
without claiming general bandwidth saturation or attributing load conversion
to generation. Tensor prefill remains material in every regime; long tensor
attention alone3652.907027ms is a second material owner. Fresh discovery follows.

### C33/C34 fresh pool and C33 selected design

C33 is a distinct synchronization/accumulator-ownership premise, not another
tile-size or quant cache retry. C32 whole fixed-work T short/medium/long is
934.260178/3911.968644/14816.681923ms; tensor p .355187704/.680126791/
.627247749. Conservative local s1.2,o.003 predicts5.954421/12.404318/
11.301720%, ideals55.083891/212.623869/168.274705%, savings52.503507/
431.703208/1504.505019ms. This is an uncertain estimate, not measured stalls.
C34 eight-real-query reuse has p0/.059279135/.246540153; s1.25,o.003 predicts
-.299103/.893495/4.855659%, ideals0/6.301459/32.721074%, savings
-2.802781/34.643718/686.131360ms. C34 remains a credible bounded long candidate
despite a prediction just below5%; C33 medium ranks first by whole-work effect.

C33 baseline is accepted10b186c, clean worktree confirmed after commit, with
immutable c32-bench/public/profile records. Change only shaders/matmul.cuh
(~355 productive lines, existing category K: one matmul operation family).
No new file/function, host dispatch, allocation, API, dependency or model change.
Replace opaque WMMA C staging with two N8 tiles per warp using
mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32. The official PTX ISA8.4
fragment and instruction sections above confirm explicit lane ownership and
sm75 admission, matching build/cuda.rs and device.rs. Precision stays half
multiplicands/f32 accumulators; instruction-internal reduction order is not
specified, so this is a numeric-gated candidate, not claimed bit-equivalent to
WMMA. Preserve increasing coefficient-group correction and input row-sum order.
Compute row sums from the same global half inputs before the first block
publication barrier; all lanes execute uniform MMA loops and the final barrier
protects replacement of A/B/coefficients/row sums. Remove shared C and its
store/reload/barrier; preserve M16/M32 and paired eligibility/tails.

Risk: register pressure, scalar shared loads, f32 summation order and warp-uniform
execution. Gate unchanged scalar .02+.02*abs bounds across107 matrix fixtures,
raw/cache bit identity, canonical and frozen129/130/549 f16/int8 token parity,
all CPU/CUDA/error tests, hybrid build and25%-mixed isolation, memcheck/synccheck.
No threshold relaxation or extra lossy conversion. Then objective medium
prompt>=5%, CV<=5%, all other controls regression<=5%, canonical rerun and
two-hour comparison rules. If rejected, preserve evidence, restore exactly and
use the observed failing owner to evaluate a genuinely bounded follow-up.

C33 draft changes only the matmul shader, +47/-35 lines. First kernel gate30419
passes all16 tests (including107 raw/cache tensor fixtures) in50.91s. This
reaches the predeclared correctness gate and counts as distinct attempt26;
real-model/frozen/canonical gates and performance remain pending. Fast68063
builds without a target change. Static ordinary/paired/wide resources become
48/56/64 registers and5696/11392/6784 shared bytes, all0 spills, versus C32
63/64/72 and9792/19584/14976. Lower static footprint is evidence of the intended
ownership change, not yet throughput or achieved-occupancy evidence.

C33 frozen5497598, frozen13041710 and frozen12967536 all pass f16/int8,
with exact frozen local token identity. Restore canonical USER_CONTENT before
the full gate. No oracle or scalar threshold changes; no performance yet.

C33 canonical90505 passes fmt, CUDA workspace check, CPU tests,11error_matrix,
all43 CUDA tests57.50s and standalone f16 parity13.72s. Hybrid build29216
passes without warnings. Standalone int8 and25%-mixed hybrid f16/int8 isolation
follow serially before sanitizers/public objective. C34 will conservatively be
a non-counting C31 tile-reuse re-evaluation if attempted; it does not inflate
the distinct-attempt quota.

C33 standalone int8 and25%-mixed hybrid f16/int8 pass58283 with unchanged
canonical IDs. Serial memcheck/synccheck5881 runs on the canonical debug binary.
Source-only follow-up noted, not bundled: explicit MMA half-pair loads now make
the shared-bank mapping inspectable. With32 four-byte banks, GROUP32 has
16-bank row stride and GROUP16 an8-bank stride; the eight lane groups can
therefore revisit banks for different rows. A GROUP+8 half row pitch would
instead step20/12 banks. Verify the official bank model, concrete same-warp
address mapping, compiler accesses and C33's measured owners before proposing
a padding experiment. No performance attribution to bank conflicts is claimed
from source arithmetic alone, and no second production change is present.

C33 memcheck/synccheck5881 pass all16 tests55.50/50.83s with0 errors. Begin
the unprofiled medium objective only after these gates. For the queued layout
hypothesis, the [CUDA Programming Guide shared-memory model](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html)
describes32 four-byte banks and broadcast only for the same word. Enumerating
the32 lane addresses for each MMA pair load gives2/4 distinct words per bank
for GROUP16/32, but1 for pitches24/40. Current PTX has ld.shared.u32 and the
tensor-specific compiled SASS has scalar LDS at the corresponding pair loads.
This establishes a concrete access mechanism, not its timing fraction or
achieved replay count; a separate padded-layout A/B would test causality.

### C33 rejected and restored; C35 bounded layout follow-up

Public objective92963: prompt263.88[CV.0047], TTFT3880.66[.0048], model
decode54.61[.0017], public51.15[.0019], wall18.91s; counts1024/32/31.
Versus C32 prompt-15.461011%, TTFT+18.292248%, model-1.425993%, public
-1.445087%. Reject both the failed objective and TTFT control. No additional
public controls or stability rerun: objective CV is below5%. Stock power-cap
increment .583349s, thermal0; no settings changed. Save c33.patch/binary/PTX/
SASS and restore the sole shader exactly to C32; worktree then contains only
this report. Counts26 distinct:15 kept,10 rejected,1 interesting/restored,
plus5 accepted non-counting re-evaluations;20 optimization commits plus S01.

Diagnostic44803 completes0 drops: prefill3899.845134ms (gap59.088779), decode
596.774243ms (gap11.940786). Authenticated shape attribution shows all tensor
shapes slower. C32->C33 Q4 total2026.242139->2516.493857ms, Q6
634.392543->703.739521; wide Q4 K3072/N9216 alone964.232695->1242.413943.
Total tensor2660.634682->3220.233378 (+21.0325%). This is consistent with,
but does not prove, the scalar-load collision mechanism documented above.

C35 changes that concrete failed premise: replay C33's direct-MMA draft with
shared A/B row pitches GROUP+8 (24 for Q6,40 for Q4/Q5), adjusting only staging
and fragment addresses. Preserve C33's arithmetic/ownership/two barriers and
C32 host/cache/dispatch. Extra padding costs16*(TOKENS+64)*STAGES bytes;
ordinary/paired/wide static allocation estimates6976/13952/8320 bytes, still
below C32. Only shaders/matmul.cuh (~365 productive lines, existing K) changes;
temporary bit-record hooks stay in existing kernel tests and are removed.
No new file/helper/API/dependency. Main risks: incorrect stage/tail addressing,
staging-store conflicts, extra shared footprint and the compiler load schedule.

Rank C35 medium first: use C32 p.680126791 and conservative net local s1.1,
o.003, predicted6.250698%, ideal212.623869%, saved230.139974ms. This requires
about1.3313x improvement over the rejected C33 tensor owner; bank-collision
removal makes that testable, not guaranteed. C34 long remains deferred at
4.855659% prediction; neither is closed by its estimate. C35 is conservatively
non-counting. Before padding, capture C33's exact matrix outputs with temporary
test-only records; require identical records after padding, unchanged scalar
and raw/cache bounds, full canonical/frozen f16/int8, hybrid isolation and
sanitizers. Performance compares only against accepted C32, medium prompt>=5%,
CV<=5%, all controls regression<=5%, canonical time/rerun rules. C33 remains
rejected even if C35 later passes; no favorable retargeting.

C35 short/long ranking with the same net s1.1,o.003 gives3.017357/5.710762%
and27.364286/800.434535ms savings; medium remains first. Exact-layout baseline
98526 replays the saved C33 shader solely for the declared test, passing the
unchanged matrix reference in47.89s. Then add only physical PITCH=GROUP+8 to
A/B stores, stage bases and MMA half-pair reads; logical groups, input addresses,
row sums, factor/bias order and output mapping stay unchanged. Test records
are temporary and must be removed before canonical/public gates. Fast64141
compiles: ordinary/paired/wide56/60/66 registers,6976/13952/8320 shared bytes,
0 spills. Exact candidate90764 is still running; no C35 performance yet.

C35 exact90764 passes in48.82s. Readback finds162 matrix records in each
baseline/candidate log; compare all records including any first record sharing
the test-harness line prefix (rg -o, not an anchored filter): byte-identical.
Both raw/cache variants and all scalar/range assertions pass. Remove both
temporary record statements and verify tests.rs equals HEAD exactly before
the next gates. C35 remains a non-counting pending candidate.

C35 frozen54925822, frozen13076539 and frozen12998002 pass both KV schemes,
all frozen local IDs identical. Restore canonical prompt before full gates.

C35 canonical95985 passes fmt, CUDA check, CPU tests,11error_matrix, all43
CUDA tests57.82s and f16 parity13.77s; hybrid check82871 passes. Command98696
correctly runs standalone int8 (saved pass record), but its unintended trailing
loop merely assigns a shell variable instead of invoking either hybrid check.
Readback confirms exit0; no extra executable was invoked by that assignment.
Do not count the empty loop as a gate or repeat the successful int8 row: run the missing
two25%-mixed hybrid checks separately with the exact declared arguments.

C35 corrected hybrid pair17227 passes f16/int8 with unchanged canonical IDs.
The loop typo did not change source, public inputs or any reference. Start
serial memcheck/synccheck41998 on the canonical C35 debug binary; no public
candidate record exists yet and production acceptance remains C32.

C35 memcheck/synccheck41998 pass all16 kernel tests55.28/50.88s, both0 errors.
Begin the unchanged unprofiled medium objective against C32 after all declared
gates. No temporary source hooks or altered parity prompt remain.

### C35 rejected and restored; C34 next, C36 queued

C35 objective50332: medium prompt276.35[CV.0001], TTFT3705.50[.0001],
model53.42[.0017], public50.06[.0017], wall18.22s, counts1024/32/31.
Against C32 prompt-11.466009%, TTFT+12.952932%, model-3.574007%, public
-3.545279%. Reject the failed objective/TTFT control; no other public rows or
stability rerun. Power-cap increment .533389s, thermal0. Preserve c35.patch/
binary/PTX and restore the shader exactly to C32; only the report remains dirty.
Distinct counts stay26 (15 kept,10 rejected,1 interesting); six non-counting
re-evaluations now comprise5 kept and1 rejected.20 optimization commits and
separate S01 remain accepted.

C35 diagnostic31599,0 drops: prefill3688.363704ms/gap55.944137,
decode587.973295/gap10.693553. Q4 prefill2324.681513ms, Q6684.167833ms,
versus C33's2516.493857/703.739521 and C32's2026.242139/634.392543.
Padding helps the losing owner but does not recover the accepted baseline.
Do not infer that bank collisions explain the entire regression.

Fresh compiled-code evidence exposes another concrete difference: C32's WMMA
lowers to HMMA.16816.F32 on the recorded capability, whereas C33's explicit
K8 instruction lowers to HMMA.1688.F32. C36 would test K16 direct MMA on
capability>=80 while preserving the accepted75-compatible WMMA fallback.
The official ISA section already consulted requires80 for f16 m16n8k16;
silently raising device admission or emitting it into the75 PTX is forbidden.
A bounded design could embed separate75/80 PTX and select exactly one at load,
with explicit capability/fallback tests. No C36 source exists yet. Tentative
C32-medium p.680126791,s1.1,o.003 gives6.250698%, ideal212.623869%,
230.139974ms, but instruction-path effect and loader/test cost are uncertain.
This is a conservative non-counting C33 re-evaluation if attempted.

C34 is selected first: its previously declared long prediction4.855659% is
close within C36 uncertainty, with much smaller experiment/validation scope.
Source bounds it to eight real rows of the existing padded M16 attention tile,
retaining F16 input/f32 probabilities/PV, dim128, rows>=4 and base>=512. Host
grid becomes ceil(rows/8)*heads; the sole kernel uses up to8 accumulators and
8 softmax coefficient slots, leaving padded/partial rows masked. No new entry,
function, file, dependency, public API or cache layout. Existing
shaders/attention.cuh (~420 productive lines, K) and kernels/attention/prefill.rs
(~60 orchestration lines) suffice. Existing tests gain8/9-query boundary cases
and a future-NaN causal check at base512 before the production edit.

Gate C34 as exact reuse: capture all selected baseline attention outputs with
temporary records, require candidate bit identity, unchanged scalar references,
all canonical/frozen129/130/549 f16/int8 and sanitizer gates. Hybrid mixed uses
at most4 rows, so check its unchanged eligibility and run25%-mixed parity.
Objective long prompt>=5%, CV<=5%, every control regression<=5%, unchanged
C32 public baseline and canonical time/rerun rules. C34 is non-counting; C36
remains deferred, not closed or silently dropped.

C34 boundary baseline4697 passes in2.66s on restored C32. Permanently add
(dim128,context529,rows8) and (128,521,9) to the existing14 cases, with a
future-NaN check for the9-query case starting at512. Original cases, scalar
references, fallback checks and bounds remain; this is support coverage, not
a production optimization. Capture temporary exact records next.

C34 baseline23446 records32 attention outputs (16 cases, both KV schemes).
Candidate11546 passes unchanged scalar/causal/fallback tests in3.70s and all32
records match byte-for-byte. A guarded extraction initially refused a wrong
source line boundary before any edit; selecting the named kernel boundaries
resolved it. Only eight-query ownership/array bounds and host ceil(rows/8)
change. Remove the temporary record line and verify tests.rs matches support
checkpoint c92005e exactly. Fast30216 compiles: tensor attention40 registers,
13408 shared bytes (+48),0 spills; admission/ABI unchanged. C34 remains pending.

C34 frozen54991285, frozen13055277 and frozen12998287 all pass both KV schemes
with exact frozen local IDs. Restore canonical prompt before full gates.

C34 canonical49646 passes fmt, CUDA check, CPU tests,11error_matrix, all43 CUDA
tests58.21s and f16 parity13.89s; hybrid check97333 passes without warnings.
To avoid repeating long hand-written isolation commands, extend the existing
private runner with one serial isolation action: the same standalone-int8 and
25%-mixed f16/int8 commands, exclusive logs, status metadata and required pass
record checks. No production/test/reference behavior changes. Run C34 isolation
through that action before the attention-specific sanitizers and public long row.

C34 isolation50279 passes standalone int8 and25%-mixed f16/int8, all unchanged
canonical IDs. Attention-specific memcheck/synccheck84992 pass all4 selected
tests7.46/3.34s with0 errors. Begin the unprofiled long objective only after
these gates; the immutable C32 baseline remains within the two-hour window.

### C34 kept — eight-query attention reuse

Objective97455 and controls46420, unprofiled immutable fast CUDA/all-GPU,
context4096 f16, warmup1/reps3/max32, same authenticated artifact and prompts:

| Regime | Prompt t/s [CV] | TTFT ms [CV] | Model decode t/s [CV] | Public delta t/s [CV] | Wall s |
|---|---:|---:|---:|---:|---:|
| short | 332.97 [.0018] | 384.42 [.0018] | 57.14 [.0018] | 55.31 [.0020] | 4.87 |
| medium | 311.78 [.0004] | 3284.32 [.0004] | 53.84 [.0019] | 50.46 [.0021] | 16.43 |
| long objective | 275.78 [.0009] | 12995.90 [.0009] | 44.08 [.0014] | 41.33 [.0013] | 55.82 |

Counts128/32/32,1024/32/31,3584/32/31 unchanged. Versus C32 long prompt
+7.705526%, TTFT-7.155033%; long model/public-2.196583/-2.154356%.
Worst control medium model-2.815884%; short public-2.434292%, short TTFT
+2.233924%, medium prompt-.115333%. Every control is within5%, objective
CV<=5%, no rerun, comparison finishes within~51 minutes of C32 baseline.
Stock power-cap increments short/medium/long .954061/0/.235741s; thermal0.
No clock/power/fan/driver or placement change. Natural environment variation
remains a caveat; CV is not an inferential confidence interval.

Keep C34 with only two production files (+12/-12 lines), no added resource
owner or interface: more real rows reuse the existing padded tensor tile and
the host geometry explicitly matches it. All earlier exact/causal/canonical/
frozen/hybrid/sanitizer gates pass.26 distinct attempts remain15 kept,
10 rejected,1 interesting;7 non-counting re-evaluations comprise6 kept and
1 rejected.21 optimization commits plus separate S01 are accepted. C36 remains
deferred pending refreshed profiles88716, running serially for all three rows.

C34 profiles88716 finish with0 dropped events in every row; these are diagnostic
timelines, not replacements for the unprofiled acceptance records. Commit this
accepted checkpoint before designing the capability-specific C36 experiment.
