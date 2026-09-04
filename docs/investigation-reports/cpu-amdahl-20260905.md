# CPU Amdahl campaign 20260905

## Recovery state

- Campaign ID: `cpu-amdahl-20260905`.
- Branch: `perf/cpu-amdahl-20260905`.
- Immutable starting revision: `62384895acfde94cf28fded1294ad860daabf2ff`.
- Retained production checkpoint: `13b2ab47da66bd190deab8138ca729fada08c2cd` (CPU25-01).
- Started: 2026-09-05T00:40:43+02:00.
- State: CPU25-01 kept; refreshed attribution complete; CPU25-02 selected.
- Deadline: none; minimum ten distinct countable attempts, two hours per comparison.
- Current attempts / kept / rejected / not_verified / closed-untried: 1 / 1 / 0 / 0 / 1.

The user explicitly selected CPU. The GPU Amdahl skill is applied to CPU
critical-path attribution and its backend-neutral campaign and correctness
rules. No GPU optimization or machine configuration change is in scope.
The initial worktree was clean on `main`; no active benchmark or build was
found. The prior CPU report and its separate continuation branch both declare
completion, so this is a new campaign with no inherited attempts in its pool.

## Predeclared tuple and gates

Linux x86_64; CPU-only placement; locked Cargo release build; default worker
policy; context 4096; F16 KV; greedy sampling; 32 requested generation tokens;
one warm-up and three measured repetitions. Short, medium and long histories
target 128, 1024 and 3584 tokens, calibrated with the engine tokenizer before
measurement. Exact prompt bytes and actual token counts will accompany rows.
Select the objective from measured removable fixed-work cost; other regimes,
TTFT and non-target throughput remain controls. No current baseline exists yet.

Retention requires at least 5% objective gain, objective CV at most 5%, no
control regression above 5%, and correctness before candidate performance.
Exactly one complete A/B rerun is allowed for objective instability. A 3–5%
signal retains evidence only. Each candidate must predeclare an exact or
risk-specific bounded gate, CPU workspace tests, CPU feature check, applicable
local tests, and authenticated real-model parity. No oracle is substituted.

The cataloged `3b-instruct` Q4_K_M artifact was found locally and authenticated:
2147023008 bytes, SHA-256
`9ed150d4367e68df0ac8e1540f6ddc65b42d0ee26378329d1ecbca60f93fc5f8`.
Its exact local path and private logs remain outside tracked documentation.

Both discovered reference-server binaries are unusable: one cannot load its
shared library, and the other crashes on `--version`. The available source
checkout is revision `9bebfcb4bc8b12a316e96ae03f33671eac1e72fd` and does not
contain the required commit. Preflight is attempting an isolated shallow fetch
of pinned revision `13f2b28b098623391b1aacfd27995e1c8b7de9a9` from the upstream
repository. This is external test support, not a production dependency change.

## Historical evidence, separate from the active pool

Consulted `cpu-amdahl-optimization.md`, reachable CPU history, and the report at
`perf/cpu-amdahl-continuation-20260904` (`c52e6a2`). Inherited production changes
include cache-sized token tiling (`c76c8b9`) and minimum-Rust CPUID compatibility
(`82634e4`), plus earlier batching, quantized decode and GQA reuse. These are not
current-campaign kept results.

Historical negative results include AVX-512 four-row Q4 (decode regression),
physical-core-only affinity (prefill regression), oversized wide-input token
tiles (medium regression), and four-row Q6 (medium decode regression). Earlier
compact repacking, 64-row batching, F16C scale widening, fixed scale blocks and
fast-profile experiments also regressed. Revisit only with a changed,
evidence-backed premise; historical sub-5% predictions alone do not close a
new bounded mechanism with a sufficient ideal ceiling.

## Active candidate pool

The active queue below is populated from the completed fresh attribution.
Fractions refer to fixed-work request time, including prompt and decode; medium
Q4 has the largest conservative percentage opportunity. Predictions are ranges
of engineering estimates, not inferential confidence intervals. All IDs belong
to this campaign; historical CPU-01 through CPU-07 are separate.

| ID / premise | Regime / phase | Conservative p | Local s | Added o | Predicted global gain / ideal ceiling | Saved time | Cost / risk / uncertainty | State |
|---|---|---:|---:|---:|---|---|---|---|
| CPU25-01: pack Q4 activation sub-blocks contiguously | medium / prefill | .55 | 1.15 | .010 | 6.58% / 2.22x | 2.02 s | initial prediction; measured medium prompt +20.681%, all gates pass | kept |
| CPU25-02: interleave two token FMA chains in Q4 | medium / prefill | .42 | 1.12 | .002 | 4.49% / 1.72x | 1.59 s | refreshed profile; small SIMD edit; exact gate; register-pressure risk | ready |
| CPU25-03: decode ephemeral weight rows once across wide token tiles | medium / prefill | .14 | 1.20 | .005 | 1.87% / 1.16x | .68 s | refreshed profile; scratch traffic may outweigh removed decode | deferred |
| CPU25-04: remove alleged duplicated SMT activation footprint | medium / prefill | 0 | n/a | 0 | 0% / 1.00x | 0 | source proves activation data is shared, not duplicated | closed |
| CPU25-05: share attention K/V reads across adjacent query positions | long / prefill | .15 | 1.30 | .005 | 3.05% / 1.18x | 3.78 s | refreshed profile; exact causal-mask and output gates needed | deferred |
| CPU25-06: bounded worker handoff spin before sleeping | short / decode-only interval | .08 | 2.0 | .005 | 3.63% / 1.09x | .17 s | worker timestamps support a bound, not causal savings; borrowed-job and idle-power risk | deferred |
| CPU25-07: direct SIMD rotation in the FP16 RoPE buffer | prefill / pending refreshed regime | unknown | unknown | unknown | not scored | unknown | target conversion/copy/rotation, not rejected coefficient reuse; exact gate required | deferred |
| CPU25-08: smaller dynamically assigned independent output chunks | medium / prefill | .04–.08 | 1.5 | .005 | .84–2.21% / 1.04–1.09x | .31–.80 s | completion spread exists; causal removable fraction uncertain; extra allocations | deferred |

CPU25-01 is selected first: its medium measured Q4 share is 63.2% of the
profiled complete request, versus 40.1% short and 55.8% long. Conservative
fractions .55/.35/.48 discount instrument and environmental uncertainty.
CPU25-02 attacks accumulator-chain scheduling rather than activation layout.
CPU25-03 uses transient per-worker rows, not the historically rejected large
persistent repacking. CPU25-04 addresses the opposite side of the rejected
oversized-tile experiment, but inspection closed its premise before trial:
both siblings read the SAME shared activation allocation and tile, so 756 KiB
does not become 1512 KiB in their shared L2. Only their small accumulator/weight
scratch differs. No duplicated activation allocation exists to remove. CPU25-05
shares across query positions, not across the already grouped four GQA heads.
CPU25-06 cannot be ranked from total kernel time alone and needs evidence first.

Non-RoPE elementwise work, KV writes, embedding, logits and public plumbing
together own only about 2.9% of the medium fixed-work interval: removing all of
them has a ceiling near 3%, so those mechanism families are not filler entries.
RoPE head-coefficient reuse has an unchanged historical negative result; revisit
only if a retained change changes its global premise or a different reuse
lifetime is evidenced. No GPU transfers exist on this selected backend.
CPU25-01 has reached correctness and started A/B; no other queue entry counts.

## Preflight checkpoint

The pinned upstream commit fetched and built successfully with the existing
CMake/GNU toolchain, CPU-only and static libraries. The shallow checkout first
reported a seven-character commit, which the unchanged parity script rejected
because it requires the nine-character prefix. Setting `core.abbrev=9` only in
that isolated checkout and rebuilding its version metadata resolves the
representation mismatch without changing the reference code or oracle gate.

Private recovery directory:
`/home/emanuele/Documenti/hack/cpu-amdahl-20260905.QFkZOu`.
It contains the isolated `oracle` checkout, build logs, `baseline-bench`, and
`screen.sh`. The model is the authenticated catalog filename in
`/home/emanuele/.local/share/Trash/files/models`; it is read in place, never
restored, moved or changed. The runner derives each deterministic prompt with
124/1020/3580 space-separated `benchmark` words and a final period, and checks
the engine tokenizer's exact count before each row. It records stdout, elapsed
resource statistics and start/end timestamps separately. No benchmark has yet
been acquired while the reference build runs.

Passed at the immutable production baseline:

```sh
cargo test --workspace --locked --release --no-default-features --features cpu
cargo check --workspace --locked --no-default-features --features cpu
cargo build --locked --release --no-default-features --features cpu \
  --example bench --example tokenize
```

The suite passed 170 root tests, 164 engine tests, one documentation integration
test, four family-agnostic tests and 12 semantic tests; seven declared external
tests were ignored. Real-model parity is running separately using
`support/testing/parity-check.sh --models-dir <catalog-directory>
--model-id 3b-instruct --backend cpu --kv f16 --reference-server
<recovery-directory>/oracle/build/bin/llama-server`.

Environment: six physical/twelve logical CPU threads, one NUMA node, 1 MiB L2
per core, shared 16 MiB L3, AVX2/FMA/F16C and AVX-512 capabilities, performance
governor, Rust 1.96.1. No settings were changed. Hardware counters are disabled
by `perf_event_paranoid=4`; temporary synchronous operation timers will be
needed after the public screen, with an explicit overhead comparison.

Pinned-reference F16 parity passed before the screen. All sixteen oracle tokens
were in the local top two and all local top-one IDs matched the oracle:
`6319,92683,1772,2007,1049,1055,1617,8379,1032,1049,1057,1918,4847,1603,1051,1050`.
Exact prompt IDs:
`1,17,18,3,6605,4842,3456,1032,1049,1055,10039,1032,1049,1057,1063,4`.
The reference's full source revision is verified by Git and its binary now
reports `13f2b28b0`; no reference implementation or threshold was changed.
The parity command's successful output is preserved in
`baseline-parity-verified-revision.log`. Oracle and build processes completed
before any public benchmark was started.

Screen command (same private recovery directory as above):

```sh
bash <recovery-directory>/screen.sh <recovery-directory>/baseline-bench baseline
```

Current measurement handle at launch: execution session `33595`. Poll the live
handle and inspect its process before treating any observation timeout as a
failure. The runner serializes the three regimes and aborts on a token-count
mismatch or benchmark failure.

### Baseline acquisition in progress

Short completed successfully with the calibrated 128 prompt tokens and 32
completion tokens / public deltas:

```text
prompt_tokens=128 completion_tokens=32 decoded_tokens=32 prompt_tps_mean=35.78 prompt_tps_median=35.78 prompt_tps_stddev=0.48 prompt_tps_cv=0.0135 ttft_ms_mean=3578.33 ttft_ms_median=3577.86 ttft_ms_stddev=48.18 ttft_cv=0.0135 model_decode_tps_mean=12.91 model_decode_tps_median=13.29 model_decode_tps_stddev=0.92 model_decode_tps_cv=0.0716 decode_tps_mean=12.50 decode_tps_median=12.87 decode_tps_stddev=0.89 decode_tps_cv=0.0715 delta_interval_ms_mean=80.28 delta_interval_ms_median=78.14 delta_interval_ms_stddev=6.74 delta_interval_ms_cv=0.0839 decode_begin_tps=12.59 decode_middle_tps=12.65 decode_end_tps=12.26
```

The prompt CV is 1.35%; model/public decode CVs are 7.16%/7.15%, a control
uncertainty to preserve, not a claimed improvement or grounds for a selective
rerun. Medium is still running. The live runner PID is `283031`; observation
session `33595` remains valid. Read-only environment observation runs under
session `34328`, automatically ending when that runner exits, and writes
`baseline-telemetry.log`. No production or test code has been edited.

Next actions: finish and inspect all three screen rows; declare and measure
temporary synchronous CPU operation attribution, including overhead; populate
and rank the active pool from the fresh profile; implement isolated candidates
with correctness before A/B. Do not count reference preparation, history or
screen rows toward the ten attempts.

### Attribution structure, declared before activation

Prepared `profile.patch` in the private recovery directory; it has not yet
changed production sources. Once the complete unprofiled baseline finishes,
apply it on this clean dedicated branch and build a separate diagnostic binary.
Only synchronized CPU backend boundaries are timed; a range ends after worker
join, so operation totals do not sum overlapping workers. Store a timeline in
memory and print it after inference, including matmul input/output dimensions
and batch count. This permits phase and shape attribution without putting
per-event output on the measured path.

Necessary temporary tree and productive-line estimates:

```text
Cargo.toml                                      (+1 diagnostic feature line)
crates/graph_horizon_engine/Cargo.toml           (+1 diagnostic feature line)
crates/graph_horizon_engine/src/backend/cpu/
  mod.rs                                       (~85 lines total; module wiring)
  backend.rs                                   (~260 lines; existing I delegator)
  profile.rs                                   (~50 lines; timeline collection)
```

Invariant: no arithmetic, scheduling, data layout, worker count or public
configuration changes. Main risk: timer calls and code placement perturb the
profiled execution. Measure the overhead with paired unprofiled/profiled
diagnostics using the same three prompts, 32 requested tokens, zero warm-up and
one measured repetition. These diagnostics are attribution only; all retained
performance decisions still use the public canonical 1/3 warm-up/repetition
tuple. Remove all temporary instrumentation before candidate correctness and
performance. No permanent profiler or dependency is introduced.

Static inspection of the baseline Q4 two-row kernel confirms a serial token
loop with four strided activation loads, two accumulator reads and two writes
per 32-value sub-block. Disassembly shows no vector stack spill inside that
token loop. After attribution, assess activation layout and accumulator reuse
as distinct mechanisms; the earlier four-output-row experiments do not test
either premise. Do not infer an Amdahl score from this inspection alone.

Medium baseline completed: 1024 prompt tokens, 32 model completion tokens and
31 public deltas; prompt 34.34 tok/s (CV 0.32%), TTFT 29823.14 ms (CV 0.32%),
model decode 11.11 tok/s (CV 0.87%), public decode 10.41 deltas/s (CV 0.88%).
The full record is preserved in `baseline-medium.out`. Long remains active.

### Complete baseline

All three benchmark children completed with exit status zero. The long row is:

```text
prompt_tokens=3584 completion_tokens=32 decoded_tokens=31 prompt_tps_mean=24.29 prompt_tps_median=24.07 prompt_tps_stddev=0.39 prompt_tps_cv=0.0162 ttft_ms_mean=147603.45 ttft_ms_median=148882.58 ttft_ms_stddev=2366.68 ttft_cv=0.0160 model_decode_tps_mean=8.78 model_decode_tps_median=8.73 model_decode_tps_stddev=0.11 model_decode_tps_cv=0.0120 decode_tps_mean=8.23 decode_tps_median=8.19 decode_tps_stddev=0.10 decode_tps_cv=0.0121 delta_interval_ms_mean=121.50 delta_interval_ms_median=117.51 delta_interval_ms_stddev=21.38 delta_interval_ms_cv=0.1759 decode_begin_tps=8.55 decode_middle_tps=8.48 decode_end_tps=7.71
```

Approximate fixed-work intervals (TTFT plus 32/model-decode-rate) are 6.06 s,
32.70 s and 151.25 s. TTFT shares are 59.1%, 91.2% and 97.6%. This is phase
screening, not isolated prefill timing or a kernel Amdahl attribution. Every
prompt objective CV is below 1.7%; the short decode variability remains noted.

The shell wrapper exited 2 after the completed long child and its saved end
timestamp: preparing optional diagnostic arguments had edited the wrapper while
Bash still held its old read offset. It printed the completed long record twice
and then reported an unexpected `done`. No inference command was repeated or
changed: all three `.time` files report child exit 0, each `.out` contains one
record and each token count matches calibration. `bash -n screen.sh` now passes.
Treat this as a wrapper bookkeeping error, not a transient inference failure;
do not selectively rerun these completed measurements. Never edit a live runner
again. Subsequent diagnostics use a fresh invocation of the verified script.

The production tree still matches the starting revision, and the worktree is
clean before applying the predeclared temporary profiler. Linux kernel is
7.0.0-30-generic and both RUSTFLAGS variables were empty. The long inference
process had about 2.2 GiB RSS and zero process swap; raw thermal/frequency
observations are preserved without changing any machine setting.

### Live attribution checkpoint

The declared profiler is now applied locally, formatted, and built successfully
with `cargo build --locked --release --no-default-features --features
cpu-profile --example bench` (1.70 s). Realized size is 50 total lines for
`profile.rs`, 36 added lines in the existing trait implementation, nine module
wiring lines and two manifest feature lines. The standalone diagnostic binary
is saved as `profile-bench`; `baseline-bench` remains unchanged. Instrumentation
is uncommitted and must be removed before any candidate's correctness gate.

Live diagnostic session: `26423`, sequentially running:

```sh
bash <recovery-directory>/screen.sh <recovery-directory>/baseline-bench diagnostic-base 32 0 1
bash <recovery-directory>/screen.sh <recovery-directory>/profile-bench diagnostic-profile 32 0 1
```

Each invocation visits short, medium and long once. This is the predeclared
overhead comparison, not an objective A/B or a countable candidate. A private
`profile.awk` parser verifies non-overlapping spans, splits prefill/decode at
the first completed logits call, and reports per-operation and matmul-shape
times plus gaps. To consume a finished row:

```sh
awk -f <recovery-directory>/profile.awk <recovery-directory>/diagnostic-profile-medium.time | sort
```

Next action is to poll the same diagnostic handle, inspect all rows and their
overhead, then remove only the temporary profiler before implementing the
highest-ranked distinct candidate. No optimization attempt has been counted.

### Completed attribution and overhead

All six diagnostic children and their wrapper completed successfully. The
private raw timeline and per-shape totals remain reproducible with `profile.awk`.

| Regime | Unprofiled TTFT ms | Profiled TTFT ms | Observed overhead | Q4 prefill ms | Q6 prefill ms | RoPE ms | Attention prefill ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| Short | 3661.80 | 4152.89 | 13.41% | 2822.473 | 676.589 | 456.708 | 40.276 |
| Medium | 35515.50 | 38727.61 | 9.04% | 26430.498 | 5623.487 | 3877.631 | 1578.518 |
| Long | 118153.48 | 134534.29 | 13.86% | 77166.736 | 18365.258 | 13983.164 | 20713.955 |

Profiled first-logits / last-operation timeline boundaries are 4.153/7.048 s,
38.727/41.853 s and 134.533/138.182 s. Prefill gaps between timed operations
are .919/7.884/42.317 ms respectively; decode gaps are 7.631/7.794/9.720 ms.
Q4's 3072x9216 batched shape owns roughly half its prefill cost. The main-loop
assembly performs 128 floating-point operations per token/two-row/sub-block,
with 128 activation bytes, 64 accumulator-read bytes and 64 accumulator-write
bytes: .5 FLOP/byte at that L1 interface, excluding weight decode. These counts
bound a plausible data-movement limiter but do not prove hardware bandwidth
saturation. No unavailable hardware counter is invented.

The no-warm-up diagnostic records differ materially from the canonical screen
even without a code change; CV within one series does not bound between-run
drift. Therefore acquire a fresh adjacent A/B from the preserved applicable
baseline for each candidate. Use canonical warm-up/repetitions and controls,
and never interpret diagnostic throughput as retained performance evidence.
During a 50-second diagnostic activity sample, inference averaged 831.62% CPU,
the editor 22.05% CPU and desktop shell 6.34% CPU. These background clients were
observed, not terminated or reconfigured; their activity is a measurement caveat.

All temporary production instrumentation has been removed with focused patches;
`git diff --quiet 6238489 -- Cargo.toml crates` passed. The private diagnostic
binary, patch and logs are retained. No profiler enters the candidate gates.

### CPU25-01 predeclaration

Objective: medium prompt throughput. Short/long prompt throughput and all TTFT,
model-decode and public-decode metrics remain controls. The intentional change
packs the already widened Q4 activation into `[32-value sub-block][token][32]`
order once per batched operation, then feeds contiguous 32-float token windows
to the existing two-output-row SIMD kernel. Single-row tails and scalar fallback
retain their existing inputs. The same weight decode, FMA order, horizontal sum,
output transpose, FP16 narrowing, worker policy and decode path remain invariant.

Smallest production structure, no new file/function/dependency/public API:

```text
crates/graph_horizon_engine/src/backend/cpu/kernels/matmul/
  q4k.rs       (~360 productive lines, existing K; replace pair dispatcher with packing/selection)
  q4k_simd.rs  (~310 productive lines, existing K; change pair-kernel activation address only)
```

Main risk: the additional activation allocation/copy and changed cache working
set may cost more than contiguous access saves. Peak extra transient storage is
`n * in_dim * 4` bytes; no persistent weight or KV representation changes.
Correctness before performance: exact batched FP16 results against the unchanged
single-row batched SIMD oracle across narrow/wide shapes, odd row counts and
tile tails; existing Q4 tests; CPU release workspace suite; CPU feature check;
pinned real-model F16 parity with baseline top-one IDs also checked. Then build
the candidate binary, run fresh medium A/B with one warm-up and three measured
repetitions, and run both control regimes after a provisional objective pass.
The two-hour comparison budget starts with the fresh objective A acquisition.

### CPU25-01 correctness and live A/B

Implemented only in the two declared kernel files: the existing SIMD pair
function now takes the full packed batch stride, and its activation address
becomes `in0 * batch + token * 32`. An obsolete pair dispatcher was removed;
scalar pairs and odd-row tails retain their original arithmetic/input. No new
production function was needed. Added transient packing increases conceptual
complexity by one explicit local representation; the experiment measures
whether its cache benefit justifies that cost.

The focused exact test passed for input widths 256, 512, 3072 and 9216; batches
2, 5, 21, 32, 33 and 65; and 13 output rows, exercising odd rows and both tile
boundaries. It compares actual batched FP16 outputs bit-for-bit against the
unchanged original single-row SIMD kernel, not against another packed index
formula. All six focused Q4 tests passed.

The CPU release workspace passed 170 root, 165 engine, one documentation
integration, four family-agnostic and 12 semantic tests (seven declared external
tests ignored). The CPU feature check passed. Pinned F16 parity passed and its
complete output is byte-identical to the initial baseline parity record,
including all sixteen local top-one IDs. The release benchmark build passed.

Evidence files: `cpu25-01-exact.log`, `cpu25-01-workspace.log`,
`cpu25-01-check.log`, `cpu25-01-parity.log`, `cpu25-01-build.log`,
`cpu25-01.patch`, and `cpu25-01-bench` in the private recovery directory.
No candidate code is committed or accepted yet.

Live objective comparison session `25513` runs these sequential commands:

```sh
bash <recovery-directory>/screen.sh <recovery-directory>/baseline-bench cpu25-01-a 32 1 3 medium
bash <recovery-directory>/screen.sh <recovery-directory>/cpu25-01-bench cpu25-01-b 32 1 3 medium
```

The runner was extended with an explicit regime argument only after the prior
runner completed; `bash -n` passed. Each command still calibrates exact prompt
IDs and uses canonical generation, warm-up and repetitions. Poll this session
before starting anything else; then classify the objective and, if provisional
keep, acquire paired short and long controls before retention.

### CPU25-01 provisional performance

The fresh medium A/B completed and passed the objective and local controls.
The paired short control then passed. No stability rerun was needed.

| Regime | Metric | A | B | Delta | CV A | CV B |
|---|---|---:|---:|---:|---:|---:|
| Medium objective | Prompt tok/s | 32.88 | 39.68 | +20.681% | .13% | 1.07% |
| Medium | TTFT ms | 31141.03 | 25810.57 | -17.117% | .13% | 1.06% |
| Medium | Model decode tok/s | 11.29 | 11.25 | -.354% | .47% | .73% |
| Medium | Public decode deltas/s | 10.59 | 10.54 | -.472% | .46% | .73% |
| Short control | Prompt tok/s | 32.65 | 43.42 | +32.986% | .12% | 1.11% |
| Short | TTFT ms | 3920.61 | 2948.05 | -24.806% | .12% | 1.11% |
| Short | Model decode tok/s | 12.58 | 12.17 | -3.259% | .40% | .85% |
| Short | Public decode deltas/s | 12.18 | 11.79 | -3.202% | .39% | .84% |

Each medium row has 1024 prompt tokens, 32 completion tokens and 31 deltas;
short has 128/32/32. Full single-line records and time/resource metadata are
saved under `cpu25-01-{a,b}-{medium,short}`. The medium fixed-work request
estimate improves from 33.975 to 28.655 seconds, distinct from prompt-only
throughput. Disassembly confirms contiguous packed activation addressing;
the existing FMA sequence and accumulator memory operations remain.

The comparison started at 2026-09-05T01:14:42+02:00, so its two-hour limit is
03:14:42+02:00. Long paired controls are now running sequentially with the same
commands and label prefixes, changing only the final regime argument to `long`.
They must pass before CPU25-01 becomes `kept` or receives a production commit.
Do not start a second candidate while this comparison is active.

Current live control session: `11862`; first child PID `295024` at launch.
The older objective session `25513` and short-control session `73799` both
completed successfully and must not be restarted. Resume by polling `11862`
and reading `cpu25-01-{a,b}-long.{out,time,start,end}`. If all controls pass,
complete retained-tree checks, commit CPU25-01 with its evidence, refresh the
affected operation profile and rerank the same pool. Otherwise restore only
CPU25-01, record its terminal reason and consider a bounded variant justified
by that reason. There are still at least nine required new attempts remaining.

Additional historical detail consulted: `dd83f5a` documents that head-level
RoPE coefficient reuse removed only 115 ms despite a much larger timed bucket;
F16 conversion/copy and rotation were the remaining cost. This supports a
future investigation of direct SIMD rotation in the FP16 buffer, a different
mechanism from coefficient caching. Do not assign it a ready score or count
an attempt before assessing the current post-decision profile and exact gate.

### CPU25-01 — kept

The long A/B completed at 2026-09-05T01:37:44+02:00, 23 minutes 2 seconds
after the declared comparison start, within the two-hour budget. Both children
and the enclosing session `11862` exited successfully. No rerun was used.

| Long control metric | A | B | Delta | CV A | CV B |
|---|---:|---:|---:|---:|---:|
| Prompt tok/s | 27.03 | 32.36 | +19.719% | .92% | .57% |
| TTFT ms | 132611.67 | 110765.72 | -16.474% | .91% | .57% |
| Model decode tok/s | 8.68 | 8.73 | +.576% | .37% | .58% |
| Public decode deltas/s | 8.13 | 8.19 | +.738% | .36% | .58% |

Both records have 3584 prompt tokens, 32 completion tokens and 31 public
deltas. The short/medium/long paired prompt gains are +32.986% / +20.681% /
+19.719%; the largest decode control regression is the short row's 3.259%,
within the 5% limit. All deciding objective and control CVs are at most 1.11%.
Correctness passed before performance. The terminal decision is `kept`.

Final formatting, warning-denied CPU Clippy and diff checks passed. The last
source edits clarify the exact input-view invariant and correct inherited
batched-versus-decode accumulation comments; no measured operation changed.
Inspection verified `BatchBuffers::scratch` creates exact `rows * width` views
even when backing buffers have larger capacity, and `CpuBuffer` confines reads
to that view. Production productive-line upper bounds, including comments, are
within the declared 360/310 estimates. No new hardware product name appears in
the added tracked lines.

The source complexity cost is one transient, explicitly indexed activation
representation; the obsolete pair-dispatch helper was removed. The gain passes
the predeclared gate, so that local cost is retained. Single-token execution,
canonical weights, KV payloads and public APIs retain their original contracts.

Current retained changes: CPU25-01 only, in the commit containing this section.
Inherited `c76c8b9` and `82634e4` remain outside current-campaign counts.
Pending: fresh attribution on the retained checkpoint, rerank CPU25-02/03/05/06,
replenish distinct candidates from that evidence, and complete at least nine
more countable attempts. This is not the campaign's final quantitative stop.

### Retained-checkpoint profiling plan

CPU25-01 is production commit `13b2ab4`. The runtime and worktree are clean at
that checkpoint. Refresh the same three regimes with unprofiled/profiled
32-token, zero-warm-up, one-repetition diagnostics. Preserve canonical benchmark
decisions separately; diagnostics are never current-campaign attempts.

The original operation spans cannot distinguish compute from worker wake/join
and unequal completion times inside an operation. Add temporary per-slot start
and finish timestamps to answer that narrower question. Worker timestamps use
separate cache-line-aligned atomic slots; the coordinator copies them only
after all slots join. This preserves the existing borrowed-job lifetime and
avoids a logging mutex on worker completion. A common monotonic origin makes
the worker records attributable to the enclosing backend span.

Temporary structure and productive-line estimates:

```text
Cargo.toml / engine Cargo.toml         (+1 diagnostic feature line each)
backend/cpu/mod.rs                    (~85 lines; diagnostic wiring)
backend/cpu/backend.rs                (~260 lines; existing I delegator)
backend/cpu/pool.rs                   (~170 lines; unchanged scheduling plus timestamps)
backend/cpu/profile.rs                (~150 lines; operation/worker timeline collection)
```

No scheduling candidate is implemented by this instrumentation. Main risk:
timestamping and post-join snapshots inflate measured handoff time. Measure
fresh overhead and treat any inferred scheduling ceiling as a bound requiring
an unprofiled reversible experiment, not as achieved savings. Remove all
instrumentation before the next candidate correctness/performance gates.

CPU25-07 is newly deferred from the historical evidence that conversion/copy
and rotation dominated the failed coefficient-reuse experiment. The existing
`CpuBuffer::with_bytes_mut` already provides a scoped exact window for such a
kernel, so it would not require a public buffer API. CPU25-08 is deferred until
worker finish-time spread proves a material tail: its chunks must remain
disjoint independent outputs, preserving reduction order. Neither has been
scored or counted. Rerank every non-terminal entry after the fresh profile.

The temporary worker profiler built successfully. Diagnostic session `52045`
runs `screen.sh cpu25-01-bench retained-base 32 0 1` followed by
`screen.sh cpu25-01-profile-bench retained-profile 32 0 1`, with all three
regimes and paths rooted in the private raw directory above. Poll that session,
do not duplicate it. The tracked instrumentation diff and its untracked module
are preserved privately as `worker-profile.diff` and `worker-profile.rs`.
No candidate timing or correctness decision uses these instrumented builds.

### Retained-checkpoint attribution results

Session `52045` and telemetry `38907` completed successfully. All temporary
profiling files and feature wiring were removed with a focused patch, restoring
the production files exactly to `13b2ab4`; only this report remained modified.
The private profiler diff/module and binary remain reproducible diagnostics.

| Regime | Unprofiled TTFT ms | Profiled TTFT ms | Observed difference | Unprofiled / profiled model decode tok/s |
|---|---:|---:|---:|---:|
| short | 3688.99 | 3643.17 | -1.24% | 6.63 / 6.72 |
| medium | 32140.86 | 31073.58 | -3.32% | 6.76 / 6.43 |
| long | 121441.71 | 123356.25 | +1.58% | 5.22 / 5.32 |

These are zero-warm-up, single-repetition diagnostics, not retention records.
The negative apparent overhead on two rows exposes environmental/run drift;
it does not mean timestamping is free or accelerates inference. Decode is
substantially slower than the earlier canonical A/B in both builds. Temperature
observations reached 92 C, desktop clients remained present, and no machine
policy was changed. Fresh adjacent unprofiled A/B remains mandatory.

| Profiled critical-path quantity, ms | short | medium | long |
|---|---:|---:|---:|
| First logits boundary | 3642.858 | 31073.210 | 123355.475 |
| Last backend operation | 8404.178 | 36050.279 | 129371.061 |
| Prefill Q4 | 2088.886 | 18198.687 | 61356.168 |
| Prefill Q6 | 899.998 | 5939.653 | 19862.679 |
| Prefill RoPE | 472.905 | 3683.026 | 13296.280 |
| Prefill attention | 40.565 | 1992.277 | 24453.685 |
| Decode Q4 | 2964.915 | 3061.148 | 3474.176 |
| Decode Q6 | 570.519 | 816.510 | 577.108 |
| Decode attention | 73.252 | 325.517 | 1246.578 |
| Prefill gaps | .884 | 7.815 | 50.336 |
| Decode gaps | 6.996 | 7.824 | 9.221 |

The Q4 prefill fractions of the full profiled request are .249/.505/.474;
CPU25-02 uses conservative medium p=.42. Its selected local s=1.12 is an
engineering hypothesis, not a counter-derived saturation claim. Assembly still
has two serial four-FMA chains per token, without inner-loop vector stack
spills. Two-token interleaving increases independent chains without arithmetic
reassociation. The medium 3072-by-9216 shape remains about half of Q4 time.

Worker records use the same monotonic origin and fit inside their enclosing
operation spans. `workers.awk` validates slot order, interval containment and
start/end bounds. For each dispatch it computes duration minus the maximum
worker duration (uncovered-time bound) and duration minus mean worker duration
(imbalance-plus-handoff bound). They overlap and MUST NOT be added. Unequal
chunk lengths, preemption, timestamp overhead and wake latency all affect them.

Representative bounds: medium prefill Q4 dispatch 17582.468 ms, uncovered
1522.677 ms, imbalance-plus-handoff 3457.925 ms, post-last-worker join 85.303 ms.
Short decode Q4 dispatch 2938.602 ms, uncovered 373.811 ms, imbalance-plus-handoff
725.773 ms, join 42.863 ms. Long prefill attention dispatch 24195.578 ms,
uncovered 677.420 ms, imbalance-plus-handoff 3928.266 ms. This supports bounded
handoff and load-balancing experiments, not a claim that all spread is removable.

Queue rerank: CPU25-02 first, CPU25-05 next by predicted full-request effect,
then CPU25-03; CPU25-06 has only about .17 s predicted savings in the short
decode-only interval (about 2% of the whole short request), and needs a bounded
spin/idle design before becoming ready. CPU25-08's conservative ceiling is low,
but its upper causal range can clear 5%; do not close it solely on the lower
prediction. CPU25-07 still needs narrower rotation/conversion attribution and
is not assigned the entire RoPE bucket. CPU25-04 remains closed unchanged.
These entries remain distinct from historical rejected experiments.

### CPU25-02 predeclaration

Baseline is retained CPU25-01 (`13b2ab4`). Objective: medium prompt throughput;
TTFT and both decode metrics are controls, followed by short and long controls
only after a provisional medium pass. Tuple, repetitions, stability/rerun rule,
5% retention threshold and 5% maximum control regression remain unchanged.

Intentional variable: execute two token iterations together inside the existing
two-output-row Q4 AVX2 kernel, interleaving four independent accumulator chains.
Keep each accumulator's exact sub-block/chunk order; retain the original odd
token tail, scalar fallback, single-token routing and packed representation.
No allocation, scheduling, weights, KV or public interface changes.

Necessary tree: `backend/cpu/kernels/matmul/q4k_simd.rs` (~350 productive lines,
existing category K). No new production file or function. Main risk is register
pressure introducing spills and reducing throughput, checked in emitted code
and the unprofiled benchmark. Complexity cost is one explicit two-token loop
plus the existing tail, acceptable only if the declared performance gate passes.

Correctness before performance: existing bit-exact packed-batch comparison
against the unchanged single-row kernel (including odd batches and tile tails),
all focused Q4 tests, release CPU workspace tests, CPU feature check, and pinned
real-model F16 parity with output compared to the baseline record. Failed
correctness rejects this candidate. No test tolerance or oracle changes.
