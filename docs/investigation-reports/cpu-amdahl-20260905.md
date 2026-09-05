# CPU Amdahl campaign 20260905

## Recovery state

- Campaign ID: `cpu-amdahl-20260905`.
- Branch: `perf/cpu-amdahl-20260905`.
- Immutable starting revision: `62384895acfde94cf28fded1294ad860daabf2ff`.
- Retained production checkpoint: `245c15f59ff8dacc0392631aec7cea04db2c2ed9` (CPU25-02); preceding checkpoint `13b2ab47da66bd190deab8138ca729fada08c2cd` (CPU25-01).
- Started: 2026-09-05T00:40:43+02:00.
- State: CPU25-05 kept after complete correctness, long objective and both controls; refreshing attribution before the next candidate.
- Deadline: none; minimum ten distinct countable attempts, two hours per comparison.
- Current attempts / kept / rejected / not_verified / closed-untried: 4 / 3 / 1 / 0 / 2.

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
| CPU25-02: interleave two token FMA chains in Q4 | medium / prefill | .42 | 1.12 | .002 | 4.49% / 1.72x | 1.59 s | measured medium +7.290% after sole rerun; controls pass; environmental caveat below | kept |
| CPU25-03: decode ephemeral weight rows once across wide token tiles | medium / prefill | .14 | 1.20 | .005 | 1.87% / 1.16x | .68 s | refreshed profile; scratch traffic may outweigh removed decode | deferred |
| CPU25-04: remove alleged duplicated SMT activation footprint | medium / prefill | 0 | n/a | 0 | 0% / 1.00x | 0 | source proves activation data is shared, not duplicated | closed |
| CPU25-05: share attention K/V reads across adjacent query positions | long / prefill | .15 | 1.30 | .005 | 3.05% / 1.18x | 3.78 s | measured long prompt +16.817%; medium prompt -2.161% within control limit; all gates pass | kept |
| CPU25-06: bounded worker handoff spin before sleeping | short / decode-only interval | .08 | 2.0 | .005 | 3.63% / 1.09x | .17 s | worker timestamps support a bound, not causal savings; borrowed-job and idle-power risk | deferred |
| CPU25-07: direct SIMD rotation in the FP16 RoPE buffer | medium / prefill | <=.03 generous upper bound | unbounded | 0 | ideal <=3.10% / 1.031x | <=1.09 s | direct ablation shows checked coefficients dominate; even generous remainder cannot clear 5% | closed |
| CPU25-08: smaller dynamically assigned independent output chunks | medium / prefill | .04–.08 | 1.5 | .005 | .84–2.21% / 1.04–1.09x | .31–.80 s | completion spread exists; causal removable fraction uncertain; extra allocations | deferred |
| CPU25-09: retain worker locality across dispatches, preserving all logical workers | pending scheduler attribution | unknown | unknown | unknown | not scored | unknown | many observed migrations; affinity portability, allowed-mask and caller-policy risks | deferred |
| CPU25-10: native VNNI integer dot with bounded transient row interleaving | medium / Q4 prefill | unknown | unknown | unknown | not scored | unknown | differs from AVX2 canonical Q8 and expanded persistent repack; numeric quality and packing cost unresolved | deferred |
| CPU25-11: per-call RoPE coefficient vector, preserving head-major traversal | medium / prefill | .10 | 5.0 | .002 | 8.46% / 1.111x | 2.83 s | correct locally, but public prompt -6.598% and TTFT +7.018%; restored | rejected |
| CPU25-12: process-local large-page backing for immutable CPU weights | pending memory attribution | unknown | unknown | unknown | not scored | unknown | >2 GiB anonymous weight storage, zero observed huge pages; page-walk cost and safe platform mechanism unresolved | deferred |

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

CPU25-02 correctness/build session is `87923`. Its chained command writes
`cpu25-02-{exact,workspace,check,parity,build}.log` in the private raw directory,
compares full parity output with `baseline-parity-verified-revision.log`, then
preserves `cpu25-02-bench`. No performance command has started. Resume this
handle and inspect the gate logs before starting the predeclared objective A/B.

Session `87923` completed successfully: six focused tests; 170 root, 165 engine,
one documentation, four family-agnostic and 12 semantic tests; seven declared
external tests ignored. CPU check passed. Pinned F16 parity passed and the full
output is byte-identical to the original baseline record. Assembly has the
intended interleaved four accumulator chains and no vector stack accesses in
the token loop. The unchanged outer decode preparation still accesses stack
constants. The source patch and assembly are preserved as `cpu25-02.patch` and
`cpu25-02-row2.asm`.

Objective A/B session `69742` started at 2026-09-05T01:58:47+02:00; its two-hour
limit is 03:58:47+02:00. It sequentially runs:

```text
screen.sh cpu25-01-bench cpu25-02-a 32 1 3 medium
screen.sh cpu25-02-bench cpu25-02-b 32 1 3 medium
awk -f compare.awk cpu25-02-a-medium.out cpu25-02-b-medium.out
```

All paths are relative to the private raw directory, with the repository as
working directory. Poll the same session. Short/long controls have not started.
CPU25-02 is the second countable attempt, not yet accepted or rejected.

Further read-only worker-gap analysis: in the retained short diagnostic, 9328
decode inter-dispatch gaps total 384.624 ms; 86.4% are at most 20 us, 87.3% at
most 50 us, 89.0% at most 100 us and 99.2% at most 250 us. This supports trying
a small bounded post-job spin for CPU25-06 rather than indefinite busy workers.
The prefill distribution differs (34.9% at most 20 us), so prefill controls are
essential. These are post-join coordinator gaps, not measurements of the full
idle interval of an early-finishing worker; the experiment must account for
that distinction. No scheduling code has been changed.

### CPU25-02 objective instability and single rerun

Session `69742` completed successfully. Initial medium A/B:

| Metric | A | B | Delta | CV A | CV B |
|---|---:|---:|---:|---:|---:|
| Prompt tok/s | 34.22 | 34.89 | +1.958% | 2.01% | 5.04% |
| TTFT ms | 29927.76 | 29396.63 | -1.775% | 2.01% | 5.01% |
| Model decode tok/s | 6.80 | 6.92 | +1.765% | 6.54% | 6.63% |
| Public decode deltas/s | 6.38 | 6.49 | +1.724% | 6.54% | 6.63% |

Work counts match (1024 prompt, 32 completion, 31 public deltas). The objective
CV in B is above 5%, so the mandatory single complete A/B rerun is used, not
a favorable-row retry. Run both medium records again under labels
`cpu25-02-rerun-a` and `cpu25-02-rerun-b`, preserving every parameter and binary.
The original two-hour comparison deadline still applies. A second unstable
objective yields `not_verified: unstable measurement`; there is no third try.

Live rerun session: `49144`. The first A/B telemetry session `12285` has ended.
Resume the rerun handle and its four-metric comparison; do not restart the
completed initial session or start controls before a provisional objective pass.

The single rerun session `49144` completed successfully:

| Metric | A | B | Delta | CV A | CV B |
|---|---:|---:|---:|---:|---:|
| Prompt tok/s | 34.84 | 37.38 | +7.290% | 3.94% | .10% |
| TTFT ms | 29417.22 | 27396.87 | -6.868% | 3.87% | .10% |
| Model decode tok/s | 7.33 | 10.37 | +41.473% | 9.39% | 1.56% |
| Public decode deltas/s | 6.87 | 9.72 | +41.485% | 9.40% | 1.55% |

The objective passes the predeclared mean/CV gate and no medium control
regresses. Counts match. The unusually large decode improvement is NOT credited
to this prefill-only mechanism: environmental drift and code-layout effects are
possible confounders, and baseline decode variability is substantial. Neither
this CV screen nor the earlier first candidate's CVs establish causal or
statistical significance. No additional objective rerun is permitted.

Run the short and long A/B controls sequentially under the original labels
`cpu25-02-{a,b}-{short,long}` with 32/1/3. Acceptance remains conditional on both
controls; the comparison deadline remains 03:58:47+02:00. Do not bundle another
candidate or claim the measured decode gain as a retained kernel speedup.

Short control session `56932` completed successfully:

| Metric | A | B | Delta | CV A | CV B |
|---|---:|---:|---:|---:|---:|
| Prompt tok/s | 39.89 | 41.04 | +2.883% | .78% | .34% |
| TTFT ms | 3209.34 | 3119.21 | -2.808% | .78% | .34% |
| Model decode tok/s | 11.00 | 10.99 | -.091% | .52% | .12% |
| Public decode deltas/s | 10.66 | 10.64 | -.188% | .52% | .11% |

Counts match: 128 prompt, 32 completion, 32 public deltas. The control passes:
no metric regresses by 5%. `compare.awk`'s generic final phrase treats every
input row as an objective and says below 3%; that phrase is inapplicable here
because the predeclared objective is medium, not short. Retain its four metric
calculations, apply the predeclared control rule. Long A/B is the last pending
performance gate. No second objective rerun will be started.

Live long-control session: `67377`, runner PID `308383` at launch. Poll this
same handle and `cpu25-02-{a,b}-long.{out,time,start,end}`. No other inference,
build or test is running alongside the control; read-only source inspection and
low-rate telemetry may continue. Only CPU25-01 remains accepted until this gate
and final retained-tree checks complete.

Read-only discovery during the long baseline: per-thread `/proc` scheduler
counters already show roughly 8,000–25,000 CPU migrations per thread and
124,000–137,000 context switches in the sampled twelve-thread process. Private
snapshot: `cpu25-02-a-long-scheduler.log`. This justifies deferred CPU25-09,
preserving all twelve logical workers while investigating locality across
dispatches. It is distinct from the historical physical-core-only experiment,
which reduced parallelism. Migration counts alone do not provide a removable
time fraction. Before any trial, assess a narrow reversible mechanism that
respects the process's allowed CPU mask, caller affinity, portability and the
no-new-dependency/public-API constraints. Do not alter the live control's
affinity. CPU25-06 may incidentally reduce migrations; propagate its evidence
before treating this as independently ready.

RoPE read-only audit: the emitted retained code calls `Yarn::pair` inside each
head/pair iteration and already uses packed SSE arithmetic for the pair's
rotation. Runtime RoPE source is unchanged from `dd83f5a` except unrelated
backend feature guards. The historical 115 ms whole-request difference is not
a direct phase timer and does not prove the conversion/rotation split. CPU25-07
therefore still needs narrow attribution; neither repeat coefficient reuse nor
assign all RoPE time to wider SIMD on this evidence alone.

Historical integer-path audit consulted `ba451b2` and `282beeb`: per-eight Q8
scales with frequent horizontal reductions lost heavily; canonical 256-value
Q8 with AVX2 maddubs/madd achieved only 1.118x local; persistent native Q4
repacking later increased RSS by 1.41 GiB and regressed. Host capability
inspection confirms AVX-512 VNNI, absent from those actual CPU implementations.
CPU25-10 is therefore deferred for a genuinely different bounded mechanism:
native integer dot with output-row lane interleaving prepared transiently per
worker, rather than the rejected expanded model-lifetime weight representation.
It is NOT permission to repeat the canonical AVX2 arithmetic substitution.
Before any implementation, prove integer range bounds, specify activation
rounding/scale layout, account for quantization and transient packing overhead,
and predeclare a risk-specific real-model quality gate. Do not assign the full
Q4 bucket a ready score before that analysis. Unresolved numerical or authority
constraints are reasons to defer or explicitly close it, not to weaken tests.

Long A has completed within session `67377`: prompt 32.29 tok/s (CV .34%),
TTFT 111005.13 ms (CV .34%), model decode 8.64 tok/s (CV 1.04%), public decode
8.10 deltas/s (CV 1.04%). B is running in the same sequential command. Do not
duplicate A, infer a final comparison from A alone, or start the next candidate.

CPU25-05 read-only feasibility analysis: the existing F16 GQA path handles four
heads per key/value load, with one independent eight-lane dot accumulator per
query. A bounded paired-position variant can handle eight queries (four heads
at each of two adjacent positions) over their common causal prefix, then use
the unchanged four-head primitives for the second position's single extra key
and value. This avoids a zero-weight operation on a masked future value and
preserves every head's dot reduction, max, exp/denominator and value-mix order.
Disjoint temporary output units `(query_pair, kv_head)` avoid reducing the
number of parallel work units to only `n/2`; narrow/reorder those units into the
existing token-major output. Limit the candidate fast path to even batches,
GQA group four and vector-aligned dimensions, keeping the existing fallback.
Before implementation, declare its exact tree/line estimates and compare
byte-for-byte with the unchanged decode attention per position across layer,
history, dimension and view offsets. No attention source has been changed yet.

### CPU25-02 — kept

Long A/B session `67377` completed at 2026-09-05T02:25:24+02:00, 26 minutes
37 seconds after the original comparison start, including the sole permitted
medium rerun. Telemetry session `15399` also completed. No process was restarted
merely because observation yielded, and no additional objective rerun was used.

| Long control metric | A | B | Delta | CV A | CV B |
|---|---:|---:|---:|---:|---:|
| Prompt tok/s | 32.29 | 32.74 | +1.394% | .34% | .09% |
| TTFT ms | 111005.13 | 109454.25 | -1.397% | .34% | .09% |
| Model decode tok/s | 8.64 | 8.74 | +1.157% | 1.04% | .62% |
| Public decode deltas/s | 8.10 | 8.19 | +1.111% | 1.04% | .62% |

Counts match: 3584 prompt, 32 completion and 31 public deltas. As with short,
the generic comparison script's below-3% objective phrase is inapplicable to
this control. Both controls pass. The medium rerun passed the declared 5%
objective threshold and 5% objective-CV screen; the largest control regression
is .188%. Correctness preceded performance. Final formatting, warning-denied
CPU Clippy and diff checks passed. Terminal decision under the predeclared
rules: `kept`.

The retained source change is only the explicit two-token loop plus unchanged
odd tail in `q4k_simd.rs`, within its ~350 productive-line estimate and existing
K category. No new function, file, dependency, allocation, public interface,
weight representation, KV layout or single-token arithmetic is introduced.
The bounded increase in loop complexity is retained because the prescribed
gates passed. Source and emitted code preserve each token's accumulation order.

Interpretation remains limited: measured short/medium/long prompt differences
are +2.883% / +7.290% / +1.394%, not evidence of a uniform 7.29% effect. The
medium rerun's large unmodified-decode gain and the initial +1.958% prompt result
show environmental/code-layout uncertainty. Do not credit its +41% decode
change to token interleaving or call these measurements statistically
significant. Future accepted checkpoints require fresh paired comparisons;
do not compound these percentages into a fabricated final speedup.

Current-campaign accepted changes are CPU25-01 and CPU25-02 only. At least eight
further distinct countable attempts remain. Next: refresh affected attribution
on this retained checkpoint, narrow CPU25-07's phase premise, rerank every
non-terminal pool entry, and continue. This is not a final quantitative stop.

### CPU25-02 retained-checkpoint profiling declaration

The worktree is clean at production `245c15f`. Reapply temporary operation
spans and add separately labeled nested RoPE parts: FP16 read/widen, mathematical
loop, FP16 narrow/write. Unlike the previous worker study, no pool edit is
needed: this refresh concerns the changed Q4 path and RoPE's unresolved internal
attribution. Nested parts are never summed with enclosing backend intervals.

Temporary tree: root/engine `Cargo.toml` (+1 diagnostic feature line each),
`backend/cpu/mod.rs` (~85 productive lines), `backend/cpu/backend.rs` (~260,
existing I), `backend/cpu/profile.rs` (~100, diagnostic collection), and
`backend/cpu/kernels/elementwise/rope.rs` (~65, existing K). No production
algorithm or scheduling changes. Risk: timestamp/log-storage overhead; measure
the same three unprofiled/profiled 32-token, zero-warm-up, one-repetition rows.
Preserve logs and diagnostic source privately, then remove instrumentation
before another candidate's correctness or performance gates.

The diagnostic build completed (session `81707`). Private reproducibility files
are `rope-parts-profile.diff`, `rope-parts-profile.rs`, `cpu25-02-profile-bench`
and `rope-parts.awk`; the parser validates three non-overlapping nested parts
inside each RoPE parent. Collection also snapshots the validated YaRN numeric
parameters once and emits them only after inference. Regular `cpu_span` records
remain compatible with the existing disjoint operation parser.

The new sequential diagnostic command runs:

```text
screen.sh cpu25-02-bench retained2-base 32 0 1
screen.sh cpu25-02-profile-bench retained2-profile 32 0 1
```

All three regimes run in each command. This is profiling, not a third attempt
or another CPU25-02 objective rerun. Do not use these single repetitions for
retention decisions.

Live diagnostic session: `96031`. Resume that same handle and the
`retained2-{base,profile}-{short,medium,long}.{out,time,start,end}` files. CPU25-02
is committed; all current source differences are temporary instrumentation.

Instrumentation defect detected by the existing disjoint-span parser on the
first profiled row: `Timer::part` used struct-update syntax on a type with
`Drop`. Its temporary logged a spurious ordinary span in addition to the
intended nested part, so the raw ordinary-span stream fails the non-overlap
invariant. Production arithmetic is untouched, but this diagnostic version is
not the deciding attribution. Correct `part` by mutating and returning one
timer, without a dropped temporary. Let the already-running diagnostic session
finish; do not modify or replace its binary. Then rebuild and repeat all three
profiled regimes as `retained2-fixed-profile`, preserving the existing matched
unprofiled diagnostic rows and reporting elapsed/environmental drift. This is
an instrumentation repair, not a benchmark stability rerun or a countable
optimization attempt. Preserve the first version's source/logs for the audit.

Session `96031` completed successfully, as did telemetry `97262`. The corrected
profiler adds a focused temporary test that drops one uniquely labeled part
and asserts exactly one matching event with the nested flag set. Only after
that test and the build pass will the fixed three-row profile run. Private
artifacts use `rope-parts-fixed-profile.{diff,rs}` and
`cpu25-02-fixed-profile-bench`; test/build logs use `retained2-fixed-profile-*`.

Live corrected test/build/profile session: `53379`. Resume it, not the completed
`96031`. No candidate is under correctness or A/B evaluation during this repair.

The one-event nested-timer test passed and the corrected short/medium records
are available; the corrected long row is still running. Narrower RoPE diagnostic
predeclaration: add one temporary ignored test in the existing RoPE test module
(zero productive production lines, no new file). Use the captured real YaRN
parameters, heads 32/query and 8/key, positions 0/127/1023/3583, head dimension
128, three rounds of 256 calls. Measure the full uninstrumented kernel,
coefficient-only checked `Yarn::pair` calls protected from elimination, and the
same conversion/rotation/write loop with precomputed coefficients. Check final
FP16 bytes against the full kernel for every round. This is a local causal
ablation for attribution, not a production coefficient-cache candidate, not a
countable attempt and not an end-to-end retention benchmark. Compiler effects
and hot-data conditions limit extrapolation. Run it only after the current
model profile finishes, with the diagnostic feature disabled; remove it with
the rest of the temporary instrumentation.

### CPU25-02 final refreshed attribution and RoPE ablation

Corrected profile session `53379` and uninstrumented ablation session `73882`
completed successfully. Every corrected operation/part parser returned zero.
All temporary source, feature wiring and the ignored diagnostic test were
removed using a focused patch, restoring production exactly to `245c15f`.
Private `rope-cost-split.rs`, `rope-cost-split.log`, and
`rope-diagnostic-final.diff` preserve the diagnostic. No instrumentation remains
in production, and no background benchmark or build is active.

| Regime | Unprofiled TTFT ms | Corrected profile TTFT ms | Apparent difference | Unprofiled / profiled model decode tok/s |
|---|---:|---:|---:|---:|
| short | 3958.44 | 3200.39 | -19.15% | 7.09 / 11.21 |
| medium | 31611.74 | 26445.37 | -16.34% | 6.82 / 10.56 |
| long | 122272.72 | 109646.06 | -10.33% | 5.93 / 8.71 |

These negative apparent overheads expose substantial environmental/code-layout
confounding, compounded by the necessary profiler repair. They do not establish
zero instrumentation cost. Use within-run attribution with conservative bounds,
not cross-run kernel speedup claims. The original faulty profiler records remain
archived but are superseded by all three corrected records. No deciding A/B
record used the faulty profiler or these single repetitions.

| Corrected profile quantity, ms | short | medium | long |
|---|---:|---:|---:|
| First logits boundary | 3200.089 | 26444.915 | 109645.525 |
| Last backend operation | 6054.672 | 29475.499 | 113319.443 |
| Prefill Q4 | 1879.241 | 14658.949 | 53246.152 |
| Prefill Q6 | 671.242 | 5085.870 | 16651.634 |
| Prefill RoPE | 453.419 | 3894.892 | 14201.722 |
| Prefill attention | 41.518 | 1644.928 | 21400.757 |
| RoPE read/widen part | 2.603 | 23.374 | 96.695 |
| RoPE mathematical part | 448.237 | 3843.949 | 13990.745 |
| RoPE narrow/write part | 1.473 | 12.303 | 44.570 |

The RoPE parts are nested inside, never added to, the parent row. Prefill Q4
still owns .497 of the medium full request; RoPE owns .132. Long attention owns
.189 of the long full request. Existing .14/.15 conservative fractions for
CPU25-03/05 remain plausible. Scheduling/locality candidates retain their
earlier measured bounds but await refreshed evidence after any relevant change.

The isolated RoPE ablation used actual parameters: dimension 128, original
context 16384, base frequency 1000000, factor 16, beta 32/1, log multiplier 1,
query temperature .1. All 24 rounds passed final FP16 byte equality after 256
repeated rotations. Observed per-call ranges across positions/rounds:

| Role / heads | Full kernel us | Checked coefficients only us | Precomputed-coefficient conversion/rotation/write us |
|---|---:|---:|---:|
| query / 32 | 80.928–95.003 | 74.229–86.868 | 1.981–2.085 |
| key / 8 | 20.108–23.877 | 18.251–21.887 | .510–.529 |

Do not add independently timed modes or assume their difference is an exact
phase duration: black-box overhead and compiler scheduling differ. Nevertheless,
checked coefficients account for roughly 90% of full-kernel time, while the
same head-major conversion/rotation/write ablation is only about 2–3%. Even a
generous remainder allowance of .03 of the full medium request has an ideal
ceiling of 3.10%, below 5%. CPU25-07 is `closed`, untried and uncounted.

This directly changes the historical premise at `dd83f5a`: its 115 ms public
difference did not establish that conversion/rotation dominated. CPU25-11 is a
bounded new variant, not an unchanged pair-major loop retry. Materialize one
small per-call coefficient vector and keep the original contiguous head-major
rotation traversal. This separates coefficient reuse from the historical
strided head traversal and is supported by the exact local ablation. Its local
s=5 discounts the much larger ideal coefficient reuse ratio and includes an
explicit overhead allowance for vector creation. It now ranks ahead of
CPU25-05 (3.05% predicted full-request gain), CPU25-03 (1.87%) and the uncertain
scheduling/locality/integer entries. CPU25-04 remains closed. No entry is closed
merely because its conservative prediction is below the retention threshold.

### CPU25-11 predeclaration

Baseline: CPU25-02 at `245c15f`. Objective: medium prompt throughput. Keep the
same artifact, runtime/build tuple, 32-token work, 1 warm-up, 3 repetitions,
5% objective threshold/CV limit and 5% control regression limit. TTFT and both
decode metrics are medium controls; short and long controls follow a provisional
objective pass. The existing sole complete objective-rerun rule applies.

Intentional variable: compute checked `Yarn::pair` results once per call into a
temporary vector, then reuse them across heads WITHOUT changing traversal,
element arithmetic, FP16 conversion, output window, post-scale or role semantics.
Do not introduce a persistent cache, backend state, new dependency or public API.
Preserve the zero-head and zero-pair behavior; invalid checked parameters must
not mutate the buffer. The graph's validated dimensions bound coefficient
storage to the existing activation geometry.

Necessary tree: `backend/cpu/kernels/elementwise/rope.rs` (~65 productive lines,
existing K), with attached tests excluded from the estimate. No new file or
production function. Complexity cost is one local coefficient vector; main
risks are allocation overhead and accidentally changing error/rounding behavior.

Correctness before performance: exact FP16 parent-buffer comparison against the
original head-major per-pair calculation, both roles, zero/single/multiple heads,
partial rotation dimensions, padded buffer views and query temperature bucket
boundaries; invalid parameters leave bytes unchanged. Then focused RoPE tests,
release CPU workspace tests, CPU feature check and pinned real-model F16 parity
with complete output compared to the original baseline record. No tolerance,
reference or workload is changed. Only after these pass build the candidate
binary and begin its fresh objective A/B.

CPU25-11 is implemented only in the declared RoPE file. Correctness/build
session `67383` runs focused tests, the CPU release workspace suite, feature
check, pinned F16 parity, byte comparison with the baseline parity log, and
candidate benchmark build in that order. Logs and the final source diff use
`cpu25-11-*` in the private raw directory. No performance measurement has
started. Resume this handle and inspect its gates before beginning A/B.

Correctness/build session `67383` completed successfully: nine focused RoPE
tests, 170 root tests, 167 engine tests, one documentation, four family-agnostic
and 12 semantic tests; seven declared external tests ignored. CPU check passed.
Pinned F16 parity passed and its complete log is byte-identical to the original
baseline. The source diff and benchmark binary were preserved privately.

CPU25-11 is now the third countable attempt. Objective session `35560` started
at 2026-09-05T02:46:06+02:00; comparison deadline 04:46:06+02:00. It runs:

```text
screen.sh cpu25-02-bench cpu25-11-a 32 1 3 medium
screen.sh cpu25-11-bench cpu25-11-b 32 1 3 medium
awk -f compare.awk cpu25-11-a-medium.out cpu25-11-b-medium.out
```

Paths are rooted in the existing private raw directory. Runner PID `320154`
owns this sequential command. Poll the same live session. No short/long control
or stability rerun has started, and CPU25-11 is not yet accepted.

### CPU25-11 — rejected and restored

Objective session `35560` completed at 2026-09-05T02:50:51+02:00 (4 minutes
45 seconds after comparison start). Telemetry `63644` also completed.

| Medium objective/control metric | A | B | Delta | CV A | CV B |
|---|---:|---:|---:|---:|---:|
| Prompt tok/s | 35.16 | 32.84 | -6.598% | 3.22% | 1.15% |
| TTFT ms | 29140.91 | 31185.90 | +7.018% | 3.24% | 1.16% |
| Model decode tok/s | 6.94 | 6.64 | -4.323% | 7.63% | 2.34% |
| Public decode deltas/s | 6.51 | 6.23 | -4.301% | 7.62% | 2.34% |

Work counts match. Both objective CVs pass, so no rerun is permitted or used.
The objective regresses and the TTFT control exceeds its maximum regression.
Terminal result: `rejected`; short/long controls are unnecessary and were not
run. The complete RoPE source and added tests were restored to `245c15f` with
a focused patch; only report changes remain. Private patch, binary, assembly,
correctness and measurement logs retain the negative result reproducibly.

The component ablation does not override the public outcome. Emitted code
confirms coefficient construction moved before the contiguous rotation loop,
but this is insufficient to explain the end-to-end regression. Allocation,
linked code layout and environmental effects are possibilities, not established
causes. Do not start stack-cache, loop-order or coefficient-lifetime variants
without evidence isolating the failing cost; the current coefficient-vector
premise is closed by its measured result. CPU25-07 remains closed on its separate
small rotation-only ceiling. This rejection increases no production complexity.

Queue rerank: CPU25-05 is now bounded and first (3.05% predicted gain with a
17.6% ideal ceiling), followed by CPU25-03. CPU25-06/08 retain timestamp-based
uncertainty; CPU25-09 needs a bounded locality mechanism; CPU25-10 needs native
packing/range/quality analysis. A read-only memory snapshot during B found
2201388 KiB RSS, 2197996 KiB anonymous memory, zero anonymous huge pages, zero
swap, normal scheduling and nice 0. System THP policy was `madvise`, unchanged.
Source copies retained weight tensors into anonymous `Vec<u8>` storage. This
supports deferred CPU25-12, distinct from arithmetic/layout/threading changes:
investigate whether process-local large-page advice can remove a material
translation cost while preserving exact bytes and allocator ownership. No
system THP policy, process mapping or affinity was changed. Page counts alone
do not justify a time fraction or guarantee a gain. Any trial must stay within
owned mapped pages, verify backing actually changes, preserve safe fallback,
avoid new dependencies, and account for load-time and memory overhead.

### CPU25-05 predeclaration

Clean production baseline remains `245c15f`. Objective: LONG prompt throughput,
because long prefill attention owns .189 of its profiled full request. Use the
conservative p=.15, s=1.30, o=.005 prediction already ranked in the pool. This
bounded experiment is eligible despite a 3.05% conservative prediction because
its ideal ceiling is 17.6% and the shared-read fraction is uncertain. Medium and
short are controls, with TTFT and both decode metrics controlled in every row.
Keep the original artifact, CPU policy, release build, context/F16 KV, 32/1/3
work and repetition tuple, 5% objective/CV/control limits and sole rerun rule.

Intentional variable: eight-query F16 attention over two adjacent positions,
four GQA heads each. Share K/V widening on the common prefix; process the second
position's one extra key/value with the unchanged four-head primitives. Preserve
each dot reduction, causal max/softmax and value accumulation order exactly.
Use disjoint `(query_pair, kv_head)` output units, then narrow/reorder them into
the existing token-major view. No retained KV-layout or payload change.

Necessary structure and productive estimates, excluding attached tests:

```text
backend/cpu/kernels/attention/
  mod.rs       (~715 lines; existing K attention family; narrow eligibility route)
  paired.rs    (~130 lines; owns temporary output/scores and paired-unit wiring)
  simd.rs      (~450 lines; existing K; one dense paired-position attention kernel)
```

The new ownership/wiring file stays below 200 lines; SIMD owns no resources and
uses caller-provided scratch. It joins the existing attention folder, not a new
prefix sibling. Keep the fast path limited to x86 AVX2/FMA/F16C, even batches
greater than one, group four, and positive vector-aligned key/value dimensions.
All other cases use the original path. Main risks: output reordering, a future
key entering the earlier query, register pressure, and extra scratch/narrowing
overhead. Added complexity is justified only if end-to-end gates pass.

Correctness before performance: compare full parent-buffer bytes with unchanged
per-position decode attention over zero/nonzero histories, multiple layers,
different key/value dimensions, multiple KV heads, even/odd batches and padded
views; preserve cache bytes. Existing attention/INT8/fallback tests remain
unchanged. Then CPU release workspace tests, CPU check and pinned real-model
F16 parity with complete log comparison against the baseline. No tolerance or
reference changes. Only after passing correctness begin a fresh LONG A/B;
provisional success requires both other regimes before retention.

CPU25-05 initial CPU compile check passed (session `64007`). The declared
implementation is in the existing attention module/SIMD file plus the new
bounded `paired.rs` ownership/wiring file. Correctness/build session `13144`
runs the exact padded-view/cache comparison first, then focused attention,
CPU release workspace tests, CPU check, pinned F16 parity and baseline log
comparison, followed by benchmark build. Private logs use `cpu25-05-*`.
No performance measurement has started; resume that handle before proceeding.

Session `13144` completed successfully. Exact paired/decode parent-buffer and
cache comparisons passed; all 12 focused attention tests passed. CPU workspace
passed 170 root, 166 engine, one documentation, four family-agnostic and 12
semantic tests (seven declared external tests ignored). CPU check passed. Pinned
F16 parity passed and its complete log matches the original baseline exactly.
Productive upper counts are 702/77/377 for mod/paired/SIMD, within 715/130/450.
CPU25-05 is the fourth countable attempt, not yet a performance success.

Warning-denied CPU Clippy also passed (session `61563`). Private reproducibility
artifacts are `cpu25-05.patch`, `cpu25-05-paired.rs`, `cpu25-05-bench` and
`cpu25-05-attention.asm`. The eight-query common-key inner loop has eight FMA
chains without accumulator spills in that loop; cross-phase/softmax stack
traffic remains and is not claimed free.

Long objective A/B session `37266` started at 2026-09-05T03:01:31+02:00;
comparison deadline 05:01:31+02:00. Runner PID `327729` sequentially executes:

```text
screen.sh cpu25-02-bench cpu25-05-a 32 1 3 long
screen.sh cpu25-05-bench cpu25-05-b 32 1 3 long
awk -f compare.awk cpu25-05-a-long.out cpu25-05-b-long.out
```

All paths are in the existing private raw directory. Resume the same handle,
not a duplicate command. No stability rerun or short/medium control has started.
Production candidate code remains unaccepted until the prescribed decisions.

CPU25-12 feasibility references, consulted without changing the running process:
[Linux THP documentation](https://www.kernel.org/doc/html/latest/admin-guide/mm/transhuge.html)
describes translation benefits and warns that frequent `smaps` reads add
overhead. The [madvise manual](https://man7.org/linux/man-pages/man2/madvise.2.html)
documents Linux 6.1's best-effort synchronous `MADV_COLLAPSE`, including possible
direct reclaim/compaction and no guarantee of permanent backing. A future
bounded trial should use only hugepage-aligned interiors of owned, fully
populated weight allocations, keep allocator ownership unchanged, fail safely
to existing backing, and measure setup cost separately. No sysfs, prctl,
advice call, affinity or mapping change has been made. The live long baseline
memory snapshot is saved privately as `cpu25-05-a-long-memory.log`; avoid
repeated expensive mapping scans during the measured repetitions.

CPU25-05 long A completed with prompt 30.03 token/s (CV 1.18%), TTFT
119339.53 ms (CV 1.18%), model decode 7.08 token/s (CV 10.63%) and public
decode 6.64 token/s (CV 10.63%). These are a partial pair, not a decision.
The objective baseline is stable; unrelated decode variability is disclosed.
Candidate B is still running in the same session. Inspection of the saved
assembly also confirms that the common-value loop widens each eight-element
KV window once for eight FMA chains, without vector stack traffic inside that
loop. Per-head softmax still calls scalar `expf` and has cross-call stack
traffic; this is an observation, not an attributed regression or a new candidate.

### CPU25-05 objective result and controls in progress

Session `37266` completed successfully. Long objective results:

| Metric | A, CPU25-02 | B, CPU25-05 | Change | CV A / B |
|---|---:|---:|---:|---:|
| Prompt token/s | 30.03 | 35.08 | +16.817% | 1.18% / 0.30% |
| TTFT ms | 119339.53 | 102177.99 | -14.380% | 1.18% / 0.30% |
| Model decode token/s | 7.08 | 9.50 | +34.181% | 10.63% / 0.83% |
| Public decode token/s | 6.64 | 8.91 | +34.187% | 10.63% / 0.83% |

Objective CV passes; no stability rerun is warranted. Unchanged decode's large
gain and baseline variability are environmental/code-layout confounds, not
claimed causal benefits of paired prefill attention. The observed prompt gain
is provisional pending both controls; this CV screen is not statistical proof.
At 03:17:19+02:00, control session `79212`, runner PID `330928`, began A then B
for short, followed by A then B for medium, using the exact existing screen
commands and binaries with `32 1 3` and the corresponding row selector.
Its deadline remains 05:01:31+02:00, inherited from the complete comparison.
The private `compare.awk` final objective label is not applied to control rows:
evaluate their reported per-metric deltas against the 5% regression limit.

Short control completed: prompt 41.82 -> 42.85 token/s (+2.463%, CV
4.94%/0.66%), TTFT 3066.03 -> 2987.32 ms (-2.567%, CV 5.08%/0.66%),
model decode 11.85 -> 11.81 (-0.338%, CV 1.50%/1.27%), public decode
11.48 -> 11.44 (-0.348%, CV 1.51%/1.27%). Counts agree at 128 prompt,
32 completion and 32 decoded tokens. This control passes; it need not deliver
an objective gain. Its TTFT CV is disclosed, not a new objective or a reason
to selectively rerun this control. Medium A/B remains live in `79212`.

### CPU25-05 final decision: keep

Session `79212` completed successfully. Medium control:

| Metric | A, CPU25-02 | B, CPU25-05 | Change | CV A / B |
|---|---:|---:|---:|---:|
| Prompt token/s | 40.25 | 39.38 | -2.161% | 0.13% / 0.09% |
| TTFT ms | 25439.55 | 26003.61 | +2.217% | 0.13% / 0.09% |
| Model decode token/s | 10.80 | 10.93 | +1.204% | 0.98% / 0.79% |
| Public decode token/s | 10.12 | 10.25 | +1.285% | 0.98% / 0.80% |

Counts agree at 1024/32/31; long counts also agree at 3584/32/31. All
controls pass their no-regression-above-5% rule. The complete comparison ran
from 03:01:31 until approximately 03:22+02:00, well inside two hours, with
no stability rerun. Keep CPU25-05: long objective +16.817%, stable objective,
all correctness gates, and both controls passed. The medium slowdown is an
explicit trade-off, not hidden by the selected long objective. Large unchanged
decode movement on long remains a confound; do not attribute that gain to this
prefill-only fast path or compound the campaign's percentage gains.

Commit the bounded paired-output owner, eight-query SIMD kernel, exact tests,
and this evidence as one accepted optimization. No instrumentation is present.
Next refresh unprofiled/profiled short, medium and long rows with one repetition
and no warm-up, solely for attribution and instrumentation-overhead disclosure.
Then rerank CPU25-03/06/08 and unresolved locality, VNNI and backing mechanisms;
none is closed by this attention result. Four attempts are not completion.
