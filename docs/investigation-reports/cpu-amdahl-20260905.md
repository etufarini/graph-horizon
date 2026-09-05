# CPU Amdahl campaign 20260905

## Retained result at a glance

CPU-only campaign using `.skills/gpu-amdahl-optimizer` with the user's explicit
backend override. Twelve new implementations reached correctness gates: four
kept, seven rejected and one interesting result whose code was restored; zero
not_verified candidates. Two additional premises closed without implementation.
Final cumulative measurement is still running; the tables below distinguish
per-candidate acceptance evidence from that separate initial/final comparison.

| Retained optimization | Commit | Objective before -> after | Objective CV A / B | Important control boundary |
|---|---|---|---|---|
| Q4 contiguous activation blocks | `13b2ab4` | medium32.88->39.68 token/s, +20.681% | .13% / 1.07% | all regimes pass; exact arithmetic and original tails |
| Q4 two-token FMA interleaving | `245c15f` | medium34.84->37.38 token/s, +7.290% | 3.94% / .10% | sole complete objective rerun; unchanged decode gain not credited |
| Paired-position attention K/V reuse | `998b189` | long30.03->35.08 token/s, +16.817% | 1.18% / .30% | medium prompt -2.161%, within5% |
| Q6 contiguous four-stream activation records | `6545e72` | medium33.63->37.36 token/s, +11.091% | 3.11% / 1.86% | short model decode -4.586%, within5% |

No gain percentages are compounded. Rejected code, profiling sources, worker
affinity, memory advice and temporary examples are absent from the accepted
branch. Raw binaries, patches and logs remain recoverable outside the repository.
This is local branch work, not a push, merge or expanded CPU qualification.

## Recovery state

- Campaign ID: `cpu-amdahl-20260905`.
- Branch: `perf/cpu-amdahl-20260905`.
- Immutable starting revision: `62384895acfde94cf28fded1294ad860daabf2ff`.
- Retained production checkpoint: `6545e72343f5daed2fe8160d3c291946771445e4` (CPU25-14); preceding checkpoints `998b189d6f68947e19ef99c757e57406f98f1522` (CPU25-05), `245c15f59ff8dacc0392631aec7cea04db2c2ed9` (CPU25-02) and `13b2ab47da66bd190deab8138ca729fada08c2cd` (CPU25-01).
- Started: 2026-09-05T00:40:43+02:00.
- State: twelve attempts and quantitative discovery stop reached; final attribution/gates complete. Sole full final rerun33770 running: short/medium pass, long pair pending; no further rerun allowed.
- Deadline: none; minimum ten distinct countable attempts, two hours per comparison.
- Current attempts / kept / rejected-code / not_verified / closed-untried: 12 / 4 / 8 / 0 / 2. Rejected-code includes seven canonical rejects and one canonical interesting result.

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
| CPU25-03: decode ephemeral weight rows once across wide token tiles | medium / prefill | .08 | 1.30 | .003 | 1.57% / 1.09x | .45 s | exact but medium prompt -6.093%, TTFT +6.513%; restored | rejected |
| CPU25-04: remove alleged duplicated SMT activation footprint | medium / prefill | 0 | n/a | 0 | 0% / 1.00x | 0 | source proves activation data is shared, not duplicated | closed |
| CPU25-05: share attention K/V reads across adjacent query positions | long / prefill | .15 | 1.30 | .005 | 3.05% / 1.18x | 3.78 s | measured long prompt +16.817%; medium prompt -2.161% within control limit; all gates pass | kept |
| CPU25-06: bounded worker handoff spin before sleeping | short / decode-only interval | .06–.11 | 2.0 | .005 | 2.56–5.26% / 1.06–1.12x | .064–.128 s | correct but model decode -3.539%; restored | rejected |
| CPU25-07: direct SIMD rotation in the FP16 RoPE buffer | medium / prefill | <=.03 generous upper bound | unbounded | 0 | ideal <=3.10% / 1.031x | <=1.09 s | direct ablation shows checked coefficients dominate; even generous remainder cannot clear 5% | closed |
| CPU25-08: smaller dynamically assigned independent output chunks | medium / prefill | .04–.08 | 1.5 | .005 | .84–2.21% / 1.04–1.09x | .24–.63 s | prompt +3.668%: canonical interesting, below keep threshold; restored | rejected (evidence only) |
| CPU25-09: retain worker locality across dispatches, preserving all logical workers | short / decode | .03–.12 uncertain bound | 1.5 | .002 | .81–3.95% / 1.03–1.14x | .02–.10 s on diagnostic interval | route verified active; model decode -10.243%; restored | rejected |
| CPU25-10: native VNNI integer dot with bounded transient row interleaving | medium / Q4 prefill | .40 | 1.5 | .040 | 10.29% / 1.67x | 2.72 s | exact integer/packing tests pass; full-path numerical tolerance fails; restored without performance | rejected |
| CPU25-11: per-call RoPE coefficient vector, preserving head-major traversal | medium / prefill | .10 | 5.0 | .002 | 8.46% / 1.111x | 2.83 s | correct locally, but public prompt -6.598% and TTFT +7.018%; restored | rejected |
| CPU25-12: process-local large-page backing for immutable CPU weights | short / decode | 0–.10 unresolved | 2.0 | .001 plus measured startup | -.10–5.15% / 1.00–1.11x | -.003–.126 s steady diagnostic interval | decode +5.161% on sole rerun, but first-request control +5.820%; restored | rejected |
| CPU25-13: asynchronous kernel promotion hint for owned weights | short / decode | 0–.10 | 2.0 | .001 plus measured startup | -.10–5.15% / 1.00–1.11x | -.003–.126 s | sole rerun model decode -6.538%, limited actual promotion; restored | rejected |
| CPU25-14: pack Q6 four-stream activation chunks across tokens | medium / prefill | .15 | 1.20 | .005 | 2.04% / 1.18x | .58 s | medium prompt +11.091%, short decode -4.586% within control limit; exact outputs, all gates pass | kept |

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

CPU25-05 retained commit is `998b189d6f68947e19ef99c757e57406f98f1522`.
On its verified clean production tree, reapplied the previously declared basic
backend-boundary profiler (same private `profile.patch`, not the faulty nested
RoPE timer). Build session `26884` completed successfully and saved
`cpu25-05-profile-bench`. At 03:23:13+02:00, session `52316` began:

```text
screen.sh cpu25-05-bench retained3-base 32 0 1
screen.sh cpu25-05-profile-bench retained3-profile 32 0 1
```

Each command covers all three regimes serially. These one-repetition rows are
diagnostics without a CV and do not replace accepted canonical A/B evidence.
Resume this session; no duplicate benchmark, correctness run or compile should
overlap it. Temporary profiler files must be restored before the next candidate.

### Post-CPU25-05 refreshed attribution

Session `52316` completed with all six diagnostic children successful. Parsed
disjoint timelines are saved as `retained3-{short,medium,long}-spans.txt`.
Unprofiled/profiled TTFT ms: short 3317.16/3174.45 (-4.30%), medium
26169.58/26288.66 (+0.46%), long 102376.82/102578.02 (+0.20%). Model decode
token/s: 12.31/11.94, 11.52/11.08, 9.63/9.49 respectively. The short apparent
negative overhead remains a drift caveat, not a profiler benefit. No CV exists
for one repetition, and these observations are not retention evidence.

| Disjoint measured ms | Short | Medium | Long |
|---|---:|---:|---:|
| First logits | 3174.136 | 26288.261 | 102577.454 |
| Whole timeline end | 5854.613 | 29176.471 | 105949.707 |
| Prefill Q4 | 1884.404 | 14413.836 | 51105.628 |
| Prefill Q6 | 659.110 | 5607.275 | 17267.597 |
| Prefill RoPE | 456.456 | 3895.808 | 14188.229 |
| Prefill attention | 34.273 | 1265.836 | 16141.260 |
| Decode Q4 | 1723.303 | 1733.124 | 1704.725 |

Medium prefill Q4 still owns .494 of the complete request; long attention now
owns .152, versus .189 in the previous corrected profile. Do not subtract
separate-run operation times as a causal proof. Conservative CPU25-03 p=.14
remains plausible within the larger Q4 interval, but dequant/scratch attribution
is uncertain. CPU25-06's earlier uncovered/imbalance bounds should be refreshed
because decode's absolute interval is now shorter (2680.477 ms on short).

Before selecting it, reuse the already inspected worker-slot timestamp
instrumentation for one short-only diagnostic. Necessary temporary files remain
the declared basic profiler tree plus `pool.rs` (~170 productive lines) and
`profile.rs` (~130, replacing its basic diagnostic contents); no permanent
structure or kernel change. Timestamp slots are cache-line separated, workers
do not allocate/log under a shared event mutex, and joined dispatch snapshots
are nested inside the existing backend ranges. Its overhead must be disclosed
against the just-completed basic-profile short row. This is attribution, not
a countable candidate or an A/B stability rerun.

Worker diagnostic session `29043` completed successfully. Its short TTFT
3013.29 ms is 5.08% below the preceding basic-profile row; model decode 12.50
is 4.69% higher. This again exposes drift/instrumentation perturbation rather
than negative overhead. The disjoint parser and worker nesting parser pass.
Its decode interval is 2561.449 ms. Q4 dispatch owns 1625.103 ms, with
188.402 ms uncovered bound, 262.658 ms imbalance bound, 286.192 ms summed
latest-start lag, and 31.256 ms join tail. These bounds overlap and must not
be added. Smaller decode dispatches also have gaps, but not every scheduler
delay is removable by spinning. Preserve the raw artifact as
`retained3-short-workers.txt`, source as `retained3-workers.diff/.rs`, and
binary as `cpu25-05-workers-bench`. All temporary profiling sources/features
were restored to `998b189`; only this report remained modified afterward.

### CPU25-06 declaration before implementation

Target: short **model decode throughput**, not prompt throughput. Public decode,
TTFT and prompt throughput are controls, along with every metric in medium and
long. Same authenticated tuple and canonical 5% gain / objective CV<=5% /
no-control-regression>5% gates. Baseline binary is `cpu25-05-bench` at
`998b189`. One complete short A/B rerun only if objective CV is unstable.

Intentional variable: workers briefly observe a generation hint after finishing
a dispatch before entering the existing condition-variable wait. Use a maximum
20 microseconds measured by `Instant`; after that, workers block as before.
The hint never grants access to a borrowed job. The mutex-protected generation,
job and remaining count stay authoritative, preserving lost-wakeup protection,
dispatch serialization, exactly-once active slots, and caller wait-before-drop.
No worker-count, affinity, kernel arithmetic or machine-policy change.

Necessary tree: existing `cpu/pool.rs` (~185 productive lines excluding tests,
below 200), with focused tests in that file. No new production file/function,
dependency or public API. Main risks are busy-wait contention/idle cost, a missed
generation, and premature return while a borrowed closure is still in use.
The explicit wall-clock cap bounds idle spinning; normal idle remains blocked.
Test rapid/changing active-slot counts, multiple concurrent callers and borrowed
outputs, blocking gaps, and caller/worker panic draining before return.

Refreshed conservative p=.06 (upper .11), credible s=2, o=.005 predicts
2.56–5.26% decode gain, ideal ceiling 6.38–12.36%, about 64–128 ms saved
per observed short decode interval. These are engineering bounds, not a claim
that all uncovered time is causally removable. This cheap bounded experiment
ranks ahead of CPU25-03's 1.87% and CPU25-08's conservative .84% estimates;
the sufficient upper ceiling permits a trial below the predicted keep threshold.
Unknown locality/VNNI/backing premises remain deferred, not closed.

Before performance: focused pool tests, unchanged exact Q4/attention tests via
the CPU release workspace suite, CPU feature check, warning-denied Clippy,
format/diff checks and pinned real-model F16 parity with full-log identity to
the initial baseline. Reject correctness failures without measuring performance.

CPU25-06 implementation is confined to `pool.rs`, 158 productive upper-count
lines before its test module (within the 185 estimate). The first compile
identified a test-fixture-only mutable-reference capture; explicitly coercing
the leaked fixture to `&'static Pool` fixes ownership without changing production
semantics or a test assertion. Original diagnostic is saved as
`cpu25-06-test-fixture-compile.log`; this is not a separate attempt or rerun.
Both focused pool tests passed. Full correctness/build session `86126` is
running the predeclared sequence, logging `cpu25-06-*`; performance has not
started. Resume this handle before any benchmark.

Session `86126` completed successfully: both focused tests, workspace 170 root
and 168 engine tests plus one documentation, four family and twelve semantic
tests passed; seven external tests remain declared ignored. CPU check, Clippy
with warnings denied, formatting, diff validation and pinned F16 parity pass.
The full parity log is byte-identical to the initial baseline. Candidate binary
and exact patch are privately saved as `cpu25-06-bench` and `cpu25-06.patch`.
This is the fifth countable attempt; it is not yet a performance success.

At 03:33:13+02:00, session `89216`, runner PID `340314`, started:

```text
screen.sh cpu25-05-bench cpu25-06-a 32 1 3 short
screen.sh cpu25-06-bench cpu25-06-b 32 1 3 short
awk -v objective=model_decode_tps -f compare.awk cpu25-06-a-short.out cpu25-06-b-short.out
```

Comparison deadline 05:33:13+02:00. Resume this same session. No stability rerun
or medium/long control has started. Thermal/frequency observations are recorded
privately by `cpu25-06-telemetry.log`; no policy changes or concurrent inference.

### CPU25-06 final decision: reject and restore

Session `89216` completed successfully within approximately two minutes.

| Metric | A, CPU25-05 | B, bounded spin | Change | CV A / B |
|---|---:|---:|---:|---:|
| Prompt token/s | 36.28 | 34.87 | -3.886% | 2.01% / 4.67% |
| TTFT ms | 3528.66 | 3676.68 | +4.195% | 2.01% / 4.80% |
| Model decode token/s, objective | 7.63 | 7.36 | -3.539% | 4.20% / 2.09% |
| Public decode token/s | 7.39 | 7.13 | -3.518% | 4.19% / 2.08% |

Counts match 128/32/32. Objective stability passes, but gain fails the 3% floor,
so reject. No rerun or other-regime controls are warranted. The one-file
candidate and its tests were restored exactly to retained `998b189`; the private
patch/binary/logs preserve reproducibility. No rejected production code remains.
The absolute decode rate shifted lower than recent diagnostics on both sides;
this is disclosed, not a reason to select a favorable repeat or infer causality.

This result falsifies the tested bounded-spin implementation's public benefit,
not every scheduler mechanism. Do not tune its timeout without new evidence
isolating why it failed. CPU25-09 retains its separately observed migration
premise; CPU25-08 targets load balance rather than sleeping; neither is closed.
Rerank: bounded CPU25-03 transient decoded rows next (largest remaining scored
conservative gain), then CPU25-08; unknown VNNI/locality/backing need additional
feasibility evidence. There are five attempts, three kept and two rejected.

### CPU25-03 declaration before implementation

Source inspection narrows the intended reuse opportunity: at the canonical
32-token batch, only input width 9216 crosses the adaptive 21-token tile;
the 3072/4096 input paths do not. Its medium Q4 shape owns 2703.093 ms,
9.26% of the refreshed 29176.471-ms complete request. Replace the initial
.14 estimate with conservative p=.08, s=1.30, o=.003: predicted 1.57% gain,
ideal 8.70%, approximately .45 s saved. The actual removable portion is
uncertain; the sufficient ceiling and bounded experiment justify testing.
This remains ahead of CPU25-08's conservative .84% estimate.

Target medium prompt throughput; unchanged tuple and canonical gates. Intentional
variable: when n exceeds the existing adaptive tile and AVX2/FMA is available,
decode a pair of Q4 rows into per-worker temporary f32 scratch once, then consume
all tokens against those rows. Preserve every fused dequant and ordered FMA
exactly; hold one accumulator per output/token in registers, interleaving two
tokens. This trades repeated quant decode, activation packing and accumulator
load/store traffic for decoded-weight reads. The measured 32-KiB L1 cannot hold
the 72-KiB two-row scratch at width 9216; L2/scratch traffic may erase any gain.
Only this crossing-tile shape changes; single-token, scalar, smaller-batch and
odd-output-row paths remain the existing implementation.

Necessary tree, all within existing matmul files:

```text
cpu/kernels/matmul/q4k.rs       (~385 productive lines; existing one-operation K kernel)
cpu/kernels/matmul/q4k_simd.rs  (~425 productive lines; existing one-operation K SIMD)
```

No new file, dependency, persistent representation or public API. Temporary
scratch is exactly two input rows per active worker; no cache or cross-call
ownership. Main risks are wrong nibble/scale mapping, reassociated reduction,
tail/view writes, packing removal affecting fallback, and excess L2 traffic.
Existing exact batched-vs-original FP16 tests cover widths 256/512/3072/9216,
batches 2/5/21/32/33/65 and odd output tails. Add a direct exact-f32 two-row
test with odd/even token tails and guarded input/output/weight windows. Then
CPU workspace tests/check, warning-denied Clippy, format/diff checks and pinned
F16 parity with full initial-log identity, all before A/B. On provisional
medium success run both other controls; otherwise restore, no selective repeat.

CPU25-03 focused Q4 session `12625` passed, including the direct exact-f32
guarded-window comparison and existing full-path exact FP16 batch/tail tests.
Realized productive upper counts are 384/412 in q4k/q4k_simd, within declared
385/425. Full workspace/check/Clippy/parity/build sequence is now running;
no performance measurement has started. The production candidate remains
isolated from the rejected pool experiment and all profiler instrumentation.

CPU25-03 full gate session `5940` completed successfully: 170 root/167 engine
tests plus the unchanged integration groups passed, CPU check and warning-denied
Clippy passed, and pinned real-model F16 parity's full log is byte-identical to
the initial baseline. Formatting/diff checks pass. Private artifacts
`cpu25-03.patch` and `cpu25-03-bench` preserve this isolated sixth attempt.

At 03:40:20+02:00, session `66505`, runner PID `346915`, began fresh A then B:

```text
screen.sh cpu25-05-bench cpu25-03-a 32 1 3 medium
screen.sh cpu25-03-bench cpu25-03-b 32 1 3 medium
awk -f compare.awk cpu25-03-a-medium.out cpu25-03-b-medium.out
```

Deadline 05:40:20+02:00. Resume the same session. No stability rerun or
short/long control has started. Telemetry is saved in `cpu25-03-telemetry.log`.

### CPU25-03 final decision: reject and restore

Session `66505` completed successfully within approximately five minutes.

| Metric | A, CPU25-05 | B, transient rows | Change | CV A / B |
|---|---:|---:|---:|---:|
| Prompt token/s | 35.45 | 33.29 | -6.093% | 1.67% / 2.72% |
| TTFT ms | 28893.58 | 30775.50 | +6.513% | 1.67% / 2.68% |
| Model decode token/s | 7.60 | 7.80 | +2.632% | 5.25% / 9.36% |
| Public decode token/s | 7.13 | 7.31 | +2.525% | 5.21% / 9.36% |

Counts agree at 1024/32/31. Stable prompt objective fails and TTFT exceeds
the control regression limit. Reject, no stability rerun and no other controls.
The new inner loop's four accumulators remain in registers (private
`cpu25-03-decoded.asm`), but this mechanism does not improve public performance.
Do not claim scratch traffic is the proven cause without causal attribution;
the result closes this exact transient-float implementation. Restore both Q4
files to `998b189`, including tests, preserving private patch/binary/evidence.
CPU25-10 remains distinct: integer VNNI with transient compact row interleaving
is not decoded-f32 scratch. CPU25-08/09 remain independent scheduler premises.

### CPU25-08 declaration before implementation

Target medium prompt throughput, baseline `998b189`/`cpu25-05-bench`, same
tuple and canonical correctness/stability/control gates. This is now the next
bounded scored mechanism; unknown locality/VNNI/backing are still deferred.
Earlier measured finish spreads and refreshed worker bounds support possible
load imbalance, not the sum of all worker durations as removable time. Keep
p=.04–.08, s=1.5, o=.005: predicted .84–2.21%, ideal 4.17–8.70%, saved
.24–.63 s on the refreshed medium request. The upper ceiling warrants a cheap
trial; the lower prediction alone does not close the candidate.

Intentional variable: divide the existing independent-output chunks into four
bounded subchunks and let workers claim them dynamically from one atomic
index. A saturating claim boundary prevents index wrap; every claimed chunk
is unique, so reconstructed mutable slices stay disjoint. The caller and all
worker slots drain through the unchanged serialized pool. Numeric reduction
order inside an output is untouched; no work is split across a reduction.
Unlike CPU25-06, this changes work assignment, not worker waiting or affinity.

Necessary tree: only existing `cpu/parallel.rs` (~125 productive lines excluding
tests, below 200). No new production helper, file, dependency or public API.
Risks: unsafe slice uniqueness, repeated/missing units, extra atomic operations,
more frequent per-closure scratch allocation, locality loss and decode regression.
The per-call claim counter lives until joined dispatch returns. Inspect all
callers: their unit is an independent output row/head/vector; none requires a
particular worker identity or global chunk execution order.

Focused gates: retain existing zero/small/strided output tests and add concurrent
callers over guarded non-Copy outputs, checking each element is visited exactly
once with the correct absolute offset. CPU release workspace includes the
unchanged exact Q4 and paired-attention tests. CPU check, warning-denied Clippy,
format/diff checks and pinned F16 parity with complete initial-log identity all
precede performance. Only provisional objective success triggers both controls.

CPU25-08 is implemented in `parallel.rs` only, 98 productive upper-count lines
(within 125), with a bounded atomic `fetch_update` claim loop and the unchanged
joined worker pool. Three focused parallel tests passed. Full gate session
`48673` is running workspace/check/Clippy/pinned-parity/build sequentially,
with logs `cpu25-08-*`. No benchmark has started. The candidate remains
unaccepted and separate from rejected spin/transient-weight code.

Session `48673` encountered a support-harness hang, not a numeric assertion:
root `support_scripts::parity_script_contract` remained over two minutes in
its final signalled mock-server case. Read-only process inspection found
test PID `349321` waiting for script `352159`, which was waiting for fixture
server `352208`; the real pinned oracle was not involved. Fixture lifecycle
records show nine completed mock servers and the final server started without
a stop. The test's readiness check accepts an earlier `started` line, so a
signal/setup race is plausible but not established as the exact cause.

Preserve `cpu25-08-workspace-hang.log`, fixture lifecycle/process snapshots and
the untouched temporary fixture. Explicitly terminated only this owned gate
runner, Cargo/test and fixture descendants (348618/348879/349321/352159/352208),
so this interrupted run cannot be credited as passing. No source/assertion or
oracle changes. Apply the skill's single identical-command retry for this
actual stuck verification command; do not start performance before success.

Single identical-tuple verification retry `83457` completed successfully:
170 root/167 engine tests plus unchanged integration groups passed, including
the previously hung script contract. All three focused parallel tests pass.
CPU check, warning-denied Clippy, pinned F16 parity/full baseline-log comparison,
benchmark build, formatting and diff checks passed. No source/assertion changed
between verification attempts; output filenames distinguish retry evidence.
Private `cpu25-08.patch` and `cpu25-08-bench` preserve the seventh attempt.

At 03:52:23+02:00, A/B session `42824`, runner PID `361358`, started:

```text
screen.sh cpu25-05-bench cpu25-08-a 32 1 3 medium
screen.sh cpu25-08-bench cpu25-08-b 32 1 3 medium
awk -f compare.awk cpu25-08-a-medium.out cpu25-08-b-medium.out
```

Comparison deadline 05:52:23+02:00. No stability rerun or short/long controls
have started. Telemetry file is `cpu25-08-telemetry.log`. Resume the same
session and keep inference/build activity serialized.

Further feasibility checks while this benchmark runs, without changing it:
[Linux affinity documentation](https://man7.org/linux/man-pages/man2/sched_setaffinity.2.html)
confirms that PID zero changes only the calling thread and that effective masks
remain constrained by permitted CPUs/cpusets. A worker-local trial can therefore
stay inside inherited masks, retain every logical participant, and fall back
unchanged if its bounded mask cannot be read/set. The caller's affinity must
remain untouched. This is distinct from the historical physical-worker reduction.

[Rust's VNNI intrinsic documentation](https://doc.rust-lang.org/core/arch/x86_64/fn._mm512_dpbusd_epi32.html)
marks `_mm512_dpbusd_epi32` stable since 1.89, later than this repository's
1.88 minimum. CPU25-10 cannot silently raise that floor. A bounded guarded
assembly implementation would need explicit register/ISA and numeric tests;
otherwise the intrinsic-only variant is infeasible under the current scope.
No affinity, ISA path, toolchain requirement or dependency was changed here.

### CPU25-08 final decision: interesting evidence only, restore code

Session `42824` completed successfully in approximately five minutes.

| Metric | A, CPU25-05 | B, dynamic chunks | Change | CV A / B |
|---|---:|---:|---:|---:|
| Prompt token/s | 32.99 | 34.20 | +3.668% | 4.46% / 1.96% |
| TTFT ms | 31075.64 | 29950.82 | -3.620% | 4.37% / 1.94% |
| Model decode token/s | 7.47 | 7.78 | +4.150% | 5.36% / 2.47% |
| Public decode token/s | 7.00 | 7.30 | +4.286% | 5.36% / 2.48% |

Counts agree 1024/32/31. Objective CV passes, controls do not regress, but
3.668% is below the 5% keep threshold. Canonical outcome is exactly
`interesting`, not `keep`; preserve evidence and restore `parallel.rs` and
its added test to `998b189`. No stability rerun or other-regime controls.
For the skill's persistent queue, its code disposition is `rejected (evidence
only)`; the count distinguishes this one interesting outcome from the three
canonical rejects. Do not misreport seven attempts as seven measured failures
or treat this result as statistical significance.

The bounded load-balancing mechanism shows a sub-threshold signal, but no
isolated cause supports varying the subdivision factor or adding a scheduler
framework. CPU25-09's independent migration/locality premise remains open;
CPU25-10/12 still require bounded feasibility. Next inspect and declare
worker-local affinity preserving the inherited allowed mask and all logical
workers. No dynamic-chunk or bounded-spin code remains in production.

### CPU25-09 declaration before implementation

Target short model decode throughput; parent is retained `998b189` and binary
`cpu25-05-bench`. Medium/long plus short prompt, TTFT and public decode remain
controls. Same artifact, default twelve participants, context/KV/build, counts,
canonical 5% gain, objective CV<=5%, <=5% control regression and single complete
instability rerun. This is not the historical six-physical-thread experiment.

Measured migrations and worker start/finish bounds justify a bounded causal
trial, not a claim that migration counts themselves are critical-path time.
Use uncertain p=.03–.12, s=1.5, o=.002: .81–3.95% predicted gain, ideal
3.09–13.64%, approximately .02–.10 s on the refreshed short decode interval.
The conservative prediction is low; the upper ceiling and cheap diagnosis keep
the trial viable. It precedes more complex unresolved VNNI/backing mechanisms.

Intentional variable: on Linux x86_64 only, each daemon CPU worker reads its
inherited allowed mask once and restricts itself to its assigned allowed logical
CPU. Activate only if the pool's participant count equals that mask's population;
otherwise leave affinity unchanged. Worker IDs 1..count map to allowed CPUs
at those indices, leaving the first CPU unassigned by workers. The caller
remains unbound within its original mask; do not claim it is forced onto the
remaining CPU. All logical workers still exist, idle behavior and pool handoff
remain unchanged. The kernel enforces any additional cpuset constraints.

Necessary tree and productive estimates:

```text
cpu/mod.rs       (~85 lines; +2 cfg/module wiring lines)
cpu/pool.rs      (~150 lines; +2 cfg/call lines before worker loop)
cpu/locality.rs  (~55 lines; one worker-local affinity/FFI boundary, plus tests)
```

The new distinct helper belongs to the existing CPU domain, not a lone-module
folder. It owns only a bounded 128-byte native Linux x86_64 mask. Read/set errors,
larger kernel masks and mismatched participant counts fall back unchanged; no
dynamic CPU-set framework or new dependency. Standard C ABI calls use PID zero,
never another process/thread ID. No global scheduling, priority, topology,
governor, cpuset or caller-affinity change. Risks: pinning competes with desktop
work, the unbound caller may contend with a worker, and setup/portability errors.

Correctness before performance: in a separate test thread, exercise mismatched
count/caller-ID no-ops and a sparse two-CPU inherited mask, verifying the selected
CPU stays inside it and parent affinity is unchanged. Full CPU workspace tests,
CPU check, Clippy, format/diff and exact pinned F16 parity follow. Verify actual
benchmark worker masks and unchanged caller mask read-only before crediting any
gain; if affinity never activates, do not claim a locality improvement.

CPU25-09 is applied in the declared three files; the new boundary has 35
productive upper-count lines (within 55). Its focused sparse-mask/caller
preservation test passed. Gate session `58631` is running the remaining full
CPU suite/check/Clippy/pinned-parity/build chain with logs `cpu25-09-*`.
No performance benchmark or external-thread affinity change has started.

CPU25-09 gate session `58631` completed successfully: focused locality test,
170 root/167 engine tests and all unchanged integration groups passed, as did
CPU check, warning-denied Clippy, pinned real-model F16 parity with full initial
log identity, benchmark build, formatting and diff checks. Reproducibility
requires both `cpu25-09.patch` and the separately saved new
`cpu25-09-locality.rs`; binary is `cpu25-09-bench`. This is attempt eight.

At 04:02:36+02:00, session `27105`, runner PID `368115`, began:

```text
screen.sh cpu25-05-bench cpu25-09-a 32 1 3 short
screen.sh cpu25-09-bench cpu25-09-b 32 1 3 short
awk -v objective=model_decode_tps -f compare.awk cpu25-09-a-short.out cpu25-09-b-short.out
```

Comparison deadline 06:02:36+02:00. No stability rerun or other controls yet.
Telemetry is `cpu25-09-telemetry.log`; read-only masks are captured separately
to verify the intended variable. Resume this session; do not duplicate inference.

The first A mask snapshot arrived after A had exited; its empty output is not
treated as evidence and no benchmark was repeated. B's executable identity was
verified at PID `368261`. `cpu25-09-b-short-masks.log` confirms eleven daemon
workers restricted individually to logical CPUs 1..11, while the benchmark
caller retains 0..11 and every thread retains NUMA allowance 0. Thus the tested
worker-local route is active, with all twelve participants and no caller pin.

### CPU25-09 final decision: reject and restore

Session `27105` completed successfully in approximately two minutes.

| Metric | A, CPU25-05 | B, worker locality | Change | CV A / B |
|---|---:|---:|---:|---:|
| Prompt token/s | 33.89 | 32.69 | -3.541% | 2.64% / 1.95% |
| TTFT ms | 3779.15 | 3915.96 | +3.620% | 2.61% / 1.93% |
| Model decode token/s, objective | 7.81 | 7.01 | -10.243% | 1.30% / 2.15% |
| Public decode token/s | 7.57 | 6.79 | -10.304% | 1.31% / 2.15% |

Counts match 128/32/32. Stable objective regresses and public decode exceeds
the control limit. Reject without rerun or other-regime controls. The observed
migrations did not make this worker-only affinity policy profitable; caller
contention and environmental interference are possible, not proven causes.
Do not automatically extend the experiment to caller affinity, topology-specific
ordering or a tuning sweep without evidence isolating a new premise.

Restore module/pool wiring to `998b189` and remove only the new `locality.rs`
candidate file, whose exact private copy and patch remain recoverable. All
candidate processes have ended, so their per-thread restrictions no longer
exist. The campaign never changed global CPU policies or other processes.
Eight attempts are complete: three kept, four canonical rejects, one interesting
evidence-only restoration; two untried closures. Next bound CPU25-12, whose
small process-local backing mechanism is cheaper than the unresolved VNNI path.

### CPU25-12 declaration before implementation

Target short model decode throughput, baseline `998b189`/`cpu25-05-bench`,
same authenticated model/context/F16/default-workers/release tuple. All usual
controls and canonical objective/CV/rerun gates apply. Additional controls:
complete benchmark process elapsed time, and mean load-plus-first-generation-
cleanup elapsed time over three fresh processes must not regress by >5%.
Record initialization separately, but do not hide its cost behind warm-up.
Only provisional canonical success requires full additional startup/control
qualification; rejection still preserves process elapsed/RSS from time(1).

Evidence: >2 GiB anonymous weight storage with zero observed huge backing,
short decode dominated by quantized weight kernels, and local Linux PMD size
2097152 with existing madvise THP policy. Hardware counters cannot isolate page
walks. Therefore p remains an honest unresolved range 0–.10, s=2, o=.001
steady-state plus explicitly measured startup: predicted -.10–5.15%, ideal
1.00–1.111x, approximately -.003–.126 s on the prior short diagnostic interval.
The upper bound is an engineering uncertainty, not an attributed 10% cost.
A cheap, bounded causal backing trial is justified; a zero lower prediction
does not itself close this material layout mechanism.

Intentional variable: on Linux x86_64, call native `madvise(MADV_COLLAPSE=25)`
once for the complete 2-MiB-aligned interior of each owned quantized weight
Vec, before sharing it through CpuBuffer. Original allocator ownership, bytes,
windowing and drop stay unchanged. Exclude partial edge pages and all scratch,
KV, F16/F32 buffers. Empty/small ranges do nothing; kernel failure falls back
to the original backing. No global THP, prctl, policy, affinity, dependency,
public API or persistent expanded representation. Local headers confirm the
advice value; the previously cited Linux manual describes its best-effort,
synchronous behavior and possible reclaim/compaction, which must be charged.

Necessary tree, declared before any production change:

```text
cpu/mod.rs        (~85 lines; Linux x86_64 module wiring)
cpu/buffer.rs     (~170 lines; owned quantized-weight setup call)
cpu/memory.rs     (~45 lines; one bounded native advice boundary, plus tests)
examples/cpu_load.rs (~55 lines; TEMPORARY load/first-request/cleanup timing)
```

The temporary example uses the existing public throughput harness with the
calibrated short prompt, context4096/F16/32 tokens, zero warmup and one first
request per new Engine. Build/save its A binary before modifying production,
then save B after correctness; keep it out of final production. Three processes
per side supply means and CV disclosure for load/generation/cleanup/total.

Invariants: advice pointer/length stays entirely inside initialized owned bytes,
aligned endpoints exclude allocator neighbors, advice never frees/discards data,
and advisory failure cannot corrupt or change logical memory. Main risks are
unsafe range arithmetic, excess synchronous load cost, memory compaction and
unproven effective promotion. Focused tests exercise empty/short/large buffers,
unaligned guarded views, exact byte preservation and unchanged Vec ownership.
CPU workspace/check/Clippy and exact pinned F16 parity precede performance.
Read actual B AnonHugePages once to verify promotion; a non-activated route
cannot be credited as a successful large-page optimization.

Temporary startup probe baseline build `23861` passed without a production
change; saved as `cpu25-12-a-load`. Private `startup.sh` runs three fresh
processes with the same calibrated 128-token prompt, 32 completion tokens and
32 decoded tokens, checking those exact counts and saving per-process time(1),
load/generation/cleanup/total measurements. Capture this baseline before
activating advice; the temporary example is not accepted production material.

Baseline startup session `28150` completed, exact counts pass in all processes:
load ms 1101.503/1096.467/1105.496; first-generation ms
7854.225/7903.586/7783.293; cleanup ms 183.071/181.880/187.193; total ms
9138.800/9181.934/9075.982. No production advice had been applied. Before
implementation, refine buffer's estimate to 170 to allow platform-local mutable
ownership without an unused-mut suppression on other targets; no responsibility
or scope expansion. Production still matches `998b189`; only the declared
temporary example and report differ. Proceed with the isolated candidate.

CPU25-12 correctness/build session `39637` completed successfully. Memory tests
preserve every byte, guarded slice and Vec pointer/capacity. Workspace passed
170 root/167 engine plus unchanged integration groups; CPU check, warning-denied
Clippy, exact pinned F16 parity/full initial-log identity, build and format/diff
checks passed. Realized productive counts are memory21/buffer160, within 45/170.
Private artifacts are `cpu25-12.patch`, new `cpu25-12-memory.rs`, candidate
`cpu25-12-bench`, both startup binaries `cpu25-12-{a,b}-load`, and temporary
probe source `cpu25-12-load.rs`. The temporary example was removed after saving
both binaries; no timing instrumentation remains in the repository candidate.

At 04:20:08+02:00, session `88917`, runner PID `376313`, began:

```text
screen.sh cpu25-05-bench cpu25-12-a 32 1 3 short
screen.sh cpu25-12-bench cpu25-12-b 32 1 3 short
awk -v objective=model_decode_tps -f compare.awk cpu25-12-a-short.out cpu25-12-b-short.out
```

Deadline 06:20:08+02:00; no stability rerun or further controls yet. A one-shot
read-only observer verifies B's executable and waits for its CPU worker threads
(after loading), then reads smaps_rollup once into `cpu25-12-b-short-backing.log`
and worker masks separately. No repeated mapping scans or policy changes.
Resume the same A/B handle; telemetry file is `cpu25-12-telemetry.log`.

Initial session `88917` completed: short prompt 33.57 -> 33.70 (+.387%, CV
2.18%/2.76%), TTFT 3814.44 -> 3800.60 (-.363%, CV 2.16%/2.81%), model
decode 7.83 -> 7.99 (+2.043%, CV .91%/5.19%), public 7.59 -> 7.74
(+1.976%, CV .92%/5.19%). Counts match 128/32/32. B's objective CV exceeds
5%, requiring the single complete A/B rerun, not a favorable-row selection.
Initial B time(1) reports peak RSS 4267780 KiB and zero swaps; this memory
cost must be disclosed and current backing verified. The observer/telemetry
launch occurred after the short pair had already ended, so no initial backing
snapshot exists and the empty telemetry is not evidence. The complete rerun
launches both observers in the same orchestration call as inference to avoid
that timing miss. No code, arguments, counts or gate changed.

Complete rerun session `4377`, runner PID `376860`, began at
04:22:58+02:00, with labels `cpu25-12-rerun-a/b`, the same baseline/candidate
binaries and `32 1 3 short` arguments. One-shot backing observer is `38095`,
telemetry `43885`; deadline remains 06:20:08+02:00. Resume these handles.
No further stability repeat is allowed after this pair. The candidate remains
unaccepted and no additional-regime or B startup control has run.

### CPU25-12 sole rerun: provisional objective pass

Session `4377` and both observers completed successfully.

| Metric | A, CPU25-05 | B, collapse | Change | CV A / B |
|---|---:|---:|---:|---:|
| Prompt token/s | 33.23 | 33.53 | +.903% | 2.20% / 3.30% |
| TTFT ms | 3853.75 | 3820.53 | -.862% | 2.23% / 3.35% |
| Model decode token/s, objective | 7.75 | 8.15 | +5.161% | .36% / 3.74% |
| Public decode token/s | 7.51 | 7.90 | +5.193% | .36% / 3.74% |

Counts agree 128/32/32; objective/CV and same-row controls pass provisionally.
Whole-process elapsed 33.08 -> 32.40 s (-2.06%) passes its additional control.
Peak RSS 4268004 -> 4267752 KiB is essentially unchanged. Initial-pair A/B
also peaked at 4268140/4267780 KiB respectively:
the high peak is present in A, not evidence of added collapse memory. B's
post-load snapshot reports RSS 2153068 KiB, anonymous 2149780 KiB,
AnonHugePages 1673216 KiB and swap0. This verifies substantial actual promotion
while resident steady-state memory is much below the measured startup peak.

Next acquire the additional first-request control as a fresh complete A/B:
three A probe processes then three B processes, identical calibrated inputs.
The earlier pre-edit startup row remains baseline screening evidence; neither
B startup nor its result has been observed yet. Given documented environmental
drift, use this adjacent pair for the control decision, not a selectively chosen
historical A. This is the first full startup A/B, not another objective rerun.
If it passes, run both medium and long canonical controls before retention.
All remaining work shares the original 06:20:08+02:00 comparison deadline.

### CPU25-12 final decision: reject on startup control, restore

Fresh startup A/B session `66108` (runner377815, telemetry35555), started
04:28:33+02:00, completed successfully. Both sides use three fresh processes
and all six fixed-work count checks pass at 128/32/32. Per-process raw ms:

| Side | Load | First generation | Cleanup | Total |
|---|---|---|---|---|
| A | 1102.209,1096.502,1099.895 | 7568.841,7524.132,7805.619 | 181.144,185.756,186.108 | 8852.194,8806.392,9091.623 |
| B | 1506.316,1510.663,1528.011 | 8026.387,7703.251,7856.463 | 57.944,59.107,58.942 | 9590.648,9273.021,9443.417 |

Mean total increases 8916.736 -> 9435.695 ms (+5.820%), exceeding the
predeclared 5% startup/first-request control limit. This overrides the
provisional steady decode gain. Mean load increases approximately 1100 ->
1515 ms; faster cleanup does not compensate, and first generation is also
slower. No medium/long control or additional objective rerun is warranted after
this terminal control failure. The original comparison stayed inside two hours.

Restore buffer/module wiring to `998b189`, remove only the new `memory.rs`
candidate, and retain its private source/binaries and the already saved/removed
temporary startup probe. No memory advice, affinity restriction, new dependency
or diagnostic source remains in production. The tested eager collapse policy
is rejected; do not tune page sizes or move its cost off the critical path
without a distinct measured premise. A background/lazy promotion worker would
introduce ownership and interference that this bounded trial did not validate.
Nine attempts are complete. CPU25-10 still requires a bounded native integer
design and risk-specific quality gate; the campaign is not yet complete.

Related-premise correction: CPU25-12's startup failure does support one bounded
variant for later evaluation. Kernel `MADV_HUGEPAGE` advice can remove synchronous
collapse from initialization without introducing an application background
worker. Its delayed activation and interference remain unresolved, so add
CPU25-13 as deferred rather than silently closing every lazy-promotion variant.
It is a different placement of work, not a page-size sweep. CPU25-10 now has
the larger credible effect and is selected first.

### CPU25-10 feasibility and declaration before implementation

The private native instruction probe passes exact signed/unsigned integer dot
results on both the normal compiler and isolated Rust1.88.0. The latter was
installed only under the private raw directory's `msrv-rustup`; the ordinary
toolchain remains unchanged. The direct intrinsic's 1.89 floor is avoided by
explicit guarded assembly, whose register clobbers and memory bounds follow
the [Rust assembly reference](https://doc.rust-lang.org/reference/inline-assembly.html).
This probe is feasibility, not a campaign attempt or performance result.

Target medium prompt throughput, parent `998b189`/`cpu25-05-bench`, same tuple
and canonical objective/CV/control/rerun rules. Q4 owns .494 of the refreshed
medium request. Conservative p=.40, s=1.5, o=.040 (activation quantization,
transient packing, conversions and cleanup included) predict +10.29%, ideal
1.67x, about 2.72 s saved. Instruction/working-set reasoning, not unavailable
hardware counters, supports the uncertain local estimate: native four-byte
integer dots and four-row reuse replace float MAC/dequant work, while packed
activations shrink fourfold. Per-call packing is not free and may erase the gain.

Intentional variable: a Q4-only prefill path for n>=8 and output counts divisible
by four, guarded by runtime AVX2/AVX-512F/VNNI availability. Quantize activations
in 256-value blocks, with signed maximum, nearest-even rounding and 32-value
sums. This mirrors the pinned reference's Q8_K scale/rounding arithmetic; the
temporary layout is internal, not a new file format. Four Q4 output rows are
unpacked/interleaved only in per-worker scratch, with exact six-bit scales/mins.
VNNI accumulates scale-weighted integer dots per 256-value block; min corrections
and original FP16 block factors reconstruct f32 output. No persistent weight
repack; single-token decode, smaller batches, output tails and nonmatching ISAs
retain the existing implementation. Nonfinite activations trigger the original
float path before any output write, not silent saturation/sanitization.

Integer bound: even permitting -128 in direct instruction tests, each lane's
absolute sum is at most 64*15*128*63=7741440; each output block sum at most
30965760, well inside i32. Min correction is likewise bounded. Assembly reads
exact typed 256-byte activation, 1024-byte interleaved-weight and eight-four-scale
windows, writes all sixteen i32 lanes, declares every vector/general-register
clobber, makes no calls and never accesses outside those arrays. Rust1.88
compatibility must remain; no target-cpu/native or global compiler flag change.

Necessary tree and productive estimates, excluding tests:

```text
cpu/kernels/matmul/mod.rs             (+2 module wiring lines, existing K)
cpu/kernels/matmul/q4k.rs             (~365 lines, existing K; one eligibility route)
cpu/kernels/matmul/integer/mod.rs     (~155 lines; per-call/per-worker ownership and writeback)
cpu/kernels/matmul/integer/activation.rs (~100 lines; bounded temporary Q8 preparation)
cpu/kernels/matmul/integer/block.rs   (~150 lines; one Q4 block pack/dot numeric kernel, K)
examples/cpu_quality.rs              (~70 lines; TEMPORARY extra public-output oracle)
```

The integer domain warrants a folder because ownership, activation preparation
and the numeric block kernel are distinct responsibilities. No external library
or public API. This complexity is retained only for a fully qualified gain.

Risk-specific gates before performance: exact packed nibble/scale/min mapping
and integer dots against independent scalar calculations, zero/tie/outlier and
nonfinite activation cases, and new full-path synthetic comparison within the
unchanged quantized per-output tolerance `8e-2 * max(abs(reference),1e-3)`.
Existing tests/oracles stay unchanged. Because activation quantization is not
bit-exact, require pinned real-model top-two parity AND byte-identical public
text/prompt/completion counts against a pre-edit baseline on four fixed greedy
64-token requests: Italian arithmetic, JSON arithmetic, Python code, and the
calibrated 1024-token benchmark history. Capture that baseline first; any
difference rejects the candidate without changing the gate. This bounded extra
check is not a universal model-quality claim. Then workspace/check/Clippy,
minimum-Rust CPU check, format/diff and the prescribed performance controls.

Quality baseline session `23597` completed successfully on unchanged `998b189`
production, producing four records in `cpu25-10-a-quality.log`; saved executable
`cpu25-10-a-quality`. Normal compiler remains rustc1.96.1. Isolated1.88 VNNI
feasibility probe passed without target-feature flags or a production dependency
change. Proceed with the declared candidate; its stricter quality baseline is
now immutable. No candidate performance has been measured.

### CPU25-10 terminal correctness failure

Focused release gate session `97540` compiled successfully and ran four tests.
Quantization edge cases, independent exact canonical packing (including nonzero
row/byte origins), and signed native integer dots all passed. The complete
numeric path failed its unchanged predeclared tolerance: in=256, out=12, n=32,
row=7, token=24 produced 1.6611328 against float reference 2.0710335, absolute
error .4099007 versus allowed .16568268 (about 19.8% relative error).
This is a genuine correctness rejection and the tenth countable attempt.
No model/performance result, MSRV production pass or extra-quality B is claimed.
The earlier integration compilation error is separately preserved, not counted.

Saved `cpu25-10.patch`, all three integer sources, the temporary quality source,
immutable four-output A and focused gate log in the private recovery directory.
Restored both existing matmul files to `998b189` and removed only those four new
candidate/temporary files; their private copies remain recoverable. No tolerance,
fixture, oracle or precision requirement was weakened. Exact integer dots do
not remove activation quantization error. A finer-block quantization sweep would
add work/layout complexity without fresh evidence overcoming the historical
fine-block integer regression; no such parameter-only successor is queued.

### CPU25-13 declaration before implementation

Replace synchronous collapse by kernel `MADV_HUGEPAGE` advice on the same fully
owned, initialized, aligned 2 MiB interior of immutable quantized weights before
publication. This is a distinct mechanism supported by CPU25-12's measured
load penalty: the application requests eligible backing without synchronously
copying every range. Actual promotion may be delayed or absent. No application
worker, machine policy, persistent repack, new dependency or public API change.
Byte contents, Vec ownership and logical results must remain unchanged; advice
failure retains the original behavior. The existing kernel policy remains fixed.

Smallest structure, productive estimates excluding tests:

```text
cpu/mod.rs       (+2 conditional wiring lines)
cpu/buffer.rs    (~170 lines; existing resource ownership)
cpu/memory.rs    (~45 lines; one best-effort owned-range backing hint)
examples/cpu_load.rs (~50 lines; TEMPORARY first-request timing oracle)
```

Risk gate: exact guarded bytes, pointer and capacity checks, full CPU workspace,
CPU check, warning-denied Clippy and unchanged pinned real-model F16 parity.
Objective short model decode, A=`cpu25-05-bench`; same 128/32/32 fixed work,
warmup1/reps3 and canonical thresholds. p=0–.10, s=2, o=.001 predicts -.10–5.15%,
ideal at most 1.11x, about -.003–.126 s on the diagnostic decode interval.
Observe actual anonymous large-page backing after loading and at the final
measured repetition without changing the workload or delaying for promotion.
If objective passes, require medium/long controls plus the same fresh-process
startup total and whole-process elapsed <=5% regression controls as CPU25-12.
An inactive hint cannot establish a causal backing gain even if unrelated drift
improves throughput. Exactly one complete objective rerun for CV>5%; no tuning
of kernel intervals or waiting selectively for favorable backing is authorized.

### CPU25-13 final rejection

Correctness completed successfully: 170 root/167 engine tests, unchanged other
integration groups, CPU check, warning-denied Clippy and byte-identical pinned
parity log. No startup probe was needed because the objective failed first.
Private source/patch and candidate executable are preserved; existing wiring
and buffer restored to `998b189`, only the new memory source removed.

Initial A/B started 05:09:40+02:00, session49001; short model decode 7.87->7.23
(-8.132%), CV5.42%/7.82%, so run exactly one complete rerun. Initial prompt
34.68->35.36 (+1.961%), TTFT3707.03->3620.59 (-2.332%), public7.63->7.00
(-8.257%). First backing snapshot reports 28672 KiB AnonHugePages; a later
snapshot missed process exit and is explicitly unavailable, not zero.

Sole rerun started 05:12:48+02:00, session6450; all 128/32/32 counts agree:

| Metric | A | B | Change | CV A / B |
|---|---:|---:|---:|---:|
| Prompt token/s | 35.04 | 35.10 | +.171% | 2.72% / 3.80% |
| TTFT ms | 3655.19 | 3650.03 | -.141% | 2.68% / 3.88% |
| Model decode token/s | 8.26 | 7.72 | -6.538% | 3.14% / 1.74% |
| Public decode token/s | 8.00 | 7.48 | -6.500% | 3.15% / 1.74% |

Reject; no medium/long/startup controls after a terminal objective failure and
no second rerun. Five-second read-only snapshots in this rerun show actual
AnonHugePages increasing from 12288 to 45056 KiB, far below synchronous collapse's
1673216 KiB. Delayed limited activation is evidenced, but no specific cause of
the throughput loss is asserted. Waiting longer or changing the kernel's scan
policy would alter the declared workload/environment; no such variant is queued.
Eleven distinct attempts are complete; all original entries are now terminal.

### Fresh discovery: CPU25-14 declaration

Source inspection of current Q4/Q6 pair kernels, their callers, RoPE and the
latest retained-profile shapes found one additional bounded material premise.
Q4 already packs and interleaves; Q6 still reads `a[i*in_dim+o+{0,32,64,96}]`,
four distant eight-float streams per token. Its dominant width9216 imposes a
36 KiB token stride. Q6 prefill owns 5607.275/29176.471=.1922 of the current
medium diagnostic request. Historical wider-Q6 rejection did not test layout.

Pack those four eight-value streams into one contiguous 32-float record per
token/chunk before dispatch, retaining original activation storage for odd-row
and scalar fallbacks. The pair kernel then reads four consecutive vectors;
weight extraction, FMA order, tile size, worker policy and decode stay unchanged.
Only supported AVX2+FMA uses the packed view. p=.15, s=1.20, o=.005 predicts
2.04%, ideal1.18x, about .58 s saved; a conservative subthreshold estimate does
not close this cheap candidate with a sufficient ideal ceiling. Packing may
instead add traffic and hurt the shared cache; no counter attribution is claimed.

No new file or function: `matmul/q6k.rs` ~260 productive lines and
`matmul/q6k_simd.rs` ~290, both existing single-operation category-K kernels.
Add exact FP16 full-path comparisons to the unchanged single-row batched kernel
across widths256/512/3072/9216, n2/5/21/32/33 and even/odd row tails, plus all
existing numeric tests, workspace/check/Clippy and pinned F16 parity before
performance. Objective medium prompt, A=`cpu25-05-bench`, canonical tuple,
thresholds, sole objective stability rerun and short/long controls. No global
flags, public API, dependency, arithmetic reassociation or larger model.

CPU25-14 correctness passed: exact packed-path test across all declared shapes,
170 root/167 engine and unchanged integration groups, CPU feature check,
warning-denied Clippy and byte-identical pinned parity output. A/B session54861,
runner395864, started 05:23:15+02:00, deadline07:23:15+02:00; commands are
`screen.sh cpu25-05-bench cpu25-14-a 32 1 3 medium`, then
`screen.sh cpu25-14-bench cpu25-14-b 32 1 3 medium`, then `compare.awk`.
No controls or rerun yet. This is the twelfth countable implementation, still
pending its mandatory performance classification. No code is accepted yet.

CPU25-14 objective session54861 completed. Medium counts1024/32/31 match.
Prompt33.63->37.36 (+11.091%, CV3.11%/1.86%); TTFT30468.97->27418.02
(-10.013%, CV3.08%/1.85%); model decode7.61->7.51 (-1.314%, CV3.47%/9.42%);
public7.14->7.04 (-1.401%, CV3.48%/9.42%). Objective is stable and provisional
gain passes; high non-objective decode CV is disclosed, not a reason to select
a new objective pair. No stability rerun. Proceed with both short/long controls.

Controls session92052, runner396839, began05:29:48+02:00. Short counts128/32/32
match: prompt35.11->35.29 (+.513%, CV1.18%/5.55%); TTFT3646.04->3634.93
(-.305%, CV1.18%/5.69%); model7.85->7.49 (-4.586%, CV3.60%/2.51%);
public7.61->7.26 (-4.599%, CV3.60%/2.52%). Control means remain within5%,
close to the limit; prompt/control CV is disclosed. `compare.awk` labels its
input prompt as objective by default, so its final short-row instability phrase
is not the campaign decision: medium prompt is the declared objective and is
stable. No selective control rerun. Long A/B remains in progress.

### Final verification plan, declared before final measurements

After CPU25-14's terminal decision, directly compare immutable initial
`baseline-bench` (6238489) with the final accepted release binary, in adjacent
A/B pairs for short, medium and long, preserving all canonical tuple fields,
warmup1/reps3. This is a descriptive cumulative campaign measurement, not a
fourteenth/fifteenth candidate or permission to multiply isolated percentages.
Disclose every metric/CV and any aggregate control weakness. If prompt objective
dispersion exceeds5%, permit only one complete six-record rerun, not a selected
regime. Preserve both complete sets. No statistical-significance claim.

Refresh final disjoint operation attribution using the same previously validated
temporary cpu-profile timers and parser, one no-warmup repetition per regime,
paired with a matching unprofiled diagnostic repetition to disclose overhead.
Reuse the declared profiler structure; remove its source and feature wiring
after saving the private executable. Finish with fresh bottleneck/source
discovery, explicit hotspot closure, final CPU gates, documentation index/status
update and clean accepted-only branch. No global configuration or external write.

### CPU25-14 retained decision

Long control completed with matching3584/32/31 counts: prompt31.24->37.70
(+20.679%, CV4.50%/.11%), TTFT114893.12->95076.98 (-17.247%, CV4.38%/.11%),
model decode7.69->9.16 (+19.116%, CV15.93%/1.86%), public7.21->8.58
(+19.001%, CV15.92%/1.85%). No control mean regresses beyond5%; the unusually
variable baseline decode is unchanged code and its apparent gain is NOT credited
to packing. Short decode's -4.586% is a measured close-to-limit trade-off.
Medium objective is stable and passes; keep this single Q6 layout optimization.
No rerun was used; the comparison completed well inside its07:23 deadline.

Realized productive upper counts q6k240/q6k_simd275 fit260/290 estimates.
Only two existing files change; no new function, dependency or public API.
Complexity increases by one transient activation view and explicit internal
stride parameter, protecting the SIMD access layout while preserving fallback
storage and exact arithmetic. Extra storage is n*in_dim*4 bytes per Q6 call,
at most1.125 MiB for the canonical32x9216 shape; it is not persistent weight
expansion. Saved executable SHA256
`e07ea430d9b809d02cfc38a4f4f084f0786e47a4e5c711bbdca759f91b88df79`.
The only post-build edit removes a duplicated test comment, not executable code.
Commit includes this evidence and unchanged exact gate; final attribution follows.

### Final execution checkpoint

Accepted production commit `6545e72343f5daed2fe8160d3c291946771445e4`.
Final CPU workspace170/167, integrations1/4/12, feature check, warning-denied
Clippy, format/diff and exact pinned parity/full initial-log identity passed.
Built the temporary profiler from the previously validated patch, saved
`final-profile-bench`, then restored all four feature/backend files and removed
its sole new source. `git status` contains only this report update.

Final uninstrumented binary SHA256
`2b89bd3acaec33e3e2ae005338d25a5b75c7e41e0020e7166b13d564a785d46d`;
initial binary SHA256
`f15102da91cae7e9482313e8863322e809c9f97126f52a386b2ff8b88c50b246`.
The final executable is rebuilt from the accepted commit, not the experimental
feature build. Compiler remains1.96.1, no RUSTFLAGS or dependency changes.

Session58448, runner405853, started05:54:40+02:00, conservatively sharing a
07:54:40+02:00 deadline for diagnostics and complete final A/B. Telemetry uses
the same read-only script. For each short/medium/long row, run
`screen.sh final-bench final-diagnostic 32 0 1 <row>`, then
`screen.sh final-profile-bench final-profile 32 0 1 <row>` and validate spans
with `profile.awk`. Then each row runs adjacent initial/final pairs:
`screen.sh baseline-bench final-initial 32 1 3 <row>` and
`screen.sh final-bench final-accepted 32 1 3 <row>`, followed by `compare.awk`.
All exact calibrated prompts/token records and per-process resource statistics
remain private. Resume this same handle; no comparison is yet classified.

### Final attribution and fresh bottleneck discovery

All three disjoint timelines passed the parser's event/ordering/boundary checks.
One diagnostic repetition has CV n/a, not zero. First-logits is the phase
boundary; prefill and decode costs are never counted twice.

| Regime | Unprofiled / profiled TTFT ms | Observed difference | Full profiled request ms | Q4 prefill ms / share | Q6 prefill ms / share | RoPE prefill ms / share | Attention prefill ms / share |
|---|---|---:|---:|---|---|---|---|
| short | 3688.95 / 3629.12 | -1.622% | 7594.012 | 2248.458 / 29.61% | 797.392 / 10.50% | 397.492 / 5.23% | 34.229 / .45% |
| medium | 29790.11 / 30640.09 | +2.853% | 34752.031 | 19530.361 / 56.20% | 4782.665 / 13.76% | 3524.517 / 10.14% | 1391.737 / 4.00% |
| long | 119494.82 / 118343.14 | -.964% | 122577.957 | 63363.983 / 51.69% | 15693.491 / 12.80% | 12637.859 / 10.31% | 22462.590 / 18.33% |

Negative differences reflect environmental variation, not beneficial timer
overhead. Short model decode8.91/8.07, medium7.58/7.78 and long7.05/7.56 likewise
show why these pairs are attribution diagnostics, never retention benchmarks.
Current short decode Q4 is2731.719 ms of3965.188 ms (68.89%); long decode
attention1062.117 ms of4235.589 ms (25.08%). No GPU queue/transfer exists here.

Fresh source/disassembly discovery inspected current Q4/Q6 callers and pair
kernels, attention's paired causal passes, RoPE and current pool/parallel code,
and rechecked historical wider-vector/repacking results. Q4 remains the largest
measured bottleneck: medium main3072x9216 batch32 owns10283.323 ms (29.59% of
the whole request), plus wide9216x3072 owns4304.860 ms (12.39%). In the final
binary, Q4's actual symbol is0x78de0..0x79570; its hot token loop0x79310..0x793c8
still has contiguous activation loads, four live FMA chains and no accumulator
stack spill. The private broad disassembly also contains following Q5/Q6
functions; their spills must not be attributed to this Q4 loop.

No new bounded mechanism emerged after the Q6 layout trial. The remaining
hotspots are closed for this campaign on the following explicit boundaries,
not on a claim that the complete kernel bucket has a sub5% ideal ceiling:

| Remaining hotspot/family | Measured boundary and closure |
|---|---|
| Q4 prefill and decode | Packing/interleaving retained; transient float expansionCPU25-03 regressed; wider row blocking and persistent repacking have unchanged historical control/memory failures; native integerCPU25-10 fails numerical tolerance. Current hot-loop inspection finds no new spill or redundant activation copy to remove. New lower-precision/external-library routes cannot bypass numerical/dependency constraints. |
| Q6 prefill/decode and vocabulary logits | Four-stream layoutCPU25-14 retained; existing two-row dequant reuse and single-token route remain. Historical wider-Q6 control failure is unchanged. No new removable copy or dispatch layer is exposed after packing; a parameter-only tile/row sweep is not a new evidence-backed premise. |
| RoPE coefficients | About10% of medium/long request is material, but exact per-call reuseCPU25-11 regressed with unchanged traversal; historical pair-major reuse also failed. No observation isolates a new lifetime/allocation cause warranting another cache variant. The separately measured pure-rotation ceiling remains<=3.10%. |
| Long attention/KV | Paired-position and inherited four-head reuse retained. Further doubling would require16 query accumulators plus a key vector, exceeding the16 AVX2 vector registers before other live state; this is not a free continuation of pair reuse. Existing causal ordering and FP16 layout remain exact. No new measured transfer, oversized allocation or synchronization boundary was found. |
| Worker scheduling/locality | CPU25-06 spin,08 dynamic chunks (interesting only),09 affinity and historical physical-core policy all terminate without a retained gain. Their overlapping timestamp bounds cannot be added to manufacture a larger opportunity; no new imbalance mechanism was exposed by this layout change. |
| Weight backing/startup | CPU25-12 synchronous promotion fails the measured first-request control;13 asynchronous advice regresses objective with limited actual backing. Waiting for kernel promotion or changing global scan/affinity policies changes the declared environment and is outside this campaign. |
| Other prefill operations/public gaps | Combined remainder is1.99% short,4.06% medium,3.41% long of each final request: even complete removal ceilings are2.03%,4.23%,3.54%, below5%. No individual member justifies a filler optimization. |

These are evidence-bounded closures, not a universal optimum or a substitute for
unavailable hardware counters. New hardware/shape evidence or an explicitly
authorized dependency/quality-contract change can justify a future campaign.
Twelve new attempts, a terminal pool and this fresh discovery satisfy the
quantitative-stop investigation conditions. Finish the already-declared final
comparison and report; do not start further micro-optimization.

The first final short A/B records prompt25.46->34.82 (+36.764%), but initial
prompt CV5.10% exceeds5% (final1.84%). Preserve this complete set and, after its
medium/long rows finish, execute exactly one complete six-record rerun as
predeclared. No individual row is repeated or substituted now.

## Complete initial/final comparison: first set

Session58448 completed all six public records; actual counts are128/32/32,
1024/32/31 and3584/32/31 on both sides. A=immutable6238489, B=accepted6545e72.
These are measured cumulative changes, not multiplied per-candidate gains.

| Regime | Metric | A | B | Change | CV A / B |
|---|---|---:|---:|---:|---|
| short | prompt token/s | 25.46 | 34.82 | 36.764% | 5.10% / 1.84% |
| short | TTFT ms | 5035.40 | 3676.85 | -26.980% | 4.96% / 1.85% |
| short | model decode token/s | 7.61 | 7.66 | 0.657% | 4.33% / 1.95% |
| short | public decode token/s | 7.37 | 7.42 | 0.678% | 4.33% / 1.92% |
| medium | prompt token/s | 24.74 | 42.06 | 70.008% | 0.60% / 2.72% |
| medium | TTFT ms | 41387.67 | 24358.45 | -41.146% | 0.60% / 2.74% |
| medium | model decode token/s | 8.17 | 11.34 | 38.800% | 8.70% / 1.34% |
| medium | public decode token/s | 7.65 | 10.63 | 38.954% | 8.71% / 1.33% |
| long | prompt token/s | 31.05 | 35.61 | 14.686% | 0.12% / 0.17% |
| long | TTFT ms | 115440.01 | 100654.11 | -12.808% | 0.12% / 0.17% |
| long | model decode token/s | 9.23 | 9.38 | 1.625% | 0.04% / 0.26% |
| long | public decode token/s | 8.65 | 8.79 | 1.618% | 0.05% / 0.26% |

The complete set is not stability-qualified because short A prompt CV=5.10%.
Medium's unchanged-decode increase also exposes substantial between-run drift;
do not interpret its70% prompt difference as an isolated causal code effect.
Keep every record. Sole full rerun session33770, runner410476, began06:22:06+02:00,
labels `final-rerun-initial/accepted`, same binaries, tuple and short/medium/long
order; deadline remains07:54:40+02:00. No code or environment change and no
further stability rerun after this set.

Sole rerun short completed: prompt30.58->41.79 (+36.658%, CV1.29%/4.70%),
TTFT4186.61->3067.25 (-26.737%, CV1.28%/4.83%), model12.39->12.48
(+.726%, CV.36%/.54%), public12.00->12.09 (+.750%, CV.35%/.52%).
Medium completed: prompt29.22->42.21 (+44.456%, CV.64%/.17%),
TTFT35040.62->24262.52 (-30.759%, CV.64%/.17%), model11.34->10.91
(-3.792%, CV1.51%/.75%), public10.63->10.23 (-3.763%, CV1.51%/.76%).
All counts match, prompt objectives stable and all current controls within5%.
Long A/B remains active on the same handle33770; no inference from partial rows.

## Campaign accounting

Current-campaign attempts12 = kept4 (CPU25-01,02,05,14) + canonical rejects7
(03,06,09,10,11,12,13) + canonical interesting1 (08, evidence only, code
restored). Candidate not_verified0. Closed-untried2 (04,07), not attempts.
The pool's `rejected (evidence only)` disposition for08 expresses restored code,
not a relabeling of its canonical3.668% interesting result. The final descriptive
matrix's verification status is separate from candidate counts.

Current-campaign retained commits: `13b2ab4`, `245c15f`, `998b189`, `6545e72`.
Six existing production files differ from the immutable start, alongside
one new paired-attention ownership file; no persistent weight repack, scheduler,
affinity, memory policy, assembly-integer path, profiler or extra example remains.
Final CPU gates pass354 tests across groups170/167/1/4/12; seven declared
external tests remain ignored. No new dependency, public API, global compiler
flag, machine setting, push, PR or merge was introduced.

## Inherited work, not counted as current retention

All of the following precede the immutable baseline; ancestry was checked:
`9dc3b01` dynamic32-row prefill, `bd62f5b` Q4 single-token latency route,
`d23801d` Q6 single-token latency route, `80eadd2` four-query GQA KV reuse,
`c76c8b9` cache-sized token tiling and `82634e4` minimum-Rust CPUID compatibility.
Their historical measurements are not current-campaign A/B results. The separate
completed continuation branch consulted for negative evidence was not merged
or counted as new work here.
