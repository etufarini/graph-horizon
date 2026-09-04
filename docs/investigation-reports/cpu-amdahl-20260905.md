# CPU Amdahl campaign 20260905

## Recovery state

- Campaign ID: `cpu-amdahl-20260905`.
- Branch: `perf/cpu-amdahl-20260905`.
- Immutable starting revision: `62384895acfde94cf28fded1294ad860daabf2ff`.
- Retained production checkpoint: starting revision; no new optimization yet.
- Started: 2026-09-05T00:40:43+02:00.
- State: baseline captured; temporary CPU timeline and overhead comparison running.
- Deadline: none; minimum ten distinct countable attempts, two hours per comparison.
- Current attempts / kept / rejected / not_verified / closed-untried: 0 / 0 / 0 / 0 / 0.

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

Not populated before the required preflight and fresh baseline attribution.
No production candidate is implicitly accepted. Amdahl fractions, benchmark
results and a quantitative stop are not yet established.

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
