# CPU Amdahl campaign 20260905

## Recovery state

- Campaign ID: `cpu-amdahl-20260905`.
- Branch: `perf/cpu-amdahl-20260905`.
- Immutable starting revision: `62384895acfde94cf28fded1294ad860daabf2ff`.
- Retained production checkpoint: starting revision; no new optimization yet.
- Started: 2026-09-05T00:40:43+02:00.
- State: preflight; authenticating the required correctness oracle.
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
