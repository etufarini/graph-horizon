# Ablation Report

## Summary

The branch audit retained the measured Vulkan improvements and their
qualification records. Rejected sparse, low-precision, M128, and diagnostic
prototypes are absent from the final source. The cleanup removed the only
Clippy complexity warning in the changed Rust code, avoided eager benchmark
error construction, made Q4 predecode validation total without `unwrap`, and
removed the installer's undeclared runtime dependency on `dirname`.

## Metrics (before → after)

| Metric | Before | After | Delta |
|---|---:|---:|---:|
| CPU all-target build | pass | pass | unchanged |
| Ordinary compiler warnings | 0 | 0 | unchanged |
| Clippy complexity warnings in changed code | 1 | 0 | -1 |
| Rust source lines | 42,056 | 42,064 | +8 test/validation lines |
| Direct CPU dependency-tree entries | 15 | 15 | 0 |
| Total CPU dependency-tree entries | 177 | 177 | 0 |
| CPU workspace tests | 1 fixture failure | 342 passed, 0 failed | green |
| Release binary size | not measured | not measured | n/a |

## Changes

### Dead code and rejected candidates

- No additional dead production path was found. The branch history already
  reverts the rejected shared-A, sparse-attention, M128, and low-precision
  candidates; source searches confirm that they are absent from the final tree.

### Dependencies and features

- No dependency or feature was added by the cleanup. `cargo-machete` and
  nightly `cargo-udeps` were unavailable; the manifest diff and dependency-tree
  counts show no dependency change relative to the reviewed branch state.

### Complexity and validation

- Median parity now uses the idiomatic integer predicate reported by Clippy.
- Benchmark argument failures are constructed only on the error path.
- Q4 predecode rejects zero dimensions and arithmetic overflow explicitly,
  without an `Option::unwrap` after a separate guard.
- The installer resolves its own directory with Bash parameter expansion, so
  its declared prerequisite checks also work under an isolated `PATH`.

## Kept despite looking removable

- Capability and route policies remain separate because they protect distinct
  Vulkan pipeline, shape, and numeric invariants and provide portable fallbacks.
- Negative investigation reports remain because they are the reproducible
  evidence that rejected candidates must not be reintroduced.
- Opt-in native Q4 predecode remains because its measured 28K TTFT improvement
  clears the repository's keep gate; canonical weights and fallback paths stay
  intact.

## Verification

- `cargo fmt --all -- --check`
- CPU workspace build and tests with locked dependencies
- Vulkan and Vulkan-profile workspace checks
- Full Vulkan tests on the available GPU: 175 passed, 0 failed, 7 ignored
- Full Vulkan-hybrid tests on the available GPU: 249 passed, 0 failed, 6 ignored
- CPU and Vulkan Clippy with all targets and warnings denied
- Frontend tests: 92 passed; Svelte check: 0 errors and 0 warnings; Vite build: pass
- Metal and Metal-hybrid checks require external verification on macOS Apple Silicon
- `git diff --check`
