# Ablation Report

## Summary

The ablation stopped compiling an x86-only SIMD test module as an empty module
on other architectures. The CPU all-target build went from two distinct
warnings to none without changing APIs, dependencies, tests, or binary size.

## Metrics (before → after)

| Metric | Before | After | Delta |
|---|---:|---:|---:|
| CPU build (`--all-targets`) | pass | pass | — |
| Compiler warning lines | 3 (2 distinct) | 0 | -3 |
| Lines of tracked Rust | 37,549 | 37,549 | 0 |
| Direct dependencies | 15 | 15 | 0 |
| Total dependencies | 176 | 176 | 0 |
| CPU release binary | 6,092,496 B | 6,092,496 B | 0 B |
| CPU tests passing | 330 | 330 | 0 |

The bundled script reports `build_ok=false` in both snapshots because it omits
the repository's mandatory backend feature. The table instead uses the smallest
supported profile, `--no-default-features --features cpu`.

## Changes

- Updated stale Italian README assertions after the engine README translation;
  the previously failing `docs_contract` is green (`c16514c`).
- Gated the complete CPU f16 SIMD test module to x86_64, matching its only test.
  On non-x86 targets this removes the empty module and two unused imports; on
  x86_64 the existing bit-exact SIMD test remains unchanged (`d3531bf`).

### Dependencies, features, and complexity

- Removed no dependencies: `cargo machete` found no candidates and
  `cargo tree --duplicates` found no duplicate versions.
- Changed no features: every explicit backend profile is live and verified.
- Applied no speculative Clippy rewrites. `clippy::complexity` was otherwise
  clean; nursery suggestions involving visibility, `const fn`, FMA, or denser
  combinators did not make the code simpler or could change numeric behavior.

## Kept Despite Looking Removable

- Public items remain API contracts; backend-gated code remains required by the
  five supported feature profiles.
- Dependency defaults remain unchanged without evidence that removing them
  preserves platform behavior.
- `cargo-udeps`, `cargo-bloat`, and `cargo-hack` were unavailable, so no
  dependency removal was attempted without confirmation tooling.

## Verification

- `cargo fmt --all --check` — pass.
- CPU all-target build and workspace tests — pass, 330 passed / 4 ignored.
- Metal build/tests — pass, 309 passed / 4 ignored.
- Metal-hybrid build/tests — pass, 374 passed / 4 ignored.
- Vulkan build/tests — pass, 311 passed / 4 ignored.
- Vulkan-hybrid build/tests — pass, 371 passed / 4 ignored.
- CPU Clippy all-targets with warnings denied — pass.
