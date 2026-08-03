<!--
This report records the reproducible Rust ablation performed on the dedicated
branch; it owns measurements and evidence, not runtime or support contracts.
-->

# Ablation Report

## Summary

The baseline documentation test was first aligned with the current Q4_K_M-only
code and made green in commit `9c51ebb`. The subsequent ablation removed one
unused engine dependency, disabled unused async and terminal features, and
simplified three control-flow sites without changing public behavior. The
supported CPU, Vulkan, and hybrid configurations remain green.

## Metrics (before → after)

| Metric | Before | After | Δ |
|---|---:|---:|---:|
| Build (`--all-targets`) | pass | pass | unchanged |
| Compiler warnings | 0 | 0 | 0 |
| Lines of Rust | 32,138 | 32,135 | -3 |
| Direct dependencies | 14 | 14 | 0 |
| Total dependencies | 186 | 185 | -1 |
| Release binary size | 8,066,952 B | 8,050,904 B | -16,048 B |
| Tests passing (`--all-features`) | 342 | 342 | 0 |

The bundled script supplied these metrics. Direct dependencies describe the root
package; the removed declaration belonged to the engine crate. Nine external
tests remain intentionally ignored in the all-feature run.

## Changes

### Removed dependencies

- Removed the engine's `serde_json`: `cargo-machete` 0.9.2 identified it and a
  full-tree search found only its manifest declaration. Final scan: clean.
  Commit `585ff39`.

### Feature trimming

- Replaced Tokio `full` with the six features used by source/tests.
- Disabled unused `tokio-stream/time`; timeout code uses `tokio::time` directly.
- Retained only Ratatui `crossterm` and `layout-cache`; calendar, macros and
  underline-color were unused, removing `ratatui-macros` from the tree.
  All three changes are in `7db1af1`.

### Complexity reductions

- Sampling performs one shared sort after optional top-k truncation (`b614ca1`).
- Transcript import derives its optional System record directly (`4effbad`).
- Streaming maps an optional delta directly to its change flag (`69ec752`).

## Kept Despite Looking Removable

- Public items remain downstream API contracts.
- Ratatui's backend still enables Crossterm defaults transitively.
- Numeric Clippy suggestions could alter floating-point results.
- Visibility, `Self`, `const fn`, and test-only style edits add no simplification.
- Significant-drop suggestions require a separate resource-lifecycle review.
- `cargo-udeps` was unavailable because no nightly toolchain is installed. The
  removal was confirmed by Machete, search, compilation and every feature gate.

## Verification

The following checks passed after the final code change:

```text
cargo fmt --all -- --check
cargo build --workspace --all-targets --all-features
cargo test --workspace --no-default-features --features cpu
cargo test --workspace --no-default-features --features vulkan
cargo test --workspace --no-default-features --features hybrid
cargo clippy --workspace --all-targets --all-features
cargo-machete
```

Clippy exits successfully with two pre-existing test warnings. Bare
`--no-default-features` is intentionally invalid: an explicit backend is required.
