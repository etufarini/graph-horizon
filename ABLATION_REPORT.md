# Ablation Report

## Summary

The current branch was simplified without changing its supported backend matrix or public API. Backend-only paths now compile only in profiles that use them, Metal hybrid placement derives its mode from the existing ownership selection, and hybrid split enumeration uses direct ranges. The result is 92 fewer lines of Rust (103 additions, 195 deletions), no new dependencies or abstractions, and a green build and test matrix across all five supported profiles.

## Metrics (before → after)

| Metric | Before | After | Δ |
|---|---:|---:|---:|
| Supported `--all-targets` builds | 5/5 green | 5/5 green | — |
| Profiles with compiler warnings during build | 5/5 | 0/5 | -5 profiles |
| Lines of Rust | 37,389 | 37,297 | -92 |
| Direct dependencies | 14 | 14 | 0 |
| Total dependencies | 176 | 176 | 0 |
| CPU release binary | 6,090,432 bytes | 6,090,432 bytes | 0 |
| Passing tests (CPU / Vulkan / Vulkan hybrid / Metal / Metal hybrid) | 320 / 301 / 376 / 298 / 368 | 320 / 301 / 361 / 298 / 363 | -20 inapplicable tests |
| Test failures | 0 | 0 | 0 |

The bundled measurement script invokes Cargo without a backend feature, so its `build_ok` is `false` both before and after by design: the crate requires exactly one of `cpu`, `vulkan`, `vulkan-hybrid`, `metal`, or `metal-hybrid`. The supported-profile matrix above is the relevant build result. The lower hybrid test counts come from no longer compiling standalone-backend loading tests in hybrid configurations; no test was removed from its applicable profile.

## Changes

### Dead and inapplicable code

- Restricted `Backend::load` and its imports to the standalone CPU and Vulkan profiles. This removed the Metal test-only full-backend loader and stopped hybrid test builds from manufacturing unused standalone paths. Every supported profile built and tested successfully after the change. (`aee4e52`)
- Removed the unused Vulkan test-buffer replacement path and narrowed standalone Vulkan memory planning to the `vulkan` profile. (`aee4e52`)
- Removed CPU F16 imports and module exports that were unused in their compiled configurations, and narrowed weight-source tensor enumeration to the profiles that consume it. (`aee4e52`)

### Feature and configuration trimming

- Replaced broad `test` and negative-feature gates with the positive backend feature that owns each path. This makes each supported configuration compile only the implementation it can exercise and eliminated compiler warnings from all five profile builds. (`aee4e52`)

### Complexity reductions

- Removed the `mixed_placement` boolean from the hybrid device contract and its callers. Metal now derives the condition from the already validated weight selection: CPU-owned leading layers plus a GPU-owned tail. This removes duplicated state while preserving the ownership invariant. (`2a23959`)
- Replaced boxed iterator dispatch in both hybrid placement planners with a directly bounded inclusive range. The range still enumerates only split zero when GPU placement is enabled and only the CPU-only split otherwise. (`d807bff`)
- Replaced a boolean equality assertion with a direct assertion. (`d8f4a8a`)

### Dependencies

- No dependency was removed. `cargo machete --with-metadata` reported `shaderc`, but `crates/graph_horizon_engine/build.rs` uses it to compile Vulkan shaders, so the report is a build-script false positive. `cargo udeps` was not installed in the environment.

## Kept despite looking unused (and why)

- The index from `enumerate()` in the Metal pipeline is referenced by a test-only failpoint. Removing it passed a normal lint path but broke Metal test compilation, so the attempted edit was reverted.
- Existing public items were retained because removing them would change the library contract without explicit API-removal approval.
- Clippy's `too_many_arguments` findings in loader, session, and kernel boundaries were retained. Grouping these cohesive values into new configuration types would add indirection and increase conceptual complexity.
- Readability-neutral nursery suggestions and findings outside the approved structure were not bulk-applied.

## Verification

The following checks were run after each relevant logical change and again for the final matrix:

```text
cargo fmt --all -- --check
cargo build --workspace --all-targets --features cpu
cargo build --workspace --all-targets --features vulkan
cargo build --workspace --all-targets --features vulkan-hybrid
cargo build --workspace --all-targets --features metal
cargo build --workspace --all-targets --features metal-hybrid
cargo test --workspace --features cpu
cargo test --workspace --features vulkan
cargo test --workspace --features vulkan-hybrid
cargo test --workspace --features metal
cargo test --workspace --features metal-hybrid
cargo clippy --workspace --all-targets --features <each supported profile>
bash .skills/rust_code_ablation/measure.sh
cargo build --release --features cpu --bin graph-horizon
cargo machete --with-metadata
```

All builds and tests passed. Clippy completed successfully in every supported profile; its remaining advisory findings are the intentionally retained cases described above or files outside the approved ablation structure.
