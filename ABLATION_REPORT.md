# Ablation Report

## Summary

The current branch was simplified without changing its supported model family, backend matrix, or public API. Backend-only paths compile only in profiles that use them, hybrid placement derives decisions from existing ownership, and the shared Ministral 3 implementation no longer carries one-call adapters or duplicate tokenizer fallback branches. Across both audited passes the result is 132 fewer lines of Rust, including 40 removed by the family-wide pass, no new dependencies or abstractions, and a green build and test matrix across all five supported profiles.

## Metrics (before → after)

| Metric | Before | After | Δ |
|---|---:|---:|---:|
| Supported `--all-targets` builds | 5/5 green | 5/5 green | — |
| Profiles with compiler warnings during build | 5/5 | 0/5 | -5 profiles |
| Lines of Rust | 37,389 | 37,257 | -132 |
| Direct dependencies | 14 | 14 | 0 |
| Total dependencies | 176 | 176 | 0 |
| CPU release binary | 6,090,432 bytes | 6,089,392 bytes | -1,040 bytes |
| Passing tests (CPU / Vulkan / Vulkan hybrid / Metal / Metal hybrid) | 320 / 301 / 376 / 298 / 368 | 320 / 301 / 361 / 298 / 363 | -20 inapplicable tests |
| Test failures | 0 | 0 | 0 |

The bundled measurement script invokes Cargo without a backend feature, so its `build_ok` is `false` both before and after by design: the crate requires exactly one of `cpu`, `vulkan`, `vulkan-hybrid`, `metal`, or `metal-hybrid`. The supported-profile matrix above is the relevant build result. The lower hybrid test counts come from no longer compiling standalone-backend loading tests in hybrid configurations; no test was removed from its applicable profile.

### Family-wide pass (37,297 → 37,257 LOC)

| Metric | Before | After | Δ |
|---|---:|---:|---:|
| Supported `--all-targets` builds | 5/5 green | 5/5 green | — |
| Profiles with compiler warnings during build | 0/5 | 0/5 | 0 |
| Lines of Rust | 37,297 | 37,257 | -40 |
| Direct dependencies | 14 | 14 | 0 |
| Total dependencies | 176 | 176 | 0 |
| CPU release binary | 6,090,432 bytes | 6,089,392 bytes | -1,040 bytes |
| Passing tests (CPU / Vulkan / Vulkan hybrid / Metal / Metal hybrid) | 320 / 301 / 361 / 298 / 363 | 320 / 301 / 361 / 298 / 363 | 0 |
| Test failures | 0 | 0 | 0 |

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
- Deleted the one-call Ministral runtime-session adapter and constructed the statically selected neutral session in the request lifecycle that owns it. (`ffb3fac`)
- Removed the `prefill_with` pass-through and exposed the existing bounded prefill implementation directly; batching policy, validation, and buffer ownership remain unchanged. (`1e8cafd`)
- Replaced the single-row MLP wrapper plus `record_rows` implementation with one recorder whose row count is explicit at both decode and prefill call sites. (`11da177`)
- Unified the identical byte fallback for unknown tokenizer symbols and literal structural spellings. Existing tests prove ordinary input still cannot emit structural IDs. (`f3bfd54`)
- Bound the optional parity-vector length directly instead of testing it and then unwrapping it. The exact validation error remains unchanged. (`5d3cced`)

### Dependencies

- No dependency was removed. `cargo machete --with-metadata` reported `shaderc`, but `crates/graph_horizon_engine/build.rs` uses it to compile Vulkan shaders, so the report is a build-script false positive. `cargo udeps` was not installed in the environment.

## Kept despite looking unused (and why)

- The index from `enumerate()` in the Metal pipeline is referenced by a test-only failpoint. Removing it passed a normal lint path but broke Metal test compilation, so the attempted edit was reverted.
- Existing public items were retained because removing them would change the library contract without explicit API-removal approval.
- Clippy's `too_many_arguments` findings in loader, session, and kernel boundaries were retained. Grouping these cohesive values into new configuration types would add indirection and increase conceptual complexity.
- `OutputTensor` was retained instead of collapsing tied and dedicated weights into an unnamed `Option`; the enum makes both valid family states explicit.
- `TextDecoder::finish` was retained as the consuming terminal boundary that documents intentional disposal of an incomplete UTF-8 suffix.
- The constant Ministral release assertion was retained because it keeps the default-context invariant executable next to the pinned version rows.
- Clippy's `redundant_pub_crate`, `missing_const_for_fn`, and test-fixture-only suggestions were retained: applying them would widen lexical visibility, add qualifiers, or make no reduction in runtime or conceptual complexity.

## Verification

The following checks were run after each relevant logical change and again for the final matrix:

```text
cargo fmt --all -- --check
cargo build --locked --workspace --all-targets --no-default-features --features <each supported profile>
cargo test --locked --workspace --no-default-features --features <each supported profile>
cargo clippy --locked --workspace --all-targets --no-default-features --features <each supported profile>
cargo clippy --locked --workspace --all-targets --no-default-features --features <each supported profile> -- -W clippy::complexity -W clippy::nursery
bash .skills/rust_code_ablation/measure.sh
cargo build --locked --release --no-default-features --features cpu --bin graph-horizon
cargo machete --with-metadata
```

Every logical family change was committed separately and passed the five-profile build and test matrix before the next change. The same full matrix passed again at the end: 320 / 301 / 361 / 298 / 363 tests, with zero failures. Default Clippy and the explicit complexity/nursery pass completed successfully in every supported profile; their remaining advisory findings are the intentionally retained cases described above or files outside the approved ablation structure. `cargo udeps`, `cargo hack`, and `cargo bloat` were not installed in the environment.
