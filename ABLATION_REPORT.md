# Ablation Report

## Summary

The branch preserves its supported model family, backend matrix, and public API.
The original ablation removed 132 Rust lines; the later repository-consistency
pass added focused tests, bounded streaming, and single-operation kernel files.
The current result is 18 lines above the pre-ablation baseline, with no new
dependencies, a stricter source-structure gate, and a green warning-free matrix
across all five supported profiles.

## Metrics (before → after)

| Metric | Before | After | Δ |
|---|---:|---:|---:|
| Supported `--all-targets` builds | 5/5 green | 5/5 green | — |
| Profiles with compiler warnings during build | 5/5 | 0/5 | -5 profiles |
| Lines of Rust | 37,389 | 37,407 | +18 |
| Direct dependencies | 14 | 14 | 0 |
| Total dependencies | 176 | 176 | 0 |
| CPU release binary | 6,090,432 bytes | 6,090,384 bytes | -48 bytes |
| Passing tests (CPU / Vulkan / Vulkan hybrid / Metal / Metal hybrid) | 320 / 301 / 376 / 298 / 368 | 322 / 303 / 363 / 300 / 365 | -12 net, from profile applicability plus 10 new passes |
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

### Repository-consistency pass (37,257 → 37,407 LOC)

| Metric | Before | After | Δ |
|---|---:|---:|---:|
| Supported `--all-targets` tests | 5/5 green | 5/5 green | — |
| Standard Clippy profiles under `-D warnings` | 0/5 | 5/5 | +5 profiles |
| Lines of Rust | 37,257 | 37,407 | +150 |
| Direct dependencies | 14 | 14 | 0 |
| Total dependencies | 176 | 176 | 0 |
| CPU release binary | 6,089,392 bytes | 6,090,384 bytes | +992 bytes |
| Passing tests (CPU / Vulkan / Vulkan hybrid / Metal / Metal hybrid) | 320 / 301 / 361 / 298 / 363 | 322 / 303 / 363 / 300 / 365 | +10 |
| Frontend parser tests | not runnable from npm | 17 | +17 |
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

### Repository consistency and security

- Bounded the server's synchronous-to-async SSE bridge to 32 frames. Slow clients now apply backpressure instead of allocating an unbounded queue, while receiver closure still cancels generation.
- Rejected invalid UTF-8 from external SSE providers instead of silently replacing bytes, and made `max_tokens` conversion platform-safe.
- Unified strict argument parsing for context and generation limits and removed stale reasoning parameters and comments.
- Split CPU elementwise operations and CPU/Vulkan KV writes into single-responsibility modules. The source audit now counts production after isolated `#[cfg(test)]` items, covers root sources and build scripts, and verifies every Metal/Vulkan shader has exactly one category-K marker.
- Made the existing frontend Reasoning suite executable through `npm test`, with 17 passing cases and an explicit Node.js minimum.
- Corrected public routes, clone instructions, validation row counts, personal paths, and historical/report wording across the documentation.

### Dependencies

- No dependency was added or removed. `shaderc` remains an optional build dependency consumed by `build.rs`; the Cargo metadata records that known cargo-machete limitation, and `cargo machete --with-metadata` is clean. `cargo udeps` was not installed in the environment.

## Kept despite looking unused (and why)

- The Metal pipeline failpoint still uses the creation index, now read directly from the number of pipelines already created instead of retaining a production-only `enumerate()` index.
- Existing public items were retained because removing them would change the library contract without explicit API-removal approval.
- Cohesive loader, session, and kernel boundaries retain their explicit arguments under narrow documented `too_many_arguments` allowances. Grouping them into duplicate configuration types would add indirection.
- `OutputTensor` was retained instead of collapsing tied and dedicated weights into an unnamed `Option`; the enum makes both valid family states explicit.
- `TextDecoder::finish` was retained as the consuming terminal boundary that documents intentional disposal of an incomplete UTF-8 suffix.
- The Ministral default-context invariant remains executable as a constant assertion next to the pinned version rows.

## Verification

The following checks were run after each relevant logical change and again for the final matrix:

```text
cargo fmt --all -- --check
cargo build --locked --workspace --all-targets --no-default-features --features <each supported profile>
cargo test --locked --workspace --no-default-features --features <each supported profile>
cargo clippy --locked --workspace --all-targets --no-default-features --features <each supported profile>
cargo clippy --locked --workspace --all-targets --no-default-features --features <each supported profile> -- -D warnings
cargo build --locked --release --no-default-features --features cpu --bin graph-horizon
cargo machete --with-metadata
cd web/frontend && npm test && npm run check && npm run build && npm audit
```

The final full matrix passed with 322 / 303 / 363 / 300 / 365 tests and zero failures. Standard Clippy completed under `-D warnings` for every supported profile. The frontend passed 17 parser tests, static checking with zero warnings, production build, and dependency audit. `cargo udeps`, `cargo hack`, and `cargo bloat` were not installed in the environment.
