<!--
This document owns repository contribution invariants, validation expectations,
and AGENTS line-limit exemptions. It does not define runtime support claims.
-->

# Contributing

Follow [AGENTS.md](AGENTS.md): keep changes small, state invariants and risks,
and add no dependency or abstraction without a concrete need.

Before changing code, identify the invariant, smallest viable change, and main
risk. Every build selects exactly one profile—`cpu`, `vulkan`,
`vulkan-hybrid`, `metal`, or `metal-hybrid`—with defaults disabled:

```sh
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --no-default-features --features cpu -- -D warnings
cargo test --workspace --no-default-features --features cpu
```

Repeat Clippy and tests for every affected profile. A missing device,
toolchain, authenticated model, or pinned oracle is `external verification`;
an assertion, comparison, placement, or lifecycle mismatch is a failure.

Orchestration stays under 200 productive lines. Every category-K file must be a
single-operation dense numeric kernel and declare exactly one
`// AGENTS deroga K: <nota>` marker. Every category-I file must contain only one
trait definition or one thin delegating `impl Trait` block and declare exactly
one `// AGENTS deroga I: <nota>` marker.

New backends implement the neutral runtime contracts; new model families do not
add backend-pair modules. Never reduce context, change placement, or retry a
different tuple to turn a failed validation into a pass. Do not commit secrets,
personal paths, generated oracle worktrees, or model artifacts.
