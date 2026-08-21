---
name: rust-code-ablation
description: >-
  Simplify a Rust workspace by removing verified dead code, unused dependencies,
  obsolete feature flags, and accidental complexity while preserving behavior.
  Use when the user explicitly requests Rust codebase simplification or ablation.
---

# Rust Code Ablation

Simplify without changing observable behavior. Prefer deletion and direct code;
Git history is the archive. Follow `AGENTS.md`, work on a dedicated branch, and
do not install tools or dependencies merely to find candidates.

## Invariants

- Begin from a clean worktree and a green baseline.
- Keep every supported feature profile explicit; default features are disabled.
- Treat public API removal as a breaking change outside scope unless authorized.
- Treat tool output as a candidate list, never as proof of dead code.
- Apply one logical removal at a time and verify it before continuing.
- Do not preserve temporary reports in the repository; record retained results
  in the relevant current-state document and report the complete diff at handoff.

## Baseline

Record the current commit, file/line counts, direct dependencies, and relevant
binary size. Run the smallest complete local baseline:

```sh
cargo fmt --all -- --check
cargo test --workspace --no-default-features --features cpu
cargo clippy --workspace --all-targets --no-default-features \
  --features cpu -- -D warnings
```

Repeat tests and Clippy for every affected profile. A missing platform or device
is external verification; a compile or test failure is a failed baseline.

## Candidate discovery

Use repository and compiler evidence already available:

- warnings from Cargo and Clippy;
- `rg` across all feature- and target-gated call sites;
- `cargo tree -e features` for dependency and feature ownership;
- manifest inspection for dependencies with no consumer;
- source-structure and removed-surface tests already in
  `crates/graph_horizon_engine/tests/family_agnostic.rs`.

Optional locally installed tools may supplement this evidence, but their
absence is not a reason to add them. Check macros, re-exports, build scripts,
tests, examples, target-specific modules, and public consumers before removal.

## Change loop

For each candidate:

1. state the responsibility or dependency believed unnecessary;
2. identify every consumer and supported feature/target;
3. remove the smallest complete unit rather than commenting it out;
4. run focused tests, then every affected profile gate;
5. retain the change only when behavior and structure remain valid;
6. otherwise restore only that candidate and record why it is load-bearing.

Do not use `cargo fix --allow-dirty` or broad formatter-driven rewrites as a
substitute for understanding the change. Do not remove a public item, numeric
fallback, resource owner, security check, or target-specific path solely because
the current host does not exercise it.

## Final verification

Re-run formatting, tests, warnings-denied Clippy, affected release builds, and
the source-structure test:

```sh
cargo test -p graph_horizon_engine --no-default-features --features cpu \
  --test family_agnostic source_structure -- --exact
git diff --check
```

Compare the final metrics with the baseline. The handoff reports removed and
retained candidates, before/after counts, all commands and results, external
verification limits, and the largest remaining source of complexity.
