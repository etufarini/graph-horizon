<!--
This document owns repository contribution invariants, validation expectations,
and AGENTS line-limit exemptions. It does not define runtime support claims.
-->

# Contributing

Follow [AGENTS.md](AGENTS.md): keep changes small, state invariants and risks,
and add no dependency or abstraction without a concrete need.
AI development documents and generic skills listed there are protected from
implicit deletion, renaming, merging, or consolidation.

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

## Documentation Changes

Use [`docs/README.md`](docs/README.md) as the documentation map. Reserve
`README.md` for repository, crate, or multi-file domain entry points; give every
leaf page a descriptive kebab-case noun name. Keep user tasks, stable reference,
development process, current project status, and historical investigation
records in their existing domains.

Each contract has one canonical owner. Summaries may link to that owner but must
not duplicate tables merely to satisfy a test. When runtime options change,
update `docs/command-line/runtime-options.md`; `docs_contract` derives the
accepted flag names from `src/app/args.rs` and checks that reference directly.

After moving or editing Markdown, run:

```sh
cargo test -p graph_horizon_engine --no-default-features --features cpu \
  --test documentation docs_contract -- --exact
```

The test includes tracked and untracked Markdown so new pages cannot bypass the
local-link check before they are staged.

Orchestration stays under 200 productive lines. Every category-K file must be a
single-operation dense numeric kernel and declare exactly one
`// AGENTS deroga K: <nota>` marker. Every category-I file must contain only one
trait definition or one thin delegating `impl Trait` block and declare exactly
one `// AGENTS deroga I: <nota>` marker.

An explicit performance investigation runs autonomously on a dedicated
`perf/<experiment>` branch, never on `main`, and follows its declared baseline,
measurement, and correctness gates. Measurements may move the investigation
across category-K kernels, cache layout, orchestration, dispatch,
synchronization, shaders, allocations, and focused test or benchmark support.
No per-candidate approval is required for local, reversible, in-scope work.

Keep candidates isolated and reproducible, commit retained checkpoints locally,
remove rejected production changes without rewriting history, and record useful
negative results. New files and structural changes must satisfy the same
single-responsibility and line-limit rules as ordinary work. This autonomy does
not authorize pushes, pull requests, merges, external side effects, unrelated
scope, dependencies, or public API changes.

New backends implement the neutral runtime contracts; new model families do not
add backend-pair modules. Never reduce context, change placement, or retry a
different tuple to turn a failed validation into a pass. Do not commit secrets,
personal paths, generated oracle worktrees, or model artifacts.
