---
name: rust-code-ablation
description: Simplify ("ablate") a Rust codebase by removing dead code, cutting unused or heavy dependencies, trimming unused feature flags, and reducing complexity — while proving nothing breaks via the compiler and tests. Produces a simplified repo AND a report of what changed and why, with before/after metrics. Use this whenever the user wants to simplify, slim down, declutter, minimize, prune, shrink, "ablate", reduce dependencies of, remove dead code from, or lower the complexity of a Rust project (Cargo workspace or crate) — even if they only say "clean up this Rust repo" or "this crate has too much stuff" without naming a specific technique.
---

# Rust Code Ablation (Simplification)

Simplify a Rust codebase without changing its observable behavior. "Ablation" here means
**removing the inessential**: dead code, unused dependencies, unneeded feature flags, and
needless complexity. The deliverables are always two things: (1) the **modified repo** and
(2) a **report** documenting every removal and the before/after metrics.

## The one non-negotiable principle

**The build and the test suite are the safety net. They must stay green at every step.**

Rust makes this unusually safe: the compiler tells you when something is unused, and a
green `cargo test` after each change is strong evidence behavior is preserved. So the entire
method is: take small, reversible bites; re-verify after each; keep a record. Never remove
something you can't justify with either a tool's output or a compile/test result.

If `cargo build` or `cargo test` is already failing before you start, **stop and tell the
user.** Without a green baseline there is no safety net, and you cannot distinguish breakage
you caused from breakage that was already there.

## Workflow

Follow these phases in order. Don't skip the baseline.

### Phase 1 — Safe workspace + green baseline

1. Work on a dedicated branch (or a copy), never directly on the user's main branch:
   `git switch -c ablation/simplify` (confirm the tree is clean first with `git status`).
2. Confirm the baseline is green across the surfaces that matter:
   - `cargo build --all-targets` (lib, bins, tests, examples, benches)
   - `cargo test`
   - If the crate has meaningful features: also `cargo test --all-features` and
     `cargo test --no-default-features`. Feature-gated code is the #1 source of false
     "dead code" findings, so know which configurations are supposed to compile.
3. Capture baseline metrics by running the bundled script (see "Measuring" below). Save its
   output — you'll diff against it at the end.

### Phase 2 — Find opportunities (measure, don't guess)

Gather candidates from tools first. Each category below lists the tool, what it catches, and
its known false-positive traps. Collect the full candidate list before removing anything.

**Dead code (unused items inside the crate):**
- The compiler is the primary source: `cargo build --all-targets 2>&1 | grep -i warning`.
  Look for `dead_code`, `unused_imports`, `unused_variables`, `unused_mut`,
  `unreachable_pub`. Temporarily add `#![warn(dead_code, unused)]` at the crate root if
  warnings are being suppressed.
- Auto-fixable subset: `cargo fix --all-targets --allow-dirty --allow-staged` cleans unused
  imports, `mut`, and similar mechanical items safely. It does NOT delete dead functions/
  types — do those by hand.
- Trap: for a **library**, `pub` items are part of the public API and are *not* reported as
  dead even when nothing internal uses them. Removing a `pub` item is a **breaking change**.
  Do not remove public API unless the user explicitly confirms it's acceptable (or the crate
  is a leaf binary with no downstream consumers).

**Unused dependencies:**
- First pass: `cargo machete` (stable, fast). Flags crates declared in `Cargo.toml` but not
  referenced.
- Confirm with: `cargo +nightly udeps --all-targets` (more precise, needs the nightly
  toolchain). Use it to double-check before deleting, since both tools can err.
- Traps: a dep may be used only under a specific `cfg`/feature/target, only in
  `dev-dependencies` (tests/examples/benches), or only via a macro. Before deleting, grep the
  whole tree (`rg crate_name`) including across feature flags and `#[cfg(...)]` blocks.

**Unused / overweight features:**
- Inspect what features pull in: `cargo tree -e features` and `cargo tree --depth 1`.
- Many deps enable heavy defaults. Try `default-features = false` and re-enable only what's
  needed — this often removes whole subtrees of transitive deps. Re-verify build+test after
  each feature change.
- For binary size impact: `cargo bloat --release --crates` shows which dependencies dominate
  the binary, pointing at the highest-value removals.

**Complexity reduction:**
- `cargo clippy --all-targets -- -W clippy::complexity -W clippy::nursery` surfaces
  needless complexity (redundant clones, manual impls of std patterns, convoluted matches,
  etc.). `cargo clippy --fix --allow-dirty` applies the mechanical ones.
- Treat clippy as suggestions, not commands: apply changes that genuinely simplify; skip
  ones that trade readability for cleverness. Complexity edits are behavior-preserving
  refactors, so the test suite must still pass unchanged.

### Phase 3 — Apply incrementally, re-verify every time

- Make **one logical change per commit** (e.g. "remove unused dep `foo`", "delete dead module
  `legacy`"). Small commits make the report trivial to write and let you bisect if something
  breaks.
- After each change run, at minimum, `cargo build --all-targets && cargo test`. For changes
  touching feature-gated code, also run the relevant `--features` / `--no-default-features`
  configurations from Phase 1.
- If a removal turns something else red, revert that one change and note it in the report as
  "looked unused but is load-bearing (reason)". That's a finding, not a failure.
- Prefer deletion over commenting-out. Dead commented code is just more clutter; git history
  is the backup.

### Phase 4 — Re-measure and write the report

1. Re-run the measurement script and compute the deltas vs. the baseline.
2. Do a final full verification: `cargo build --all-targets`, `cargo test`,
   `cargo clippy --all-targets`, plus the feature configurations from Phase 1.
3. Write `ABLATION_REPORT.md` at the repo root using the template below.

## Measuring

Use the bundled `scripts/measure.sh` to capture comparable before/after numbers. Run it from
the repo root in Phase 1 (baseline) and again in Phase 4 (final):

```bash
bash scripts/measure.sh > /tmp/ablation_before.json   # Phase 1
bash scripts/measure.sh > /tmp/ablation_after.json    # Phase 4
```

It reports: build status, warning count, lines of Rust, direct dependency count, total
(transitive) dependency count, and release binary size when applicable. It degrades
gracefully if optional tools (`tokei`) are missing. Read its header comment for details.

If a metric the user cares about isn't covered (e.g. compile time, specific binary), measure
it explicitly and add it to the report.

## Rust-specific traps (read before deleting anything)

These cause most mistakes. Internalize them.

- **Conditional compilation.** Code under `#[cfg(feature = "x")]`, `#[cfg(test)]`,
  `#[cfg(target_os = "...")]` can look dead in your current configuration yet be essential in
  another. Check across the feature/target matrix before concluding something is unused.
  `cargo hack check --feature-powerset` (from `cargo-hack`) is the thorough way to verify
  feature combinations still compile.
- **Macros.** Names referenced only inside macro expansions can fool both dead-code analysis
  and `grep`. When a tool flags something that a macro might use, verify with a real build,
  not just search.
- **Public API = contract.** For libraries, removing or changing `pub` items breaks
  downstream users and is a semver-major change. Flag these to the user; don't silently
  remove them.
- **Tooling false positives.** `cargo machete` and `cargo udeps` both occasionally
  mis-report. Treat their output as candidates to confirm, never as a delete list to apply
  blindly.
- **Re-exports and the prelude.** An item may exist only to be re-exported (`pub use`); that's
  a real use even if nothing else in the crate touches it.

## Report template

Write this to `ABLATION_REPORT.md` at the repo root. Keep it factual and skimmable.

```markdown
# Ablation Report

## Summary
One paragraph: what was simplified and the headline result (e.g. "Removed 4 unused
dependencies and ~600 lines of dead code; build and full test suite remain green").

## Metrics (before → after)
| Metric                  | Before | After | Δ |
|-------------------------|--------|-------|---|
| Build (`--all-targets`) |        |       |   |
| Compiler warnings       |        |       |   |
| Lines of Rust           |        |       |   |
| Direct dependencies     |        |       |   |
| Total dependencies      |        |       |   |
| Release binary size     |        |       |   |
| Tests passing           |        |       |   |

## Changes
Group by category. For each item: what was removed and the evidence it was safe.
### Removed dependencies
- `crate_x` — flagged by machete + udeps, no references across any feature/target. (commit abc123)
### Removed dead code
- module `legacy` (412 LOC) — `dead_code` warnings, no callers. (commit def456)
### Feature trimming
- disabled default features of `serde`, re-enabled only `derive`. (commit ...)
### Complexity reductions
- simplified `parse_config` per clippy::complexity; behavior covered by existing tests. (commit ...)

## Kept despite looking unused (and why)
- `pub fn foo` — part of the public API; removal would be a breaking change.
- `helper()` — appears unused but is referenced via the `bar!` macro.

## Verification
Commands run and their results (build, test, clippy, feature configs).
```

## Quick checklist

- [ ] Clean tree, working on a dedicated branch
- [ ] Green baseline (build + test, incl. feature configs) confirmed and measured
- [ ] Candidates gathered from tools (dead code, deps, features, complexity)
- [ ] Each removal applied in its own commit, re-verified green
- [ ] Public-API removals flagged to the user, not done silently
- [ ] Final verification across the feature/target matrix
- [ ] `ABLATION_REPORT.md` written with before→after metrics and per-change evidence
