---
name: deep-clean
description: >-
  Deeply clean and simplify an existing code change or explicitly requested
  repository scope without altering observable behavior. Use when the user asks
  for a thorough cleanup of dead code, duplication, stale comments, unnecessary
  indirection, or accidental complexity; use rust-code-ablation for exhaustive
  Rust workspace ablation.
---

<!--
This skill owns focused behavior-preserving cleanup inside the requested scope;
feature work, broad redesign, and speculative optimization remain separate.
-->

# Deep Clean

Always communicate in English. Keep commands, identifiers, and technical terms
in the project's language where appropriate.

Obey `AGENTS.md` and more specific repository instructions. Cleanup must reduce
conceptual complexity without changing behavior, public APIs, persistent
formats, dependencies, or architectural responsibilities.

## Scope

Use the scope specified by the user. If the request refers generally to the
current change, limit the work to the branch diff against its base and any
uncommitted changes. Do not extend a local cleanup to adjacent files or domains
merely because they could also be improved.

Before editing:

1. read the diff and the affected files;
2. identify the behavior and invariants that must remain unchanged;
3. find the tests and checks already used by the project;
4. separate demonstrable simplifications from design changes.

## Eligible candidates

- genuinely unused private code, after checking every consumer;
- duplication that can be replaced by an existing, more direct form;
- indirection layers, wrappers, or conversions that protect no invariant;
- obsolete, redundant, or contradictory comments;
- imports, variables, features, or local configuration with no remaining use;
- equivalent branches or repeated checks that can be unified without changing
  effect ordering, errors, or validation.

Do not remove or rewrite something merely because it appears unnecessary. Check
macros, re-exports, feature flags, target-specific paths, tests, examples, build
scripts, and external consumers before declaring it unused. A public API,
security validation, or path not exercised on the current machine is not dead
code for that reason alone.

## Cleanup loop

For each logical unit:

1. briefly state the complexity being removed;
2. apply the smallest complete diff, preferring deletion and direct code over
   new abstractions;
3. avoid unrelated formatting, renaming, or refactoring;
4. run the narrowest check first, then the gates applicable to the scope;
5. retain the candidate only when behavior and invariants remain proven;
6. if it fails, restore only that candidate and continue with independent ones.

Do not introduce dependencies, generic helpers, or structure "for the future."
Do not turn cleanup into performance optimization: if the benefit depends on
measurements, use an optimization skill and establish a baseline.

## Final verification

Run the project's existing formatter in check mode, tests, lint/type checks, and
build, scoped where possible. Also run `git diff --check` and reread the diff to
confirm that it contains only authorized simplifications.

The final report lists what was removed or simplified, every command and its
result, unavailable external verification, and candidates rejected because
their equivalence could not be proven.
