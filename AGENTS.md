# AGENTS.md

## Purpose

This repository should be developed in a simple, direct style: small understandable programs, explicit invariants, few moving parts, and low accidental complexity.

Guided by **KISS** (*Keep It Simple, Stupid*) and the **Unix philosophy**: *"Do one thing and do it well."*

## Core Principles

- Make minimal, focused changes.
- Do not create new files or functions unless requested or clearly beneficial.
- Avoid micro-functions and overly utility-driven abstractions. Keep code clear, cohesive, and maintainable.
- Prefer idiomatic code and follow the project's established naming conventions and language-specific best practices.
- Keep code readable and make syntax choices consistent where possible.
- Prefer simple, direct, self-contained solutions.
- Optimize for understandability first; optimize performance only where it matters.
- Treat unnecessary complexity as a design bug.
- Cut or defer features that add disproportionate complexity.
- Favor designs that can be reasoned about locally.
- Keep interfaces narrow, predictable, and explicit.
- Add inline comments to clarify non-obvious logic, decisions, or trade-offs.
- Avoid duplicated or unnecessary code; keep the implementation as simple as the problem allows.

## Cybersecurity

- Write code with cybersecurity in mind.
- Treat all external data (user input, APIs, files) as untrusted; validate, parse, and sanitize strictly before processing.
- Avoid insecure defaults, unsafe input handling, data leaks, injection risks, and unnecessary exposure of sensitive information.
- Fail securely and ensure error messages or logs do not expose system internals, paths, or stack traces.
- Never commit hardcoded secrets, API keys, credentials, or production tokens.
- Minimize attack surface: do not introduce third-party dependencies unless absolutely necessary and thoroughly verified.

## AI Operating Mode

- Do not vibe code.
- Stay within explicit human and repository intent.
- Before changing code, understand existing constraints, invariants, and success criteria.
- Prefer small, well-understood changes over broad rewrites.
- Avoid new abstractions, helpers, frameworks, or dependencies unless they clearly reduce complexity.
- State assumptions, trade-offs, invariants, and risks when relevant.

### Phase 1: Structure Proposal
**Action:** Propose the software structure according to the following strict constraints:
- **File and Directory Tree:** Include only what is strictly necessary for the approved solution.
- **Self-Explanatory Names:** Use clear, direct nouns (e.g., `console`, `runtime`, `parser`). Do not use abstract or verb-based names.
- **Line Count Estimation:** For each file, provide an explicit estimation of productive lines of code, excluding tests and test fixtures (e.g., `parser.py (~120 lines, excluding tests)`).
- **200-Line Limit — orchestration code (the limit that earns its keep):** Code that *orchestrates* — dispatch, resource ownership, configuration/budget arithmetic, I/O, wiring of the model graph — is bound to **200 productive lines**. If the estimate exceeds 200, the file must be split into submodules/subfiles **before approval**, not during implementation. Tests and test fixtures never count toward this limit. **Keep this limit.** This is exactly the code where 200 lines does real work: it forces apart responsibilities that were only accidentally cohabiting (the god-file splits that produced `init/{device,bootstrap,caps}.rs`, `mem/{memory,budget}.rs`, and the `resident/ops_*` family), and the result is genuinely clearer than the file it came from.
- **No line limit — structurally irreducible files (categorie K and I):** Two kinds of file are irreducible *by nature* and carry **no line limit at all**. Splitting them does not separate responsibilities — it fragments a *single* responsibility, touching the file's own tests and visibility for zero benefit. For these the exemption is the **rule, declared up front** — not a concession argued file by file. Each such file *classifies itself* with exactly one inline comment `// AGENTS deroga K: <nota>` or `// AGENTS deroga I: <nota>`: the marker **declares the category**, and the note is a short reminder, not a justification to defend.
  - **Categoria K (dense numeric kernel):** a single dense numeric kernel, or a cohesive family of per-format variants of **one** operation (matmul / dequant / attention / elementwise), with **no** cross-operation dispatch, no I/O, and no resource ownership. A file mixing two operations (e.g. matmul + dequant + I/O) is **not** K — that is orchestration, and the 200-line limit applies.
  - **Categoria I (single irreducible language construct):** a single trait *definition*, or one `impl Trait` *block* of thin delegators (e.g. the ~25-method `impl Backend`), whose split would require super-traits or abstractions this document forbids.

  The category is **narrow and never a blanket**: K is the kernel of **one** operation, I is **one** trait or **one** `impl Trait` block of delegators. "For GPU files" or "for dispatch" is never a category — those are orchestration. The boundary is the same line the 200-line limit defends: a file that *dispatches, owns resources, or does I/O* is orchestration and must fit 200; a file that is *one kernel* or *one language construct* is K/I and has no limit.
- **Single Responsibility:** Each file must do one thing and one thing only. If its purpose cannot be described in a single sentence, it must be split.
- **Folders Only When Warranted:** Introduce a directory only when its domain genuinely splits into more than one file. A domain that fits in a single file stays a single file with a clear, domain-named filename — never a folder containing a lone `mod.rs`.
- **Split by folder, not by prefix:** when a file is split, group the resulting parts in a domain subfolder (`domain/{mod.rs, part.rs, …}`), never as prefix-named siblings (`domain.rs` + `domain_part.rs`). Prefix-named siblings sharing a stem are a smell that the parts are one domain — express it as a folder. A part belonging to a *different* existing domain joins that domain instead; a single distinct-named helper may stay a sibling in the existing domain folder — do not create a folder to isolate one file.

> **Workflow Boundary:** Wait for explicit approval or feedback on the proposed structure before moving to the implementation phase or writing any production code.

#### Deroga E: Autonomous Kernel Experiments

An agent may run an optimization experiment without per-candidate approval when
the work is limited to an existing category-K kernel, its directly related
tests, and the existing measurement path. Before the first production edit, it
must declare the experiment under `docs/performance-investigation-process.md`
and verify that `HEAD` is not on `main`. Otherwise it must create or switch to a
dedicated `perf/<experiment>` branch, or stop if that cannot be done safely.

This is the only exception to the Workflow Boundary. It authorizes autonomous
implementation, measurement, rollback, and local commits, but not new files,
dependencies, public APIs, orchestration changes, pushes, pull requests, or
merges. Each self-contained candidate must be committed once it reaches a
reproducible checkpoint, whether it is retained or rejected; broken intermediate
states need not be committed. Rejected candidates must be removed by a later
commit rather than by rewriting history; do not wait until the end of the search
to create one aggregate commit.

The final branch state must contain either the accepted candidate or the
restored baseline, and the report must identify the relevant commit revisions.
All correctness, security, scope, and performance gates remain in force.

### Phase 2: Implementation Phase
Once the structure is approved, ensure that every file adheres to the following checklist:
- [ ] **Initial architectural comment:** A clear multiline comment defining the file's exact purpose, main responsibilities, and system context.
- [ ] **Compliant implementation:** Stay strictly within the approved line limit estimation, counting only productive code and non-test comments.
- [ ] **Inline invariants:** Documented and explained at critical logical and state-mutation points.
- [ ] **Minimal tests:** Attached, implemented, or described contextually within or alongside the file block.

## Design Rules

- Start from the most direct working model.
- Use data representations that make invalid states hard to express.
- Use dedicated types, structures, interfaces, modules, and restricted visibility to protect invariants.
- Make ownership, mutation, side effects, dependencies, and error boundaries explicit.
- Prefer simple data flow and clear responsibility boundaries.
- Use shared mutable state, global state, concurrency primitives, async flows, or framework-specific mechanisms only when there is a real design need.
- Use abstractions, generics, metaprogramming, decorators, macros, or dynamic dispatch only when they simplify the model or express a real boundary.
- Prefer safe, idiomatic language features; use low-level, unsafe, reflective, or escape-hatch mechanisms only when necessary, narrow, documented, and backed by clear invariants.

## Performance

- Do not optimize blindly.
- First choose a design that is small, correct, and inspectable.
- During implementation and profiling, use the smallest supported model that exercises the affected path.
- Validate the other supported models only after the candidate is complete on the smallest applicable model.
- Validate a larger model earlier when the change directly depends on model size, capacity, layout, or memory pressure.
- Run larger-model performance benchmarks only when an approved acceptance criterion explicitly requires them.
- Optimize only after identifying a real bottleneck.
- Prefer structural performance improvements over micro-optimizations.
- Use copies, clones, and allocation avoidance intentionally.
- If a simple solution is fast enough, stop there.

## Change Policy

Before editing, identify:
- The invariant that must remain true;
- The smallest viable change;
- The main risk introduced.

After editing, verify whether the change reduced or increased conceptual complexity. If complexity increased, explain why it was unavoidable. If a feature request conflicts with simplicity, propose a reduced version first.

## Non-Goals

- No fashionable abstractions for their own sake.
- No premature generality.
- No unnecessary indirection.
- No broad rewrites without clear structural benefit.
- No code the AI cannot explain line by line.

---

Once changes are complete, update the documentation as needed.
