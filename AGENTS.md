# AGENTS.md

## Purpose

This repository should be developed in a simple, direct style: small understandable programs, explicit invariants, few moving parts, and low accidental complexity.

Guided by **KISS** (*Keep It Simple, Stupid*) and the **Unix philosophy**: *"Do one thing and do it well."*

## Core Principles

- Make minimal, focused changes.
- Do not create new files or functions unless requested or clearly beneficial.
- Avoid micro-functions and utility abstractions that protect no invariant.
- Prefer idiomatic, direct, self-contained code and established naming.
- Optimize for understandability first; optimize performance only where it matters.
- Treat unnecessary complexity as a design bug.
- Cut or defer features that add disproportionate complexity.
- Favor designs that can be reasoned about locally.
- Keep interfaces narrow, predictable, and explicit.
- Add inline comments to clarify non-obvious logic, decisions, or trade-offs.

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

### Protected AI Development Material

Markdown files used to instruct, coordinate, plan, review, or preserve context
for AI-assisted development are protected repository material.

Never delete, rename, merge, replace, or consolidate `AGENTS.md`, `CLAUDE.md`,
`GEMINI.md`, `.skills/**`, `plans/**`, `DECISIONS.md`, or another Markdown file
explicitly used by an AI development workflow unless the user explicitly
requests that exact operation.

Documentation cleanup must not classify these files as public-documentation
noise. Their wording may be simplified and contradictions may be corrected, but
their entry points, responsibilities, generic skills, references, and historical
development context must remain available.

### Phase 1: Structure Proposal
**Action:** When the user asks for a proposal, plan, or design review, propose the
software structure according to the following strict constraints:
- **File and Directory Tree:** Include only what is strictly necessary for the approved solution.
- **Self-Explanatory Names:** Use clear, direct nouns (e.g., `console`, `runtime`, `parser`). Do not use abstract or verb-based names.
- **Line Count Estimation:** For each file, provide an explicit estimation of productive lines of code, excluding tests and test fixtures (e.g., `parser.py (~120 lines, excluding tests)`).
- **200-Line Limit — orchestration code (the limit that earns its keep):** Code that *orchestrates* — dispatch, resource ownership, configuration/budget arithmetic, I/O, wiring of the model graph — is bound to **200 productive lines**. If the estimate exceeds 200, the file must be split into submodules/subfiles **before implementation**, not after the file has grown. Tests and test fixtures never count toward this limit. **Keep this limit.** This is exactly the code where 200 lines does real work: it forces apart responsibilities that were only accidentally cohabiting (the god-file splits that produced `init/{device,bootstrap,caps}.rs`, `mem/{memory,budget}.rs`, and the `resident/ops_*` family), and the result is genuinely clearer than the file it came from.
- **No line limit — structurally irreducible files (categorie K and I):** Two kinds of file are irreducible *by nature* and carry **no line limit at all**. Splitting them does not separate responsibilities — it fragments a *single* responsibility, touching the file's own tests and visibility for zero benefit. For these the exemption is the **rule, declared up front** — not a concession argued file by file. Each such file *classifies itself* with exactly one inline comment `// AGENTS deroga K: <nota>` or `// AGENTS deroga I: <nota>`: the marker **declares the category**, and the note is a short reminder, not a justification to defend.
  - **Categoria K (dense numeric kernel):** a single dense numeric kernel, or a cohesive family of per-format variants of **one** operation (matmul / dequant / attention / elementwise), with **no** cross-operation dispatch, no I/O, and no resource ownership. A file mixing two operations (e.g. matmul + dequant + I/O) is **not** K — that is orchestration, and the 200-line limit applies.
  - **Categoria I (single irreducible language construct):** a single trait *definition*, or one `impl Trait` *block* of thin delegators (e.g. the ~25-method `impl Backend`), whose split would require super-traits or abstractions this document forbids.

  The category is **narrow and never a blanket**: K is the kernel of **one** operation, I is **one** trait or **one** `impl Trait` block of delegators. "For GPU files" or "for dispatch" is never a category — those are orchestration. The boundary is the same line the 200-line limit defends: a file that *dispatches, owns resources, or does I/O* is orchestration and must fit 200; a file that is *one kernel* or *one language construct* is K/I and has no limit.
- **Single Responsibility:** Each file must do one thing and one thing only. If its purpose cannot be described in a single sentence, it must be split.
- **Folders Only When Warranted:** Introduce a directory only when its domain genuinely splits into more than one file. A domain that fits in a single file stays a single file with a clear, domain-named filename — never a folder containing a lone `mod.rs`.
- **Split by folder, not by prefix:** when a file is split, group the resulting parts in a domain subfolder (`domain/{mod.rs, part.rs, …}`), never as prefix-named siblings (`domain.rs` + `domain_part.rs`). Prefix-named siblings sharing a stem are a smell that the parts are one domain — express it as a folder. A part belonging to a *different* existing domain joins that domain instead; a single distinct-named helper may stay a sibling in the existing domain folder — do not create a folder to isolate one file.

> **Workflow Boundary:** Wait only when the user requested a structure proposal,
> plan, or design review without authorizing implementation. An explicit request
> to implement, fix, optimize, benchmark, investigate, or execute an existing
> specification is itself approval to enter Phase 2 within that request's scope.
> Do not ask for a duplicate approval, sign-off, or per-step confirmation.

If implementation reveals an in-scope structural change that was not knowable
up front, record the proposed tree and productive-line estimates before making
that change, then continue without waiting when it is local, reversible, and
verifiable. Ask for human intervention only when the choice expands the stated
scope, has an external or irreversible effect, or cannot be resolved safely by
inspection, tests, benchmarks, or a reversible experiment.

#### Autonomous Performance Investigations

An explicit performance-investigation or optimization request authorizes the
complete local cycle of baseline, instrumentation, profiling, hypothesis,
reversible experiment, correctness check, benchmark, retention or rollback,
and renewed profiling. This autonomy follows the measured bottleneck across
category-K kernels, KV/cache layout, orchestration, dispatch, command buffers,
barriers, synchronization, shaders, allocations, and test or benchmark support.
It is not limited to an existing kernel or measurement path.

The agent may add temporary instrumentation, test-only paths, focused benchmark
support, and production candidates that are necessary to test an in-scope
hypothesis. New files or structural changes remain subject to the repository's
single-responsibility and line-limit rules. Before the first production edit,
verify a clean worktree, capture the requested baseline, and work on a dedicated
`perf/<experiment>` branch rather than `main`.

Do not request approval between candidates. Let evidence select the next
experiment and continue until the task's stated completion or stop conditions
are met. Keep each candidate isolated and reproducible. Commit useful
checkpoints; remove rejected production changes without destructive history
rewrites, and retain concise negative results in the investigation record.

This autonomy does not authorize pushes, pull requests, merges, secrets,
external side effects, destructive operations, or scope unrelated to the
performance target. Dependencies and public API changes require explicit task
scope because they enlarge the durable attack surface. The final branch state
must contain only accepted changes or the restored baseline, and the report must
identify baseline, retained candidates, rejected candidates, correctness
results, before/after measurements, and the largest remaining bottleneck.

### Phase 2: Implementation Phase
Once implementation is explicitly requested or a proposed structure is
approved, ensure that every file adheres to the following checklist:
- [ ] **Architectural comment where needed:** Explain purpose and context for orchestration, resource ownership, unsafe code, K/I exemptions, and non-obvious security or format boundaries. Simple files should remain self-explanatory.
- [ ] **Compliant implementation:** Stay strictly within the declared line limit estimation, counting only productive code and non-test comments.
- [ ] **Inline invariants:** Documented and explained at critical logical and state-mutation points.
- [ ] **Minimal tests:** Attached, implemented, or described contextually within or alongside the file block.

## Design Rules

### Web UI visual style

- For every component and UI change under `web/frontend`, preserve the product's established pixel-art aesthetic. Reuse the existing design tokens, typography, icons, square borders, hard shadows, discrete spacing grid, and interaction patterns.
- Do not replace this visual language with generic SaaS styling, glassmorphism, soft gradients, diffuse shadows, excessive rounded corners, incompatible icon sets, or decorative effects.
- Controls inside panels, including close buttons, must use the same pixel-art component language, spacing grid, dimensions, alignment, accessible names, focus treatment, and interaction states.
- Any intentional departure from the established pixel-art style requires explicit user approval.

- Start from the most direct working model and make invalid states hard to express.
- Use dedicated types and restricted visibility only where they protect invariants.
- Make ownership, mutation, side effects, dependencies, and error boundaries explicit.
- Prefer simple data flow and clear responsibility boundaries.
- Use shared state, concurrency, generics, macros, or dynamic dispatch only for a concrete boundary.
- Prefer safe, idiomatic language features; use low-level, unsafe, reflective, or escape-hatch mechanisms only when necessary, narrow, documented, and backed by clear invariants.

## Performance

- Do not optimize blindly.
- First choose a design that is small, correct, and inspectable.
- During implementation and profiling, use the smallest supported model that exercises the affected path.
- Validate other models after the candidate is complete; use a larger model earlier only for size, capacity, layout, or memory behavior.
- Run larger-model performance benchmarks only when the task's acceptance
  criteria or the measured size/capacity behavior requires them.
- Optimize only after identifying a real bottleneck.
- Prefer structural performance improvements over micro-optimizations.
- Stop when the task's explicit completion conditions are satisfied; do not keep
  optimizing after the measured result is sufficient.

## Hardware-Neutral Documentation

- Do not add commercial GPU, accelerator, or SoC model names to current or
  public documentation, installation and release commands, qualification
  tables, or versioned benchmark artifacts unless the user explicitly requests
  that exact disclosure.
- Express support and qualification boundaries through operating system,
  architecture, backend capabilities, memory, driver or toolchain, model,
  context, and data-format requirements.
- Refer to machine-dependent results as a recorded validation environment. Keep
  exact device identity in private raw logs rather than tracked repository
  material.
- Before committing documentation or benchmark results, search the changed
  tracked files for newly introduced hardware product names.

## Change Policy

Before editing, identify:
- The invariant that must remain true;
- The smallest viable change;
- The main risk introduced.

After editing, verify whether the change reduced or increased conceptual complexity. If complexity increased, explain why it was unavoidable. If a feature request conflicts with simplicity, prefer the smallest version that still satisfies the explicit request; do not pause merely to propose a reduction when the requested implementation is already authorized.

## Non-Goals

- No fashionable abstractions for their own sake.
- No premature generality.
- No unnecessary indirection.
- No broad rewrites without clear structural benefit.
- No code the AI cannot explain line by line.

---

Once changes are complete, update the documentation as needed.
