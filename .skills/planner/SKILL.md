---
name: planner
description: >-
  Design a software change as an implementation-ready Markdown specification
  without modifying code. Use when the user requests a proposal, plan, or design
  review that needs explicit requirements, decisions, milestones, and objective
  validation criteria before implementation.
---

<!--
This protocol owns interactive requirements gathering and approved Markdown
specification structure; it does not implement or verify repository changes.
-->

# Planner

Design an implementation-ready change without implementing it. Communicate and
write plan documents in English. Obey `AGENTS.md`, including its protection of
AI-development Markdown and generic skills: planning may update these files but
must never delete, rename, merge, replace, or consolidate them unless the user
requests the exact operation explicitly.

## Boundary

The planner may inspect the repository and write or update Markdown plans. It
does not modify source, tests, configuration, dependencies, or generated assets.
If a request combines planning and implementation, finish the requested plan and
hand implementation to the normal implementation workflow; do not reinterpret a
planning-only invocation as authorization to edit code.

## Right-size the Plan

Match detail to risk and ambiguity:

- **Small, obvious change:** a concise in-chat plan is enough. Persist it under
  `plans/` only when the user requests a durable specification or it materially
  improves handoff.
- **Standard change:** capture scope, invariants, affected files, risks, tasks,
  and validation in one document.
- **Large or multi-area change:** split the persisted plan into ordered milestone
  files with a `README.md` index.

Do not require separate approval after every phase. Ask only when an unresolved
choice would materially change behavior, public API, architecture, dependency
surface, ownership, or an irreversible/external action. Otherwise state the
safe assumption and continue.

## Required Content

A plan contains only the detail needed to implement safely:

1. problem and observable outcome;
2. in-scope and out-of-scope behavior;
3. invariants, external inputs, and concrete error handling;
4. minimal file tree with one-sentence responsibilities and productive-line
   estimates required by `AGENTS.md`;
5. ordered tasks and material dependencies;
6. objective acceptance checks and exact validation commands;
7. decisions that are important enough to prevent implementation drift;
8. risks or external verification that cannot be completed locally.

Use EARS (`WHEN ... THE SYSTEM SHALL ...`, `IF ... THEN ...`) only when it makes
a behavior or error contract clearer. Plain testable language is preferred for
simple criteria. Avoid vague terms such as "robust" or "handle gracefully".

Plans may name files, types, interfaces, commands, and short non-executable
signatures. They do not include paste-ready source, tests, diffs, or complete
configuration.

## Structure Rules

Persisted plans use the smallest warranted structure. A single plan stays one
file. Multiple milestones use `plans/<NN>-<slug>/{README.md,m0.md,...}`. Never
create an index plus a milestone file for a trivial plan merely to satisfy a
template.

Apply the repository's file rules directly:

- orchestration files are estimated at no more than 200 productive lines;
- category K and I exceptions are declared exactly as `AGENTS.md` requires;
- each file has one responsibility and a self-explanatory noun name;
- a domain becomes a folder only when it genuinely contains multiple files;
- split one domain into a folder, not prefix-named siblings;
- plan architectural comments only where `AGENTS.md` requires them.

## Completeness Gate

Before emitting the specification (or any updated version of it), the planner self-reviews it against this checklist — the planner's analogue of catching a defect before it ships. A spec defect is a future runtime block for the implementer, so this gate is where most implementer blocks are prevented. If any item fails, fix the spec before presenting it.

* **No open decisions.** Every point with more than one valid approach is resolved in "Decisions already made".
* **No vague language.** No "appropriate/reasonable/robust/handle gracefully"; every behavior and error is concrete and, where applicable, measurable.
* **Every relevant error point handled.** External inputs and failure paths have
  concrete handling.
* **Objective DoD on every task.** No task is "done" by judgement; each has a runnable check.
* **Full coverage / traceability.** Every acceptance criterion is covered by at least one task ("Satisfies"); no task implements behavior absent from the criteria.
* **Dependencies stated.** Every task lists its dependencies (or "none"); independents are ordered early.
* **Safety Floor respected.** No DoD or validation command requires a forbidden action; external-resource checks are labelled `external verification`.
* **Consistent terminology.** Domain terms are used consistently; ambiguous ones are defined in the glossary.
* **All plan documents are English.** Persisted plans and their indexes are
  written entirely in English.

State briefly in English that the gate passed, or what you fixed, when presenting the spec.

---

## Re-planning

When implementation reveals that the plan is incompatible with the repository,
update only the affected scope, structure, decisions, and checks. Preserve prior
plan history and all protected AI-development material. Re-run the completeness
gate, then return the revised plan to the implementation workflow.
