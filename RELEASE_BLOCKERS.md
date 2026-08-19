<!--
This file is the temporary v0.1.0 release-blocker register. It records only
issues that affect the release contract, their evidence, and their closure
gates; qualification results and release notes belong in their own documents.
-->

# v0.1.0 release blockers

Initial main SHA: `a8b5a16d0c2197a7bcdc46a72546a71556aa5a2f`  
Release branch: `release/v0.1.0-qualification`

Severity meanings:

- P0: correctness, security, integrity, or reproducibility failure.
- P1: a claimed model/backend/configuration cannot yet satisfy the contract.
- P2: important installer, package, version, or documentation inconsistency.
- P3: explicit non-blocking limitation or post-release work.

| ID | Severity | Description and evidence | Release claim affected | Status | Closure / requalification gate |
|---|---|---|---|---|---|
| RLS-001 | P0 | Root bootstrap downloads mutable `main` from the obsolete `gh-zero-engine-ministral3` repository URL and performs no source checksum verification. | Immutable, reproducible installation. | OPEN | Pin acquisition to `v0.1.0`, use the current repository identity, publish the source checksum procedure, and pass bootstrap tests plus a clean install from the local tag. |
| RLS-002 | P1 | 8B Reasoning is authoritatively `not-qualified`; historical and later observations disagree. Baseline run 1 on the initial SHA passed 9/9, which is insufficient by itself. | Six-artifact support matrix. | OPEN | Apply the predeclared repetition contract below on the final RC. Otherwise mark the artifact `NOT SUPPORTED`. |
| RLS-003 | P1 | Current qualification evidence predates the release branch and does not cover one immutable final SHA across the complete model/backend/platform matrix. | Same-SHA model and backend claims. | OPEN | Freeze the final RC and rerun all applicable build, runtime, teacher-forced, generation, serving, and failure-path gates from a clean checkout. |
| RLS-004 | P2 | Package metadata says `0.1.0`, but the user-facing binary rejects `--version`. | Installed-version identification. | OPEN | Add a package-derived `--version`, test it, and verify it after clean installation. |
| RLS-005 | P1 | No immutable local `v0.1.0` tag, tag-built artifact checksum, or clean tag installation exists. | Source/artifact provenance and final installation. | OPEN | Create the final release commit, annotated local tag, build from that tag, record checksums, and complete post-tag installation smoke qualification. |
| RLS-006 | P1 | Sequential request reuse, clean-process repetition, real entry points, and obvious failure paths have not yet been qualified on the final RC. | Serving and user-visible runtime contract. | OPEN | Pass the documented sequential/clean-process/CLI/server test matrix with no intermittent failures. |

## Frozen 8B Reasoning acceptance contract

This rule was recorded after one diagnostic baseline run and before the final
qualification repetitions. It must not be relaxed after results are known.

- Target: the exact 8B Reasoning artifact in `support/models.tsv` on the final
  RC, Vulkan all-GPU on the recorded RTX 3060 host, F16 KV, context 4096,
  `temperature=0.7`, `seed=0`, `top_p=1`, `top_k=0`, `min_p=0`, and
  `repeat_penalty=1`.
- Execute three independent fresh-process runs of all nine cases, with no retry.
- Every run must have `critical=4/4`, `semantic>=8/9`, nine complete marker
  pairs, successful execution, and EOS termination for every case.
- Per-case prompt token count, completion token count, terminal reason, and
  captured user-visible output bytes must be identical across all three runs.
- Teacher-forced parity is a distinct required gate and cannot be replaced by
  generation success.
- Any engine error, incomplete generation, marker failure, semantic gate miss,
  or cross-run divergence yields final status `NOT SUPPORTED` for v0.1.0.

## P3 post-v0.1.0 work

- Runtime backend selection.
- Concurrent multi-request scheduling.
- Qualification on unavailable AMD and Metal hardware.
- Performance work beyond catastrophic-regression smoke checks.

