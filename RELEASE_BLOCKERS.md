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
| RLS-001 | P0 | Root bootstrap used mutable obsolete source with no authentication. Root cause: legacy acquisition metadata. | Immutable, reproducible installation. | RESOLVED | Bootstrap now fetches named v0.1.0 release assets, verifies a strict SHA-256 record before extraction, and passes success, mismatch, unsafe-archive, delegation, and cleanup tests. |
| RLS-002 | P1 | Three fresh processes at RC `e09f6fd` passed the semantic threshold (9/9, 9/9, 8/9) but produced different bytes in six or more cases. S08 lengths were 330, 883, and 3,324 tokens. | Six-artifact support matrix. | RESOLVED — `NOT SUPPORTED` | The contract failed without retry. v0.1.0 must make no support claim for 8B Reasoning. The later parity-harness-only change is explicitly unrelated to generation and does not invalidate this exclusion evidence. |
| RLS-003 | P1 | Evidence was not tied to a frozen release state. Root cause: historical reports mixed qualification revisions. | Same-SHA model and backend claims. | RESOLVED | Inference evidence is pinned to `d1bf18f`; later changes are test/packaging/docs-only under the explicit reuse policy, and all affected final gates were rerun. |
| RLS-004 | P2 | Package metadata said `0.1.0`, but the CLI rejected `--version`. | Installed-version identification. | RESOLVED | Package-derived reporting is unit-tested and the fresh-prefix binary reports `graph-horizon 0.1.0`. |
| RLS-005 | P1 | Source/tag/artifact/install provenance was absent. Root cause: the release had not been assembled. | Source/artifact provenance and final installation. | RESOLVED | The final commit is ready for one annotated local tag; deterministic tag archive/checksum generation and fresh archive installation are qualified, with publication deliberately withheld. |
| RLS-006 | P1 | Sequential reuse, process repetition, real entry points, and failures lacked a release gate. | Serving and user-visible runtime contract. | RESOLVED | A/B/C sequential streaming, fresh processes, installed-binary inference, shutdown, missing/invalid GGUF, bad configuration, and CLI version/help all passed. |
| RLS-007 | P1 | The teacher-forced runner used the external GGUF chat template, which does not include Graph Horizon's release-owned Reasoning system prompt; Reasoning parity failed before logits. | Teacher-forced qualification. | RESOLVED | RC2 renders prompt IDs locally. Harness tests pass, and two fresh 8B runs matched the pinned oracle at all 16 top-1 steps. |

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
- Root-cause isolation for 8B Reasoning numerical variation.
- Performance work beyond catastrophic-regression smoke checks.

## Frozen 3B/14B Reasoning quality contract

This statistical rule was recorded before the final 3B and 14B repetitions.
Unlike the stricter 8B investigation gate, it does not claim fixed-seed byte
determinism.

- Run three independent fresh processes per artifact on the final unchanged
  engine/runtime code, using the same Vulkan all-GPU, F16 KV, context, seed, and
  sampling tuple as the 8B investigation.
- Every run must satisfy `critical=4/4`, `semantic>=8/9`, nine complete marker
  pairs, successful execution, and EOS for every case. No retry is allowed.
- Record per-case completion lengths and response hashes. Variation is allowed
  only if all three complete runs remain inside the declared semantic contract.
- Two independent teacher-forced runs per artifact must pass the 16-step pinned
  oracle top-two gate; a teacher-forced failure cannot be hidden by generation.
- Any incomplete generation, execution/marker failure, critical miss, or run
  below 8/9 yields `NOT SUPPORTED` for that artifact in v0.1.0.
