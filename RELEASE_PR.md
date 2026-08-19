<!--
This is the ready-to-paste review description for the v0.1.0 release branch.
It contains no publication authorization or mutable release state.
-->

# Purpose

Freeze and qualify the first immutable Graph Horizon Ministral 3 release.

# Release contract

- `0.1.0`, annotated tag `v0.1.0`.
- Supported tuple: Linux x86_64, RTX 3060, Vulkan-hybrid 100% all-GPU.
- Qualified: 3B/8B/14B Instruct and 3B/14B Reasoning Q4_K_M by SHA-256.
- 8B Reasoning: `NOT SUPPORTED`.

# Changes

- Added package-derived `--version` reporting.
- Corrected Reasoning teacher-forced prompt rendering and evidence capture.
- Pinned bootstrap acquisition to immutable v0.1.0 assets with SHA-256
  verification before extraction.
- Reconciled README, validation registry, qualification report, and notes.

# Qualification

3B and 14B Reasoning passed three fresh-process semantic and two 16-step oracle
runs each. Instruct artifacts passed current parity with preserved Plan 05
semantic evidence under the unrelated-change policy. The server passed A/B/C
streaming, reset, error-path, and shutdown smokes. Final build/test/lint,
frontend, tag, checksum, clean-install, version, and inference gates are in
`RELEASE_QUALIFICATION.md`.

# 8B Reasoning resolution

Three no-retry fixed-seed processes passed the semantic floor but emitted
different bytes and S08 lengths of 330, 883, and 3,324 tokens. Teacher-forced
and greedy controls passed. The fixed reproducibility contract failed.

# Limitations and follow-up

Serving is serialized and backend selection compile-time. Unlisted hardware,
placement, model, long-context, Q8, and MoE combinations are outside support.
Follow-up may investigate 8B variation and broaden qualification; scheduling,
runtime selection, packaging, and performance remain separate work.
