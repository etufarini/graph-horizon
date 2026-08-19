<!--
This report is the persistent v0.1.0 qualification checkpoint and, once the RC
is frozen, the reproducibility record. It owns release state and evidence
summaries; VALIDATION.md remains the authoritative support registry.
-->

# Ministral 3 v0.1.0 release qualification

## Current checkpoint

| Field | Value |
|---|---|
| Qualification date | 2026-08-19 |
| Initial main SHA | `a8b5a16d0c2197a7bcdc46a72546a71556aa5a2f` |
| Branch | `release/v0.1.0-qualification` |
| Current RC SHA | not frozen |
| 8B Reasoning | `UNQUALIFIED` pending frozen repetition contract |
| Installer | blocked: mutable and obsolete source URL |
| Current determination | qualification in progress |
| Next action | close packaging/version blockers, then qualify one frozen RC from scratch |

## Initial environment

- Linux x86_64, kernel `7.0.0-29-generic`.
- Intel Core i5-9600K, 6 cores.
- NVIDIA GeForce RTX 3060 12 GiB, driver 595.84, Vulkan 1.4.329.
- Rust 1.95.0, Cargo 1.95.0, Node.js 24.15.0, npm 11.12.1, GCC 15.2.0.
- Metal runtime hardware and macOS toolchain unavailable on this host.

## Authenticated model inventory

All six files under `/home/emanuele/Documenti/models` matched the byte counts
and SHA-256 values in `support/models.tsv` on 2026-08-19. The external paths are
not part of the release; qualification identity is the filename, byte count,
and digest recorded in `VALIDATION.md`.

## Initial-SHA baseline

- `cargo fmt --all -- --check`: PASS.
- CPU workspace tests: PASS; 166 binary tests, 159 engine tests, 5 integration
  tests, and 12 semantic-harness tests; four real-artifact tests remained
  intentionally ignored.
- CPU warnings-denied Clippy: PASS.
- Vulkan and Vulkan-hybrid workspace tests and warnings-denied Clippy: PASS.
- Frontend: 92 tests PASS; Svelte check 0 errors/0 warnings; production build
  PASS; npm audit reported zero known vulnerabilities.
- 8B Reasoning diagnostic run 1: 9/9 semantic, 4/4 critical, 9/9 complete
  markers, all EOS; this is diagnostic evidence, not final qualification.

Raw machine-local logs are retained under `target/release-qualification/` and
are not release inputs.

## Open qualification matrix

No cell is qualified until it is rerun on the frozen final RC.

| Model | CPU Linux x86_64 | Vulkan RTX 3060 | Vulkan-hybrid RTX 3060 | Metal | Final status |
|---|---|---|---|---|---|
| 3B Instruct | pending | pending | pending | UNAVAILABLE | UNQUALIFIED |
| 3B Reasoning | pending | pending | pending | UNAVAILABLE | UNQUALIFIED |
| 8B Instruct | pending | pending | pending | UNAVAILABLE | UNQUALIFIED |
| 8B Reasoning | pending | pending | pending | UNAVAILABLE | UNQUALIFIED |
| 14B Instruct | pending | pending | pending | UNAVAILABLE | UNQUALIFIED |
| 14B Reasoning | pending | pending | pending | UNAVAILABLE | UNQUALIFIED |

