<!--
This log preserves durable repository implementation decisions that are not
already expressed by a narrower contract. Temporary task status is excluded.
-->

# Implementation Decision Log

## Explicit CPU Feature For Focused Rust Tests

- Decision: run installer and documentation-contract tests with
  `--no-default-features --features cpu`.
- Reason: the workspace intentionally rejects every build without one selected backend.
- Reversible: remove the option only if the crates later define a default profile.

## Frontend Directory

- Decision: run frontend gates in `web/frontend` and document that exact path.
- Reason: this is the only tracked frontend; creating a duplicate path or symlink would add ambiguity.
- Reversible: update paths if the repository structure changes.

## Performance Stability Gate

- Decision: apply the 2.00% coefficient-of-variation limit to TTFT while retaining decode samples above that value for median regression checks.
- Reason: the approved reference tuple defines stability on TTFT and treats decode as a median guard.
- Reversible: repeat the baseline under a specification that also requires decode stability.

## Eight-Thousand-Token Candidate Gate

- Decision: reject the recorded candidate after a 3.418725% median 8K gain,
  below its binding 10.00% threshold, without running 16K or 28K.
- Reason: the investigation authorized one candidate and a mandatory early gate.
- Reversible: only through a newly authorized performance investigation.

## Single Internal Reserve Unit

- Decision: convert `vram_reserve_mib` once during validation and pass only byte-valued `Option<u64>` data to Vulkan, hybrid, and Metal planners.
- Reason: one checked conversion prevents competing units without changing public API or defaults.
- Reversible: technically yes, but reintroducing two internal units would violate the invariant.
