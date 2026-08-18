# Ablation Report

## Summary

The audit retained the measured Vulkan performance paths, their capability
gates, generic fallbacks, correctness tests, benchmark improvements, and the
reviewed validation register. It removed 23 phase diaries, restored the prior
long-context investigation, and replaced the 1,492-line rolling optimization
log with a compact final record. Rejected prototypes were already absent from
production source and remain absent.

## Metrics (before → after cleanup)

| Metric | Before | After | Delta |
|---|---:|---:|---:|
| CPU all-target build | pass | pass | unchanged |
| Compiler/Clippy warnings | 0 | 0 | unchanged |
| Rust source lines | 42,165 | 42,165 | 0 |
| Direct dependencies | unchanged | unchanged | 0 |
| Total dependency-tree entries | 177 | 177 | 0 |
| Net branch line delta vs `origin/main` | +19,054 | +4,382 | -14,672 |
| Added phase diaries | 23 | 0 | -23 |
| CPU workspace tests | 342 passed | 342 passed | unchanged |
| Release binary size | not measured | not measured | n/a |

## Retained Changes

### Vulkan execution

- Capability- and shape-gated Matrix2 and cooperative paths for Q4, Q6, and
  attention; every route preserves an existing generic fallback.
- Direct canonical-Q4 Matrix2 prefill, per-8 Q8/DP4A Q4 decode, vectorized GQA
  decode, native FP16 weight views, and request-local KV reuse. Each cleared its
  recorded performance and correctness gates.
- Focused CPU/Vulkan numeric tests, six-model teacher-forced qualification
  support, and overflow/resource checks around all new eligibility decisions.

### Simplicity and tooling

- Batched matmul, attention policy, pipeline capability, and weight-conversion
  responsibilities remain separated and satisfy the repository's productive
  line limits.
- Benchmark output includes medians and constructs parse errors lazily.
- Q4 predecode validation is total and overflow-safe; the installer no longer
  relies on an undeclared `dirname` executable.
- Validation evidence is consolidated in `CURRENT_OPTIMIZATION_STATE.md` and
  `VALIDATION.md`; detailed experiment chronology remains available in Git.

## Removed

- Sparse attention, low-precision weight, direct-Q6, oversized Matrix2,
  experimental tracing, and one-shot diagnostic source paths: none were found
  in the final production tree.
- Phase-by-phase Markdown diaries and the appended 2,756-line follow-up in the
  existing long-context report. They documented rejected or superseded steps
  and were not needed to operate, validate, or understand the retained design.

## Kept despite looking removable

- Capability and route policies stay separate because they protect different
  device, shape, resource, and numeric invariants.
- Numeric oracle and teacher-forced test support stays because retained FP16
  accumulation and quantized decode paths do not promise bit identity.
- `VALIDATION.md` is tracked because it is the reviewed support and qualification
  register referenced by the public documentation.

## Verification

- `cargo fmt --all -- --check`
- CPU workspace tests: 342 passed, 0 failed, 4 ignored
- Vulkan workspace tests: 341 passed, 0 failed, 7 ignored
- Vulkan-hybrid workspace tests: 417 passed, 0 failed, 6 ignored
- CPU and Vulkan Clippy with all targets and warnings denied
- Fresh 3B/128 release benchmark against `origin/main`: TTFT 1,356.07 →
  78.70 ms and decode 43.27 → 69.81 tok/s over three measured repetitions
- `git diff --check`
- Real-model and Metal qualification remain external because they require the
  authenticated artifacts or macOS hardware named by the validation process.
