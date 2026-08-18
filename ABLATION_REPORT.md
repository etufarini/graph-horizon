# Production Refactor Report

## Baseline And Scope

- Cleanup branch: `perf/production-delta-cleanup`; code checkpoint before this
  report: `ec0c37c88559177f77f20ee47370dccd58e26983`.
- Pre-refactor source: `033996f22361331075feac901b0750a80136c868`; qualified
  optimization checkpoint: `99a2ea8a3b338dc17fdd1b7951485f8aa2c94aaa`.
- Local `main` and merge-base: `43f2da5437a150a9dbe6cd350b2e2e3186534b51`.
  Current upstream `origin/main` and merge-base:
  `bc0145dec73931cfc0bb572ad05420249c1ca853`.
- Audit reference: `origin/main`, because the local `main` predates the upstream
  merge already contained in the qualified branch.
- Hardware: NVIDIA RTX 3060 12 GiB, driver 595.84; pure Vulkan, full offload,
  F16 KV, one warm-up and three measured repetitions.

The final upstream delta contains 39 Rust files, including two dedicated test
files and the benchmark/build support, 16 compute shaders, six Markdown files,
one shell tool, and `.gitignore`. All 37 non-dedicated-test Rust files and all
16 shaders were reviewed against their active route, fallback, tests, and
performance purpose.

## Removed

- The superseded native-FP16 predecode representation: two Rust modules, one
  Matrix2 shader, native offsets/budget state, upload/loader branches, routing,
  and tests that existed only for that representation.
- Seven experiment environment controls covering attention, cooperative and
  Matrix2 matmul, Q4 metadata, GQA decode, and MMVQ decode. Qualified routing is
  now unconditional after its production capability/shape gates.
- The branch-added profiler path matrix, gap timers, maximum-gap state,
  per-layer detail, and the associated report/test machinery.
- Historical candidate comments, duplicate format/capability checks, and the
  quality test's obsolete fallback-generation mode.

This removed two experiment-only enums, seven persistent/summary fields, one
shader variant, three files, and the native conversion helpers. Tests removed
with them exclusively exercised the deleted controls or native representation;
numeric, fallback, resource, and route coverage remains.

## Simplified Retained Features

- **14B and canonical Q4_K Matrix2:** format is established once at the
  format-specific dispatch boundary; built-pipeline presence is the final
  Matrix2 capability signal. Canonical compressed weights feed the retained
  direct Q4 shader with no expanded native representation.
- **GQA decode:** the host keeps one exact 4:1 eligibility check, scratch bounds,
  device limits, split/reduce kernels, and the generic fallback. An attempted
  shader-index inlining was rejected after an initial long-decode signal and the
  qualified indexing was restored.
- **Resources:** persistent MMVQ/GQA scratch and the single serialized Vulkan
  request-session allocation remain because they eliminate qualified hot-path
  allocation. Ownership and failure cleanup remain explicit.
- **Other retained work:** Matrix2/cooperative/tiled attention, Q6 staged
  Matrix2, per-8 Q8/DP4A decode, batched RoPE, skipped intermediate logits,
  capability discovery, benchmark medians, and feature-gated profiling all map
  directly to active production behavior.

## Production Delta

| Comparison with `origin/main` | Files | Insertions | Deletions |
|---|---:|---:|---:|
| Before refactor (`033996f`) | 73 | 5,256 | 874 |
| After code cleanup (`ec0c37c`) | 62 | 4,536 | 876 |
| Cleanup itself | 27 | 123 | 845 |

No dependency or public API was added. Conceptual complexity decreased: one
execution representation, its policy surface, experiment switches, and detailed
profiling state disappeared; no replacement abstraction was introduced.

## Performance

The definitive long rows pair the final binary with the immutable pre-refactor
binary under the same late-session system state. This control was necessary
because both binaries had drifted about 1.2–1.5% from the earlier frozen long
measurements. Short rows use the original same-session baseline. CV is shown as
a fractional ratio.

| Workload / metric | Before (CV) | After (CV) | Delta |
|---|---:|---:|---:|
| 8B/128 TTFT | 150.70 ms (.0021) | 150.78 ms (.0057) | +0.05% |
| 8B/128 decode | 34.93 tok/s (.0009) | 34.91 tok/s (.0006) | -0.06% |
| 14B/128 TTFT | 250.48 ms (.0024) | 249.73 ms (.0043) | -0.30% |
| 14B/128 decode | 23.44 tok/s (.0019) | 23.63 tok/s (.0034) | +0.81% |
| 8B/28K TTFT | 52,238.30 ms (.0011) | 52,339.23 ms (.0024) | +0.19% |
| 8B/28K decode | 23.81 tok/s (.0014) | 23.82 tok/s (.0023) | +0.04% |
| 3B/28K TTFT | 29,157.76 ms (.0005) | 29,113.38 ms (.0002) | -0.15% |
| 3B/28K decode | 42.54 tok/s (.0007) | 42.30 tok/s (.0020) | -0.56% |

Every final comparison is within 1%; no qualified performance regression
remains.

## Quality And Validation

- Six-artifact 128-step teacher-forced top-two matrix: **PASS** for 3B, 8B,
  and 14B Instruct and Reasoning. One 3B-Reasoning step-28 miss did not reproduce
  in three immediate repeats; the subsequent complete six-artifact matrix passed.
- Exact/local oracles: CPU/Vulkan dense, RoPE, attention, Q4, Q6, Matrix2, and
  format-coverage tests passed. The pinned external llama.cpp server was absent,
  so external oracle comparison was not rerun.
- Root suite: 166 passed. Engine suites: CPU 159 passed / 2 ignored; Vulkan 155
  passed / 5 ignored; Vulkan-hybrid 228 passed / 4 ignored.
- Integration: CPU and Vulkan 5 passed / 1 ignored; Vulkan-hybrid 6 passed /
  1 ignored. Synthetic semantic suite: 12 passed / 1 real-model ignored.
- Real semantic no-retry campaign: 3B and 14B Reasoning qualified 9/9; 8B
  Reasoning passed the 4/4 critical and 8/9 semantic threshold but is
  **not-qualified** because S08 reached context with incomplete reasoning
  markers. The immutable pre-refactor source qualified 9/9 on a diagnostic
  comparison but followed substantially different sampled trajectories; this
  does not replace the current campaign's terminal state.
- Formatting, `git diff --check`, shell syntax, shader compilation, and the
  integration documentation/source contracts passed.
- CPU, Vulkan, and Vulkan-hybrid Clippy passed for the workspace and all targets
  with warnings denied. Metal remains external to this Linux host.

## Remaining Differences From Main

| Component | Why it remains |
|---|---|
| Matrix2 ABI, device bootstrap, pipeline caps/registry | Safe discovery and construction of retained NVIDIA Matrix2 paths with absence-based fallback |
| Q4/Q6 cooperative and Matrix2 matmul host/shaders | Retained canonical-Q4, staged-Q6, metadata, and 14B prefill execution |
| Tiled/Matrix2 prefill attention and routing | Qualified short/long prefill paths plus portable successive fallbacks |
| Per-8 Q8/DP4A MMVQ and packed Q6/logits shaders | Qualified decode throughput with format/device fallback |
| Vectorized GQA split/reduce and shared scratch | Qualified exact-4:1 decode path with generic attention fallback |
| Batched RoPE, prefill chunk/tail changes | Fewer dispatches and no overwritten intermediate logits |
| Session cache and fixed-resource accounting | Retained request allocation reuse with explicit failure ownership |
| Numeric/quality tests and capability predicates | Protect lossy numeric paths, supported shapes, and fallback boundaries |
| Benchmark median and documentation/tool updates | Reproducible performance decisions and maintained interfaces |

## Rejected Cleanup Candidates

| Candidate | Reason rejected |
|---|---|
| Inline GQA split/reduce index arithmetic | Optional hot-shader edit showed an initial >1% long-decode signal; qualified source was restored |
| Collapse capability and route policy | They protect distinct hardware-resource and runtime-shape invariants and are independently tested |
| Remove cooperative/tiled fallbacks | Required on capable devices without the narrower Matrix2 pipeline |
| Remove feature-gated profiler or baseline matmul trace | Profiler is an intentional supported build; the negligible-cost opt-in trace predates this branch |
| Run external llama.cpp or Metal checks locally | Required server and macOS hardware were unavailable; no substitute was used |

## Final Assessment

Every remaining production delta maps to a retained optimization, its
capability/shape/format routing, a portable fallback, resource ownership,
correctness coverage, or maintained benchmark/documentation support. No block
remains solely for investigation history. The worktree is intended to finish
clean on this unpushed local branch. The required deterministic quality and
performance gates pass; the separate stochastic 8B semantic qualification is
recorded as `not-qualified` above rather than being masked by a retry.
