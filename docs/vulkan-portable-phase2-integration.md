<!--
This report owns the portable Vulkan Phase 2 integration decision, current
qualification evidence, and the common NVIDIA/AMD specialization handoff.
-->

# Portable Vulkan Phase 2 Integration

## Baseline

| Item | Value |
|---|---|
| Date | 2026-08-20 Europe/Rome |
| Main before integration | `4bc6c7f07684ebc8f6bf7080a4ad77db6a9354a4` |
| Phase 2 source | `e5dbe8e8ec26c747a18be2a8e35b55da070efc6c` |
| Qualified source commits | `36a73a2`, `ea7ca1a` |
| Integration branch | `perf/vulkan-portable-phase2-integration` |
| Hardware-specialization baseline | **SELF: the commit containing this report** |

`SELF` is intentional: a Git commit cannot contain its own literal object ID.
Resolve the immutable baseline with `git rev-parse HEAD` while this report is at
`HEAD`; the release handoff records that resolved ID. NVIDIA and AMD must use
that same commit, not one of the source commits above.

The measured host is Linux x86_64 with an NVIDIA GeForce RTX 3060 12 GiB,
driver 595.84, Vulkan device API 1.4.329, and loader 1.4.341. Local `main`
exactly matched `origin/main`; no fetch, rebase, push, or remote mutation was
needed. The catalogued 3B and 8B artifacts matched both byte count and SHA-256.

## Retained production change

Matrix2-capable standalone Vulkan prefill request ownership changes from 256
to 512 rows. The invariant is unchanged canonical GGUF bytes, Q4_K/Q6_K
arithmetic, accumulation and output precision, exact causal attention, F16/INT8
KV layouts, decode, tile geometry, public interfaces, and fallbacks.

The complete production-change map is:

```text
PipelineRegistry contains direct Q4 Matrix2
AND PipelineRegistry contains exact Q64 Matrix2 attention
    -> standalone Vulkan prefill ownership = 512 rows
    -> existing graph batching records half as many full request batches
    -> existing Q4_K/Q6_K Matrix2 shape routing remains unchanged
    -> existing attention policy selects Q64 only for exact eligible rows
otherwise
    -> existing 256-row portable ownership and kernel fallbacks remain
```

The only production file that differs from `main` is
`crates/graph_horizon_engine/src/backend/vulkan/mod.rs`. It owns the row policy,
the two-pipeline conjunction, and its boundary test. No shader, kernel ABI,
allocation type, quantized representation, decode route, or dependency differs.
Vulkan-hybrid keeps its existing effective-placement row policy and is not
selected by this standalone-only constant.

## Routing qualification

All catalogued Q4_K_M models contain Q4_K and Q6_K projection tensors. On the
measured device both required pipelines built successfully, so actual request
ownership was 512 rows. Operation routing remained capability, format, and
shape based:

| Model | Operation | Shape (in -> out) | Quantization | Matrix2 | Actual operation path |
|---|---|---:|---|---|---|
| 3B | Q/O | 3072 -> 4096 | Q4_K | yes | direct Q4 Matrix2 |
| 3B | K/V | 3072 -> 1024 | Q4_K or Q6_K | yes | matching quantized Matrix2 |
| 3B | gate/up, down | 3072 -> 9216, 9216 -> 3072 | Q4_K or Q6_K | yes | matching quantized Matrix2 |
| 8B | Q/O, K/V | 4096 -> 4096, 4096 -> 1024 | Q4_K or Q6_K | yes | matching quantized Matrix2 |
| 8B | gate/up, down | 4096 -> 14336, 14336 -> 4096 | Q4_K or Q6_K | yes | matching quantized Matrix2 |
| 14B | Q/O, K/V | 5120 -> 4096, 5120 -> 1024 | Q4_K or Q6_K | yes | matching quantized Matrix2 |
| 14B | gate/up, down | 5120 -> 16384, 16384 -> 5120 | Q4_K or Q6_K | yes | matching quantized Matrix2 |
| any | incompatible format or shape | outside the exact predicates | other | no | established batched/cooperative/portable fallback |
| any | partial or absent required capability | any | Q4_K_M | no | 256-row ownership and established kernel fallback |

The 128-token performance row verifies below-boundary neutrality, 512 verifies
the exact ownership boundary, and 2K/8K/28K verify repeated full batches plus
tails. Attention tests passed 63/64/65, 127/128/129, and 255/256/257 rows;
nonmultiples use the established Q32 or generic fallback. Existing AMD watchdog
bounds remain 128 rows, or 64 for deep 16K+ graphs, before the 512-row branch.
No model name, GPU marketing name, or device ID was added to the new policy.

## Same-session performance

Graph Horizon values use one warm-up and three measured requests, exact
chat-tokenized repeated `" a"` prompts, context 32768, F16 KV, and 32 requested
tokens. Prefill values are milliseconds; decode values are tokens/second. For
8B/28K, `main` is the mean of A/B/A bookends (52010.81 and 52079.89 ms).

| Model/workload | Main mean (median) | Integrated mean (median) | Delta | CV main / integrated |
|---|---:|---:|---:|---:|
| 3B / 28K prefill | 28860.70 (28830.60) ms | 27344.01 (27380.15) ms | -5.26% | 0.21% / 0.27% |
| 8B / 128 prefill | 151.50 (151.28) ms | 151.04 (151.05) ms | -0.30% | 0.54% / 0.46% |
| 8B / 512 prefill | 572.16 (571.35) ms | 530.17 (529.86) ms | -7.34% | 0.25% / 0.31% |
| 8B / 2K prefill | 2354.39 (2354.34) ms | 2184.86 (2184.86) ms | -7.20% | 0.01% / 0.14% |
| 8B / 8K prefill | 10755.32 (10751.87) ms | 10015.93 (10013.96) ms | -6.87% | 0.24% / 0.28% |
| 8B / 28K prefill | 52045.35 (bookends 52036.97/52077.90) ms | 48726.50 (48713.41) ms | -6.38% | 0.10%/0.03% / 0.08% |
| 8B decode / KV128 | 34.52 (34.57) | 34.91 (34.93) | +1.13% | 0.29% / 0.12% |
| 8B decode / KV2K | 33.32 (33.36) | 33.51 (33.55) | +0.57% | 0.25% / 0.52% |
| 8B decode / KV28K | 23.70 (bookends 23.72/23.67) | 23.65 (23.65) | -0.19% | 0.16%/0.03% / 0.25% |

The primary improvement materially exceeds noise, generalizes to 3B, protects
short prefill, and leaves decode neutral. No important unaffected workload has
a reproducible regression above 1%.

The acquired Phase 2 result remains 52.044 -> 48.453 seconds (-6.90%) for
8B/28K and -5.55% for 3B/28K. Its matched llama.cpp reference is 24.579387
seconds, giving the historical 1.971x ratio and 6.667892 seconds remaining to
1.7x. Applying that unchanged reference to this integration run gives 1.982x
and 6.941542 seconds remaining; llama.cpp was not rerun for this integration.

## Correctness, quality, and validation

- Pure Vulkan all-targets: root 166 pass; engine 156 pass/5 ignored; family 5
  pass/1 ignored; semantic protocol 12 pass/1 ignored.
- Vulkan-hybrid all-targets: root 166 pass; engine 229 pass/4 ignored; family 6
  pass/1 ignored; semantic protocol 12 pass/1 ignored.
- CPU all-targets: root 166 pass; engine 159 pass/2 ignored; family 5 pass/1
  ignored; semantic protocol 12 pass/1 ignored.
- Q4_K and Q6_K: all current Matrix2 CPU-oracle families pass, including block
  tails, large-K, 8B gate/up/down, and 14B shapes. No tolerance changed.
- F16/INT8, Matrix2, GQA, and causal attention: Q64 boundaries and the ignored
  long-context numeric matrix through 28K pass; every reported INT8 comparison
  is exact.
- Known token: the authenticated 8B prompt still yields exactly
  `2757,10637,2479,1636`.
- Teacher-forced quality: all pure-Vulkan rows pass 16/16 against pinned
  llama.cpp `13f2b28b`. Across the 40 Linux-applicable main/endpoint rows, 37
  pass and three fixed 14B mixed rows are external for capacity; no numeric row
  fails. The four homogeneous hybrid endpoint sequences equal their standalone
  controls.
- Semantic quality: the current six-row runner reports `qualified=6`, with all
  Reasoning rows at 9/9 and 4/4 critical. The single current 8B pass does not
  replace the validation register's historical multi-run release status.
- Lifecycle: request cancellation/failpoints, KV allocation/release, cached
  prompt refill, sequential graph batching, and homogeneous/mixed placement
  tests pass in the all-target suites.
- Clippy: pure Vulkan and Vulkan-hybrid pass with warnings denied.
- Formatting/docs: rustfmt, rustdoc, `git diff --check`, docs contract, source
  structure, and removed-surface scans pass.
- Metal: externally unavailable by the repository's Linux platform contract.
  The monolithic 74-row script stops at its first Metal build guard instead of
  classifying it external; applicable rows were therefore run directly. Six Q8
  negative artifacts are also unavailable. Neither condition is a regression.

## Rejected architecture audit

Production source contains no Phase 2 candidate flag, benchmark-only route,
temporary profiler, or experimental dispatch. The rejected 1024-row request
ownership is absent. The pre-existing 1024-thread decode kernel is unrelated
and remains qualified. The Q6-to-F16 shaders and transient route module are
absent, as are rejected Q4 representation and attention-dataflow prototypes.
Historical documentation retains their negative evidence.

The production diff against `main` was reviewed line by line. Every changed
line is required by the 512-row ownership policy, its exact capability gate, or
its regression test. Conceptual complexity rises only by one constant and one
two-pipeline conjunction; no new moving part or public surface was introduced.

## Freeze and handoff

**PORTABLE VULKAN STRUCTURAL FLOOR REACHED** remains valid. No integration
result invalidates the Phase 2 screens. Further large gains belong to NVIDIA
and AMD hardware specialization, not another portable search.

```text
NVIDIA STARTING SHA: SELF (the commit containing this report)
AMD STARTING SHA:    SELF (the commit containing this report)

NVIDIA_PARENT_SHA = AMD_PARENT_SHA = HARDWARE_SPECIALIZATION_BASELINE_SHA
```

Both handoffs use portable Vulkan 8B/28K = 48.726500 seconds in this integration
session; historical llama.cpp = 24.579387 seconds; current inferred ratio =
1.982x; portable status = FLOOR REACHED; current equivalent remaining to 1.7x
= 6.941542 seconds (historical Phase 2 remainder 6.667892 seconds).

> Do not attempt to rediscover or undo the portable 512-row ownership optimization. Treat it as part of the portable baseline. Hardware-specific paths may override its execution architecture only when a measured vendor-specific fast path provides a material additional end-to-end benefit and preserves the portable fallback.

## PR description

**Portable Vulkan long-prefill structural consolidation**

Integrates the qualified Phase 2 policy that raises standalone prefill ownership
from 256 to 512 rows only when direct-Q4 Matrix2 and exact-Q64 Matrix2 attention
pipelines are both available. Same-session A/B/A shows a 6.38% 8B/28K TTFT
reduction and a 5.26% 3B/28K reduction, with short prefill and decode protected.
Current Q4/Q6, long-context attention, known-token, teacher-forced, semantic,
pure Vulkan, hybrid, CPU, lint, formatting, and documentation gates pass. The
portable structural floor remains reached; subsequent gap closure belongs to
vendor-specific NVIDIA and AMD branches from this exact common commit.
