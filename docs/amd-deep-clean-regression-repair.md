# AMD deep-clean regression repair

This report is the persistent evidence registry for the correction of the final
backend/model-family cleanup on AMD Vulkan. The failed candidate remains
immutable in its detached qualification worktree. All experiments run from the
authoritative cleanup lineage on `perf/amd-deep-clean-regression-repair`.

## Qualification target

```text
CLEANUP_BASELINE_SHA = 6e514c1eaba02b8dcb6d7202bfd13fcee0317f36
FAILED_CLEANUP_CANDIDATE_SHA = 1bfe33753c8b7cfb21b6c254800a3970a1cdf5fd

DEVICE = AMD Radeon RX 6750 XT, PCI 1002:73df
AMD GPU FAMILY / ARCHITECTURE = Navi 22 / RDNA2 / gfx1031
DRIVER / VULKAN RUNTIME = Mesa RADV 26.0.3-1ubuntu1; device API 1.4.335;
                          loader 1.4.341

BLOCKING MODEL = Ministral-3-3B-Instruct-2512-Q4_K_M.gguf
BACKEND = standalone Vulkan
KV FORMAT = F16
PROMPT / CONTEXT = pinned 16-token teacher-forced prompt; context 4096
BASELINE RESULT = 50/50 fresh-process parity runs passed
FAILED CANDIDATE RESULT = original matrix failure; focused fresh builds fail
                          intermittently at varying teacher-forced steps
DELTA = correctness failure-rate difference; no qualified performance delta
CV / NOISE = not measured: qualification stopped before performance
```

The failed qualification report is authoritative. It did **not** report an AMD
performance regression: the performance phase was not run after the blocking
correctness failure. Latency, throughput delta, and CV therefore remain
unavailable until correctness is repaired. They must not be inferred from the
correctness reproducer.

## Original failure

The immutable candidate passed formatting, shell syntax, warning-denied Vulkan
and Vulkan-hybrid Clippy, and both complete workspace/all-target test profiles.
All six Q4_K_M artifacts matched the catalog byte counts and SHA-256 hashes. The
pinned llama.cpp oracle was
`13f2b28b098623391b1aacfd27995e1c8b7de9a9`.

The full authenticated model-family matrix stopped at its first physical
standalone Vulkan row:

```text
Q8 negative-artifact rows: 6 external verification
3B Instruct CPU F16: PASS, 16/16 IDs
3B Instruct CPU INT8: PASS, 16/16 IDs
3B Instruct Vulkan F16: FAILURE
terminal summary: pass=2 external_verification=6 failure=1 total=9
```

The runner did not preserve the first assertion text. A direct diagnostic then
passed once, but repeated fresh-process execution reproduced the blocker:

- candidate build 1: pass, pass, then failure at teacher-forced step 13;
- cleanup baseline: 50/50 passes of the identical tuple;
- candidate build 1 bookend: pass, then failure at step 3;
- clean candidate rebuild: pass, then failure at step 3.

Each recorded failure says that the oracle completion ID is absent from the
local top two. The changing step rules out one stable near-tie at a fixed token.
No GPU reset, hang, or `VK_ERROR_DEVICE_LOST` was recorded.

## Acquired qualification evidence

The following evidence remains valid and is not reopened without causal data:

- F16 and INT8 long-context numeric oracles passed at 2K, 8K, and 28K;
- 64-row and 32-row AMD-specialized boundaries passed;
- the 16-row portable fallback boundary passed;
- Q4_K prefill and Q6_K projection/logits physical CPU-oracle tests passed;
- GQA attention, RoPE, KV append, lifecycle, and failure cleanup tests passed;
- standalone Vulkan and Vulkan-hybrid workspace tests passed;
- AMD capability routing and generic Vulkan fallback policy tests passed;
- NVIDIA-only Matrix2 and Metal-only routes remained excluded;
- the device exposes subgroup 64, subgroup-size control, FP16, accelerated
  packed integer dot product, 1,024 workgroup invocations, and 64 KiB shared
  memory;
- AMD cooperative matrix is not exposed on this device;
- six absent Q8_0 negative artifacts remain external verification, not failures;
- Apple Metal qualification remains external and unavailable on this machine.

No known baseline-identical test failure was established by the original AMD
qualification. The current focused baseline stress has not reproduced the
candidate's teacher-forced failure.

## Workload classification

Passing workloads include CPU F16/INT8 teacher-forced parity, focused Vulkan
Q4_K/Q6_K kernel parity, F16/INT8 long-context attention at all tested row
boundaries, GQA decode tests, and portable fallback tests.

The only failing workload is the 3B Instruct standalone Vulkan teacher-forced
F16-KV row. It fails at varying decode steps in fresh processes. There are no
measured borderline or noisy performance rows because performance qualification
did not begin. No favorable or unfavorable cleanup performance movement was
measured. Focused prefill/decode numeric guards outside the real-model row are
unaffected.

The narrowest demonstrated property is therefore real-model, standalone AMD
Vulkan, Q4_K_M weights, F16 KV, after a 16-token prompt and during
teacher-forced decode. INT8 KV is the nearest required paired guard but has not
yet been stressed through this exact real-model protocol.

## Cleanup diff inventory

The cleanup range contains three commits:

| Commit | Classification | Production effect |
|---|---|---|
| `f563c3e` | test/docs only | Adds the architecture inventory. |
| `a2e12f3` | harness cleanup, comments, benchmark removal | Removes two redundant throughput segment fields/calculation and the standalone decode benchmark; rewrites invariant comments. |
| `1bfe337` | test/docs only | Classifies Metal rows unavailable off Darwin arm64 and adds its script test. |

The Vulkan diff changes comments only. It does not change a capability query,
routing predicate, pipeline, shader instruction, specialization constant,
workgroup size, dispatch size, descriptor binding, offset, scratch allocation,
command submission, barrier, or resource lifetime. All 48 release SPIR-V modules
are byte-identical between baseline and candidate.

Changed production blocks are classified as follows:

| Class | Files / change | Plausibility |
|---|---|---|
| model-family cleanup | tokenizer comment only | none |
| shared runtime cleanup | no semantic runtime change | none observed |
| Vulkan routing cleanup | comments only | none |
| AMD capability cleanup | comments only | none |
| resource/lifetime cleanup | no executable change | none |
| weight/representation cleanup | no executable change | none |
| Vulkan host/plumbing cleanup | comments only | none |
| AMD shader/kernel cleanup | INT8 shader comments only; SPIR-V identical | none for F16 blocker |
| generic Vulkan cleanup | comments only | none |
| harness/test/docs | throughput report field removal, benchmark deletion, docs and script tests | only executable cleanup delta |

## Suspect ranking

1. A latent Vulkan ordering or lifetime defect exposed by the changed host binary
   layout or timing. This fits identical SPIR-V, varying failure step, and
   candidate-only reproduction.
2. A whole-crate code-generation interaction from removing the throughput fields
   and calculations. These functions are outside the parity call graph, so this
   is expected to be an exposure mechanism rather than the underlying defect.
3. Build-cache contamination. A clean candidate rebuild reproduced the same
   failure rate, so this is rejected.
4. Changed AMD route, shader, capability state, or F16-KV representation. The
   source and SPIR-V diffs contradict these hypotheses.
5. The INT8 KV shader comment cleanup. The blocking row uses F16 KV and the
   emitted shader is identical, so this is rejected.

## Experiment registry

| ID | Suspected change | Ablation / comparison | Result | CPU/GPU attribution | Decision |
|---|---|---|---|---|---|
| E00 | Qualification fluke | Original direct retry | 1 pass after matrix failure | unresolved | insufficient; retry cannot qualify |
| E01 | Candidate-associated instability | Candidate fresh-process stress | failure on run 3 at step 13 | standalone Vulkan only | KEEP evidence |
| E02 | Baseline-identical instability | Baseline fresh-process stress | 50/50 pass | same host and GPU | REJECT as observed baseline behavior |
| E03 | Thermal/session drift | Candidate after baseline | failure on run 2 at step 3 | same-session bookend | REJECT |
| E04 | Stale target/cache | Candidate in new target directory | failure on run 2 at step 3 | fresh host binary and shaders | REJECT |
| E05 | Changed shader bytes | SHA-256 comparison of 48 SPIR-V modules | all byte-identical | excludes shader build delta | REJECT |

Performance mean, median, standard deviation, CV, GPU critical-path time, and
component timings are intentionally blank at this stage: they were not captured
by the failed report, and correctness isolation must finish before performance
qualification.

## Root cause

Not yet established. Production code must not be corrected until an ablation
identifies the execution invariant that is violated.

## Correction and requalification

Pending causality. The final section will record the minimal correction, why
each restored state/check is necessary, focused A/B/A evidence, immediate
neighboring and paired-KV guards, full correctness/model-family qualification,
performance guards, the corrected immutable SHA, and the remaining cleanup
delta justification.
