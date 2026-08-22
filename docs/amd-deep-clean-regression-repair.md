# AMD deep-clean regression repair

This report records the investigation and correction performed on the final
backend/model-family cleanup lineage. It deliberately separates the failure
that stopped AMD qualification from the performance regression assumed by the
repair mission.

## Verdict

No cleanup change caused an AMD performance regression. The failed
qualification stopped on an intermittent standalone Vulkan correctness failure
before the performance phase. The failed cleanup candidate subsequently
benchmarked within noise of the cleanup baseline, and a final untouched
baseline bookend reproduced the same correctness failure.

The investigation did uncover and repair the pre-existing defect that stopped
qualification: a one-shot barrier-elision hint was consumed by the first
dispatch inside the composite AMD Q4_K decode MMVQ operation. That removed the
required quantizer-write to MMVQ-read dependency on shared Q8 scratch. The
minimal correction inserts a buffer-scoped dependency for those scratch
windows while preserving the caller's trailing-barrier coalescing request.

```text
CLEANUP_BASELINE_SHA = 6e514c1eaba02b8dcb6d7202bfd13fcee0317f36
FAILED_CLEANUP_CANDIDATE_SHA = 1bfe33753c8b7cfb21b6c254800a3970a1cdf5fd
REPAIR_BASE_SHA = ce75b03af9dc822a89658078fa989faf8c0b6ad3
CORRECTED_CANDIDATE_SHA = e7edc8315d397f6eb34c5efb91a9ae20b9b59bc4

DEVICE = AMD Radeon RX 6750 XT, PCI 1002:73df
AMD GPU FAMILY / ARCHITECTURE = Navi 22 / RDNA2 / gfx1031
DRIVER / VULKAN RUNTIME = Mesa RADV 26.0.3-1ubuntu1; device API 1.4.335;
                          loader 1.4.341

BLOCKING MODEL = Ministral-3-3B-Instruct-2512-Q4_K_M.gguf
BACKEND = standalone Vulkan
KV FORMAT = F16
PROMPT / CONTEXT = pinned 16-token teacher-forced prompt / 4096
BASELINE RESULT = initially 50/50 pass; final bookend failed on run 4, step 3
FAILED CANDIDATE RESULT = intermittent failures at steps 13, 3, 3, and 12
DELTA = no performance regression; both revisions expose the same correctness bug
CV / NOISE = failed report had no performance data; focused failed-candidate
             benchmark was within 0.27% of baseline with low CV
```

The failed candidate remains unchanged in its detached qualification worktree.
All production experiments were performed on disposable worktrees before the
accepted correction was committed on `perf/amd-deep-clean-regression-repair`.
Nothing was pushed.

## Original failure and preserved evidence

The immutable failed candidate passed formatting, shell syntax, warning-denied
Vulkan and Vulkan-hybrid Clippy, both workspace/all-target test profiles, and
all six Q4_K_M artifact byte-count and SHA-256 checks. The pinned llama.cpp
oracle revision was `13f2b28b098623391b1aacfd27995e1c8b7de9a9`.

The authenticated matrix stopped at its first physical standalone Vulkan row:

```text
Q8 negative-artifact rows: 6 external verification
3B Instruct CPU F16: PASS, 16/16 IDs
3B Instruct CPU INT8: PASS, 16/16 IDs
3B Instruct Vulkan F16: FAILURE
summary: pass=2 external_verification=6 failure=1 total=9
```

The report therefore contained no AMD performance regression, component
timing, or CV. Those values cannot be inferred from the correctness failure.
The following acquired evidence remained valid and was not reopened during
causal isolation:

- F16 and INT8 long-context numeric oracles at 2K, 8K, and 28K;
- AMD-specialized 64-row and 32-row boundaries and the 16-row fallback;
- Q4_K prefill and Q6_K projection/logits CPU-oracle tests;
- GQA attention, RoPE, KV append, lifecycle, and failure-cleanup tests;
- standalone Vulkan and Vulkan-hybrid workspace suites;
- AMD capability routing and generic Vulkan fallback policy tests;
- exclusion of NVIDIA-only Matrix2 and Metal-only routes;
- six missing Q8_0 artifacts as external verification;
- Apple Metal physical qualification as external on this Linux host.

## Reproduction and workload classification

Repeated fresh-process execution established the following sequence:

| Revision / experiment | Result |
|---|---|
| failed candidate, original focused build | pass, pass, then fail at teacher-forced step 13 |
| cleanup baseline, initial stress | 50/50 pass |
| failed-candidate bookend | pass, then fail at step 3 |
| failed candidate, clean target rebuild | pass, then fail at step 3 |
| candidate with removed throughput fields restored | pass, then fail at step 12 |
| cleanup baseline, final bookend | pass three times, then fail at step 3 |
| corrected runtime | F16 50/50 pass; INT8 20/20 pass |

Every failure reported that the oracle token was absent from the local top two.
The varying step and final baseline reproduction reject a stable near-tie,
stale build artifact, and cleanup-only behavior. No run recorded a GPU reset,
hang, validation error, or `VK_ERROR_DEVICE_LOST`.

The narrow failing property was an AMD DP4A Q4_K single-token decode projection
after prefill. F16 KV was the blocking row, but KV representation was not the
causal component: both formats use the same projection composite, and the
paired INT8 guard passes after the correction. Prefill, long-context attention,
Q6_K, CPU, hybrid placement, and portable fallback paths were not implicated.

## Cleanup diff inventory

The cleanup range contains three commits:

| Commit | Classification | Executable effect |
|---|---|---|
| `f563c3e` | test/docs only | architecture inventory only |
| `a2e12f3` | harness cleanup and comments | removes two redundant throughput segment fields/calculations and the obsolete decode benchmark |
| `1bfe337` | test/docs only | classifies Metal parity unavailable off Darwin arm64 |

The Vulkan source delta is comments only. All 48 release SPIR-V modules are
byte-identical between baseline and failed candidate. The range changes no
capability query, routing predicate, pipeline, specialization constant,
workgroup or dispatch size, descriptor binding, offset, scratch allocation,
command submission, barrier, or resource lifetime.

| Requested classification | Cleanup delta | Causal assessment |
|---|---|---|
| model-family cleanup | tokenizer comment | none |
| shared runtime cleanup | no semantic runtime change | none |
| Vulkan routing cleanup | comments only | none |
| AMD capability cleanup | comments only | none |
| resource/lifetime cleanup | none | none |
| weight/representation cleanup | none | none |
| Vulkan host/plumbing cleanup | comments only | none |
| AMD shader/kernel cleanup | INT8 shader comments; identical SPIR-V | none |
| generic Vulkan cleanup | comments only | none |
| test/docs/harness | report-field deletion, obsolete example deletion, docs/tests | outside parity call graph |

The only executable cleanup deletion was restored in isolation and still
failed. Combined with the final baseline failure, this proves there is no exact
cleanup change to name as the cause.

## Suspect ranking and ablations

The initial ranking was: latent Vulkan ordering/lifetime defect exposed by
binary timing; whole-crate code-generation interaction from harness removal;
stale cache; changed route/shader/capability state; and INT8 comment cleanup.
Evidence retained only the first hypothesis, but the baseline bookend refined
it from “candidate-exposed” to “pre-existing in both revisions.”

| ID | Ablation / comparison | Blocking result | Performance / attribution | Decision |
|---|---|---|---|---|
| E00 | original matrix retry | one pass after failure | retry cannot qualify | retain failure evidence |
| E01 | failed-candidate fresh-process stress | fail run 3, step 13 | standalone Vulkan | retain |
| E02 | initial cleanup-baseline stress | 50/50 pass | same host/GPU | insufficient to prove absence |
| E03 | failed-candidate same-session bookend | fail run 2, step 3 | rejects thermal drift | reject drift |
| E04 | failed candidate in new target directory | fail run 2, step 3 | rejects stale cache | reject cache |
| E05 | compare 48 release SPIR-V hashes | all identical | excludes shader delta | reject shader change |
| E06 | restore removed throughput fields/calculations | fail run 2, step 12 | outside parity call graph | reject cleanup deletion |
| E07 | disable all Vulkan barrier elision | 30/30 pass | synchronization defect | keep causal region |
| E08 | preserve all composite MMVQ barriers | 50/50 pass | global barrier; prompt regression | correctness keep, production reject |
| E09 | apply preservation only to decode composite | 30/30 pass | proves blocking row is decode path | keep scope |
| E10 | buffer-scoped scratch barrier in decode and batch | 30/30 pass | prompt regression remains | reject broad scope |
| E11 | decode-only buffer-scoped scratch barrier | 50/50 F16, 20/20 INT8 | performance within guard | accept |
| E12 | final cleanup-baseline bookend | fail run 4, step 3 | proves pre-existing defect | keep conclusion |

### Performance falsification and rejected corrections

The failed cleanup candidate itself showed no performance regression on the
3B/128 F16 control:

| Build | Prompt tok/s | TTFT ms | Model decode tok/s |
|---|---:|---:|---:|
| cleanup baseline bookend average | 576.52 | 222.02 | 83.84 |
| failed cleanup candidate | 576.63 | 221.98 | 84.06 |
| delta | +0.02% | -0.02% | +0.26% |

The first correctness-preserving fix retained a global barrier between the
quantizer and MMVQ. It passed 50/50 but changed prompt throughput from a
575.80 tok/s baseline average to 550.88 tok/s (-4.33%) and TTFT from 222.30ms
to 232.36ms (+4.53%). A buffer-scoped barrier applied to both decode and batch
also regressed prompt throughput from 575.83 to 547.29 tok/s (-4.96%) and TTFT
from 222.29ms to 233.89ms (+5.22%). Both were rejected before full
qualification. This attributes their lost time to prefill command
synchronization, not shader execution.

## Runtime path and root cause

Temporary trace instrumentation on the exact blocking tuple recorded:

```text
prefill projection = AMD Q4_K MMQ, rows=16, in_dim=3072, out_dim=4096
prefill quant groups = 96; MMQ groups = 128x2
prefill attention = tiled, dispatch_rows=2, heads=32/8, head_dim=128
decode projection = AMD Q4_K MMVQ, in_dim=3072, out_dim=4096
decode quant groups = 6; MMVQ groups = 128
decode attention = AMD GQA wave32 split-reduce, heads=32/8, parts=8
scratch = qs 8 MiB + ds 8 MiB; cooperative matrix = unavailable
```

Baseline and failed candidate selected the same routes, shaders, dispatch
dimensions, scratch sizes, and capability decisions. The device reports
subgroup size 64, subgroup-size control, FP16, accelerated packed integer dot,
1,024 maximum workgroup invocations, and 64 KiB shared memory. AMD cooperative
matrix is not exposed. No cleanup capability field was merged or moved.

`Backend::no_barrier()` sets the one-shot `Device::skip_next_barrier` flag.
Projection orchestration uses it before independent Q and K projections so the
logical operation's trailing barrier can be coalesced. AMD Q4_K decode MMVQ is
not one dispatch, however:

```text
QuantAQ8F16 writes qs/ds scratch
        ↓ required compute-write → compute-read dependency
MatmulQ4KMmvqF16Out reads qs/ds scratch
```

Before the correction, `QuantAQ8F16` consumed the one-shot flag. The generic
recorder therefore omitted the internal dependency, and the final MMVQ dispatch
could read stale or partially written scratch. This corrupted Q/K projection
values nondeterministically and explains the varying teacher-forced failure
step. It is a GPU command-ordering defect, not host setup, memory migration,
capability routing, shader code, or a cleanup-induced cost change.

Fine-grained per-layer GPU timings were not needed or valid for a nonexistent
performance regression. The failed-candidate benchmark falsifies a lost-time
component, while barrier ablations and the physical oracle attribute the real
failure to the internal decode scratch dependency.

## Minimal correction

The production correction is limited to the Q4_K MMVQ decode composite:

1. take and clear the caller's pending trailing-barrier request before the
   internal quantizer dispatch;
2. suppress the quantizer recorder's global trailing memory barrier;
3. insert one compute-shader write-to-read `VkBufferMemoryBarrier` for each of
   the exact `qs` and `ds` scratch ranges;
4. restore the caller's request before the final MMVQ dispatch so it still
   applies to the logical projection's trailing edge.

Every piece is required. Capturing and restoring the flag preserves the public
one-shot invariant. The two buffer barriers establish visibility for both
scratch representations. Suppressing the global barrier avoids ordering
unrelated buffers and preserves prefill performance. No model name, product
name, device ID, context threshold, dependency, shader, or public API was added.

The batched MMQ path remains unchanged. E09 proved it was outside the blocking
decode path, its physical numeric tests and long prefill guards pass, and
applying the correction to it caused a reproducible roughly 5% prompt loss.
Changing it would violate the minimum-fix rule.

The existing physical `mmvq_q4k_matches_cpu_oracle` test now enters through the
real `Backend::no_barrier()` plus `Backend::matmul()` path. It therefore guards
the composite operation and the exact one-shot interaction instead of invoking
its two kernels manually.

## Focused A/B/A acceptance

On the committed corrected runtime, the 3B/128 F16 same-session bookend was:

| Build | Prompt tok/s | Prompt CV | TTFT ms | TTFT CV | Model decode tok/s | Decode CV |
|---|---:|---:|---:|---:|---:|---:|
| baseline B1 | 576.07 | 0.10% | 222.20 | 0.10% | 83.84 | 0.15% |
| corrected | 576.25 | 0.05% | 222.13 | 0.05% | 84.28 | 0.30% |
| baseline B2 | 576.25 | 0.11% | 222.12 | 0.11% | 84.06 | 0.30% |
| corrected vs baseline average | +0.02% | | -0.01% | | +0.39% | |

The corrected reproducer then passed 50/50 fresh F16 processes. The closest
paired-KV reproducer passed 20/20 INT8 processes. The physical MMVQ CPU oracle
also passed with the pending barrier-elision hint enabled.

## Full performance guards

Every row used the unmodified `examples/bench.rs`, exact `N - 3` copies of
`" a"`, 32 requested tokens, one warm-up, and three measured repetitions. The
first complete A/B set saw a -1.35% model-decode result on the 3B long INT8 row
with 1.77% candidate CV. Per the packet, exactly one complete five-row A/B
rerun was performed; no favorable row was rerun alone. The complete rerun is
the qualification result:

| Artifact / prompt / KV | Prompt tok/s B → C | Delta | TTFT ms B → C | Delta | Model decode tok/s B → C | Delta | Max primary CV |
|---|---:|---:|---:|---:|---:|---:|---:|
| 3B Instruct / 128 / F16 | 576.12 → 576.10 | -0.003% | 222.18 → 222.18 | +0.000% | 84.06 → 84.43 | +0.440% | 0.15% |
| 8B Instruct / 2,048 / F16 | 186.74 → 186.54 | -0.107% | 10,966.90 → 10,978.86 | +0.109% | 38.39 → 39.36 | +2.527% | 2.37% |
| 8B Instruct / 15,435 / F16 | 131.76 → 131.79 | +0.023% | 117,146.88 → 117,116.86 | -0.026% | 32.99 → 33.37 | +1.152% | 1.91% |
| 3B Instruct / 15,435 / INT8 | 130.36 → 130.45 | +0.069% | 118,406.42 → 118,324.43 | -0.069% | 62.79 → 63.37 | +0.924% | 1.23% |
| 14B Instruct / 128 / F16 | 136.62 → 136.21 | -0.300% | 936.90 → 939.71 | +0.300% | 23.95 → 23.95 | +0.000% | 0.69% |

Every mature regression metric is within 1%, every CV is below the packet's
5% noise ceiling, and every emitted `prompt_tokens` equals the required N.
There was no device loss, validation error, or unacknowledged fallback.

These rows provide the immediate neighbors requested by the repair mission:
short 3B control, 8B crossover, deep 8B long context, long paired INT8, and the
largest-family 14B projection/decode guard.

## Correctness, routing, and model-family requalification

All commands below ran against corrected runtime SHA `e7edc83`:

| Gate | Result |
|---|---|
| `cargo fmt --all -- --check`; shell syntax | pass |
| warning-denied workspace/all-target Clippy, `vulkan` | pass |
| warning-denied workspace/all-target Clippy, `vulkan-hybrid` | pass |
| workspace/all-target tests, `vulkan` | pass; main 178, engine 162, integration 5, semantic 12; zero failures |
| workspace/all-target tests, `vulkan-hybrid` | pass; main 178, engine 233, integration 6, semantic 12; zero failures |
| ignored long-context physical oracle | pass; F16/INT8 at 2K, 8K, 28K and 64/32/16 rows |
| fresh-process blocking F16 stress | 50/50 pass |
| fresh-process paired INT8 stress | 20/20 pass |
| authenticated model-family matrix | `pass=40 external_verification=34 failure=0 total=74` |
| semantic quality gate | `qualified=6 not_qualified=0 external_verification=0 total=6` |

The matrix passed all 36 main local CPU, standalone Vulkan, and mixed
Vulkan-hybrid rows for six artifacts and both KV formats, plus four all-GPU and
CPU-only endpoint controls. It covers Q4_K model execution, Q6_K
projection/logits tensors, F16/INT8 KV, GQA, generic and AMD routes,
multi-token generation, sequential lifecycle, placement, and exact oracle
parity. The semantic Reasoning rows each passed critical 4/4, semantic 9/9,
and reasoning format 9/9.

## Known pre-existing and external results

`KNOWN PRE-EXISTING FAILURE`: the intermittent MMVQ scratch-ordering failure
exists at both cleanup baseline and failed candidate. The correction removes it;
it is not evidence that the deep clean introduced a regression.

The six Q8_0 negative artifacts are absent and remain external verification.
All 28 Metal/Metal-hybrid matrix rows are external because the backend is
unavailable on Linux. These are the 34 external rows in the zero-failure
matrix, not corrected-candidate failures.

## Remaining cleanup-delta justification

All intended cleanup remains intact. The obsolete decode benchmark, redundant
throughput fields/calculations, comments, architecture inventory, and Metal
platform classification were not restored. The final runtime delta is one
buffer-scoped dependency in `mmvq.rs` plus one strengthened existing physical
test in `matmul/decode.rs`. Temporary tracing, ablation switches, counters, and
experimental environment variables were removed before qualification.

The qualified architecture is therefore unchanged: AMD DP4A Q4_K MMVQ decode,
AMD MMQ prefill, wave32 split-history GQA where eligible, generic Vulkan
fallbacks, shared model-family logic, and hybrid placement all remain in their
existing narrow routes.

## Final proof obligation

**Which exact cleanup change caused the AMD regression?** None. Diff inventory,
identical SPIR-V, restoration of the only executable harness deletion, neutral
failed-candidate benchmarks, and the final baseline failure jointly disprove
cleanup causality.

**Why did the blocking workload fail more readily?** It exercises the AMD Q4_K
single-token composite immediately under a caller barrier-elision hint. That
hint accidentally removed the internal quantizer-to-MMVQ scratch dependency;
surrounding CPU, prefill, Q6_K, and fallback routes do not share that exact
interaction.

**What is the smallest structural production correction?** Preserve the
logical operation's one-shot trailing hint and insert a compute-write to
compute-read dependency only on the two scratch buffers between the composite's
two dispatches.

**Why is every restored piece necessary?** The two scratch dependencies prevent
stale Q8 data, while capturing/restoring the one-shot hint retains projection
coalescing. Removing either side loses correctness or the measured performance
guard. No deleted cleanup surface was restored.

**Does the result preserve the cleanup and AMD/Vulkan architecture?** Yes. The
full matrix, semantic gate, physical numeric oracles, both profile suites,
focused stress, and all five performance guards pass on the immutable corrected
runtime.
