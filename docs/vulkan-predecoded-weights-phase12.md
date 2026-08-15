<!--
This report owns Phase 12 evidence for persistent Vulkan prefill weights: the
quantized-matmul attribution bounds, representation economics, retained
gate/up implementation, qualification, and stop decision. It defines no model
serialization or public support contract.
-->

# Vulkan Predecoded Weights Phase 12

## Outcome

Phase 12 keeps a narrow, opt-in execution representation for the 52 Q4_K MLP
gate/up tensors. The GGUF bytes remain canonical at offset zero for decode and
fallback. At model load, pure Vulkan can append the same dequantized FP16
values that the old prefill shader reconstructed repeatedly; prefill then uses
direct Matrix2 tensor loads from that retained region.

The diagnostics-free 28K A/B/A result is 49,740.01 ms to 45,963.65 ms,
**-7.59%**. This clears the required 5% gate and misses the desirable 8% target
by 0.41 percentage points. Same-session profiling measures gate/up at
10,649.916 ms to 6,493.897 ms (-39.02%), total matmul at 22,660.709 ms to
18,577.191 ms (-18.02%), GPU prefill at 49,754.024 ms to 45,680.769 ms
(-8.19%), and attention at 26,166.727 ms to 26,131.123 ms (-0.14%).

The cost is intentionally visible rather than hidden: +2.742188 GiB persistent
VRAM, 5,452 to 8,260 MiB application VRAM at 28K, and +9.013 s measured model
initialization. The gain is 2.77 TTFT percentage points per added GiB. The
feature therefore remains explicit through
`GRAPH_HORIZON_PREFILL_PREDECODE_MLP=1`; default and hybrid behavior remain the
compressed path. Nothing changes the model file, public API, Q6/F16/INT8
paths, or M=1 decode routing.

The retained branch is `perf/vulkan-predecoded-weights-phase12`, based on clean
Phase 11 commit `3a05b01fee9d09449fbd90bb75c00b058f2d56b6`. No change was
pushed.

## Baseline, method, and invariants

- Phase 11 baseline: 49.746 s TTFT, 50.112 s GPU prefill, 26.344 s attention,
  22.831 s quantized matmul, and 0.937 s other GPU work at 28K.
- Fresh diagnostics-free control before edits: 49,714.10 ms TTFT, 151.72 ms
  standard deviation, 0.31% CV, 563.22 prompt tok/s, and 29.20 decode tok/s.
- Model: `Ministral-3-3B-Instruct-2512-Q4_K_M.gguf`, 2,147,023,008 bytes,
  SHA-256 `9ed150d4367e68df0ac8e1540f6ddc65b42d0ee26378329d1ecbca60f93fc5f8`.
- Device: RTX 3060 12 GiB (`0x2487`), driver 595.84, 7,501 MHz memory clock,
  170 W board limit.
- Runtime: release pure Vulkan, context 32,768, F16 KV, greedy sampling, exact
  synthetic prompts made from `" a"` repeated `tokens - 3` times.
- Diagnostics-free qualification uses independent processes. The mandatory
  128 and 28K points use baseline/candidate/baseline recheck. Long points use
  one warm-up and three measured repetitions; 128 uses two warm-ups and 15.
- Profiler results are a separate build and are never subtracted from public
  TTFT. The 28K macro result uses candidate-surrounding controls.

The invariant is byte-identical canonical Q4_K/Q6_K storage, the same per-value
Q4 dequantization formula and FP16 narrowing, FP16 activations/output, FP32
Matrix2 accumulation, exact causal attention, original decode kernels, and
unchanged tolerances. The smallest viable production change is a shape-gated
secondary view for `[3072, 9216]` Q4_K gate/up weights. The main risk is cache
and capacity pressure from expanded weights; the attention, short, decode, and
VRAM guards measure it directly.

## What can and cannot be attributed

The driver exposes timestamps and executable statistics, but no usable Vulkan
physical-byte, cache-hit, or per-pipeline instruction-sampling counters on this
device. Shader-visible traffic is heavily cacheable: treating it as physical
DRAM traffic would be false. Load, unpack, conversion, staging, MMA, and issue
also overlap in one kernel timestamp. The tables below therefore distinguish:

1. direct operation timestamps;
2. exact static SPIR-V sites and logical bytes;
3. a controlled “metadata is free” program delta; and
4. the full native gate/up causal delta.

No unmeasured physical-load seconds are invented. “Unresolved” means the named
stages share a timestamp, not that the work is absent.

### Current Q4/Q6 inventory

At 28K there are 438 M64 row workgroups across command batches. “Executed” is
one logical traversal per workgroup, not a DRAM counter.

| Component | Q4_K | Q6_K | Total |
|---|---:|---:|---:|
| Recurrent operation time, fresh same-session | 19.258 s | 3.399 s | 22.657 s |
| Minimum quant payload | about 1.309 GB | about 0.306 GB | 1.615 GB |
| Minimum metadata | about 0.164 GB | about 0.028 GB | 0.192 GB |
| Executed quant payload | about 573.177 GB | about 134.338 GB | 707.515 GB |
| Executed metadata | about 71.647 GB | about 12.595 GB | 84.242 GB |
| Static shifts, Q4/Q6 shader | 12 | 16 | format-specific sites |
| Static integer multiplies | 10 | 18 | format-specific sites |
| Static integer divisions | 3 | 4 | format-specific sites |
| Registers/thread | 98 | 108 | two resident WG for both |

The full seven-operation logical movement from Phase 11 is 6.138 TB in
22.831 s, 269 GB/s including activations and outputs. Its nominal 360 GB/s
logical-traffic floor is 17.05 s; the 169.467 TFLOP dense Matrix2 floor at
61.1 TFLOP/s is 2.77 s. These overlap and are not additive application floors.
The path is neither proven physical-bandwidth-bound nor dense-compute-bound.

### Per-operation causal decomposition

The following uses the fresh control surrounding the full prototype. The
Matrix2 column is the optimistic useful-FLOP/61.1-TFLOP/s floor, not measured
MMA-only time. The metadata column is an upper bound from a temporary shader in
which Q4 metadata loads, scale/min reconstruction, and their conversions were
made free while payload unpack and staging remained. Q6 metadata was not
ablated. Only gate/up have a full native measurement.

| Operation | Format | Total s/request | Physical load | Metadata upper | Unpack/dequant | Staging | Matrix2 floor | Other / unresolved |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Q | Q4 | 2.420 | unmeasured | 0.149 | unresolved | unresolved | 0.300 | 1.971 joint remainder |
| K | Q4 | 0.668 | unmeasured | 0.046 | unresolved | unresolved | 0.075 | 0.547 joint remainder |
| V | 13 Q4 + 13 Q6 | 0.707 | unmeasured | 0.013 Q4 only | unresolved | unresolved | 0.075 | 0.619 joint remainder |
| O | Q4 | 2.507 | unmeasured | 0.159 | unresolved | unresolved | 0.300 | 2.048 joint remainder |
| Gate | Q4 | 5.334 | included in native residual | 0.311 | 1.770 combined with staging | included | 0.675 | 2.578 native residual |
| Up | Q4 | 5.316 | included in native residual | 0.331 | 1.744 combined with staging | included | 0.675 | 2.566 native residual |
| Down | 13 Q4 + 13 Q6 | 5.709 | unmeasured | 0.146 Q4 only | unresolved | unresolved | 0.675 | 4.888 joint remainder |

The gate/up rows are an additive causal model: current total = metadata upper
+ net removable unpack/dequant/staging + optimistic Matrix2 floor + measured
native residual. The residual still contains expanded weight loads, tensor
load/MMA issue, output conversion/store, synchronization, cache behavior, and
the gap between attainable and theoretical Matrix2 throughput. Splitting it
further without counters would fabricate attribution.

The metadata-free probe removed about 1.153 s across all Q4 operation
timestamps, only 2.31% of the roughly 49.94 s request before secondary effects.
Its raw wall delta was larger, but simultaneous attention drift made that wall
number non-causal. Partial metadata predecode therefore cannot reach 5%.

### Regimes

| Operation class | Regime | Evidence |
|---|---|---|
| Gate/up Q4 | A: dequant/instruction/staging dominated | Native is 39.02% faster despite 3.555x logical weight bytes and unchanged modeled occupancy |
| Q/K/O Q4 | A, lower removable seconds | Same Q4 static path and low dense-compute floor; native form not prototyped |
| V/down mixed Q4/Q6 | A/issue mixed, not proven bandwidth-bound | Q6 has more decode sites; physical bytes unavailable and no native Q6 prototype |
| Dense Matrix2 itself | not C at current total | 2.77 s optimistic compute floor versus 22.831 s measured matmul |

No operation is classified B solely from shader-visible bytes; cache prevents
that inference. Gate/up empirically falsify B for the retained shape because
expanded traffic is faster.

## Current and predecoded floors

For gate/up, the dominant clean pair:

| Floor model | Runtime weight representation | Weight traffic | Runtime decode | Practical measured/modelled time |
|---|---|---:|---|---:|
| Current | Q4 payload + 16 B metadata / 256 values | 362.713 GB logical/request | unpack, scale/min, convert, shared weight staging | 10.650 s current; 10.008 s if metadata work were free |
| Predecoded | two-byte FP16 value, no per-tile metadata | 1,289.648 GB logical/request | none | 6.494 s measured |

The predecoded practical floor is not the 1.350 s dense-compute floor; direct
tensor loads and MMA still run at two-WG residency. Measured removable time is
4.156 s/request. Current compressed effective logical weight rate is 34.1
GB/s; native is 198.6 GB/s. The 3.555x byte penalty is outweighed by removed
decode, shared staging, barriers, and associated issue pressure.

## Representation models before implementation

The model contains 2,617,245,696 Q4 and 811,597,824 Q6 matrix values. Dual
full-FP16 execution storage would add 6.386719 GiB before runtime scratch/KV.

| Candidate | Representation and scope | Added persistent | Total weight VRAM / expansion | Predecode time | Predicted 28K gain | Decision |
|---|---|---:|---:|---:|---:|---|
| A | Full materialized FP16 for every quantized matrix, including tied embedding | 6.387 GiB | 8.378 GiB / 4.206x | about 21 s CPU, extrapolated | at most 17.8% ideal from 39% local; cache/capacity excluded | reject before full build: leaves no safe 28K headroom |
| B | FP16 Matrix2-tiled exact values for every quantized matrix | at least 6.387 GiB | at least 8.378 GiB / at least 4.206x | at least A plus tiling | no demonstrated gain beyond A | reject: direct tensor transpose already consumes row-major values without padding |
| C | Quant payload plus expanded/rearranged Q4 scales/mins | about 0.152 GiB all Q4; 0.086 GiB gate/up | about 2.144 GiB all-Q4 weights / 1.077x | low, not measured | less than 2.31% empirical upper | reject below GO gate |
| D | Full FP16 only for 26 gate + 26 up tensors | 2.742 GiB | 4.734 GiB / 2.377x | +9.013 s measured CPU load | 8.32% from profiled removable time; 7.59% achieved TTFT | **keep, explicit opt-in** |

Candidate B cannot preserve exact dequantized values in fewer than two bytes
per value without becoming another runtime numeric decode format. Every target
dimension is already a multiple of N32/K128, so a tiled permutation adds no
padding or traffic advantage. Candidate C received priority through the free-
metadata upper-bound experiment and failed before durable infrastructure.

Gate alone was the required one-group prototype: it adds 1.371094 GiB, reduces
the gate timestamp about 37.81%, and improves 28K TTFT 3.28%. That is below the
whole-request gate but predicts a qualifying extension to the shape-identical
up tensors. Gate+up then achieved the gate. The phase stops there rather than
materializing the rest of the model.

## Retained representation and layout

The retained execution path is:

```text
GGUF Q4_K bytes (canonical, offset 0)
        + eager CPU exact Q4 dequant and FP16 narrowing
same Vulkan allocation, native_offset = 15,925,248 bytes
        + row-major FP16 W[out, in]
direct tensor load through transposed view
        + Matrix2 B[K128, N32]
```

- Shape: `in=3072`, `out=9216`; each native tensor is 56,623,104 bytes.
- Layout: two-byte little-endian FP16, row-major `[out][in]`. Rows are output
  channels; columns are input channels.
- Alignment: the canonical offset 15,925,248 is 256-byte aligned. Native size
  and all N32/K128 tile boundaries are exact; alignment padding is zero.
- Workgroup: 256 threads own one M64 x N32 output tile. X traverses prompt rows,
  Y traverses output rows, and K advances by 128.
- Matrix2 mapping: activation is a direct M64 x K128 tensor load. A transposed
  tensor view maps persistent `W[out, in]` to a direct K128 x N32 B fragment.
  FP32 accumulation and the existing FP16 output/store order are retained.
- Lifetime: preprocessing occurs once in weight upload. Native bytes live in
  the parent weight allocation, are reused across requests, and are destroyed
  with that parent. No per-request conversion exists.
- Allocation economics: 52 existing allocations grow; buffer count and
  allocation count do not increase. Canonical bytes remain addressable for
  decode/fallback. There are no thousands of tile buffers and no fragmentation
  beyond normal Vulkan allocation requirements.
- Routing: pure Vulkan, Q4_K, exact `[3072, 9216]`, native offset present, and
  Matrix2 capability are all required. Hybrid and default runs retain the old
  path. `GRAPH_HORIZON_PREFILL_PREDECODE_GATE` and `_UP` remain narrow A/B
  controls; `_MLP=1` selects the qualified pair.

### Compiler and resource audit

Static optimized SPIR-V changes as follows:

| Static site | Q4 runtime shader | Native shader | Eliminated |
|---|---:|---:|---:|
| Loads | 106 | 44 | 62 |
| Stores | 70 | 24 | 46 |
| Access chains | 18 | 15 | 3 |
| Shifts | 12 | 0 | 12 |
| Bit operations | 15 | 0 | 15 |
| Integer multiplies | 10 | 3 | 7 |
| Integer divisions | 3 | 1 | 2 |
| Numeric conversions | 5 | 1 | 4 |
| Function calls | 11 | 0 | 11 |
| Subgroup shuffles | 2 | 0 | 2 |
| Barriers | 3 | 1 | 2 |

Both have one Matrix2 multiply-add and one cooperative store. Q4 uses an
activation tensor load plus shared cooperative weight load after decode;
native uses direct tensor loads for both A and W and declares 8,192 instead of
16,384 bytes shader shared memory.

A temporary `VK_KHR_pipeline_executable_properties` capture reports native at
99 registers/thread, 34,304 bytes driver shared/WG, zero stack, subgroup 32,
and 58,240-byte binary. GA106 limits give two resident 256-thread workgroups
by both registers and shared memory: 16 warps, 33.3% modeled occupancy. Q4 was
98 registers and 33,792 bytes driver shared/WG, also two WG. The gain is not an
occupancy claim; it is the removed instruction/staging path. The temporary
probe was removed.

## Performance qualification

Stable diagnostics-free prompt results:

| Context | Surrounding/control TTFT | Candidate TTFT | SD / CV candidate | Delta |
|---:|---:|---:|---:|---:|
| 128 | 115.88 ms | 97.63 ms | 0.35 ms / 0.36% | -15.75% |
| 512 | 433.55 ms | 365.19 ms | 0.66 ms / 0.18% | -15.77% |
| 2K | 1,829.46 ms | 1,556.09 ms | 1.25 ms / 0.08% | -14.94% |
| 8K | 8,962.15 ms | 7,921.39 ms | 24.56 ms / 0.31% | -11.61% |
| 16K | 21,941.91 ms | 19,766.83 ms | 27.68 ms / 0.14% | -9.91% |
| 28K | 49,740.01 ms | 45,963.65 ms | 32.02 ms / 0.07% | **-7.59%** |

The 28K controls were 49,705.80 ms (0.12% CV) and 49,774.22 ms (0.11% CV).
The short controls were 115.97 and 115.79 ms. Expanded weights improve rather
than sacrifice short prefill; the declining gain with context is consistent
with exact attention taking a larger share.

### Full macro reprofile

| Context / component | Baseline | Candidate | Delta |
|---|---:|---:|---:|
| 8K GPU prefill | 9,028.178 ms | 8,208.170 ms | -9.08% |
| 8K attention | 2,217.846 ms | 2,236.708 ms | +0.85% |
| 8K seven matmuls | 6,540.342 ms | 5,640.186 ms | -13.76% |
| 8K other | 269.990 ms | 331.276 ms | +61.286 ms |
| 28K GPU prefill | 49,754.024 ms | 45,680.769 ms | -8.19% |
| 28K attention | 26,166.727 ms | 26,131.123 ms | -0.14% |
| 28K seven matmuls | 22,660.709 ms | 18,577.191 ms | -18.02% |
| 28K other | 926.589 ms | 972.455 ms | +45.866 ms |

At 8K the untargeted rows moved together in the single profiler A/B sample;
their 61 ms increase is included in the end-to-end result and is not assigned
to a fabricated cache counter. At 28K the mandatory surrounding controls show
attention unchanged within noise. The candidate therefore passes the cache/
attention guard at the decision point.

## Decode and correctness guards

Decode remains on canonical compressed weights. Throughput excludes TTFT:

| Prompt history | Surrounding baseline | Candidate | Delta |
|---:|---:|---:|---:|
| 128 | 47.205 tok/s | 47.16 tok/s | -0.10% |
| 2K | 46.325 tok/s | 46.37 tok/s | +0.10% |
| 28K | 29.155 tok/s | 29.18 tok/s | +0.09% |

All are within the required 1% regression guard.

Correctness evidence is:

- CPU Q4 predecode matches the established CPU Q4 dequantizer after identical
  FP16 narrowing for every value in a synthetic block.
- The focused Matrix2 qualification runs compressed and native paths over the
  same Q4 weights and activations and requires raw FP16 output bytes to be
  exactly equal. It passes. Both paths have max relative error 0.000493355 and
  mean relative error 0.000194311 against the FP32 CPU oracle.
- The authenticated 128-token full-model greedy result remains exactly
  `2757,10637,2479,1636` with and without native gate/up.
- Q6, F16, INT8, logits, attention, and decode source paths are unchanged. The
  Vulkan-profile engine suite passes 154 tests with four intentional ignores,
  including CPU/Vulkan dense and attention parity and retained-format smoke.
- No tolerance, sampling, model-format, or public behavior change was made.

## Memory economics and preprocessing

### Persistent and peak memory

| Measure | Compressed | Gate/up native | Increase |
|---|---:|---:|---:|
| Model weight bytes | 2,138,615,808 | 5,083,017,216 | 2,944,401,408 |
| Model weight GiB | 1.991741 | 4.733929 | 2.742188 |
| Weight expansion | 1.000x | 2.376779x | — |
| 28K application VRAM | 5,452 MiB | 8,260 MiB | 2,808 MiB |
| Added buffers / allocations | 0 / 0 | 0 / 0 | none |
| Native alignment padding | 0 | 0 | none |

Canonical and native uploads are sequential, so the new per-tensor peak
staging requirement is the larger region, 56,623,104 bytes (54 MiB), not their
sum. The memory preflight now accounts persistent bytes by summing both and
staging by taking the larger upload. A 100 ms `nvidia-smi` load trace rose
monotonically to about 4,855 MiB application memory before request allocation;
the request then peaked at 8,258 MiB. The independently sampled 28K resident
request was 8,260 MiB. There is no hidden per-request native allocation.

Candidate A's 8.378 GiB of weights alone would put this 12 GiB device at the
28K capacity edge before a safe reserve and is rejected regardless of its
idealized speedup. Candidate D retains about 4 GiB application headroom at the
measured 28K point.

### CPU/GPU and eager/lazy economics

A temporary timer around `Engine::new` measured 1.841 s and 1.758 s for
surrounding compressed controls versus 10.813 s native. The incremental eager
CPU predecode/upload cost is therefore **9.013 s**. The conversion reads about
0.828 GB canonical gate/up storage, materializes 2.944 GB on the host, and
uploads 2.944 GB once. The timer was removed after capture.

A Vulkan conversion could read canonical blocks already on-device and write
2.944 GB native data, avoiding the expanded PCIe upload. Its memory-traffic
lower bound is tens of milliseconds rather than seconds, but it was not
measured and would require another exact Q4 GPU kernel, startup dispatch/
synchronization, and error lifecycle. CPU preprocessing is kept because model
load is one-time, runtime has qualified, and the simpler converter shares the
CPU oracle semantics. Optimizing startup does not justify that production
complexity in this phase.

Eager conversion is predictable and keeps conversion outside every request.
Lazy conversion has the same final memory, adds state and synchronization, and
would move up to 9.013 s into first-use latency; it is rejected. The benchmark
constructs the engine before starting TTFT, so steady-state TTFT never includes
predecode accidentally.

At the measured 28K saving of 3.77636 s, initialization amortizes after 2.39
28K requests, or about 66,830 prefetched tokens if the 28K rate is used only as
a linear economic approximation. Equivalent request counts are about 494 at
128, 132 at 512, 33 at 2K, 8.7 at 8K, and 4.1 at 16K. Persistent inference
servers are the intended economic regime.

## Experiment registry

| ID | Representation / target | VRAM delta | Target/matmul result | TTFT result | Correctness | Decision |
|---|---|---:|---:|---:|---|---|
| P12-META | Synthetic free Q4 metadata, all Q4 ops | 0 synthetic | 1.153 s Q4 timestamp upper | at most 2.31% attributable | unchanged arithmetic outside ablation | reject below gate; removed |
| P12-GATE | FP16 native gate only | +1.371 GiB | gate about -37.81% | 28K -3.28% | pass | do not keep alone; evidence supports identical up extension |
| P12-GATE-UP | FP16 native gate + up | +2.742 GiB | pair -39.02%; all matmul -18.02% | 28K -7.59% | bit-exact focused output and greedy pass | **keep opt-in** |
| P12-ALL | Full dual FP16 model | +6.387 GiB model | ideal at most about -39% matmul | ideal at most -17.8%, capacity/cache excluded | exact possible | reject before build on capacity/economics |
| P12-TILED | Full exact FP16 tile permutation | at least +6.387 GiB | no byte or instruction advantage over direct tensor view | no credible incremental 5% | exact possible | reject before build |

Rejected shaders, instrumentation, and temporary executable-statistics code
are absent from the final tree.

## Final decision and next boundary

**KEEP** candidate D as an explicit pure-Vulkan gate/up prefill option. It
satisfies stop condition 1: at least 5% whole-request improvement, correctness,
short-prefill, decode, attention, lifecycle, and memory cost are quantified.
The implementation adds one numeric conversion file and one single-operation
Matrix2 shader rather than a general execution-format framework.

The accepted 28K GPU critical path is now approximately 26.131 s attention,
18.577 s matmul, and 0.972 s other. Exact attention is the largest remaining
bottleneck at 57.2% of GPU prefill; Phase 10 already closed its local exact-
representation variants. Down is the largest remaining projection at 5.732 s,
but a gate/up-like 39% local gain would remove only about 2.24 s, below the 5%
whole-request gate by itself, and Q6 support would add another representation
and memory cost. It is not pursued after this phase's stop condition.

The next genuinely architectural step is either a compact native numeric
weight format that removes block reconstruction without FP16's byte expansion,
or an explicitly authorized change to attention representation (windowed,
sparse, or otherwise non-exact). The latter has the larger Amdahl target but
changes semantics; neither is implicitly authorized by Phase 12.

## Verification and repository state

- `cargo fmt --all --check`: pass.
- Workspace Clippy, all targets, warnings denied: pass for pure Vulkan and
  Vulkan-hybrid.
- Pure Vulkan and Vulkan-hybrid release checks: pass.
- Vulkan-profile engine library: 154 passed, four intentional ignores.
- Focused CPU predecode and raw compressed/native Matrix2 equality: pass.
- Full workspace run: 165 unrelated application tests passed and one
  installer-fixture test failed because its isolated PATH returned status 1
  before the expected missing-`npm` status 2. The same isolated test repeats
  the failure and does not touch Vulkan or this patch.
- `family_agnostic::docs_contract` cannot reach link validation because the
  repository checkout has no root `VALIDATION.md`; it fails first at that
  pre-existing documentation-register dependency.
- `git diff --check`: pass.

Production implementation and evidence are committed separately. The final
worktree is clean and nothing is pushed.
