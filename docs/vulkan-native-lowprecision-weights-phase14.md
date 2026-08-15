<!--
This report owns Phase 14 evidence for native low-precision Vulkan execution
weights: capability discovery, representation screening, isolated prototypes,
numeric qualification, performance economics, and the final keep/stop decision.
It defines no GGUF serialization or public support contract.
-->

# Vulkan Native Low-Precision Weights Phase 14

## Status

Investigation complete on branch
`perf/vulkan-native-lowprecision-weights-phase14`, created from the required
clean Phase 13 commit `16dd03e`. The effective production baseline is Phase 12
commit `6cb170f`. The INT8 prototypes are rejected on quality, all production
source is restored to that baseline, and nothing has been pushed.

The canonical Q4_K/Q6_K model representation remains at offset zero. Phase 14
may append an opt-in execution representation only after the actual device,
Matrix2, compiler, tensor-load, performance, memory, and numerical gates pass.
Decode remains canonical compressed.

## Capability audit before production code

The qualification device is the NVIDIA GeForce RTX 3060 12 GiB (`0x2487`),
driver 595.84, Vulkan device API 1.4.329. The local loader is Vulkan 1.4.341;
`glslc` is shaderc 2026.1-1 with glslang 16.1.0-1.

`VK_NV_cooperative_matrix2` is advertised. All seven queried Matrix2 features
are true: workgroup scope, flexible dimensions, reductions, conversions,
per-element operations, tensor addressing, and block loads. Limits are 256
workgroup invocations, dimension 1,024, and 8,192 B driver-reserved shared
memory. The device also exposes storage-buffer 8-bit access and shader INT8.
It does **not** expose `VK_EXT_shader_float8`; `shaderFloat8` is false.

The live flexible-dimensions query returns 25 records: five type families at
subgroup scope and at workgroup sizes 32, 64, 128, and 256. The 256-invocation
records relevant to the retained M64/N32 design are:

| A / B | C / result | M granularity | N granularity | K granularity | Saturating |
|---|---|---:|---:|---:|---:|
| FP16 / FP16 | FP16 / FP16 | 32 | 32 | 16 | no |
| FP16 / FP16 | FP32 / FP32 | 32 | 32 | 16 | no |
| UINT8 / UINT8 | UINT32 / UINT32 | 32 | 32 | 32 | no |
| SINT8 / SINT8 | SINT32 / SINT32 | 32 | 32 | 32 | no |
| BF16 / BF16 | FP32 / FP32 | 32 | 32 | 16 | no |

The fifth record was called `other` and described as float8 in an older report.
The live raw value is `1000141000`, which current Vulkan headers define as
`VK_COMPONENT_TYPE_BFLOAT16_KHR`; it is **BF16, not float8**. There is no mixed
FP16-activation / INT8-weight Matrix2 record.

A compiler probe using `int8_t` storage buffers and Matrix2 tensor addressing
succeeds. Its SPIR-V has one-byte array stride, two direct
`OpCooperativeMatrixLoadTensorNV` operations, and a signed
`OpCooperativeMatrixMulAddKHR` into SINT32. This proves that INT8 storage does
not need unpacking or an intermediate FP16 tile. It does not remove the central
constraint: both operands must be INT8, so FP16 activations require runtime
quantization and the I32 result requires scale application.

### Supported execution types

| Input type | Matrix2 native | Direct tensor load | Per-tile FP16 conversion | Unpack | Accumulator | Usable conclusion |
|---|---|---|---|---|---|---|
| FP16 | yes | yes | no | no | FP16 or FP32 | retained exact control |
| BF16 | yes | compiler/device support present | FP16 A must change to BF16 | no | FP32 | two bytes/value and lower precision; no memory benefit |
| SINT8 | yes, only SINT8 x SINT8 | yes, compiler probe | no | no | SINT32 | viable only with activation quantization and output scales |
| UINT8 | yes, only UINT8 x UINT8 | yes by the same storage/tensor mechanism | no | no | UINT32 | signed activations need offsets and correction; dominated by SINT8 |
| FP8 E4M3/E5M2 | no | no | n/a | n/a | n/a | device extension, feature, and Matrix2 records absent |
| Mixed FP16 x 8-bit | no | no | n/a | n/a | n/a | no advertised Matrix2 type pair |

Tensor addressing directly consumes row-major storage with the existing
transposed B view. For the supported 256-thread path, M and N must be multiples
of 32 and K a multiple of 16 for FP16/BF16 or 32 for INT8. Gate/up dimensions
M64/N32/K128 satisfy both. Persistent allocation starts and descriptor offsets
must retain the device's 256-byte storage-buffer alignment; rows have no
padding because 3,072 and 9,216 are exact multiples of every relevant tile.

## Representation table before prototype code

Gate/up contain 1,472,200,704 execution values. Phase 12 FP16 therefore uses
2,944,401,408 B / 2.742188 GiB.

| Representation | Bytes/weight | Matrix2 direct | Runtime conversion | Scale metadata | Expected native gate/up | Expected weight traffic | Numerical risk | Decision before code |
|---|---:|---|---|---|---:|---:|---|---|
| FP16 row-major | 2 | yes, FP16 x FP16 -> FP32 | none | none | 2.742188 GiB | 1,289.648 GB/request logical | exact acquired narrowing | control |
| BF16 row-major | 2 | yes, BF16 x BF16 -> FP32 | FP16 activation to BF16 | none | 2.742188 GiB | about FP16 | extra activation and weight rounding with no residency gain | reject before prototype |
| FP8 E4M3/E5M2 | 1 | no | unsupported | format-dependent | unavailable | unavailable | high | reject: hardware path absent |
| INT8 weight-only feeding FP16 Matrix2 | 1 plus scales | no | unpack/convert each weight tile to FP16 | required | about 1.373 GiB | half persistent bytes, but FP16 staging returns | weight rounding | reject: recreates runtime conversion/staging |
| UINT8 asymmetric direct | 1 plus scales/zero points | yes only with UINT8 A | FP16 A quantization plus zero-point corrections | required | about 1.373 GiB | about half FP16 weights | activation, weight, and correction error | reject as dominated by signed symmetric INT8 |
| INT8 symmetric direct, per-output weight scale and per-row activation scale | 1 plus FP32 scales | yes, INT8 x INT8 -> I32 | one activation quantization and one I32-to-F32 scaled epilogue | 4 B/output channel/tensor plus one FP32/activation row | 1,474,117,632 B / 1.372879 GiB | about 644.824 GB weights plus under 1 GB logical scale reads | controlled weight and activation rounding | **single gate/up prototype** |

The direct INT8 candidate removes 49.93% of Phase 12 native residency. Weight
scales are per output channel because that is compatible with the persistent
row-major B layout and adds only 1,916,928 B across 52 tensors. One symmetric
activation scale per row permits a single I32 accumulator across the full K
dimension; per-K-block scales would require repeated I32-to-F32 conversion and
scaled partial accumulation inside the K loop. Per-tensor weight scale has less
metadata but needlessly enlarges error. These three granularities are therefore
screened analytically rather than brute-forced.

The prototype conversion contract is:

```text
canonical Q4_K -> established FP32 dequantized value
               -> scale = max(abs(row)) / 127
               -> round-to-nearest-even(value / scale)
               -> clamp to [-127, 127]
               -> persistent SINT8 payload + FP32 row scale
```

A zero row maps to zero payload and zero scale. Non-finite dequantized weights
must reject the opt-in representation rather than create an unusable native
buffer. No value may wrap on conversion. Activation quantization uses the same
nearest-even/clamp rule with one scale per prompt row; its finite-input behavior
and any non-finite guard must be explicit in the prototype.

Conversion is one-time CPU work at model initialization; inference performs no
weight conversion. Because the scale is defined by the finite row maximum,
finite weights cannot exceed the endpoint clamp; the largest magnitude maps to
`+/-127`. Subnormal and other sufficiently small finite values naturally round
to zero. The prototype rejected non-finite weights and mapped non-finite runtime
activations to zero before quantization, although authenticated-model evidence
never exercised either guard.

## Approved isolated prototype structure

The first production experiment is limited to Q4 gate/up and remains behind a
new explicit flag. The anticipated minimal touched tree and productive code
sizes, excluding tests and test fixtures, is:

```text
crates/graph_horizon_engine/src/backend/vulkan/
├── coopmat2.rs (~190 productive lines)
├── init/bootstrap.rs (~150 productive lines)
├── mem/
│   ├── buffers.rs (~190 productive lines)
│   ├── native.rs (~80 productive lines)
│   ├── predecode.rs (category K; no line limit)
│   └── weights.rs (~200 productive lines)
├── kernels/
│   ├── coopmat.rs (~155 productive lines)
│   └── matmul/prefill/mod.rs (~190 productive lines, excluding tests)
├── pipeline/kernel.rs (~175 productive lines)
└── shaders/
    ├── matmul/matmul_i8_matrix2_f16out.comp (category K; ~85 lines)
    └── quant/quant_a_i8_f16.comp (category K; ~65 lines)
```

If runtime wiring proves that activation quantization belongs at the graph
gate/up pair rather than inside generic matmul dispatch, that local structural
change will be recorded here before editing. No public API, dependency, model
serialization, decode path, Q/K/V/O/down route, or default behavior is in scope
for this first prototype.

## Acquired baseline

The Phase 13 final diagnostics-free 28K TTFT is 45,879.40 ms (45,878.72 ms
median), with 46,159.432 ms profiled GPU prefill: 26,477.321 ms attention,
6,554.180 ms native gate/up, 12,184.511 ms other recurrent matmul, and
943.420 ms other work. Canonical weights occupy 1.991741 GiB; native gate/up
adds 2.742188 GiB; 28K application VRAM is 8,260 MiB; incremental native
initialization is 9.013 s.

Fresh same-session baselines and all prototype evidence follow after the
capability-gated build.

### Fresh Phase 14 pre-code control

The artifact was re-authenticated before execution: 2,147,023,008 bytes,
SHA-256 `9ed150d4367e68df0ac8e1540f6ddc65b42d0ee26378329d1ecbca60f93fc5f8`.
The release pure-Vulkan build uses F16 KV, greedy sampling,
`GRAPH_HORIZON_PREFILL_PREDECODE_MLP=1`, and exact prompts made from `" a"`
repeated `tokens - 3` times.

| Context | Warm-up / reps | TTFT mean | SD | CV | Prompt tok/s | Decode tok/s |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 2 / 15 | 97.81 ms | 0.28 ms | 0.28% | 1,308.62 | 46.56 |
| 28K | 1 / 3 | **45,886.31 ms** | 48.43 ms | 0.11% | 610.20 | 29.15 |

The fresh 28K denominator is within 6.91 ms / 0.02% of the Phase 13 final
45,879.40 ms result. A qualifying further improvement therefore requires at
least 2,294.32 ms from this same-session control before the numerical and
memory gates are considered.

## Prototype P14-I8-GATE-UP

The first direct INT8 prototype passes the existing focused CPU-oracle matmul
gate without a tolerance change: max absolute error 17.71875, mean absolute
error 3.84602, max relative error 0.00191465, and mean relative error
0.00037214 on the repository's established random Q4/activation case. Absolute
outputs in this synthetic case are large; the existing combined absolute-or-
relative assertion passes every value.

Diagnostics-free A/B/A at 28K is 45,886.31 / 44,145.20 / 46,001.88 ms. The
surrounding FP16-native control is 45,944.10 ms, so INT8 gate/up removes
1,798.90 ms / **3.92%** with 0.26% candidate CV. At 128, the initial control
and candidate are 97.81 and 89.78 ms (-8.21%); decode moves from 46.56 to
46.81 tok/s.

The four-request profiler comparison (one warm-up plus three measurements;
totals divided by four) is:

| 28K component | FP16 gate/up | INT8 gate/up | Delta |
|---|---:|---:|---:|
| GPU prefill | 46,092.984 ms | 44,062.249 ms | -2,030.735 ms / -4.41% |
| Attention | 26,398.985 ms | 26,206.616 ms | -192.369 ms / -0.73% |
| Gate | 3,279.740 ms | 2,398.697 ms | -881.043 ms / -26.86% |
| Up | 3,275.138 ms | 2,380.029 ms | -895.110 ms / -27.33% |
| INT8 activation quantization | 0 | 69.421 ms | +69.421 ms |
| Down | 5,781.402 ms | 5,729.878 ms | -51.524 ms / -0.89% |

The candidate is faster than FP16 and uses 1.372879 instead of 2.742188 GiB
native gate/up memory (-49.93%), satisfying prototype GO condition A. It is a
Pareto replacement but does not alone satisfy the Phase-wide 5% further TTFT
target, so it is not yet a final KEEP.

### Pre-code down extension gate

Fresh down time remains 5.729878 s. Phase 13 measured exact FP16 down at
3.313502 s. Applying the measured gate/up INT8/FP16 local ratio (0.729) gives
about 2.416 s Matrix2 down; scaling the measured activation-quantization cost
from K=3072 to K=9216 adds about 0.104 s. The predicted INT8 down total is
therefore about 2.52 s, removing about 3.21 s from the new request and clearing
the required 5% marginal whole-request coding gate.

Down adds 736,100,352 INT8 values and 319,488 B of FP32 per-output scales:
736,419,840 B / 0.685844 GiB. Combined gate/up+down native residency is
2.058723 GiB, 75.08% of Phase 12 gate/up FP16 and 0.683464 GiB smaller. The
extension touches no new files: Q6 joins the existing category-K Q4 execution
converter family in `mem/predecode.rs`; `mem/native.rs` widens only target and
shape policy; the existing INT8 shader already accepts the transposed down
shape. These files remain within their recorded category/line estimates.

### Structural correction before the down candidate

The gate/up prototype made two orchestration files exceed the mandatory
200-productive-line ceiling before tests (`mem/weights.rs`: 237 lines;
`matmul/prefill/mod.rs`: 223 lines). Before extending either responsibility,
the accepted structure is corrected as follows:

```text
mem/weights/
├── mod.rs       (~130 productive lines; selected-set assembly and ownership)
└── tensor.rs    (~130 productive lines; one-tensor conversion and upload)
kernels/matmul/prefill/
├── mod.rs       (~35 productive lines; exports and tests)
├── policy.rs    (existing policy)
├── q4.rs        (~145 productive lines; Q4_K route)
└── q6.rs        (~110 productive lines; Q6_K route)
```

Each file has one sentence-worth of responsibility, the existing `weights`
and `prefill` directories contain multiple warranted parts, and no public API
or new dependency results. The down-candidate additions then remain within
these estimates.

## Final prototype results

### Layout and shader contract

Both prototypes stored weights row-major as `W[out][in]`, without padding.
Matrix2 loaded a M64 x K128 activation tile and a transposed K128 x N32 view of
the row-major weight tile directly from one-byte storage. One 256-thread
workgroup owned M64 x N32 output values, accumulated SINT32 across the complete
K dimension, multiplied by one FP32 activation-row and output-row weight scale,
and narrowed once to FP16. The separate quantizer used one workgroup per input
row. Payload and scale descriptor offsets were 256-byte aligned.

`glslc -S` emitted two `OpCooperativeMatrixLoadTensorNV`, one
`OpCooperativeMatrixMulAddKHR`, and one workgroup barrier in the 354-op INT8
matmul module. The 335-op activation quantizer has two barriers. Explicit
shared memory is 8,192 B/WG for the matmul SINT32 result and 1,024 B/WG for
quantization; Matrix2 adds the device's 8,192 B reserved allocation. This gives
the matmul a shared-memory upper bound of three resident workgroups / 50%
thread occupancy on this SM before register limits. The rejected pipeline was
not retained long enough to add executable-statistics plumbing, so
registers/thread, the actual resident-WG limit, and spill count are recorded as
unavailable rather than inferred. No occupancy claim is used in the decision.

### Weight error

The authenticated model diagnostic samples 64 evenly spaced output rows from
every affected tensor and every value in those rows: 10,223,616 gate/up and
15,335,424 down values. Relative error uses `max(abs(reference), 1e-6)`.

| Group / representation | Max abs | Mean abs | RMSE | Max relative | Mean relative |
|---|---:|---:|---:|---:|---:|
| Gate/up FP16 control | 0.00004959 | 0.00000096 | 0.00000146 | 0.000488 | 0.000167 |
| Gate/up INT8 | 0.00056604 | 0.00005208 | 0.00006199 | 1.000000 | 0.037920 |
| Down FP16 diagnostic control | 0.00004578 | 0.00000084 | 0.00000131 | 0.000488 | 0.000157 |
| Down INT8 | 0.00146091 | 0.00005479 | 0.00006863 | 1.000000 | 0.033926 |

The 1.0 maxima are real zeroing of sufficiently small values, not NaN/Inf.
No sampled source value or reconstructed value is non-finite.

### Target matmul output

The existing random Q4 gate/up oracle remains at its unchanged historical
absolute-or-relative assertion. Q4 INT8 passes it. Q6 down does not pass that
same tight assertion; it was measured as a diagnostic and was not used to
weaken the accepted gate.

| Source | Max abs | Mean abs | RMSE | Max relative | Mean relative |
|---|---:|---:|---:|---:|---:|
| Q4 gate/up INT8 | 17.71875 | 3.84602 | 4.77179 | 0.001915 | 0.000372 |
| Q6 down INT8 | 0.15930 | 0.04650 | 0.05848 | 0.131647 | 0.015105 |

The different absolute scales reflect different synthetic oracle outputs. The
Q6 result is the first direct warning that one scale per whole output channel
does not preserve the current numerical contract across formats.

### Full-model layer, logits, and sequence evidence

The test-only diagnostic compares the strongest accepted Phase 12 FP16-native
gate/up control with each prototype. It reads the final transformer's FP32
residual after submission, then compares all 131,072 logits, top-10 membership,
and eight greedy tokens. Relative maxima are dominated by near-zero references;
absolute/mean/RMSE and ranking evidence drive the verdict.

Gate/up INT8 with F16 KV:

| Prompt/context | Layer max / mean / RMSE | Logit max / mean / RMSE | Top-1 | Top-10 | First sequence disagreement |
|---|---|---|---|---:|---:|
| 128 repeated | 100.264 / 1.6795 / 3.0330 | 0.3705 / 0.04867 / 0.06162 | yes | 10/10 | none in 8 |
| 2K repeated | 0.1984 / 0.02649 / 0.03445 | 0.3759 / 0.05355 / 0.06833 | yes | 10/10 | none in 8 |
| 8K repeated | 0.2458 / 0.02298 / 0.02935 | 0.4123 / 0.07220 / 0.08835 | yes | 10/10 | none in 8 |
| 28K repeated | 0.5228 / 0.02748 / 0.03732 | **5.6787 / 2.0269 / 2.2573** | **no** | **1/10** | **token 0** |
| math oracle prompt | 100.264 / 1.6795 / 3.0330 | 0.2949 / 0.04402 / 0.05507 | **no** | 10/10 | **token 0** |
| greeting | 298.810 / 1.8061 / 5.9770 | 10.3596 / 2.1730 / 2.5429 | yes | 1/10 | token 6 |
| explanation | 100.264 / 1.6795 / 3.0330 | 0.3847 / 0.03639 / 0.04650 | yes | 9/10 | none in 8 |

The 128 regression control remains exactly
`2757,10637,2479,1636,6483,60210,1261,6715`, demonstrating why that sequence
alone cannot qualify a new numeric format. The 28K control/candidate sequences
start `2405,...` and `1402,...`; the math prompt starts `6319,...` and
`8396,...`.

Adding INT8 down worsens the already failing result: at 28K, layer max/mean/RMSE
is 1.3063/0.05748/0.07777, logit max/mean/RMSE is
6.3106/1.8744/2.1022, top-10 overlap is 1/10, and the first token differs.
Its 128, 2K, and 8K repeated prompts keep top-1, but the math sequence diverges
at token 3 and only 5/10 greeting logits remain in the control top ten.

With INT8 KV, gate/up INT8 was additionally checked through 8K. The repeated
128/2K/8K rows keep top-1 and all eight greedy tokens, with logit
max/mean/RMSE respectively 0.4506/0.05566/0.07043,
0.3313/0.05477/0.06825, and 0.5360/0.08371/0.10525. The greeting still
diverges at token 6 and has 1/10 top-ten overlap. A candidate 28K INT8-KV run
was stopped after the FP16-native control alone exceeded 11 minutes; F16 KV had
already produced a terminal quality failure. The accepted source restored at
the end retains the already-qualified Phase 12 F16/INT8 KV behavior unchanged.

No historical tolerance was widened. These results fail the precision gate.

## Performance and attribution of rejected prototypes

### Same-session decision points

Gate/up INT8 A/B/A at 28K is 45,886.31 / 44,145.20 / 46,001.88 ms. Its
surrounding FP16 control is 45,944.10 ms, giving -1,798.90 ms / **-3.92%**.
It is locally 26.9--27.3% faster for gate/up and uses 49.93% less native VRAM,
so it passes prototype GO condition A before quality qualification.

| Gate/up comparison | Compressed control | Phase 12 FP16 native | INT8 candidate |
|---|---:|---:|---:|
| Native value bytes | 0 | 2,944,401,408 | 1,472,200,704 |
| Native metadata | 0 | 0 | 1,916,928 B scales |
| Runtime conversion | Q4 dequant/unpack per K tile | none | FP16 activation to INT8 once/projection |
| Matrix2 weight input | shared dequantized FP16 | direct FP16 tensor | direct SINT8 tensor |
| Gate + up GPU | 10.650 s acquired | 6.555 s | 4.779 s |
| 28K TTFT | 49.714 s acquired | 45.944 s surrounding | 44.145 s |
| Native residency | 0 | 2.742188 GiB | 1.372879 GiB |

Gate/up+down A/B/A is 45,886.31 / 40,875.27 / 45,959.58 ms. The surrounding
control is 45,922.95 ms: -5,047.68 ms / **-10.99%**, with 0.22% candidate CV.
At 128, the surrounding 97.81/96.58 ms controls average 97.20 ms and the
candidate is 75.10 ms (-22.73%, 0.54% CV).

A separate 32-token baseline/candidate/baseline guard isolates decode from the
three-token TTFT runs:

| Prompt history | Controls | Candidate | Delta vs surrounding control |
|---:|---:|---:|---:|
| 128 | 46.28 / 46.46 tok/s | 46.35 tok/s | -0.04% |
| 2K | 46.56 / 46.37 tok/s | 46.44 tok/s | -0.05% |
| 28K | 29.30 / 29.04 tok/s | 29.41 tok/s | +0.82% |

All are within the 1% guard and confirm that decode remains canonical.

The complete diagnostics-free shape curve is:

| Context | Phase 13/FP16 control | Gate/up+down INT8 | Delta | Candidate CV |
|---:|---:|---:|---:|---:|
| 128 | 97.20 ms surrounding | 75.10 ms | -22.73% | 0.54% |
| 512 | 369.92 ms | 281.21 ms | -23.98% | 0.35% |
| 2K | 1,562.99 ms | 1,207.42 ms | -22.75% | 0.19% |
| 8K | 7,930.87 ms | 6,483.95 ms | -18.24% | 0.28% |
| 16K | 20,414.46 ms | 17,521.66 ms | -14.17% | 0.34% |
| 28K | 45,922.95 ms surrounding | 40,875.27 ms | **-10.99%** | 0.22% |

The separate profile build records this per-request 28K comparison (four
requests divided by four):

| Component | FP16 gate/up control | INT8 gate/up | INT8 gate/up+down |
|---|---:|---:|---:|
| GPU prefill | 46,092.984 ms | 44,062.249 ms | **40,948.975 ms** |
| Attention | 26,398.985 | 26,206.616 | 26,266.221 |
| Q | 2,459.938 | 2,436.140 | 2,440.906 |
| K | 678.656 | 669.920 | 674.978 |
| V | 721.659 | 712.491 | 714.194 |
| O | 2,544.872 | 2,515.926 | 2,525.832 |
| Gate | 3,279.740 | 2,398.697 | 2,401.307 |
| Up | 3,275.138 | 2,380.029 | 2,384.595 |
| Down | 5,781.402 | 5,729.878 | **2,415.153** |
| Activation quantization | 0 | 69.421 | 175.776 |
| Activation | 218.389 | 214.089 | 215.584 |
| Other | about 734 | about 735 | about 734 |

Attention moves by less than 0.51%; the gain is attributable to direct compact
MLP Matrix2, not an attention artifact. At 8K the combined candidate records
6,492.360 ms GPU prefill, 2,213.313 attention, 710.217/194.139/207.241/738.641
ms Q/K/V/O, 693.072/693.460/701.000 ms gate/up/down, 52.250 ms quantization,
and 289.027 ms remaining work. The Phase 13 8K control is 8,008.914 ms GPU,
2,264.105 ms attention, and 1,691.809 ms down.

## Memory and initialization economics

| Configuration | Native values | Scale metadata | Native total | Total weight residency | 28K application VRAM |
|---|---:|---:|---:|---:|---:|
| Canonical compressed | 0 | 0 | 0 | 1.991741 GiB | 5,452 MiB acquired |
| Phase 12 FP16 gate/up | 2,944,401,408 B | 0 | 2.742188 GiB | 4.733929 GiB | 8,260 MiB acquired |
| INT8 gate/up | 1,472,200,704 B | 1,916,928 B | 1.372879 GiB | 3.364620 GiB | not separately sampled |
| INT8 gate/up+down | 2,208,301,056 B | 2,236,416 B | **2.058723 GiB** | **4.050465 GiB** | **7,564 MiB measured** |

The combined candidate is 0.683464 GiB smaller than Phase 12 native and its
observed steady request is 696 MiB smaller. The 2-second process trace rises
through 4,002 MiB during model initialization and then holds 7,564 MiB after
request/KV allocation. Native uploads are sequential; the maximum native
staging payload is 27 MiB, giving a conservative sampled initialization peak
of about 4,029 MiB. Canonical residency remains byte-identical and
decode continues to use it.

Independent same-process initialization measurements are 11.30 s for the FP16
control, 14.81 s for INT8 gate/up, and 20.91 s for INT8 gate/up+down. Focused
timing of the native tensor work is:

| Candidate | Tensors | Conversion | Buffer creation | Upload |
|---|---:|---:|---:|---:|
| INT8 gate/up | 52 | 10.935 s | 0.050 s | 2.392 s |
| INT8 gate/up+down | 78 | 17.055 s | 0.081 s | 3.957 s |

These are independent runs, so their component sum is not subtracted from the
total-init run. Against FP16 gate/up, combined INT8 adds about 9.61 s startup
and saves 5.048 s per 28K request, a performance-only break-even of 1.90
requests. Gate/up alone adds about 3.51 s and saves 1.799 s, or 1.95 requests.
Phase 12's acquired incremental native initialization remains 9.013 s. The
break-even calculation is informational only because both lossy candidates
fail quality.

## Pareto and experiment registry

| ID | Representation / target | Direct Matrix2 | TTFT delta | Native delta vs Phase 12 | Quality | Decision |
|---|---|---|---:|---:|---|---|
| P14-BF16 | BF16 gate/up | yes | not built | 0 | strictly less precise, no memory gain | reject analytically |
| P14-FP8 | E4M3/E5M2 | no device record | not built | -50% theoretical | unsupported | reject |
| P14-I8-WO | INT8 weights into FP16 MMA | no | not built | about -50% | runtime reconversion required | reject analytically |
| P14-U8 | asymmetric UINT8 | yes, with UINT8 A | not built | about -50% | correction and zero-point risk; dominated | reject analytically |
| P14-I8-GU | signed symmetric INT8 gate/up | yes | -3.92% at 28K | -1.369309 GiB | top-1/sequence failure | **reject** |
| P14-I8-GUD | same plus Q4/Q6 down | yes | -10.99% at 28K | -0.683464 GiB | larger 28K logits failure | **reject** |

The speed/memory coordinates of P14-I8-GU and P14-I8-GUD dominate Phase 12
mechanically, but neither is on the usable Pareto frontier because quality is a
hard dimension. Per-tensor scaling was rejected analytically as less accurate;
per-K-block scaling would break the single-I32-accumulator contract and add
scaled partial accumulation to every K loop. The measured per-output strategy
already fails, so no numerical sweep is justified.

## Final decision

**STOP — no Phase 14 production candidate is retained.**

The direct INT8 route proves useful hardware capability and excellent local
economics, but fails the unchanged full-model precision gate in both F16 and
INT8 KV evidence. This is stop condition 3. It also reaches condition 6:
gate/up passes the mechanical GO gate, while down is faster but not an
acceptable additional group. All prototype source, shaders, diagnostic hooks,
flags, tests, and structural splits are removed. Production source returns to
the Phase 12 commit content (`6cb170f`), as carried by the Phase 13 baseline.

### Final bottleneck

The restored 28K profile remains 46.159 s GPU prefill with about 26.48 s exact
attention, 18.74 s seven matmuls, and 0.94 s other. Attention is again the
largest bottleneck at roughly 57%.

### Remaining headroom

Further native-weight progress requires a materially more accurate compact
numeric scheme, such as block-scaled partial accumulation. That reintroduces
scale work inside K and is likely to erase much of direct INT8's advantage; the
row-scaled numeric frontier is closed and should not be swept again.

### Recommended Phase 15

Return to an attention representation or algorithm change rather than another
weight-format sweep. Any future weight experiment must first offer a stronger
quality model than whole-row scaling and retain the Phase 14 long-context logit
and multi-prompt sequence gates.

## Final verification

- `git diff 6cb170f` is empty for all production, workspace, support, example,
  manifest, and build-script paths; only this report and `docs/README.md`
  survive after the evidence commits.
- `cargo fmt --all -- --check` passes.
- workspace/all-target Vulkan Clippy passes with `-D warnings`.
- the workspace/all-target Vulkan-hybrid compile passes.
- the release Vulkan engine library reports 147 passed, zero failed, and four
  intentional authenticated/long-context ignores.
- the broader application run reports 165 passed and one pre-existing fixture
  failure: its isolated installer `PATH` omits `dirname`, so the script exits 1
  before reaching the expected missing-`npm` exit 2.
- the engine integration run reports four passed, one intentional oracle
  ignore, and one pre-existing documentation-fixture failure because this
  baseline does not contain the `VALIDATION.md` file that `docs_contract`
  attempts to open. Neither failure is touched by Phase 14.

The worktree is clean after the final evidence commit, and no branch, commit,
or artifact is pushed.
