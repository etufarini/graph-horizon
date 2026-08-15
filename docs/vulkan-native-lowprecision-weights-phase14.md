<!--
This report owns Phase 14 evidence for native low-precision Vulkan execution
weights: capability discovery, representation screening, isolated prototypes,
numeric qualification, performance economics, and the final keep/stop decision.
It defines no GGUF serialization or public support contract.
-->

# Vulkan Native Low-Precision Weights Phase 14

## Status

Investigation in progress on branch
`perf/vulkan-native-lowprecision-weights-phase14`, created from the required
clean Phase 13 commit `16dd03e`. The effective production baseline is Phase 12
commit `6cb170f`. Nothing has been pushed.

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
