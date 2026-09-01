// AGENTS deroga K: kernel della sola operazione SiLU multiplication.
#pragma once

extern "C" __global__ void cuda_silu_mul(
    const __half *gate, const __half *up, __half *out, uint32_t length) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= length) return;
    const float value = __half2float(gate[i]);
    const __half activated = __float2half_rn(value / (1.0f + expf(-value)));
    out[i] = __float2half_rn(__half2float(activated) * __half2float(up[i]));
}
