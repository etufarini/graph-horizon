// AGENTS deroga K: kernel della sola operazione RMS normalization.
#pragma once

extern "C" __global__ void cuda_rmsnorm(
    const float *input, const __half *weight, __half *out,
    uint32_t width, float epsilon, uint32_t rows) {
    const uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    const uint64_t base = uint64_t(row) * width;
    float sum = 0.0f;
    for (uint32_t i = 0; i < width; ++i) sum += input[base + i] * input[base + i];
    const float inverse = rsqrtf(sum / float(width) + epsilon);
    for (uint32_t i = 0; i < width; ++i) {
        out[base + i] = __float2half_rn(input[base + i] * inverse * __half2float(weight[i]));
    }
}
