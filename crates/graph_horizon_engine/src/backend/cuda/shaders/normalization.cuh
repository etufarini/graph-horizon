// AGENTS deroga K: kernel della sola operazione RMS normalization.
#pragma once

extern "C" __global__ void cuda_rmsnorm(
    const float *input, const __half *weight, __half *out,
    uint32_t width, float epsilon, uint32_t rows) {
    // One block owns the row; padded lanes contribute zero to its width sum.
    const uint32_t row = blockIdx.x;
    if (row >= rows) return;
    const uint64_t base = uint64_t(row) * width;
    float sum = 0.0f;
    for (uint32_t i = threadIdx.x; i < width; i += blockDim.x) {
        sum += input[base + i] * input[base + i];
    }
    __shared__ float partial[128];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    const float inverse = rsqrtf(partial[0] / float(width) + epsilon);
    for (uint32_t i = threadIdx.x; i < width; i += blockDim.x) {
        out[base + i] = __float2half_rn(input[base + i] * inverse * __half2float(weight[i]));
    }
}
