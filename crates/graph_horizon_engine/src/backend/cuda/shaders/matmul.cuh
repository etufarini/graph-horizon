// AGENTS deroga K: famiglia coesa della sola operazione matmul.
#pragma once

__device__ __forceinline__ float cuda_dot(
    const __half *input, const unsigned char *weight, uint32_t format,
    uint32_t row, uint32_t width) {
    float sum = 0.0f;
    for (uint32_t i = 0; i < width; ++i) {
        sum += __half2float(input[i]) * cuda_weight_value(weight, format, row, i, width);
    }
    return sum;
}

extern "C" __global__ void cuda_matmul(
    const __half *input, const unsigned char *weight, __half *out,
    uint32_t input_width, uint32_t output_width, uint32_t format) {
    const uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < output_width) out[row] = __float2half_rn(cuda_dot(input, weight, format, row, input_width));
}

extern "C" __global__ void cuda_matmul_batched(
    const __half *input, const unsigned char *weight, __half *out,
    uint32_t input_width, uint32_t output_width, uint32_t rows, uint32_t format) {
    const uint64_t index = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint64_t total = uint64_t(rows) * output_width;
    if (index >= total) return;
    const uint32_t token = uint32_t(index / output_width);
    const uint32_t row = uint32_t(index % output_width);
    out[index] = __float2half_rn(cuda_dot(
        input + uint64_t(token) * input_width, weight, format, row, input_width));
}

extern "C" __global__ void cuda_logits(
    const __half *input, const unsigned char *weight, float *out,
    uint32_t input_width, uint32_t output_width, uint32_t format) {
    const uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < output_width) out[row] = cuda_dot(input, weight, format, row, input_width);
}
