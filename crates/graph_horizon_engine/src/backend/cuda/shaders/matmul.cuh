// AGENTS deroga K: famiglia coesa della sola operazione matmul.
#pragma once

__device__ __forceinline__ float cuda_dot(
    const __half *input, const unsigned char *weight, uint32_t format,
    uint32_t row, uint32_t width) {
    float sum = 0.0f;
    for (uint32_t i = threadIdx.x; i < width; i += blockDim.x) {
        sum += __half2float(input[i]) * cuda_weight_value(weight, format, row, i, width);
    }
    __shared__ float partial[256];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    return partial[0];
}

extern "C" __global__ void cuda_matmul(
    const __half *input, const unsigned char *weight, __half *out,
    uint32_t input_width, uint32_t output_width, uint32_t format) {
    const uint32_t row = blockIdx.x;
    const float value = cuda_dot(input, weight, format, row, input_width);
    if (threadIdx.x == 0) out[row] = __float2half_rn(value);
}

extern "C" __global__ void cuda_matmul_batched(
    const __half *input, const unsigned char *weight, __half *out,
    uint32_t input_width, uint32_t output_width, uint32_t rows, uint32_t format) {
    const uint32_t row = blockIdx.x;
    const uint32_t token = blockIdx.y;
    const float value = cuda_dot(
        input + uint64_t(token) * input_width, weight, format, row, input_width);
    if (threadIdx.x == 0) {
        out[uint64_t(token) * output_width + row] = __float2half_rn(value);
    }
}

extern "C" __global__ void cuda_logits(
    const __half *input, const unsigned char *weight, float *out,
    uint32_t input_width, uint32_t output_width, uint32_t format) {
    const uint32_t row = blockIdx.x;
    const float value = cuda_dot(input, weight, format, row, input_width);
    if (threadIdx.x == 0) out[row] = value;
}
