// AGENTS deroga K: famiglia coesa della sola operazione matmul.
#pragma once

template<uint32_t FORMAT>
__device__ __forceinline__ float cuda_dot_format(
    const __half *input, const unsigned char *weight,
    uint32_t row, uint32_t width, float *partial) {
    float sum = 0.0f;
    for (uint32_t i = threadIdx.x; i < width; i += blockDim.x) {
        sum += __half2float(input[i]) * cuda_weight_value(weight, FORMAT, row, i, width);
    }
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    return partial[0];
}

__device__ __forceinline__ float cuda_dot(
    const __half *input, const unsigned char *weight, uint32_t format,
    uint32_t row, uint32_t width) {
    // One scratch allocation and one uniform format choice, outside the K loop.
    __shared__ float partial[256];
    if (format == 0) return cuda_dot_format<0>(input, weight, row, width, partial);
    if (format == 1) return cuda_dot_format<1>(input, weight, row, width, partial);
    if (format == 2) return cuda_dot_format<2>(input, weight, row, width, partial);
    return cuda_dot_format<3>(input, weight, row, width, partial);
}

extern "C" __global__ void cuda_matmul(
    const __half *input, const unsigned char *weight, __half *out,
    uint32_t input_width, uint32_t output_width, uint32_t format) {
    const uint32_t row = blockIdx.x;
    const float value = cuda_dot(input, weight, format, row, input_width);
    if (threadIdx.x == 0) out[row] = __float2half_rn(value);
}

template<uint32_t FORMAT>
__device__ __forceinline__ void cuda_matmul_batched_format(
    const __half *input, const unsigned char *weight, __half *out,
    uint32_t input_width, uint32_t output_width, uint32_t rows, float partial[4][256]) {
    const uint32_t row = blockIdx.x;
    const uint32_t token = blockIdx.y * 4;
    float sums[4] = {};
    for (uint32_t i = threadIdx.x; i < input_width; i += blockDim.x) {
        const float value = cuda_weight_value(weight, FORMAT, row, i, input_width);
        for (uint32_t offset = 0; offset < 4 && token + offset < rows; offset++) {
            sums[offset] += __half2float(
                input[uint64_t(token + offset) * input_width + i]) * value;
        }
    }
    for (uint32_t offset = 0; offset < 4; offset++) {
        partial[offset][threadIdx.x] = sums[offset];
    }
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            for (uint32_t offset = 0; offset < 4; offset++) {
                partial[offset][threadIdx.x] += partial[offset][threadIdx.x + stride];
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        for (uint32_t offset = 0; offset < 4 && token + offset < rows; offset++) {
            out[uint64_t(token + offset) * output_width + row] =
                __float2half_rn(partial[offset][0]);
        }
    }
}

extern "C" __global__ void cuda_matmul_batched(
    const __half *input, const unsigned char *weight, __half *out,
    uint32_t input_width, uint32_t output_width, uint32_t rows, uint32_t format) {
    __shared__ float partial[4][256];
    if (format == 0) {
        cuda_matmul_batched_format<0>(input, weight, out, input_width, output_width, rows, partial);
    } else if (format == 1) {
        cuda_matmul_batched_format<1>(input, weight, out, input_width, output_width, rows, partial);
    } else if (format == 2) {
        cuda_matmul_batched_format<2>(input, weight, out, input_width, output_width, rows, partial);
    } else {
        cuda_matmul_batched_format<3>(input, weight, out, input_width, output_width, rows, partial);
    }
}

extern "C" __global__ void cuda_logits(
    const __half *input, const unsigned char *weight, float *out,
    uint32_t input_width, uint32_t output_width, uint32_t format) {
    const uint32_t row = blockIdx.x;
    const float value = cuda_dot(input, weight, format, row, input_width);
    if (threadIdx.x == 0) out[row] = value;
}
