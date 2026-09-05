// AGENTS deroga K: famiglia coesa della sola operazione matmul.
#pragma once
#include <mma.h>

template<uint32_t FORMAT>
__device__ __forceinline__ float cuda_dot_format(
    const __half *input, const unsigned char *weight,
    uint32_t row, uint32_t width, float *partial) {
    float sum = 0.0f;
    if (FORMAT == 0) {
        for (uint32_t i = threadIdx.x; i < width; i += blockDim.x) {
            sum += __half2float(input[i]) * cuda_weight_value(weight, FORMAT, row, i, width);
        }
        partial[threadIdx.x] = sum;
    } else {
        // 64 physical threads own all 128 original leaves, paired by shared quant bytes.
        const uint32_t lane = FORMAT == 3 ? threadIdx.x
            : (threadIdx.x / 32) * 64 + threadIdx.x % 32;
        constexpr uint32_t DISTANCE = FORMAT == 3 ? 64 : 32;
        const uint32_t blocks = width / 256;
        float paired_sum = 0.0f;
        for (uint32_t group = 0; group < blocks; ++group) {
            const uint64_t block = (uint64_t(row) * blocks + group)
                * (FORMAT == 1 ? 144 : FORMAT == 2 ? 176 : 210);
            const uint32_t i = group * 256 + lane;
            float first, first_pair, second, second_pair;
            cuda_weight_pair<FORMAT>(weight, block, lane, first, first_pair);
            cuda_weight_pair<FORMAT>(weight, block, lane + 128, second, second_pair);
            sum += __half2float(input[i]) * first;
            sum += __half2float(input[i + 128]) * second;
            paired_sum += __half2float(input[i + DISTANCE]) * first_pair;
            paired_sum += __half2float(input[i + DISTANCE + 128]) * second_pair;
        }
        partial[lane] = sum;
        partial[lane + DISTANCE] = paired_sum;
    }
    __syncthreads();
    // Restore the same logical tree for packed 64-thread and F16 128-thread blocks.
    for (uint32_t stride = 64; stride > 0; stride /= 2) {
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

template<uint32_t FORMAT>
__device__ __forceinline__ void cuda_matmul_tensor_format(
    const __half *input, const unsigned char *weight, __half *out,
    uint32_t width, uint32_t outputs, uint32_t rows,
    __half *a, __half *b, float *c, float *row_sums, float *factors, float *biases) {
    using namespace nvcuda;
    const uint32_t token = blockIdx.y * 16;
    const uint32_t output = blockIdx.x * 64;
    const uint32_t warp = threadIdx.x / 32;
    // Q4/Q5 share coefficients over 32 values; Q6 changes them every 16.
    constexpr uint32_t GROUP = FORMAT == 3 ? 16 : 32;
    float sums[8] = {};
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> af;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> bf;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> cf;
    // Packed widths are positive multiples of 256, so every group is full.
    for (uint32_t group = 0; group < width / GROUP; ++group) {
        const uint32_t base = group * GROUP;
        // Two complete warps own coefficients; B staging consumes only quants.
        // Unused helper results disappear after inlining, preserving the same math.
        if (threadIdx.x < 64) {
            const uint32_t column = output + threadIdx.x;
            float factor = 0.0f, bias = 0.0f;
            if (column < outputs) {
                cuda_weight_parts<FORMAT>(weight, column, base, width, factor, bias);
            }
            factors[threadIdx.x] = factor;
            biases[threadIdx.x] = bias;
        }
        for (uint32_t i = threadIdx.x; i < 16 * GROUP; i += 128) {
            const uint32_t row = token + i / GROUP;
            a[i] = row < rows ? input[uint64_t(row) * width + base + i % GROUP]
                              : __float2half_rn(0.0f);
        }
        for (uint32_t i = threadIdx.x; i < 64 * GROUP; i += 128) {
            const uint32_t column = output + i / GROUP;
            float factor = 0.0f, bias = 0.0f;
            const float quant = column < outputs
                ? cuda_weight_parts<FORMAT>(weight, column, base + i % GROUP, width, factor, bias)
                : 0.0f;
            b[i] = __float2half_rn(quant);
        }
        __syncthreads();
        if (threadIdx.x < 16) {
            float sum = 0.0f;
            for (uint32_t i = 0; i < GROUP; ++i) {
                sum += __half2float(a[threadIdx.x * GROUP + i]);
            }
            row_sums[threadIdx.x] = sum;
        }
        // No tail lane exits: the entire warp uses identical aligned pointers.
        wmma::fill_fragment(cf, 0.0f);
        for (uint32_t slice = 0; slice < GROUP; slice += 16) {
            wmma::load_matrix_sync(af, a + slice, GROUP);
            wmma::load_matrix_sync(bf, b + warp * 16 * GROUP + slice, GROUP);
            wmma::mma_sync(cf, af, bf, cf);
        }
        wmma::store_matrix_sync(c + warp * 16, cf, 64, wmma::mem_row_major);
        __syncthreads();
        for (uint32_t part = 0; part < 8; ++part) {
            const uint32_t i = threadIdx.x + part * 128;
            sums[part] += factors[i % 64] * c[i] + biases[i % 64] * row_sums[i / 64];
        }
        // All consumers finish before the next group replaces shared staging.
        __syncthreads();
    }
    for (uint32_t part = 0; part < 8; ++part) {
        const uint32_t i = threadIdx.x + part * 128;
        const uint32_t row = token + i / 64;
        const uint32_t column = output + i % 64;
        if (row < rows && column < outputs) {
            out[uint64_t(row) * outputs + column] = __float2half_rn(sums[part]);
        }
    }
}

extern "C" __global__ void cuda_matmul_tensor(
    const __half *input, const unsigned char *weight, __half *out,
    uint32_t width, uint32_t outputs, uint32_t rows, uint32_t format) {
    // One staging set per block, aligned for the opaque WMMA fragment API.
    __shared__ __align__(32) __half a[512], b[2048];
    __shared__ __align__(32) float c[1024];
    __shared__ float row_sums[16], factors[64], biases[64];
    if (format == 1) {
        cuda_matmul_tensor_format<1>(input, weight, out, width, outputs, rows,
            a, b, c, row_sums, factors, biases);
    } else if (format == 2) {
        cuda_matmul_tensor_format<2>(input, weight, out, width, outputs, rows,
            a, b, c, row_sums, factors, biases);
    } else {
        cuda_matmul_tensor_format<3>(input, weight, out, width, outputs, rows,
            a, b, c, row_sums, factors, biases);
    }
}
