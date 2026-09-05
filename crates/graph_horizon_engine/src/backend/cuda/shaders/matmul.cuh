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
        // One warp owns four original leaves per lane, without regrouping K sums.
        const uint32_t lane = threadIdx.x % 32;
        constexpr uint32_t PAIR = FORMAT == 3 ? 64 : 32;
        constexpr uint32_t SECOND = FORMAT == 3 ? 32 : 64;
        const uint32_t blocks = width / 256;
        float second_sum = 0.0f, paired_sum = 0.0f, second_paired_sum = 0.0f;
        for (uint32_t group = 0; group < blocks; ++group) {
            const uint64_t block = (uint64_t(row) * blocks + group)
                * (FORMAT == 1 ? 144 : FORMAT == 2 ? 176 : 210);
            #pragma unroll
            for (uint32_t half = 0; half < 256; half += 128) {
                const uint32_t i = group * 256 + half + lane;
                float first, first_pair, second, second_pair;
                cuda_weight_pair<FORMAT>(weight, block, half + lane, first, first_pair);
                cuda_weight_pair<FORMAT>(weight, block, half + lane + SECOND, second, second_pair);
                sum += __half2float(input[i]) * first;
                second_sum += __half2float(input[i + SECOND]) * second;
                paired_sum += __half2float(input[i + PAIR]) * first_pair;
                second_paired_sum += __half2float(input[i + SECOND + PAIR]) * second_pair;
            }
        }
        // Restore levels64 then32 before reducing the remaining32 logical lanes.
        sum = FORMAT == 3 ? (sum + paired_sum) + (second_sum + second_paired_sum)
                          : (sum + second_sum) + (paired_sum + second_paired_sum);
        for (uint32_t stride = 16; stride > 0; stride /= 2) {
            const float other = __shfl_down_sync(0xffffffff, sum, stride);
            if (lane < stride) sum += other;
        }
        return sum;
    }
    __syncthreads();
    // Only F16 reaches this block-wide tree; packed outputs use independent warps.
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
    const uint32_t row = format == 0 ? blockIdx.x : blockIdx.x * 4 + threadIdx.x / 32;
    // Packed output tails are whole warps; no block barrier occurs on their path.
    if (row >= output_width) return;
    const float value = cuda_dot(input, weight, format, row, input_width);
    if ((format == 0 ? threadIdx.x : threadIdx.x % 32) == 0) out[row] = __float2half_rn(value);
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
    const uint32_t row = format == 0 ? blockIdx.x : blockIdx.x * 4 + threadIdx.x / 32;
    if (row >= output_width) return;
    const float value = cuda_dot(input, weight, format, row, input_width);
    if ((format == 0 ? threadIdx.x : threadIdx.x % 32) == 0) out[row] = value;
}

template<uint32_t FORMAT, uint32_t TOKENS, uint32_t STAGES>
__device__ __forceinline__ void cuda_matmul_tensor_format(
    const __half *input, const unsigned char *weight, __half *out,
    uint32_t width, uint32_t outputs, uint32_t rows,
    __half *a, __half *b, float *c, float *row_sums, float *factors, float *biases) {
    using namespace nvcuda;
    const uint32_t token = blockIdx.y * TOKENS;
    const uint32_t output = blockIdx.x * 64;
    const uint32_t warp = threadIdx.x / 32;
    // Q4/Q5 share coefficients over 32 values; Q6 changes them every 16.
    constexpr uint32_t GROUP = FORMAT == 3 ? 16 : 32;
    constexpr uint32_t STRIDE = GROUP * STAGES;
    float sums[TOKENS / 2] = {};
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> af;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> bf;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> cf;
    // Packed widths are positive multiples of 256, so every group is full.
    for (uint32_t group = 0; group < width / STRIDE; ++group) {
        const uint32_t base = group * STRIDE;
        // Two complete warps own coefficients; B staging consumes only quants.
        // Unused helper results disappear after inlining, preserving the same math.
        if (threadIdx.x < 64) {
            const uint32_t column = output + threadIdx.x;
            #pragma unroll
            for (uint32_t stage = 0; stage < STAGES; ++stage) {
                float factor = 0.0f, bias = 0.0f;
                if (column < outputs) {
                    cuda_weight_parts<FORMAT>(weight, column, base + stage * GROUP,
                        width, factor, bias);
                }
                factors[stage * 64 + threadIdx.x] = factor;
                biases[stage * 64 + threadIdx.x] = bias;
            }
        }
        // Each stage keeps the original K32 row pitch; pairing changes only ownership.
        for (uint32_t i = threadIdx.x; i < TOKENS * GROUP; i += 128) {
            const uint32_t row = token + i / GROUP;
            #pragma unroll
            for (uint32_t stage = 0; stage < STAGES; ++stage) {
                a[stage * TOKENS * GROUP + i] = row < rows
                    ? input[uint64_t(row) * width + base + stage * GROUP + i % GROUP]
                    : __float2half_rn(0.0f);
            }
        }
        for (uint32_t i = threadIdx.x; i < 64 * GROUP; i += 128) {
            const uint32_t column = output + i / GROUP;
            if (STAGES == 2) {
                float low = 0.0f, high = 0.0f;
                if (column < outputs) {
                    cuda_quant_pair<FORMAT>(weight, column, base + i % GROUP, width, low, high);
                }
                b[i] = __float2half_rn(low);
                b[64 * GROUP + i] = __float2half_rn(high);
            } else {
                float factor = 0.0f, bias = 0.0f;
                const float quant = column < outputs
                    ? cuda_weight_parts<FORMAT>(weight, column, base + i % GROUP,
                        width, factor, bias) : 0.0f;
                b[i] = __float2half_rn(quant);
            }
        }
        __syncthreads();
        if (threadIdx.x < TOKENS) {
            #pragma unroll
            for (uint32_t stage = 0; stage < STAGES; ++stage) {
                float sum = 0.0f;
                for (uint32_t i = 0; i < GROUP; ++i) {
                    sum += __half2float(a[stage * TOKENS * GROUP + threadIdx.x * GROUP + i]);
                }
                row_sums[stage * TOKENS + threadIdx.x] = sum;
            }
        }
        // No tail lane exits: the entire warp uses identical aligned pointers.
        // Each M16 fragment keeps its K order while sharing the staged weights.
        #pragma unroll
        for (uint32_t stage = 0; stage < STAGES; ++stage) {
            for (uint32_t m = 0; m < TOKENS; m += 16) {
                wmma::fill_fragment(cf, 0.0f);
                for (uint32_t slice = 0; slice < GROUP; slice += 16) {
                    wmma::load_matrix_sync(af, a + stage * TOKENS * GROUP + m * GROUP + slice, GROUP);
                    wmma::load_matrix_sync(bf, b + stage * 64 * GROUP + warp * 16 * GROUP + slice, GROUP);
                    wmma::mma_sync(cf, af, bf, cf);
                }
                wmma::store_matrix_sync(c + stage * TOKENS * 64 + m * 64 + warp * 16,
                    cf, 64, wmma::mem_row_major);
            }
        }
        __syncthreads();
        // Correct each K32 group separately, in the original increasing-K order.
        #pragma unroll
        for (uint32_t stage = 0; stage < STAGES; ++stage) {
            for (uint32_t part = 0; part < TOKENS / 2; ++part) {
                const uint32_t i = threadIdx.x + part * 128;
                sums[part] += factors[stage * 64 + i % 64] * c[stage * TOKENS * 64 + i]
                    + biases[stage * 64 + i % 64] * row_sums[stage * TOKENS + i / 64];
            }
        }
        // All consumers finish before the next group replaces shared staging.
        __syncthreads();
    }
    for (uint32_t part = 0; part < TOKENS / 2; ++part) {
        const uint32_t i = threadIdx.x + part * 128;
        const uint32_t row = token + i / 64;
        const uint32_t column = output + i % 64;
        if (row < rows && column < outputs) {
            out[uint64_t(row) * outputs + column] = __float2half_rn(sums[part]);
        }
    }
}

template<uint32_t TOKENS, uint32_t STAGES = 1>
__device__ __forceinline__ void cuda_matmul_tensor_tile(
    const __half *input, const unsigned char *weight, __half *out,
    uint32_t width, uint32_t outputs, uint32_t rows, uint32_t format) {
    // One staging set per block, aligned for the opaque WMMA fragment API.
    __shared__ __align__(32) __half a[TOKENS * 32 * STAGES], b[2048 * STAGES];
    __shared__ __align__(32) float c[TOKENS * 64 * STAGES];
    __shared__ float row_sums[TOKENS * STAGES], factors[64 * STAGES], biases[64 * STAGES];
    if (format == 1) {
        cuda_matmul_tensor_format<1, TOKENS, STAGES>(input, weight, out, width, outputs, rows,
            a, b, c, row_sums, factors, biases);
    } else if (format == 2) {
        cuda_matmul_tensor_format<2, TOKENS, STAGES>(input, weight, out, width, outputs, rows,
            a, b, c, row_sums, factors, biases);
    } else if (STAGES == 1) {
        cuda_matmul_tensor_format<3, TOKENS, STAGES>(input, weight, out, width, outputs, rows,
            a, b, c, row_sums, factors, biases);
    }
}

// Separate entries keep M32's larger staging out of small-batch resource usage.
extern "C" __global__ void cuda_matmul_tensor(
    const __half *input, const unsigned char *weight, __half *out,
    uint32_t width, uint32_t outputs, uint32_t rows, uint32_t format) {
    cuda_matmul_tensor_tile<16>(input, weight, out, width, outputs, rows, format);
}

extern "C" __global__ void cuda_matmul_tensor_wide(
    const __half *input, const unsigned char *weight, __half *out,
    uint32_t width, uint32_t outputs, uint32_t rows, uint32_t format) {
    cuda_matmul_tensor_tile<32>(input, weight, out, width, outputs, rows, format);
}

// Paired staging is isolated from Q6 and the larger M32 resource footprint.
extern "C" __global__ void cuda_matmul_tensor_paired(
    const __half *input, const unsigned char *weight, __half *out,
    uint32_t width, uint32_t outputs, uint32_t rows, uint32_t format) {
    cuda_matmul_tensor_tile<16, 2>(input, weight, out, width, outputs, rows, format);
}
