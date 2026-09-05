// AGENTS deroga K: famiglia coesa della sola operazione causal attention.
#pragma once
#include <mma.h>

template <bool INT8>
__device__ __forceinline__ float cuda_cache_value(
    const unsigned char *cache, uint64_t index, uint64_t metadata, uint64_t vector) {
    if (!INT8) return __half2float(reinterpret_cast<const __half *>(cache)[index]);
    const uint64_t offset = metadata + vector * 4;
    const uint16_t min_bits = uint16_t(cache[offset]) | (uint16_t(cache[offset + 1]) << 8);
    const uint16_t scale_bits = uint16_t(cache[offset + 2]) | (uint16_t(cache[offset + 3]) << 8);
    const float minimum = __half2float(*reinterpret_cast<const __half *>(&min_bits));
    const float scale = __half2float(*reinterpret_cast<const __half *>(&scale_bits));
    return minimum + float(cache[index]) * scale;
}

template <bool INT8>
__device__ void cuda_attention_online(
    const __half *query, const unsigned char *k_cache, const unsigned char *v_cache,
    __half *out, uint32_t dim, uint32_t kv_heads, uint32_t q_heads,
    uint32_t base, uint32_t rows, uint32_t layer, uint32_t context,
    uint64_t k_metadata, uint64_t v_metadata) {
    // Four warps score four context tokens; each thread owns up to two output
    // dimensions. Validation bounds dim to 256 and dispatch fixes 128 threads.
    const uint32_t id = blockIdx.x;
    const uint32_t row = id / q_heads;
    const uint32_t q_head = id % q_heads;
    const uint32_t kv_head = q_head / (q_heads / kv_heads);
    const uint32_t position = base + row;
    const uint64_t q_base = uint64_t(id) * dim;
    const uint32_t lane = threadIdx.x % 32;
    const uint32_t warp = threadIdx.x / 32;
    float accumulator[2] = {};
    float maximum = -CUDART_INF_F;
    __shared__ float scores[4];
    __shared__ float previous;
    __shared__ float weights[4];
    __shared__ float denominator;
    if (threadIdx.x == 0) denominator = 0.0f;
    const float scale = rsqrtf(float(dim));
    // Tile indices avoid wrapping the token increment near the u32 boundary.
    for (uint32_t tile = 0; tile <= position / 4; ++tile) {
        const uint32_t token = tile * 4 + warp;
        const uint64_t vector = (uint64_t(layer) * context + token) * kv_heads + kv_head;
        const uint64_t cache_base = vector * dim;
        float sum = 0.0f;
        if (token <= position) {
            for (uint32_t i = lane; i < dim; i += 32) {
                sum += __half2float(query[q_base + i])
                    * cuda_cache_value<INT8>(k_cache, cache_base + i, k_metadata, vector);
            }
        }
        for (uint32_t stride = 16; stride > 0; stride /= 2) {
            const float other = __shfl_down_sync(0xffffffff, sum, stride);
            if (lane < stride) sum += other;
        }
        if (lane == 0) scores[warp] = token <= position ? sum * scale : -CUDART_INF_F;
        __syncthreads();
        if (threadIdx.x == 0) {
            float next_maximum = maximum;
            for (uint32_t offset = 0; offset < 4; ++offset) {
                next_maximum = fmaxf(next_maximum, scores[offset]);
            }
            previous = expf(maximum - next_maximum);
            float next_denominator = denominator * previous;
            for (uint32_t offset = 0; offset < 4; ++offset) {
                weights[offset] = expf(scores[offset] - next_maximum);
                next_denominator += weights[offset];
            }
            denominator = next_denominator;
            maximum = next_maximum;
        }
        __syncthreads();
        for (uint32_t part = 0; part < 2; ++part) {
            const uint32_t i = threadIdx.x + part * 128;
            if (i < dim) {
                float value = accumulator[part] * previous;
                for (uint32_t offset = 0; offset < 4 && tile * 4 + offset <= position; ++offset) {
                    const uint64_t v_vector =
                        (uint64_t(layer) * context + tile * 4 + offset) * kv_heads + kv_head;
                    value += weights[offset] * cuda_cache_value<INT8>(
                        v_cache, v_vector * dim + i, v_metadata, v_vector);
                }
                accumulator[part] = value;
            }
        }
        // All readers finish before the next tile overwrites shared weights.
        __syncthreads();
    }
    for (uint32_t part = 0; part < 2; ++part) {
        const uint32_t i = threadIdx.x + part * 128;
        if (i < dim) out[q_base + i] = __float2half_rn(accumulator[part] / denominator);
    }
}

template <bool INT8, uint32_t THREADS = 128>
__device__ void cuda_attention_buffered(
    const __half *query, const unsigned char *k_cache, const unsigned char *v_cache,
    __half *out, uint32_t dim, uint32_t kv_heads, uint32_t q_heads,
    uint32_t base, uint32_t rows, uint32_t layer, uint32_t context,
    uint64_t k_metadata, uint64_t v_metadata) {
    const uint32_t id = blockIdx.x;
    const uint32_t position = base + id / q_heads;
    // Decode only: prefill calls online directly and does not own score storage.
    // Bounded shared scores preserve the online path for larger positions.
    if (THREADS == 128 && position >= 4096) {
        cuda_attention_online<INT8>(query, k_cache, v_cache, out, dim, kv_heads,
            q_heads, base, rows, layer, context, k_metadata, v_metadata);
        return;
    }
    const uint32_t kv_head = (id % q_heads) / (q_heads / kv_heads);
    const uint64_t q_base = uint64_t(id) * dim;
    const uint32_t lane = threadIdx.x % 32;
    const uint32_t warp = threadIdx.x / 32;
    __shared__ float scores[4096];
    __shared__ float partial[128];
    const float scale = rsqrtf(float(dim));
    for (uint32_t token = warp; token <= position; token += THREADS / 32) {
        const uint64_t vector = (uint64_t(layer) * context + token) * kv_heads + kv_head;
        float sum = 0.0f;
        for (uint32_t i = lane; i < dim; i += 32) {
            sum += __half2float(query[q_base + i])
                * cuda_cache_value<INT8>(k_cache, vector * dim + i, k_metadata, vector);
        }
        for (uint32_t stride = 16; stride > 0; stride /= 2) {
            const float other = __shfl_down_sync(0xffffffff, sum, stride);
            if (lane < stride) sum += other;
        }
        if (lane == 0) scores[token] = sum * scale;
    }
    __syncthreads();
    float maximum = -CUDART_INF_F;
    // Preserve exactly128 logical softmax leaves even in the512-thread entry.
    if (threadIdx.x < 128) {
        for (uint32_t token = threadIdx.x; token <= position; token += 128) {
            maximum = fmaxf(maximum, scores[token]);
        }
        partial[threadIdx.x] = maximum;
    }
    __syncthreads();
    for (uint32_t stride = 64; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) partial[threadIdx.x] =
            fmaxf(partial[threadIdx.x], partial[threadIdx.x + stride]);
        __syncthreads();
    }
    maximum = partial[0];
    // All warps must read the maximum before partial[0] becomes a sum.
    __syncthreads();
    float sum = 0.0f;
    if (threadIdx.x < 128) {
        for (uint32_t token = threadIdx.x; token <= position; token += 128) {
            const float weight = expf(scores[token] - maximum);
            scores[token] = weight;
            sum += weight;
        }
        partial[threadIdx.x] = sum;
    }
    __syncthreads();
    for (uint32_t stride = 64; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    const float denominator = partial[0];
    if (THREADS == 512) {
        // Four contiguous history segments change only the f32 PV reduction.
        __shared__ float values[4 * 256];
        const uint32_t group = threadIdx.x / 128;
        const uint32_t begin = (position + 1) * group / 4;
        const uint32_t end = (position + 1) * (group + 1) / 4;
        for (uint32_t part = 0; part < 2; ++part) {
            const uint32_t i = threadIdx.x % 128 + part * 128;
            if (i < dim) {
                float value = 0.0f;
                for (uint32_t token = begin; token < end; ++token) {
                    const uint64_t vector = (uint64_t(layer) * context + token) * kv_heads + kv_head;
                    value += scores[token]
                        * cuda_cache_value<INT8>(v_cache, vector * dim + i, v_metadata, vector);
                }
                values[group * 256 + i] = value;
            }
        }
        __syncthreads();
        if (threadIdx.x < dim) {
            const uint32_t i = threadIdx.x;
            const float value = ((values[i] + values[256 + i]) + values[512 + i]) + values[768 + i];
            out[q_base + i] = __float2half_rn(value / denominator);
        }
        return;
    }
    for (uint32_t part = 0; part < 2; ++part) {
        const uint32_t i = threadIdx.x + part * 128;
        if (i < dim) {
            float value = 0.0f;
            for (uint32_t token = 0; token <= position; ++token) {
                const uint64_t vector = (uint64_t(layer) * context + token) * kv_heads + kv_head;
                value += scores[token]
                    * cuda_cache_value<INT8>(v_cache, vector * dim + i, v_metadata, vector);
            }
            out[q_base + i] = __float2half_rn(value / denominator);
        }
    }
}

template <bool INT8>
__device__ void cuda_attention_prefill_warp(
    const __half *query, const unsigned char *k_cache, const unsigned char *v_cache,
    __half *out, uint32_t dim, uint32_t kv_heads, uint32_t q_heads,
    uint32_t base, uint32_t rows, uint32_t layer, uint32_t context,
    uint64_t k_metadata, uint64_t v_metadata) {
    // One independent warp owns a query head; no block barrier may couple different rows.
    const uint64_t id = uint64_t(blockIdx.x) * 4 + threadIdx.x / 32;
    if (id >= uint64_t(rows) * q_heads) return;
    const uint32_t lane = threadIdx.x % 32;
    const uint32_t position = base + uint32_t(id / q_heads);
    const uint32_t kv_head = uint32_t(id % q_heads) / (q_heads / kv_heads);
    const uint64_t q_base = id * dim;
    const float scale = rsqrtf(float(dim));
    float accumulator[8] = {};
    float maximum = -CUDART_INF_F, denominator = 0.0f;
    for (uint32_t tile = 0; tile <= position / 4; ++tile) {
        const uint32_t token = tile * 4 + lane / 8;
        const uint64_t vector = (uint64_t(layer) * context + token) * kv_heads + kv_head;
        float partial[4] = {};
        if (token <= position) {
            for (uint32_t i = lane % 8; i < dim; i += 32) {
                for (uint32_t leaf = 0; leaf < 4; ++leaf) {
                    const uint32_t d = i + leaf * 8;
                    if (d < dim) partial[leaf] += __half2float(query[q_base + d])
                        * cuda_cache_value<INT8>(k_cache, vector * dim + d, k_metadata, vector);
                }
            }
        }
        // Preserve the original 32-leaf tree: levels16/8 locally, then4/2/1 in eight lanes.
        float sum = (partial[0] + partial[2]) + (partial[1] + partial[3]);
        for (uint32_t stride = 4; stride > 0; stride /= 2) {
            const float other = __shfl_down_sync(0xffffffff, sum, stride, 8);
            if (lane % 8 < stride) sum += other;
        }
        const float score = token <= position ? sum * scale : -CUDART_INF_F;
        float weights[4];
        for (uint32_t offset = 0; offset < 4; ++offset) {
            weights[offset] = __shfl_sync(0xffffffff, score, offset * 8);
        }
        float previous = 0.0f;
        if (lane == 0) {
            float next_maximum = maximum;
            for (uint32_t offset = 0; offset < 4; ++offset) {
                next_maximum = fmaxf(next_maximum, weights[offset]);
            }
            previous = expf(maximum - next_maximum);
            float next_denominator = denominator * previous;
            // Reuse score slots only after fixing the maximum for all four tokens.
            for (uint32_t offset = 0; offset < 4; ++offset) {
                weights[offset] = expf(weights[offset] - next_maximum);
                next_denominator += weights[offset];
            }
            denominator = next_denominator;
            maximum = next_maximum;
        }
        previous = __shfl_sync(0xffffffff, previous, 0);
        for (uint32_t offset = 0; offset < 4; ++offset) {
            weights[offset] = __shfl_sync(0xffffffff, weights[offset], 0);
        }
        for (uint32_t part = 0; part < 8; ++part) {
            const uint32_t i = lane + part * 32;
            if (i < dim) {
                float value = accumulator[part] * previous;
                for (uint32_t offset = 0; offset < 4 && tile * 4 + offset <= position; ++offset) {
                    const uint64_t v_vector =
                        (uint64_t(layer) * context + tile * 4 + offset) * kv_heads + kv_head;
                    value += weights[offset] * cuda_cache_value<INT8>(
                        v_cache, v_vector * dim + i, v_metadata, v_vector);
                }
                accumulator[part] = value;
            }
        }
    }
    denominator = __shfl_sync(0xffffffff, denominator, 0);
    for (uint32_t part = 0; part < 8; ++part) {
        const uint32_t i = lane + part * 32;
        if (i < dim) out[q_base + i] = __float2half_rn(accumulator[part] / denominator);
    }
}

extern "C" __global__ void cuda_attention_prefill_tensor_f16(
    const __half *query, const unsigned char *k, const unsigned char *v, __half *out,
    uint32_t dim, uint32_t kvh, uint32_t qh, uint32_t base, uint32_t rows,
    uint32_t layer, uint32_t context) {
    using namespace nvcuda;
    // Checked dispatch fixes dim=128 and128 threads. Eight queries share the padded M16 QK tile.
    const uint32_t first_row = (blockIdx.x / qh) * 8;
    const uint32_t head = blockIdx.x % qh, kv_head = head / (qh / kvh);
    const uint32_t last_row = first_row + 7 < rows ? first_row + 7 : rows - 1;
    const uint32_t last_position = base + last_row;
    __shared__ __align__(32) __half q_tile[16 * 128], k_tile[16 * 128], v_tile[16 * 128];
    __shared__ __align__(32) float scores[16 * 16];
    __shared__ float maximum[8], denominator[8], previous[8];
    float accumulator[8] = {};
    const float scale = rsqrtf(float(dim));
    for (uint32_t i = threadIdx.x; i < 16 * 128; i += 128) {
        const uint32_t row = i / 128;
        q_tile[i] = row < 8 && first_row + row < rows
            ? query[(uint64_t(first_row + row) * qh + head) * 128 + i % 128]
            : __float2half_rn(0.0f);
    }
    if (threadIdx.x < 8) {
        maximum[threadIdx.x] = -CUDART_INF_F;
        denominator[threadIdx.x] = 0.0f;
    }
    // A padded M16 QK tile uses existing F16 inputs; probabilities and PV stay f32.
    for (uint32_t tile = 0; tile <= last_position / 16; ++tile) {
        for (uint32_t i = threadIdx.x; i < 16 * 128; i += 128) {
            const uint32_t token = tile * 16 + i / 128;
            const uint64_t vector = (uint64_t(layer) * context + token) * kvh + kv_head;
            k_tile[i] = token <= last_position
                ? reinterpret_cast<const __half *>(k)[vector * 128 + i % 128]
                : __float2half_rn(0.0f);
            v_tile[i] = token <= last_position
                ? reinterpret_cast<const __half *>(v)[vector * 128 + i % 128]
                : __float2half_rn(0.0f);
        }
        __syncthreads();
        if (threadIdx.x < 32) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> af;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> bf;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> cf;
            wmma::fill_fragment(cf, 0.0f);
            for (uint32_t slice = 0; slice < 128; slice += 16) {
                wmma::load_matrix_sync(af, q_tile + slice, 128);
                wmma::load_matrix_sync(bf, k_tile + slice, 128);
                wmma::mma_sync(cf, af, bf, cf);
            }
            wmma::store_matrix_sync(scores, cf, 16, wmma::mem_row_major);
        }
        __syncthreads();
        if (threadIdx.x < 8) {
            const uint32_t q = threadIdx.x;
            const uint32_t position = base + (first_row + q < rows ? first_row + q : rows - 1);
            float next_maximum = maximum[q];
            for (uint32_t offset = 0; offset < 16; ++offset) {
                const float score = tile * 16 + offset <= position
                    ? scores[q * 16 + offset] * scale : -CUDART_INF_F;
                scores[q * 16 + offset] = score;
                next_maximum = fmaxf(next_maximum, score);
            }
            previous[q] = expf(maximum[q] - next_maximum);
            float next_denominator = denominator[q] * previous[q];
            for (uint32_t offset = 0; offset < 16; ++offset) {
                const float weight = expf(scores[q * 16 + offset] - next_maximum);
                scores[q * 16 + offset] = weight;
                next_denominator += weight;
            }
            maximum[q] = next_maximum;
            denominator[q] = next_denominator;
        }
        __syncthreads();
        for (uint32_t q = 0; q < 8; ++q) {
            if (first_row + q < rows) {
                const uint32_t position = base + first_row + q;
                float value = accumulator[q] * previous[q];
                // Skip masked V entirely: zero times a future NaN is not causal.
                for (uint32_t offset = 0; offset < 16 && tile * 16 + offset <= position; ++offset) {
                    value += scores[q * 16 + offset]
                        * __half2float(v_tile[offset * 128 + threadIdx.x]);
                }
                accumulator[q] = value;
            }
        }
        // Every PV consumer finishes before shared K/V, scores and coefficients change.
        __syncthreads();
    }
    for (uint32_t q = 0; q < 8; ++q) {
        if (first_row + q < rows) {
            out[(uint64_t(first_row + q) * qh + head) * 128 + threadIdx.x] =
                __float2half_rn(accumulator[q] / denominator[q]);
        }
    }
}

extern "C" __global__ void cuda_attention_decode_f16(
    const __half *q, const unsigned char *k, const unsigned char *v, __half *out,
    uint32_t dim, uint32_t kvh, uint32_t qh, uint32_t position,
    uint32_t layer, uint32_t context) {
    cuda_attention_buffered<false>(q, k, v, out, dim, kvh, qh, position, 1, layer, context, 0, 0);
}

extern "C" __global__ void cuda_attention_decode_int8(
    const __half *q, const unsigned char *k, const unsigned char *v, __half *out,
    uint32_t dim, uint32_t kvh, uint32_t qh, uint32_t position,
    uint32_t layer, uint32_t context, uint64_t kmeta, uint64_t vmeta) {
    cuda_attention_buffered<true>(q, k, v, out, dim, kvh, qh, position, 1, layer, context, kmeta, vmeta);
}

extern "C" __global__ void cuda_attention_decode_split_f16(
    const __half *q, const unsigned char *k, const unsigned char *v, __half *out,
    uint32_t dim, uint32_t kvh, uint32_t qh, uint32_t position,
    uint32_t layer, uint32_t context) {
    cuda_attention_buffered<false, 512>(q, k, v, out, dim, kvh, qh, position, 1, layer, context, 0, 0);
}

extern "C" __global__ void cuda_attention_decode_split_int8(
    const __half *q, const unsigned char *k, const unsigned char *v, __half *out,
    uint32_t dim, uint32_t kvh, uint32_t qh, uint32_t position,
    uint32_t layer, uint32_t context, uint64_t kmeta, uint64_t vmeta) {
    cuda_attention_buffered<true, 512>(q, k, v, out, dim, kvh, qh, position, 1, layer, context, kmeta, vmeta);
}

extern "C" __global__ void cuda_attention_prefill_f16(
    const __half *q, const unsigned char *k, const unsigned char *v, __half *out,
    uint32_t dim, uint32_t kvh, uint32_t qh, uint32_t base, uint32_t rows,
    uint32_t layer, uint32_t context) {
    cuda_attention_prefill_warp<false>(
        q, k, v, out, dim, kvh, qh, base, rows, layer, context, 0, 0);
}

extern "C" __global__ void cuda_attention_prefill_int8(
    const __half *q, const unsigned char *k, const unsigned char *v, __half *out,
    uint32_t dim, uint32_t kvh, uint32_t qh, uint32_t base, uint32_t rows,
    uint32_t layer, uint32_t context, uint64_t kmeta, uint64_t vmeta) {
    cuda_attention_prefill_warp<true>(
        q, k, v, out, dim, kvh, qh, base, rows, layer, context, kmeta, vmeta);
}
