// AGENTS deroga K: famiglia coesa della sola operazione causal attention.
#pragma once

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
__device__ void cuda_attention_body(
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

extern "C" __global__ void cuda_attention_decode_f16(
    const __half *q, const unsigned char *k, const unsigned char *v, __half *out,
    uint32_t dim, uint32_t kvh, uint32_t qh, uint32_t position,
    uint32_t layer, uint32_t context) {
    cuda_attention_body<false>(q, k, v, out, dim, kvh, qh, position, 1, layer, context, 0, 0);
}

extern "C" __global__ void cuda_attention_decode_int8(
    const __half *q, const unsigned char *k, const unsigned char *v, __half *out,
    uint32_t dim, uint32_t kvh, uint32_t qh, uint32_t position,
    uint32_t layer, uint32_t context, uint64_t kmeta, uint64_t vmeta) {
    cuda_attention_body<true>(q, k, v, out, dim, kvh, qh, position, 1, layer, context, kmeta, vmeta);
}

extern "C" __global__ void cuda_attention_prefill_f16(
    const __half *q, const unsigned char *k, const unsigned char *v, __half *out,
    uint32_t dim, uint32_t kvh, uint32_t qh, uint32_t base, uint32_t rows,
    uint32_t layer, uint32_t context) {
    cuda_attention_body<false>(
        q, k, v, out, dim, kvh, qh, base, rows, layer, context, 0, 0);
}

extern "C" __global__ void cuda_attention_prefill_int8(
    const __half *q, const unsigned char *k, const unsigned char *v, __half *out,
    uint32_t dim, uint32_t kvh, uint32_t qh, uint32_t base, uint32_t rows,
    uint32_t layer, uint32_t context, uint64_t kmeta, uint64_t vmeta) {
    cuda_attention_body<true>(
        q, k, v, out, dim, kvh, qh, base, rows, layer, context, kmeta, vmeta);
}
