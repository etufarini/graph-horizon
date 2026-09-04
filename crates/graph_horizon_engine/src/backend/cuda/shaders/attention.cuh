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
    const uint32_t id = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t total = rows * q_heads;
    if (id >= total) return;
    const uint32_t row = id / q_heads;
    const uint32_t q_head = id % q_heads;
    const uint32_t kv_head = q_head / (q_heads / kv_heads);
    const uint32_t position = base + row;
    const uint64_t q_base = uint64_t(id) * dim;
    float maximum = -CUDART_INF_F;
    float denominator = 0.0f;
    float accumulator[256];
    for (uint32_t d = 0; d < dim; ++d) accumulator[d] = 0.0f;
    const float scale = rsqrtf(float(dim));
    for (uint32_t token = 0; token <= position; ++token) {
        const uint64_t vector = (uint64_t(layer) * context + token) * kv_heads + kv_head;
        const uint64_t cache_base = vector * dim;
        float dot = 0.0f;
        for (uint32_t d = 0; d < dim; ++d) {
            dot += __half2float(query[q_base + d])
                * cuda_cache_value<INT8>(k_cache, cache_base + d, k_metadata, vector);
        }
        const float score = dot * scale;
        const float next_maximum = fmaxf(maximum, score);
        const float previous = expf(maximum - next_maximum);
        const float weight = expf(score - next_maximum);
        denominator = denominator * previous + weight;
        for (uint32_t d = 0; d < dim; ++d) {
            accumulator[d] = accumulator[d] * previous + weight
                * cuda_cache_value<INT8>(v_cache, cache_base + d, v_metadata, vector);
        }
        maximum = next_maximum;
    }
    for (uint32_t d = 0; d < dim; ++d) {
        out[q_base + d] = __float2half_rn(accumulator[d] / denominator);
    }
}

template <bool INT8>
__device__ void cuda_attention_prefill_body(
    const __half *query, const unsigned char *k_cache, const unsigned char *v_cache,
    __half *out, uint32_t dim, uint32_t kv_heads, uint32_t q_heads,
    uint32_t base, uint32_t rows, uint32_t layer, uint32_t context,
    uint64_t k_metadata, uint64_t v_metadata) {
    const uint32_t id = blockIdx.x;
    const uint32_t row = id / q_heads;
    const uint32_t q_head = id % q_heads;
    const uint32_t kv_head = q_head / (q_heads / kv_heads);
    const uint32_t position = base + row;
    const uint64_t q_base = uint64_t(id) * dim;
    const bool active = threadIdx.x < dim;
    float accumulator = 0.0f;
    float maximum = -CUDART_INF_F;
    __shared__ float partial[256];
    __shared__ float previous;
    __shared__ float weight;
    __shared__ float denominator;
    if (threadIdx.x == 0) denominator = 0.0f;
    const float scale = rsqrtf(float(dim));
    for (uint32_t token = 0; token <= position; ++token) {
        const uint64_t vector = (uint64_t(layer) * context + token) * kv_heads + kv_head;
        const uint64_t cache_base = vector * dim;
        partial[threadIdx.x] = active
            ? __half2float(query[q_base + threadIdx.x])
                * cuda_cache_value<INT8>(
                    k_cache, cache_base + threadIdx.x, k_metadata, vector)
            : 0.0f;
        __syncthreads();
        for (uint32_t stride = blockDim.x / 2; stride > 0; stride /= 2) {
            if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            const float score = partial[0] * scale;
            const float next_maximum = fmaxf(maximum, score);
            previous = expf(maximum - next_maximum);
            weight = expf(score - next_maximum);
            denominator = denominator * previous + weight;
            maximum = next_maximum;
        }
        __syncthreads();
        if (active) {
            accumulator = accumulator * previous + weight
                * cuda_cache_value<INT8>(
                    v_cache, cache_base + threadIdx.x, v_metadata, vector);
        }
        __syncthreads();
    }
    if (active) out[q_base + threadIdx.x] = __float2half_rn(accumulator / denominator);
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
    cuda_attention_prefill_body<false>(
        q, k, v, out, dim, kvh, qh, base, rows, layer, context, 0, 0);
}

extern "C" __global__ void cuda_attention_prefill_int8(
    const __half *q, const unsigned char *k, const unsigned char *v, __half *out,
    uint32_t dim, uint32_t kvh, uint32_t qh, uint32_t base, uint32_t rows,
    uint32_t layer, uint32_t context, uint64_t kmeta, uint64_t vmeta) {
    cuda_attention_prefill_body<true>(
        q, k, v, out, dim, kvh, qh, base, rows, layer, context, kmeta, vmeta);
}
