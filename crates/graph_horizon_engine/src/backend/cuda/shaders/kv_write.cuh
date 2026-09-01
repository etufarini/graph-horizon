// AGENTS deroga K: famiglia coesa della sola operazione KV encoding.
#pragma once

__device__ __forceinline__ uint8_t cuda_kv_code(float value, float minimum, float scale) {
    if (scale == 0.0f) return 0;
    uint32_t code = 0;
    for (uint32_t step = 128; step > 0; step >>= 1) {
        const uint32_t candidate = code + step;
        const float boundary = minimum + (float(candidate) - 0.5f) * scale;
        if (candidate <= 255 && value >= boundary) code = candidate;
    }
    return uint8_t(code);
}

__device__ __forceinline__ void cuda_quantize_kv(
    const __half *source, unsigned char *target, uint64_t payload,
    uint64_t metadata, uint32_t vector, uint32_t dim) {
    const uint64_t base = uint64_t(vector) * dim;
    float minimum = CUDART_INF_F;
    float maximum = -CUDART_INF_F;
    for (uint32_t i = 0; i < dim; ++i) {
        const float value = __half2float(source[base + i]);
        minimum = fminf(minimum, value);
        maximum = fmaxf(maximum, value);
    }
    const __half stored_minimum = __float2half_rn(minimum);
    const __half stored_scale = __float2half_rn((maximum - minimum) * (1.0f / 255.0f));
    const float rounded_minimum = __half2float(stored_minimum);
    const float rounded_scale = __half2float(stored_scale);
    for (uint32_t i = 0; i < dim; ++i) {
        target[payload + base + i] = cuda_kv_code(
            __half2float(source[base + i]), rounded_minimum, rounded_scale);
    }
    const uint16_t min_bits = *reinterpret_cast<const uint16_t *>(&stored_minimum);
    const uint16_t scale_bits = *reinterpret_cast<const uint16_t *>(&stored_scale);
    const uint64_t meta = metadata + uint64_t(vector) * 4;
    target[meta] = uint8_t(min_bits);
    target[meta + 1] = uint8_t(min_bits >> 8);
    target[meta + 2] = uint8_t(scale_bits);
    target[meta + 3] = uint8_t(scale_bits >> 8);
}

extern "C" __global__ void cuda_kv_write_f16(
    const __half *k, const __half *v, unsigned char *k_cache, unsigned char *v_cache,
    uint64_t k_payload, uint64_t v_payload, uint32_t vectors, uint32_t dim) {
    const uint32_t vector = blockIdx.x * blockDim.x + threadIdx.x;
    if (vector >= vectors) return;
    for (uint32_t i = 0; i < dim; ++i) {
        reinterpret_cast<__half *>(k_cache + k_payload)[uint64_t(vector) * dim + i] =
            k[uint64_t(vector) * dim + i];
        reinterpret_cast<__half *>(v_cache + v_payload)[uint64_t(vector) * dim + i] =
            v[uint64_t(vector) * dim + i];
    }
}

extern "C" __global__ void cuda_kv_write_int8(
    const __half *k, const __half *v, unsigned char *k_cache, unsigned char *v_cache,
    uint64_t k_payload, uint64_t v_payload, uint64_t k_metadata, uint64_t v_metadata,
    uint32_t vectors, uint32_t dim) {
    const uint32_t vector = blockIdx.x * blockDim.x + threadIdx.x;
    if (vector >= vectors) return;
    cuda_quantize_kv(k, k_cache, k_payload, k_metadata, vector, dim);
    cuda_quantize_kv(v, v_cache, v_payload, v_metadata, vector, dim);
}
