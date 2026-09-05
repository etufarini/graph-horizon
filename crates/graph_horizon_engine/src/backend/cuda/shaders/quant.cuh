// AGENTS deroga K: famiglia coesa di dequantizzazione dei pesi.
#pragma once
#include <cuda_fp16.h>
#include <math_constants.h>
#include <stdint.h>

__device__ __forceinline__ float cuda_half_at(const unsigned char *w, uint64_t b) {
    const uint16_t bits = uint16_t(w[b]) | (uint16_t(w[b + 1]) << 8);
    return __half2float(*reinterpret_cast<const __half *>(&bits));
}

__device__ __forceinline__ void cuda_scale_min(
    const unsigned char *w, uint64_t b, uint32_t j, uint32_t &scale, uint32_t &minimum) {
    if (j < 4) {
        scale = w[b + j] & 63;
        minimum = w[b + j + 4] & 63;
    } else {
        const uint32_t high = w[b + j + 4];
        const uint32_t low = w[b + j - 4];
        scale = (high & 15) | ((low >> 6) << 4);
        minimum = (high >> 4) | ((uint32_t(w[b + j]) >> 6) << 4);
    }
}

__device__ __forceinline__ float cuda_q4_value(
    const unsigned char *w, uint64_t block, uint32_t i, bool five) {
    const uint32_t group = i / 64;
    const uint32_t lane = i % 64;
    uint32_t scale, minimum;
    cuda_scale_min(w, block + 4, group * 2 + lane / 32, scale, minimum);
    const uint64_t qoff = block + (five ? 48 : 16) + group * 32 + lane % 32;
    uint32_t q = lane < 32 ? (w[qoff] & 15) : (w[qoff] >> 4);
    if (five) {
        q += ((w[block + 16 + lane % 32] >> (group * 2 + lane / 32)) & 1) * 16;
    }
    return cuda_half_at(w, block) * float(scale) * float(q)
        - cuda_half_at(w, block + 2) * float(minimum);
}

__device__ __forceinline__ float cuda_q6_value(
    const unsigned char *w, uint64_t block, uint32_t i) {
    const uint32_t segment = i / 128;
    const uint32_t category = (i % 128) / 32;
    const uint32_t lane = i % 32;
    const uint64_t qoff = block + segment * 64 + (category & 1) * 32 + lane;
    const uint32_t low = category < 2 ? (w[qoff] & 15) : (w[qoff] >> 4);
    const uint32_t high =
        (w[block + 128 + segment * 32 + lane] >> (category * 2)) & 3;
    const int scale = int(reinterpret_cast<const int8_t *>(w)[
        block + 192 + segment * 8 + lane / 16 + category * 2]);
    return cuda_half_at(w, block + 208) * float(scale)
        * (float(low | (high << 4)) - 32.0f);
}

__device__ __forceinline__ float cuda_weight_value(
    const unsigned char *w, uint32_t format, uint32_t row, uint32_t i, uint32_t width) {
    if (format == 0) {
        return __half2float(reinterpret_cast<const __half *>(w)[uint64_t(row) * width + i]);
    }
    const uint64_t blocks = width / 256;
    const uint64_t block_index = uint64_t(row) * blocks + i / 256;
    const uint32_t q = i % 256;
    if (format == 1) return cuda_q4_value(w, block_index * 144, q, false);
    if (format == 2) return cuda_q4_value(w, block_index * 176, q, true);
    return cuda_q6_value(w, block_index * 210, q);
}

template<uint32_t FORMAT>
__device__ __forceinline__ float cuda_weight_parts(
    const unsigned char *w, uint32_t row, uint32_t i, uint32_t width,
    float &factor, float &bias) {
    // Every aligned K16 group has constant factor/bias. The returned integer
    // fits half exactly; reconstructed weights need not fit half's range.
    const uint64_t index = uint64_t(row) * (width / 256) + i / 256;
    const uint32_t qindex = i % 256;
    if (FORMAT == 1 || FORMAT == 2) {
        const uint64_t block = index * (FORMAT == 1 ? 144 : 176);
        const uint32_t group = qindex / 64;
        const uint32_t lane = qindex % 64;
        uint32_t scale, minimum;
        cuda_scale_min(w, block + 4, group * 2 + lane / 32, scale, minimum);
        const uint64_t qoff = block + (FORMAT == 1 ? 16 : 48) + group * 32 + lane % 32;
        uint32_t q = lane < 32 ? (w[qoff] & 15) : (w[qoff] >> 4);
        if (FORMAT == 2) {
            q += ((w[block + 16 + lane % 32] >> (group * 2 + lane / 32)) & 1) * 16;
        }
        factor = cuda_half_at(w, block) * float(scale);
        bias = -cuda_half_at(w, block + 2) * float(minimum);
        return float(q);
    }
    const uint64_t block = index * 210;
    const uint32_t segment = qindex / 128;
    const uint32_t category = (qindex % 128) / 32;
    const uint32_t lane = qindex % 32;
    const uint64_t qoff = block + segment * 64 + (category & 1) * 32 + lane;
    const uint32_t low = category < 2 ? (w[qoff] & 15) : (w[qoff] >> 4);
    const uint32_t high = (w[block + 128 + segment * 32 + lane] >> (category * 2)) & 3;
    const int scale = int(reinterpret_cast<const int8_t *>(w)[
        block + 192 + segment * 8 + lane / 16 + category * 2]);
    factor = cuda_half_at(w, block + 208) * float(scale);
    bias = 0.0f;
    return float(low | (high << 4)) - 32.0f;
}
