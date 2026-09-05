// AGENTS deroga K: famiglia coesa di dequantizzazione dei pesi.
#pragma once
#include <cuda_fp16.h>
#include <math_constants.h>
#include <stdint.h>

__device__ __forceinline__ float cuda_half_at(const unsigned char *w, uint64_t b) {
    // Packed strides/metadata offsets are even; checked views preserve allocation alignment.
    return __half2float(*reinterpret_cast<const __half *>(w + b));
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
__device__ __forceinline__ void cuda_weight_pair(
    const unsigned char *w, uint64_t block, uint32_t i, float &first, float &paired) {
    // i owns the low nibble; its paired logical leaf owns the same byte's high nibble.
    if (FORMAT == 1 || FORMAT == 2) {
        const uint32_t group = i / 64, lane = i % 32;
        const uint32_t packed = w[block + (FORMAT == 1 ? 16 : 48) + group * 32 + lane];
        uint32_t low = packed & 15, high = packed >> 4;
        if (FORMAT == 2) {
            const uint32_t bits = w[block + 16 + lane];
            low += ((bits >> (group * 2)) & 1) * 16;
            high += ((bits >> (group * 2 + 1)) & 1) * 16;
        }
        uint32_t scale, minimum, paired_scale, paired_minimum;
        cuda_scale_min(w, block + 4, group * 2, scale, minimum);
        cuda_scale_min(w, block + 4, group * 2 + 1, paired_scale, paired_minimum);
        const float d = cuda_half_at(w, block), dmin = cuda_half_at(w, block + 2);
        first = d * float(scale) * float(low) - dmin * float(minimum);
        paired = d * float(paired_scale) * float(high) - dmin * float(paired_minimum);
    } else {
        const uint32_t segment = i / 128, category = (i % 128) / 32, lane = i % 32;
        const uint32_t low = w[block + segment * 64 + category * 32 + lane];
        const uint32_t high = w[block + 128 + segment * 32 + lane];
        const uint64_t offset = block + 192 + segment * 8 + lane / 16 + category * 2;
        const int scale = int(reinterpret_cast<const int8_t *>(w)[offset]);
        const int paired_scale = int(reinterpret_cast<const int8_t *>(w)[offset + 4]);
        const float d = cuda_half_at(w, block + 208);
        first = d * float(scale)
            * (float((low & 15) | (((high >> (category * 2)) & 3) << 4)) - 32.0f);
        paired = d * float(paired_scale)
            * (float((low >> 4) | (((high >> (category * 2 + 4)) & 3) << 4)) - 32.0f);
    }
}

template<uint32_t FORMAT>
__device__ __forceinline__ void cuda_quant_pair(
    const unsigned char *w, uint32_t row, uint32_t i, uint32_t width,
    float &first, float &paired) {
    // Q4/Q5 only: i is in the low K32 half of a full aligned K64 group.
    const uint64_t block = (uint64_t(row) * (width / 256) + i / 256)
        * (FORMAT == 1 ? 144 : 176);
    const uint32_t group = (i % 256) / 64, lane = i % 32;
    const uint32_t packed = w[block + (FORMAT == 1 ? 16 : 48) + group * 32 + lane];
    uint32_t low = packed & 15, high = packed >> 4;
    if (FORMAT == 2) {
        const uint32_t bits = w[block + 16 + lane];
        low += ((bits >> (group * 2)) & 1) * 16;
        high += ((bits >> (group * 2 + 1)) & 1) * 16;
    }
    first = float(low);
    paired = float(high);
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
