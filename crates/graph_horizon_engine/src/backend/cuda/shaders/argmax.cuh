// AGENTS deroga K: kernel della sola operazione argmax reduction.
#pragma once

__device__ __forceinline__ int32_t cuda_total_key(float value) {
    int32_t bits = __float_as_int(value);
    bits ^= int32_t(uint32_t(bits >> 31) >> 1);
    return bits;
}

extern "C" __global__ void cuda_argmax(const float *values, uint32_t *out, uint32_t length) {
    if (blockIdx.x != 0 || threadIdx.x != 0 || length == 0) return;
    uint32_t best = 0;
    int32_t key = cuda_total_key(values[0]);
    for (uint32_t i = 1; i < length; ++i) {
        const int32_t candidate = cuda_total_key(values[i]);
        if (candidate > key) {
            best = i;
            key = candidate;
        }
    }
    out[0] = best;
}
