// AGENTS deroga K: kernel della sola operazione argmax reduction.
#pragma once

__device__ __forceinline__ int32_t cuda_total_key(float value) {
    int32_t bits = __float_as_int(value);
    bits ^= int32_t(uint32_t(bits >> 31) >> 1);
    return bits;
}

extern "C" __global__ void cuda_argmax(const float *values, uint32_t *out, uint32_t length) {
    if (blockIdx.x != 0 || length == 0) return;
    uint32_t best = UINT32_MAX;
    int32_t key = INT32_MIN;
    // A valid minimum-key NaN still beats an inactive lane by its lower index.
    for (uint64_t i = threadIdx.x; i < length; i += 256) {
        const int32_t candidate = cuda_total_key(values[i]);
        if (candidate > key || (candidate == key && i < best)) {
            best = uint32_t(i);
            key = candidate;
        }
    }
    __shared__ int32_t keys[256];
    __shared__ uint32_t indices[256];
    keys[threadIdx.x] = key;
    indices[threadIdx.x] = best;
    __syncthreads();
    for (uint32_t stride = 128; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            const uint32_t other = threadIdx.x + stride;
            if (keys[other] > keys[threadIdx.x] ||
                (keys[other] == keys[threadIdx.x] && indices[other] < indices[threadIdx.x])) {
                keys[threadIdx.x] = keys[other];
                indices[threadIdx.x] = indices[other];
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) out[0] = indices[0];
}
