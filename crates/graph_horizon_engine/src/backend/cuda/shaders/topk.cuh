// AGENTS deroga K: kernel della sola operazione top-k reduction.
#pragma once

extern "C" __global__ void cuda_topk(
    const float *values, unsigned char *storage, uint32_t vocab, uint32_t requested) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    const uint32_t count = requested < vocab ? requested : vocab;
    uint32_t *indices = reinterpret_cast<uint32_t *>(storage);
    float *selected = reinterpret_cast<float *>(indices + count);
    for (uint32_t rank = 0; rank < count; ++rank) {
        uint32_t best = UINT32_MAX;
        int32_t best_key = INT32_MIN;
        for (uint32_t i = 0; i < vocab; ++i) {
            bool used = false;
            for (uint32_t prior = 0; prior < rank; ++prior) used |= indices[prior] == i;
            if (used) continue;
            const int32_t key = cuda_total_key(values[i]);
            if (best == UINT32_MAX || key > best_key || (key == best_key && i < best)) {
                best = i;
                best_key = key;
            }
        }
        indices[rank] = best;
        selected[rank] = values[best];
    }
}
