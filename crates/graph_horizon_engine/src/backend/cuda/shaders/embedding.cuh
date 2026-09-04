// AGENTS deroga K: famiglia della sola operazione embedding.
#pragma once

extern "C" __global__ void cuda_embedding(
    const unsigned char *weight, float *out, uint32_t token, uint32_t width, uint32_t format) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < width) out[i] = cuda_weight_value(weight, format, token, i, width);
}
