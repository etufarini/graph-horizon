// AGENTS deroga K: kernel della sola operazione residual addition.
#pragma once

extern "C" __global__ void cuda_residual_add(float *x, const __half *y, uint32_t length) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < length) x[i] += __half2float(y[i]);
}
