/*
 * graph_horizon_engine — Metal RMS normalization kernel
 * Owns only FP32 accumulation and FP16 row output; no dispatch or resource ownership.
 */
// AGENTS deroga K: kernel della sola operazione RMS normalization.
#include <metal_stdlib>
using namespace metal;

struct Params {
    uint dim;
    float eps;
    uint rows;
};

kernel void metal_rmsnorm(
    device const float *x [[buffer(0)]],
    device const half *w [[buffer(1)]],
    device half *out [[buffer(2)]],
    constant Params &p [[buffer(3)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]])
{
    threadgroup float sums[256];
    uint base = row * p.dim;
    float sum = 0.0f;
    for (uint i = lane; i < p.dim; i += 256) {
        float value = x[base + i];
        sum += value * value;
    }
    sums[lane] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Every lane reaches every barrier; the lower half owns each reduction
    // step until sums[0] contains the complete row sum.
    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (lane < stride) {
            sums[lane] += sums[lane + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float inverse_rms = rsqrt(sums[0] / float(p.dim) + p.eps);
    for (uint i = lane; i < p.dim; i += 256) {
        out[base + i] = half(x[base + i] * inverse_rms * float(w[i]));
    }
}
