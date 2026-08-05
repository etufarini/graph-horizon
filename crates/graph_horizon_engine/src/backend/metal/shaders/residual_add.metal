/*
 * graph_horizon_engine — Metal residual-add kernel
 * Owns only FP16-to-FP32 in-place addition; no dispatch or resource ownership.
 */
// AGENTS deroga K: kernel della sola operazione residual addition.
#include <metal_stdlib>
using namespace metal;
kernel void metal_residual_add(device float*x[[buffer(0)]],device const half*y[[buffer(1)]],constant uint&n[[buffer(2)]],uint i[[thread_position_in_grid]]){if(i<n)x[i]+=float(y[i]);}
