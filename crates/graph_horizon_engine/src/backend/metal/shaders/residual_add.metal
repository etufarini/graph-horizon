/*
 * graph_horizon_engine — Metal residual-add kernel
 * Owns only FP16-to-FP32 in-place addition; no dispatch or resource ownership.
 */
// AGENTS deroga K: kernel della sola operazione residual addition.
#include <metal_stdlib>
using namespace metal;
struct P{uint n;uint fp32;};
kernel void metal_residual_add(device float*x[[buffer(0)]],device const uchar*y[[buffer(1)]],constant P&p[[buffer(2)]],uint i[[thread_position_in_grid]]){if(i<p.n){half rounded=p.fp32?half(((device const float*)y)[i]):((device const half*)y)[i];x[i]+=float(rounded);}}
