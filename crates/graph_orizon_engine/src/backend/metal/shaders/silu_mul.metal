/*
 * graph_orizon_engine — Metal SiLU-multiply kernel
 * Owns only the FP16-rounded fused numeric operation; no dispatch or resources.
 */
// AGENTS deroga K: kernel della sola operazione SiLU multiplication.
#include <metal_stdlib>
using namespace metal;
kernel void metal_silu_mul(device const half*g[[buffer(0)]],device const half*u[[buffer(1)]],device half*out[[buffer(2)]],constant uint&n[[buffer(3)]],uint i[[thread_position_in_grid]]){if(i<n){float v=float(g[i]);half a=half(v/(1.0f+exp(-v)));out[i]=half(float(a)*float(u[i]));}}
