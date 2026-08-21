/*
 * graph_horizon_engine — Metal SiLU-multiply kernel
 * Owns only the FP16-rounded fused numeric operation; no dispatch or resources.
 */
// AGENTS deroga K: kernel della sola operazione SiLU multiplication.
#include <metal_stdlib>
using namespace metal;
struct P{uint n;uint fp32;};
inline half input(device const uchar*x,uint i,uint fp32){return fp32?half(((device const float*)x)[i]):((device const half*)x)[i];}
kernel void metal_silu_mul(device const uchar*g[[buffer(0)]],device const uchar*u[[buffer(1)]],device half*out[[buffer(2)]],constant P&p[[buffer(3)]],uint i[[thread_position_in_grid]]){if(i<p.n){float v=float(input(g,i,p.fp32));half a=half(v/(1.0f+exp(-v)));out[i]=half(float(a)*float(input(u,i,p.fp32)));}}
