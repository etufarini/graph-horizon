/*
 * graph_horizon_engine — Metal RMS normalization kernel
 * Owns only FP32 accumulation and FP16 row output; no dispatch or resource ownership.
 */
// AGENTS deroga K: kernel della sola operazione RMS normalization.
#include <metal_stdlib>
using namespace metal;
struct Params{uint dim;float eps;uint rows;};
kernel void metal_rmsnorm(device const float*x[[buffer(0)]],device const half*w[[buffer(1)]],device half*out[[buffer(2)]],constant Params&p[[buffer(3)]],uint row[[thread_position_in_grid]]){if(row>=p.rows)return;uint b=row*p.dim;float s=0;for(uint i=0;i<p.dim;i++){float v=x[b+i];s+=v*v;}float inv=rsqrt(s/float(p.dim)+p.eps);for(uint i=0;i<p.dim;i++)out[b+i]=half(x[b+i]*inv*float(w[i]));}
