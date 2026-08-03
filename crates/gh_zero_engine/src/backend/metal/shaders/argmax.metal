/*
 * gh_zero_engine — Metal argmax kernel
 * Owns one deterministic finite-logit reduction; no dispatch or resource ownership.
 */
// AGENTS deroga K: kernel della sola operazione argmax reduction.
#include <metal_stdlib>
using namespace metal;
kernel void metal_argmax(device const float*x[[buffer(0)]],device uint*out[[buffer(1)]],constant uint&n[[buffer(2)]],uint id[[thread_position_in_grid]]){if(id)return;float best=-INFINITY;uint index=0;for(uint i=0;i<n;i++)if(x[i]>best){best=x[i];index=i;}out[0]=index;}
