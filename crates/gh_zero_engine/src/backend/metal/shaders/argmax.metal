/*
 * gh_zero_engine — Metal argmax kernel
 * Owns one deterministic SIMD-group finite-logit reduction; no dispatch or resources.
 */
// AGENTS deroga K: kernel della sola operazione argmax reduction.
#include <metal_stdlib>
using namespace metal;
kernel void metal_argmax(device const float*x[[buffer(0)]],device uint*out[[buffer(1)]],constant uint&n[[buffer(2)]],uint lane[[thread_index_in_simdgroup]],uint lanes[[threads_per_simdgroup]]){
 float best=-INFINITY;uint index=lane<n?lane:UINT_MAX;
 for(uint i=lane;i<n;i+=lanes){float value=x[i];if(value>best){best=value;index=i;}}
 float maximum=simd_max(best);
 // Local scans are ascending; the second reduction preserves first-index ties globally.
 uint candidate=best==maximum?index:UINT_MAX;
 uint winner=simd_min(candidate);
 if(lane==0)out[0]=winner;
}
