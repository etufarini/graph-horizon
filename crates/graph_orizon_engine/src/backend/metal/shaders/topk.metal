/*
 * graph_orizon_engine — Metal top-k kernel
 * Owns one deterministic bounded ordering; no dispatch or resource ownership.
 */
// AGENTS deroga K: kernel della sola operazione top-k reduction.
#include <metal_stdlib>
using namespace metal;
struct P{uint vocab;uint k;};struct Pair{uint index;float value;};
kernel void metal_topk(device const float*x[[buffer(0)]],device Pair*out[[buffer(1)]],constant P&p[[buffer(2)]],uint id[[thread_position_in_grid]]){if(id)return;uint k=min(p.k,p.vocab);for(uint r=0;r<k;r++){float best=-INFINITY;uint bi=UINT_MAX;for(uint i=0;i<p.vocab;i++){bool used=false;for(uint j=0;j<r;j++)used|=out[j].index==i;if(!used&&(x[i]>best||(x[i]==best&&i<bi))){best=x[i];bi=i;}}out[r]={bi,best};}}
