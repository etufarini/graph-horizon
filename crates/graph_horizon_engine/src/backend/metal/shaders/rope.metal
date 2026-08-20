/*
 * graph_horizon_engine — Metal YaRN RoPE kernel
 * Owns only role-scaled in-place rotary math; no dispatch or resource ownership.
 */
// AGENTS deroga K: kernel della sola operazione YaRN rotary embedding.
#include <metal_stdlib>
using namespace metal;
struct P{uint heads;uint dim;uint rope;uint pos;uint rows;float base;float factor;float fast;float slow;float original;float scale;};
kernel void metal_rope(device half*x[[buffer(0)]],constant P&p[[buffer(1)]],uint id[[thread_position_in_grid]]){
 uint halfrope=p.rope/2,rowpairs=p.heads*halfrope,total=p.rows*rowpairs;if(id>=total||halfrope==0)return;
 uint row=id/rowpairs,pair=id%rowpairs,head=pair/halfrope,j=pair%halfrope,b=(row*p.heads+head)*p.dim;
 float corr1=float(p.rope)*log(p.original/(p.fast*2.0f*M_PI_F))/(2.0f*log(p.base));float corr2=float(p.rope)*log(p.original/(p.slow*2.0f*M_PI_F))/(2.0f*log(p.base));
 float ramp=1.0f-clamp((float(j)-max(floor(corr1),0.0f))/max(min(ceil(corr2),float(p.rope-1))-max(floor(corr1),0.0f),0.001f),0.0f,1.0f);
 float ex=float(p.pos+row)*pow(p.base,-2.0f*float(j)/float(p.rope)),theta=(ex/p.factor)*(1.0f-ramp)+ex*ramp;float c=cos(theta),s=sin(theta),a=float(x[b+2*j]),d=float(x[b+2*j+1]);
 x[b+2*j]=half((a*c-d*s)*p.scale);x[b+2*j+1]=half((a*s+d*c)*p.scale);
}
