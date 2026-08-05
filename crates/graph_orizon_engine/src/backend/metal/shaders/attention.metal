/*
 * graph_orizon_engine — Metal causal attention kernel family
 * Owns F16/int8 GQA decode and prefill math only; no dispatch or resources.
 */
// AGENTS deroga K: varianti coese della sola operazione causal attention.
#include <metal_stdlib>
using namespace metal;
struct P{uint dim;uint kvh;uint qh;uint base;uint rows;uint layer;uint context;uint scheme;ulong kmeta;ulong vmeta;float scale;uint parallel;};
inline float hv(device const uchar*c,ulong i){return float(((device const half*)c)[i]);}
inline float qv(device const uchar*c,ulong i,ulong meta,uint vec){ulong m=meta+ulong(vec)*4;ushort mb=ushort(uint(c[m])|(uint(c[m+1])<<8)),sb=ushort(uint(c[m+2])|(uint(c[m+3])<<8));return float(as_type<half>(mb))+float(c[i])*float(as_type<half>(sb));}
kernel void metal_attention(device const half*q[[buffer(0)]],device const uchar*k[[buffer(1)]],device const uchar*v[[buffer(2)]],device half*out[[buffer(3)]],constant P&p[[buffer(4)]],uint tid[[thread_position_in_grid]],uint id[[threadgroup_position_in_grid]],uint lane[[thread_index_in_simdgroup]],uint lanes[[threads_per_simdgroup]]){
 uint total=p.rows*p.qh;
 if(!p.parallel){
  if(tid>=total)return;uint row=tid/p.qh,h=tid%p.qh,pos=p.base+row,kh=h/(p.qh/p.kvh),qb=(row*p.qh+h)*p.dim;float m=-INFINITY,l=0.0f;float acc[256];
  for(uint d=0;d<p.dim;d++)acc[d]=0.0f;
  for(uint t=0;t<=pos;t++){uint vec=(p.layer*p.context+t)*p.kvh+kh;ulong b=ulong(vec)*p.dim;float dot=0.0f;for(uint d=0;d<p.dim;d++)dot+=float(q[qb+d])*(p.scheme==0?hv(k,b+d):qv(k,b+d,p.kmeta,vec));float score=dot*p.scale,nm=max(m,score),a=exp(m-nm),w=exp(score-nm);l=l*a+w;for(uint d=0;d<p.dim;d++)acc[d]=acc[d]*a+w*(p.scheme==0?hv(v,b+d):qv(v,b+d,p.vmeta,vec));m=nm;}
  for(uint d=0;d<p.dim;d++)out[qb+d]=half(acc[d]/l);return;
 }
 if(id>=total)return;uint row=id/p.qh,h=id%p.qh,pos=p.base+row,kh=h/(p.qh/p.kvh),qb=(row*p.qh+h)*p.dim;
 // One SIMD group owns one head. Each lane retains at most eight output values,
 // avoiding the serial kernel's 256-float private accumulator and register spills.
 float m=-INFINITY,l=0.0f,acc[8];uint parts=(p.dim+lanes-1)/lanes;
 for(uint part=0;part<parts;part++)acc[part]=0.0f;
 for(uint t=0;t<=pos;t++){
  uint vec=(p.layer*p.context+t)*p.kvh+kh;ulong b=ulong(vec)*p.dim;float dot=0.0f;
  for(uint d=lane;d<p.dim;d+=lanes)dot+=float(q[qb+d])*(p.scheme==0?hv(k,b+d):qv(k,b+d,p.kmeta,vec));
  float score=simd_sum(dot)*p.scale,nm=max(m,score),a=exp(m-nm),w=exp(score-nm);l=l*a+w;
  for(uint part=0;part<parts;part++){uint d=lane+part*lanes;if(d<p.dim)acc[part]=acc[part]*a+w*(p.scheme==0?hv(v,b+d):qv(v,b+d,p.vmeta,vec));}
  m=nm;
 }
 for(uint part=0;part<parts;part++){uint d=lane+part*lanes;if(d<p.dim)out[qb+d]=half(acc[part]/l);}
}
