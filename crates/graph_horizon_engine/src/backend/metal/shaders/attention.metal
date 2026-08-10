/*
 * graph_horizon_engine — Metal causal attention kernel family
 * Owns F16/int8 GQA decode and prefill math only; no dispatch or resources.
 */
// AGENTS deroga K: varianti coese della sola operazione causal attention.
#include <metal_stdlib>
using namespace metal;
struct P{uint dim;uint kvh;uint qh;uint base;uint rows;uint layer;uint context;uint scheme;ulong kmeta;ulong vmeta;float scale;uint mode;};
inline float hv(device const uchar*c,ulong i){return float(((device const half*)c)[i]);}
inline float qv(device const uchar*c,ulong i,ulong meta,uint vec){ulong m=meta+ulong(vec)*4;ushort mb=ushort(uint(c[m])|(uint(c[m+1])<<8)),sb=ushort(uint(c[m+2])|(uint(c[m+3])<<8));return float(as_type<half>(mb))+float(c[i])*float(as_type<half>(sb));}
kernel void metal_attention(device const half*q[[buffer(0)]],device const uchar*k[[buffer(1)]],device const uchar*v[[buffer(2)]],device half*out[[buffer(3)]],constant P&p[[buffer(4)]],uint tid[[thread_position_in_grid]],uint id[[threadgroup_position_in_grid]],uint lane[[thread_index_in_simdgroup]],uint lanes[[threads_per_simdgroup]],uint group[[simdgroup_index_in_threadgroup]]){
 uint total=p.rows*p.qh;
 if(p.mode==0){
  if(tid>=total)return;uint row=tid/p.qh,h=tid%p.qh,pos=p.base+row,kh=h/(p.qh/p.kvh),qb=(row*p.qh+h)*p.dim;float m=-INFINITY,l=0.0f;float acc[256];
  for(uint d=0;d<p.dim;d++)acc[d]=0.0f;
  for(uint t=0;t<=pos;t++){uint vec=(p.layer*p.context+t)*p.kvh+kh;ulong b=ulong(vec)*p.dim;float dot=0.0f;for(uint d=0;d<p.dim;d++)dot+=float(q[qb+d])*(p.scheme==0?hv(k,b+d):qv(k,b+d,p.kmeta,vec));float score=dot*p.scale,nm=max(m,score),a=exp(m-nm),w=exp(score-nm);l=l*a+w;for(uint d=0;d<p.dim;d++)acc[d]=acc[d]*a+w*(p.scheme==0?hv(v,b+d):qv(v,b+d,p.vmeta,vec));m=nm;}
  for(uint d=0;d<p.dim;d++)out[qb+d]=half(acc[d]/l);return;
 }
 if(p.mode==4){
  // Four query rows and the four GQA heads sharing one KV head reuse each
  // K/V tile. Every thread participates in both barriers before tile reuse.
  threadgroup half tk[32*128],tv[32*128];uint local=group*32+lane,row0=(id/p.kvh)*4,kh=id%p.kvh,row=row0+group,h=kh*4+(lane>>3),sub=lane&7,pos=p.base+row,qb=(row*p.qh+h)*128;float m=-INFINITY,l=0.0f,acc[16];
  for(uint part=0;part<16;part++)acc[part]=0.0f;
  for(uint tile=0;tile<=p.base+row0+3;tile+=32){
   for(uint index=local;index<32*128;index+=128){uint t=tile+index/128,d=index%128;if(t<p.context){uint vec=(p.layer*p.context+t)*p.kvh+kh;ulong b=ulong(vec)*128+d;tk[index]=((device const half*)k)[b];tv[index]=((device const half*)v)[b];}else{tk[index]=half(0.0h);tv[index]=half(0.0h);}}
   threadgroup_barrier(mem_flags::mem_threadgroup);
   for(uint offset=0;offset<32&&tile+offset<=pos;offset++){float dot=0.0f;for(uint part=0;part<16;part++){uint d=sub+part*8;dot+=float(q[qb+d])*float(tk[offset*128+d]);}dot+=simd_shuffle_xor(dot,4);dot+=simd_shuffle_xor(dot,2);dot+=simd_shuffle_xor(dot,1);float score=dot*p.scale,nm=max(m,score),a=exp(m-nm),weight=exp(score-nm);l=l*a+weight;for(uint part=0;part<16;part++){uint d=sub+part*8;acc[part]=acc[part]*a+weight*float(tv[offset*128+d]);}m=nm;}
   threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  for(uint part=0;part<16;part++){uint d=sub+part*8;out[qb+d]=half(acc[part]/l);}return;
 }
 if(id>=total)return;uint row=id/p.qh,h=id%p.qh,pos=p.base+row,kh=h/(p.qh/p.kvh),qb=(row*p.qh+h)*p.dim;
 if(p.mode==3||p.mode==5){
  // Long F16 decode uses four SIMD-groups; prefill and INT8 retain two. Both
  // routes preserve the eight-lane accumulator and merge every partial state.
  threadgroup float pm[4],pl[4],pa[512];uint groups=p.mode==5?4:2,segment=group*4+(lane>>3),sub=lane&7;float m=-INFINITY,l=0.0f,acc[16];
  for(uint part=0;part<16;part++)acc[part]=0.0f;
  for(uint t=segment;t<=pos;t+=groups*4){
   uint vec=(p.layer*p.context+t)*p.kvh+kh;ulong b=ulong(vec)*p.dim;float dot=0.0f;
   for(uint d=sub;d<128;d+=8)dot+=float(q[qb+d])*(p.scheme==0?hv(k,b+d):qv(k,b+d,p.kmeta,vec));
   dot+=simd_shuffle_xor(dot,4);dot+=simd_shuffle_xor(dot,2);dot+=simd_shuffle_xor(dot,1);
   float score=dot*p.scale,nm=max(m,score),a=exp(m-nm),w=exp(score-nm);l=l*a+w;
   for(uint part=0;part<16;part++){uint d=sub+part*8;acc[part]=acc[part]*a+w*(p.scheme==0?hv(v,b+d):qv(v,b+d,p.vmeta,vec));}m=nm;
  }
  float m0=simd_shuffle(m,0),m1=simd_shuffle(m,8),m2=simd_shuffle(m,16),m3=simd_shuffle(m,24),gm=max(max(m0,m1),max(m2,m3));
  float r0=exp(m0-gm),r1=exp(m1-gm),r2=exp(m2-gm),r3=exp(m3-gm),gl=simd_shuffle(l,0)*r0+simd_shuffle(l,8)*r1+simd_shuffle(l,16)*r2+simd_shuffle(l,24)*r3;
  if(lane==0){pm[group]=gm;pl[group]=gl;}
  for(uint part=0;part<16;part++){float merged=simd_shuffle(acc[part],sub)*r0+simd_shuffle(acc[part],8+sub)*r1+simd_shuffle(acc[part],16+sub)*r2+simd_shuffle(acc[part],24+sub)*r3;if(segment==group*4)pa[group*128+sub+part*8]=merged;}
  // Every thread reaches this barrier before group zero reads the partial states.
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if(group==0&&segment==0){
   float top=max(pm[0],pm[1]);if(groups==4)top=max(top,max(pm[2],pm[3]));float s0=exp(pm[0]-top),s1=exp(pm[1]-top),s2=groups==4?exp(pm[2]-top):0.0f,s3=groups==4?exp(pm[3]-top):0.0f,den=pl[0]*s0+pl[1]*s1+(groups==4?pl[2]*s2+pl[3]*s3:0.0f);
   for(uint part=0;part<16;part++){uint d=sub+part*8;out[qb+d]=half((pa[d]*s0+pa[128+d]*s1+(groups==4?pa[256+d]*s2+pa[384+d]*s3:0.0f))/den);}
  }return;
 }
 if(p.mode==2){
  // Four eight-lane subgroups own disjoint KV positions for one query head.
  uint segment=lane>>3,sub=lane&7;float m=-INFINITY,l=0.0f,acc[16];
  for(uint part=0;part<16;part++)acc[part]=0.0f;
  for(uint t=segment;t<=pos;t+=4){
   uint vec=(p.layer*p.context+t)*p.kvh+kh;ulong b=ulong(vec)*p.dim;float dot=0.0f;
   for(uint d=sub;d<128;d+=8)dot+=float(q[qb+d])*(p.scheme==0?hv(k,b+d):qv(k,b+d,p.kmeta,vec));
   dot+=simd_shuffle_xor(dot,4);dot+=simd_shuffle_xor(dot,2);dot+=simd_shuffle_xor(dot,1);
   float score=dot*p.scale,nm=max(m,score),a=exp(m-nm),w=exp(score-nm);l=l*a+w;
   for(uint part=0;part<16;part++){uint d=sub+part*8;acc[part]=acc[part]*a+w*(p.scheme==0?hv(v,b+d):qv(v,b+d,p.vmeta,vec));}m=nm;
  }
  // All lanes merge partial online-softmax states; segment zero alone writes.
  float m0=simd_shuffle(m,0),m1=simd_shuffle(m,8),m2=simd_shuffle(m,16),m3=simd_shuffle(m,24),gm=max(max(m0,m1),max(m2,m3));
  float r0=exp(m0-gm),r1=exp(m1-gm),r2=exp(m2-gm),r3=exp(m3-gm),gl=simd_shuffle(l,0)*r0+simd_shuffle(l,8)*r1+simd_shuffle(l,16)*r2+simd_shuffle(l,24)*r3;
  for(uint part=0;part<16;part++){float merged=simd_shuffle(acc[part],sub)*r0+simd_shuffle(acc[part],8+sub)*r1+simd_shuffle(acc[part],16+sub)*r2+simd_shuffle(acc[part],24+sub)*r3;if(segment==0)out[qb+sub+part*8]=half(merged/gl);}return;
 }
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
