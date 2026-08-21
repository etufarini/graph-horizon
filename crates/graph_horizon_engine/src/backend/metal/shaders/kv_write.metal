/*
 * graph_horizon_engine — Metal KV-write kernel family
 * Owns F16 copy and per-vector int8 encoding only; no dispatch or resource ownership.
 */
// AGENTS deroga K: varianti coese della sola operazione KV encoding.
#include <metal_stdlib>
using namespace metal;
struct P{ulong kp;ulong vp;ulong km;ulong vm;uint vectors;uint dim;uint scheme;uint fp32;};
inline uchar code(float v,float mn,float sc){if(sc==0)return 0;return uchar(clamp(round((v-mn)/sc),0.0f,255.0f));}
inline half source(device const uchar*src,uint i,uint fp32){return fp32?half(((device const float*)src)[i]):((device const half*)src)[i];}
inline void quant(device const uchar*src,device uchar*dst,ulong po,ulong mo,uint vec,uint dim,uint fp32){uint b=vec*dim;float mn=INFINITY,mx=-INFINITY;for(uint i=0;i<dim;i++){float v=float(source(src,b+i,fp32));mn=min(mn,v);mx=max(mx,v);}half hm=half(mn),hs=half((mx-mn)*(1.0f/255.0f));float rm=float(hm),rs=float(hs);for(uint i=0;i<dim;i++)dst[po+b+i]=code(float(source(src,b+i,fp32)),rm,rs);ushort mb=as_type<ushort>(hm),sb=as_type<ushort>(hs);dst[mo+vec*4]=uchar(mb);dst[mo+vec*4+1]=uchar(mb>>8);dst[mo+vec*4+2]=uchar(sb);dst[mo+vec*4+3]=uchar(sb>>8);}
kernel void metal_kv_write(device const uchar*k[[buffer(0)]],device const uchar*v[[buffer(1)]],device uchar*kc[[buffer(2)]],device uchar*vc[[buffer(3)]],constant P&p[[buffer(4)]],uint vec[[thread_position_in_grid]]){if(vec>=p.vectors)return;if(p.scheme==0){for(uint i=0;i<p.dim;i++){((device half*)(kc+p.kp))[vec*p.dim+i]=source(k,vec*p.dim+i,p.fp32);((device half*)(vc+p.vp))[vec*p.dim+i]=source(v,vec*p.dim+i,p.fp32);}}else{quant(k,kc,p.kp,p.km,vec,p.dim,p.fp32);quant(v,vc,p.vp,p.vm,vec,p.dim,p.fp32);}}
