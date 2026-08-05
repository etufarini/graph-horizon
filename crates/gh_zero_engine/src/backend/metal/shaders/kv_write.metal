/*
 * gh_zero_engine — Metal KV-write kernel family
 * Owns F16 copy and per-vector int8 encoding only; no dispatch or resource ownership.
 */
// AGENTS deroga K: varianti coese della sola operazione KV encoding.
#include <metal_stdlib>
using namespace metal;
struct P{ulong kp;ulong vp;ulong km;ulong vm;uint vectors;uint dim;uint scheme;};
inline uchar code(float v,float mn,float sc){if(sc==0)return 0;return uchar(clamp(round((v-mn)/sc),0.0f,255.0f));}
inline void quant(device const half*src,device uchar*dst,ulong po,ulong mo,uint vec,uint dim){uint b=vec*dim;float mn=INFINITY,mx=-INFINITY;for(uint i=0;i<dim;i++){float v=float(src[b+i]);mn=min(mn,v);mx=max(mx,v);}half hm=half(mn),hs=half((mx-mn)*(1.0f/255.0f));float rm=float(hm),rs=float(hs);for(uint i=0;i<dim;i++)dst[po+b+i]=code(float(src[b+i]),rm,rs);ushort mb=as_type<ushort>(hm),sb=as_type<ushort>(hs);dst[mo+vec*4]=uchar(mb);dst[mo+vec*4+1]=uchar(mb>>8);dst[mo+vec*4+2]=uchar(sb);dst[mo+vec*4+3]=uchar(sb>>8);}
kernel void metal_kv_write(device const half*k[[buffer(0)]],device const half*v[[buffer(1)]],device uchar*kc[[buffer(2)]],device uchar*vc[[buffer(3)]],constant P&p[[buffer(4)]],uint vec[[thread_position_in_grid]]){if(vec>=p.vectors)return;if(p.scheme==0){for(uint i=0;i<p.dim;i++){((device half*)(kc+p.kp))[vec*p.dim+i]=k[vec*p.dim+i];((device half*)(vc+p.vp))[vec*p.dim+i]=v[vec*p.dim+i];}}else{quant(k,kc,p.kp,p.km,vec,p.dim);quant(v,vc,p.vp,p.vm,vec,p.dim);}}
