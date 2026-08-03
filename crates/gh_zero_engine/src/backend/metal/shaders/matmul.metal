/*
 * gh_zero_engine — Metal projection kernel family
 * Computes F16/Q4_K/Q5_K/Q6_K row projections with FP32 accumulation and
 * explicit FP16 activation or FP32 logits output. It owns no dispatch or I/O.
 */
// AGENTS deroga K: varianti coese della sola operazione projection.
#include <metal_stdlib>
using namespace metal;
struct Params { uint input; uint output; uint format; uint fp32; };
inline float h(device const uchar *w,uint b){return float(as_type<half>(ushort(uint(w[b])|(uint(w[b+1])<<8))));}
inline void sm(device const uchar*w,uint b,uint j,thread uint&sc,thread uint&mn){if(j<4){sc=w[b+j]&63;mn=w[b+j+4]&63;}else{uint hi=w[b+j+4],lo=w[b+j-4];sc=(hi&15)|((lo>>6)<<4);mn=(hi>>4)|((uint(w[b+j])>>6)<<4);}}
inline float q4(device const uchar*w,uint b,uint i,bool five){uint g=i/64,l=i%64,sc,mn;sm(w,b+4,g*2+l/32,sc,mn);uint qo=b+(five?48:16)+g*32+l%32;uint q=l<32?(w[qo]&15):(w[qo]>>4);if(five)q+=((w[b+16+l%32]>>(g*2+l/32))&1)*16;return h(w,b)*float(sc)*float(q)-h(w,b+2)*float(mn);}
inline float q6(device const uchar*w,uint b,uint i){uint s=i/128,r=i%128,c=r/32,l=r%32,qb=b+s*64+l+(c&1)*32;uint lo=c<2?(w[qb]&15):(w[qb]>>4),hi=(w[b+128+s*32+l]>>(c*2))&3;int sc=int(as_type<char>(w[b+192+s*8+l/16+c*2]));return h(w,b+208)*float(sc)*(float(lo|(hi<<4))-32.0f);}
kernel void metal_matmul(device const half*a[[buffer(0)]],device const uchar*w[[buffer(1)]],device uchar*out[[buffer(2)]],constant Params&p[[buffer(3)]],uint o[[thread_position_in_grid]]){
 if(o>=p.output)return;float acc=0.0f;uint ns=p.input/256;
 for(uint i=0;i<p.input;i++){float v;if(p.format==0)v=float(((device const half*)w)[o*p.input+i]);else{uint b=o*ns+i/256,q=i%256;v=p.format==1?q4(w,b*144,q,false):p.format==2?q4(w,b*176,q,true):q6(w,b*210,q);}acc+=float(a[i])*v;}
 if(p.fp32)((device float*)out)[o]=acc;else((device half*)out)[o]=half(acc);
}
