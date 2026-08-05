/*
 * gh_zero_engine — Metal projection kernel family
 * Computes scalar and cooperative F16/Q4_K/Q5_K/Q6_K projections with FP32
 * accumulation and explicit FP16 activation or FP32 logits output. No dispatch.
 */
// AGENTS deroga K: varianti coese della sola operazione projection.
#include <metal_stdlib>
using namespace metal;
struct Params { uint input; uint output; uint format; uint fp32; };
struct BatchParams { uint input; uint output; uint format; uint rows; };
inline float h(device const uchar *w,uint b){return float(as_type<half>(ushort(uint(w[b])|(uint(w[b+1])<<8))));}
inline void sm(device const uchar*w,uint b,uint j,thread uint&sc,thread uint&mn){if(j<4){sc=w[b+j]&63;mn=w[b+j+4]&63;}else{uint hi=w[b+j+4],lo=w[b+j-4];sc=(hi&15)|((lo>>6)<<4);mn=(hi>>4)|((uint(w[b+j])>>6)<<4);}}
inline float q4(device const uchar*w,uint b,uint i,bool five){uint g=i/64,l=i%64,sc,mn;sm(w,b+4,g*2+l/32,sc,mn);uint qo=b+(five?48:16)+g*32+l%32;uint q=l<32?(w[qo]&15):(w[qo]>>4);if(five)q+=((w[b+16+l%32]>>(g*2+l/32))&1)*16;return h(w,b)*float(sc)*float(q)-h(w,b+2)*float(mn);}
inline float q6(device const uchar*w,uint b,uint i){uint seg=i/128,c=(i%128)/32,l=i%32;uint qb=b+seg*64+(c&1)*32+l,hb=b+128+seg*32+l;uint lo=c>=2?(w[qb]>>4):(w[qb]&15),hi=(w[hb]>>(c*2))&3;float dl=h(w,b+208)*float(int(as_type<char>(w[b+192+seg*8+l/16+c*2])));return dl*(float(lo|(hi<<4))-32.0f);}
kernel void metal_matmul(device const half*a[[buffer(0)]],device const uchar*w[[buffer(1)]],device uchar*out[[buffer(2)]],constant Params&p[[buffer(3)]],uint o[[thread_position_in_grid]]){
 if(o>=p.output)return;float acc=0.0f;uint ns=p.input/256;
 if(p.format==1){
  for(uint s=0;s<ns;s++){
   uint b=(o*ns+s)*144;float d=h(w,b),dm=h(w,b+2);
   for(uint j=0;j<8;j++){
    uint sc,mn;sm(w,b+4,j,sc,mn);
    float dl=d*float(sc),ml=dm*float(mn);uint qo=b+16+(j/2)*32;
    bool high=(j&1)!=0;uint i=s*256+j*32;
    // Preserve natural input order so the FP32 accumulation sequence is unchanged.
    for(uint l=0;l<32;l++){
     uint q=high?(w[qo+l]>>4):(w[qo+l]&15);
     float v=dl*float(q)-ml;acc+=float(a[i+l])*v;
    }
   }
  }
 }else if(p.format==3){
  for(uint s=0;s<ns;s++){
   uint b=(o*ns+s)*210;float d=h(w,b+208);
   for(uint g=0;g<16;g++){
    uint c=(g/2)&3,seg=g/8,l0=(g&1)*16;
    float dl=d*float(int(as_type<char>(w[b+192+g])));
    uint qb=b+seg*64+(c&1)*32+l0,hb=b+128+seg*32+l0;
    uint shift=c*2,i=s*256+g*16;bool high=c>=2;
    // Each group covers the next 16 inputs, preserving the FP32 accumulation order.
    for(uint l=0;l<16;l++){
     uint lo=high?(w[qb+l]>>4):(w[qb+l]&15),hi=(w[hb+l]>>shift)&3;
     float v=dl*(float(lo|(hi<<4))-32.0f);acc+=float(a[i+l])*v;
    }
   }
  }
 }else{
  for(uint i=0;i<p.input;i++){float v;if(p.format==0)v=float(((device const half*)w)[o*p.input+i]);else{uint b=o*ns+i/256,q=i%256;v=q4(w,b*176,q,true);}acc+=float(a[i])*v;}
 }
 if(p.fp32)((device float*)out)[o]=acc;else((device half*)out)[o]=half(acc);
}
kernel void metal_matmul_batched(device const half*a[[buffer(0)]],device const uchar*w[[buffer(1)]],device half*out[[buffer(2)]],constant BatchParams&p[[buffer(3)]],uint group[[threadgroup_position_in_grid]],ushort lane[[thread_index_in_threadgroup]]){
 threadgroup half weights[64],acts[4*64];threadgroup float result[4*64];
 // One SIMD group owns one 8-output tile and up to four 8-token tiles.
 simdgroup_float8x8 acc[4];uint tiles=(p.rows+7)/8,out_base=group*8,ns=p.input/256;
 for(uint t=0;t<tiles;t++)acc[t]=make_filled_simdgroup_matrix<float,8>(0.0f);
 for(uint base=0;base<p.input;base+=8){
  for(uint index=lane;index<64;index+=32){
   uint row=out_base+index/8,k=base+index%8;
   if(row<p.output){
    uint block=(row*ns+k/256)*(p.format==1?144:210);
    weights[index]=half(p.format==1?q4(w,block,k&255,false):q6(w,block,k&255));
   }else weights[index]=half(0.0h);
   for(uint t=0;t<tiles;t++){
    uint token=t*8+index/8;
    acts[t*64+index]=token<p.rows?a[token*p.input+k]:half(0.0h);
   }
  }
  // All 32 lanes reach both barriers before shared tiles can be overwritten.
  threadgroup_barrier(mem_flags::mem_threadgroup);
  simdgroup_half8x8 wm;simdgroup_load(wm,weights,8,0,true);
  for(uint t=0;t<tiles;t++){
   simdgroup_half8x8 am;simdgroup_load(am,acts+t*64,8,0,false);
   simdgroup_multiply_accumulate(acc[t],am,wm,acc[t]);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
 }
 for(uint t=0;t<tiles;t++)simdgroup_store(acc[t],result+t*64,8,0,false);
 threadgroup_barrier(mem_flags::mem_threadgroup);
 for(uint t=0;t<tiles;t++)for(uint index=lane;index<64;index+=32){
  uint token=t*8+index/8,column=out_base+index%8;
  if(token<p.rows&&column<p.output)out[token*p.output+column]=half(result[t*64+index]);
 }
}
