/*
 * graph_horizon_engine — Metal projection kernel family
 * Computes scalar and cooperative F16/Q4_K/Q5_K/Q6_K projections with packed
 * dequantization, FP32 accumulation, and FP16 or FP32 output. No dispatch.
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
kernel void metal_matmul(device const half*a[[buffer(0)]],device const uchar*w[[buffer(1)]],device uchar*out[[buffer(2)]],constant Params&p[[buffer(3)]],uint tid[[thread_position_in_grid]],ushort sl[[thread_index_in_simdgroup]]){
 if(p.format==1){
  // One SIMD-group emits two rows; four lane octets own disjoint input blocks.
  uint first=(tid>>5)*2;if(first>=p.output)return;uint ix=sl>>3,it=sl&7,iq=it>>2,ir=it&3,ns=p.input/256;float sum[2]={0.0f,0.0f};
  for(uint s=ix;s<ns;s+=4){
   uint yb=s*256+64*iq+8*ir;float yl[16],yh[16];float4 sumy=float4(0.0f);
   for(uint i=0;i<8;i++){yl[i]=float(a[yb+i]);sumy.x+=yl[i];yl[i+8]=float(a[yb+32+i]);sumy.y+=yl[i+8];yh[i]=float(a[yb+128+i]);sumy.z+=yh[i];yh[i+8]=float(a[yb+160+i]);sumy.w+=yh[i+8];}
   for(uint row=0;row<2&&first+row<p.output;row++){
    uint b=((first+row)*ns+s)*144;device const ushort*sc=(device const ushort*)(w+b+4)+iq;ushort packed_sc[4];thread const uchar*sc8=(thread const uchar*)packed_sc;
    packed_sc[0]=sc[0]&0x3f3f;packed_sc[1]=sc[2]&0x3f3f;packed_sc[2]=((sc[4]&0x0f0f)|((sc[0]&0xc0c0)>>2));packed_sc[3]=(((sc[4]>>4)&0x0f0f)|((sc[2]&0xc0c0)>>2));
    device const ushort*q1=(device const ushort*)(w+b+16)+16*iq+4*ir;device const ushort*q2=q1+32;float4 acc1=float4(0.0f),acc2=float4(0.0f);
    for(uint i=0;i<4;i++){acc1.x+=yl[2*i]*(q1[i]&0x000f);acc1.y+=yl[2*i+1]*(q1[i]&0x0f00);acc1.z+=yl[2*i+8]*(q1[i]&0x00f0);acc1.w+=yl[2*i+9]*(q1[i]&0xf000);acc2.x+=yh[2*i]*(q2[i]&0x000f);acc2.y+=yh[2*i+1]*(q2[i]&0x0f00);acc2.z+=yh[2*i+8]*(q2[i]&0x00f0);acc2.w+=yh[2*i+9]*(q2[i]&0xf000);}
    float d=h(w,b),dm=h(w,b+2);sum[row]+=d*((acc1.x+acc1.y/256.0f)*sc8[0]+(acc1.z+acc1.w/256.0f)*sc8[1]/16.0f+(acc2.x+acc2.y/256.0f)*sc8[4]+(acc2.z+acc2.w/256.0f)*sc8[5]/16.0f)-dm*dot(sumy,float4(sc8[2],sc8[3],sc8[6],sc8[7]));
   }
  }
  for(uint row=0;row<2&&first+row<p.output;row++){float total=simd_sum(sum[row]);if(sl==0){if(p.fp32)((device float*)out)[first+row]=total;else((device half*)out)[first+row]=half(total);}}return;
 }
 if(p.format==3){
  // Lane pairs split input blocks and reuse each activation slice for both rows.
  uint first=(tid>>5)*2;if(first>=p.output)return;uint pair=sl>>1,ix=sl&1,part=pair>>3,l0=4*(pair&7),scale_index=8*part+l0/16,ns=p.input/256;float sum[2]={0.0f,0.0f};
  for(uint s=ix;s<ns;s+=2){
   uint yb=s*256+128*part+l0;float values[16];for(uint l=0;l<4;l++){values[4*l]=float(a[yb+l]);values[4*l+1]=float(a[yb+32+l]);values[4*l+2]=float(a[yb+64+l]);values[4*l+3]=float(a[yb+96+l]);}
   for(uint row=0;row<2&&first+row<p.output;row++){
    uint b=((first+row)*ns+s)*210;device const uchar*q1=w+b+64*part+l0;device const uchar*q2=q1+32;device const uchar*qh=w+b+128+32*part+l0;device const char*sc=(device const char*)(w+b+192+scale_index);float4 sums=float4(0.0f);
    for(uint l=0;l<4;l++){sums.x+=values[4*l]*float(int((q1[l]&15)|((qh[l]&3)<<4))-32);sums.y+=values[4*l+1]*float(int((q2[l]&15)|((qh[l]&12)<<2))-32);sums.z+=values[4*l+2]*float(int((q1[l]>>4)|(qh[l]&48))-32);sums.w+=values[4*l+3]*float(int((q2[l]>>4)|((qh[l]&192)>>2))-32);}
    sum[row]+=h(w,b+208)*dot(sums,float4(sc[0],sc[2],sc[4],sc[6]));
   }
  }
  for(uint row=0;row<2&&first+row<p.output;row++){float total=simd_sum(sum[row]);if(sl==0){if(p.fp32)((device float*)out)[first+row]=total;else((device half*)out)[first+row]=half(total);}}return;
 }
 uint o=tid;if(o>=p.output)return;float acc=0.0f;uint ns=p.input/256;
 for(uint i=0;i<p.input;i++){float v;if(p.format==0)v=float(((device const half*)w)[o*p.input+i]);else{uint b=o*ns+i/256,q=i%256;v=q4(w,b*176,q,true);}acc+=float(a[i])*v;}
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
