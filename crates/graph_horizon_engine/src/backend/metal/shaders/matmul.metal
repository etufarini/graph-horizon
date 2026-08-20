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
inline void q4x16(device const uchar*w,uint b,uint i,thread half4x4&r){uint g=i/64,l=i%64,sc,mn;sm(w,b+4,g*2+l/32,sc,mn);uint qo=b+16+g*32+l%32;float ds=h(w,b)*float(sc),ms=h(w,b+2)*float(mn);for(uint j=0;j<16;j++){uint q=l<32?(w[qo+j]&15):(w[qo+j]>>4);r[j/4][j%4]=half(ds*float(q)-ms);}}
inline void q6x16(device const uchar*w,uint b,uint i,thread half4x4&r){uint seg=i/128,c=(i%128)/32,l=i%32;uint qb=b+seg*64+(c&1)*32+l,hb=b+128+seg*32+l;float dl=h(w,b+208)*float(int(as_type<char>(w[b+192+seg*8+l/16+c*2])));for(uint j=0;j<16;j++){uint lo=c>=2?(w[qb+j]>>4):(w[qb+j]&15),hi=(w[hb+j]>>(c*2))&3;r[j/4][j%4]=half(dl*(float(lo|(hi<<4))-32.0f));}}
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
kernel void metal_matmul_batched(device const half*a[[buffer(0)]],device const uchar*w[[buffer(1)]],device half*out[[buffer(2)]],constant BatchParams&p[[buffer(3)]],uint group[[threadgroup_position_in_grid]],ushort tid[[thread_index_in_threadgroup]],ushort sg[[simdgroup_index_in_threadgroup]]){
 threadgroup half weights[64*32],acts[32*32];threadgroup float result[32*64];
 simdgroup_float8x8 acc[8];for(uint i=0;i<8;i++)acc[i]=make_filled_simdgroup_matrix<float,8>(0.0f);
 uint ns=p.input/256,group_out=group*64,local_out=tid/2,chunk=tid&1;
 for(uint base=0;base<p.input;base+=32){
  uint column=group_out+local_out;
  if(column<p.output){
   uint block=(column*ns+base/256)*(p.format==1?144:210);half4x4 values;
   if(p.format==1)q4x16(w,block,(base&255)+chunk*16,values);else q6x16(w,block,(base&255)+chunk*16,values);
   for(uint i=0;i<16;i++)weights[local_out*32+chunk*16+i]=values[i/4][i%4];
  }else for(uint i=0;i<16;i++)weights[local_out*32+chunk*16+i]=half(0.0h);
  for(uint index=tid;index<32*32;index+=128){uint token=index/32,k=index%32;acts[index]=token<p.rows?a[token*p.input+base+k]:half(0.0h);}
  // The complete K32 tiles are immutable until all four SIMD groups finish.
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint out_tile=(sg&1)*32,token_tile=(sg>>1)*16;
  for(uint k=0;k<32;k+=8){
   simdgroup_half8x8 wm[4],am[2];
   for(uint i=0;i<4;i++)simdgroup_load(wm[i],weights+(out_tile+i*8)*32+k,32,0,true);
   for(uint i=0;i<2;i++)simdgroup_load(am[i],acts+(token_tile+i*8)*32+k,32,0,false);
   for(uint token=0;token<2;token++)for(uint column=0;column<4;column++){uint i=token*4+column;simdgroup_multiply_accumulate(acc[i],am[token],wm[column],acc[i]);}
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
 }
 uint out_tile=(sg&1)*32,token_tile=(sg>>1)*16;
 for(uint token=0;token<2;token++)for(uint column=0;column<4;column++)simdgroup_store(acc[token*4+column],result+(token_tile+token*8)*64+out_tile+column*8,64,0,false);
 threadgroup_barrier(mem_flags::mem_threadgroup);
 for(uint index=tid;index<32*64;index+=128){uint token=index/64,column=index%64,dst_column=group_out+column;if(token<p.rows&&dst_column<p.output)out[token*p.output+dst_column]=half(result[index]);}
}
kernel void metal_matmul_batched_wide(device const half*a[[buffer(0)]],device const uchar*w[[buffer(1)]],device half*out[[buffer(2)]],constant BatchParams&p[[buffer(3)]],uint group[[threadgroup_position_in_grid]],ushort tid[[thread_index_in_threadgroup]],ushort sg[[simdgroup_index_in_threadgroup]]){
 threadgroup half weights[64*32],acts[64*32];threadgroup float result[64*64];
 simdgroup_float8x8 acc[8];for(uint i=0;i<8;i++)acc[i]=make_filled_simdgroup_matrix<float,8>(0.0f);
 uint ns=p.input/256,group_out=group*64;
 for(uint base=0;base<p.input;base+=32){
  if(tid<128){
   uint local_out=tid/2,chunk=tid&1,column=group_out+local_out;half4x4 values;
   if(column<p.output){uint block=(column*ns+base/256)*(p.format==1?144:210);if(p.format==1)q4x16(w,block,(base&255)+chunk*16,values);else q6x16(w,block,(base&255)+chunk*16,values);}
   else for(uint i=0;i<16;i++)values[i/4][i%4]=half(0.0h);
   for(uint i=0;i<16;i++){uint sx=2*chunk+i/8,sy=local_out/8,lx=local_out&7,ly=i&7;weights[64*(8*sx+sy)+8*ly+lx]=values[i/4][i%4];}
  }
  for(uint index=tid;index<64*32;index+=256){uint token=index/32,k=index%32,sx=k/8,sy=token/8;acts[64*(8*sx+sy)+8*(token&7)+(k&7)]=token<p.rows?a[token*p.input+base+k]:half(0.0h);}
  // The swizzle keeps each SIMD matrix load contiguous across one K8 tile.
  threadgroup_barrier(mem_flags::mem_threadgroup);
  threadgroup const half*wm_base=weights+4*64*(sg&1);threadgroup const half*am_base=acts+2*64*(sg>>1);
  for(uint k=0;k<4;k++){
   simdgroup_half8x8 wm[4],am[2];
   for(uint i=0;i<4;i++)simdgroup_load(wm[i],wm_base+64*i,8,0,false);
   for(uint i=0;i<2;i++)simdgroup_load(am[i],am_base+64*i,8,0,false);
   for(uint token=0;token<2;token++)for(uint column=0;column<4;column++){uint i=token*4+column;simdgroup_multiply_accumulate(acc[i],am[token],wm[column],acc[i]);}
   wm_base+=8*64;am_base+=8*64;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
 }
 uint out_tile=(sg&1)*32,token_tile=(sg>>1)*16;
 for(uint token=0;token<2;token++)for(uint column=0;column<4;column++)simdgroup_store(acc[token*4+column],result+(token_tile+token*8)*64+out_tile+column*8,64,0,false);
 threadgroup_barrier(mem_flags::mem_threadgroup);
 for(uint index=tid;index<64*64;index+=256){uint token=index/64,column=index%64,dst_column=group_out+column;if(token<p.rows&&dst_column<p.output)out[token*p.output+dst_column]=half(result[index]);}
}
