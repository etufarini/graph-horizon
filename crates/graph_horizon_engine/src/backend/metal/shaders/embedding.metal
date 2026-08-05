/*
 * graph_horizon_engine — Metal embedding kernel family
 * Dequantizes one selected F16/Q4_K/Q5_K/Q6_K row into the FP32 residual.
 */
// AGENTS deroga K: varianti coese della sola operazione embedding.
#include <metal_stdlib>
using namespace metal;

struct Params { uint row; uint width; uint format; };
inline float h(device const uchar *w, uint b) { return float(as_type<half>(ushort(uint(w[b]) | (uint(w[b+1]) << 8)))); }
inline void sm(device const uchar *w, uint b, uint j, thread uint &sc, thread uint &mn) {
    if (j < 4) { sc=w[b+j]&63; mn=w[b+j+4]&63; }
    else { uint hi=w[b+j+4], lo=w[b+j-4]; sc=(hi&15)|((lo>>6)<<4); mn=(hi>>4)|((uint(w[b+j])>>6)<<4); }
}
inline float q4(device const uchar *w, uint block, uint i, bool five) {
    uint g=i/64, lane=i%64, sc, mn; sm(w,block+4,g*2+lane/32,sc,mn);
    uint qoff=block+(five?48:16)+g*32+lane%32;
    uint q=(lane<32)?(w[qoff]&15):(w[qoff]>>4);
    if (five) q += ((w[block+16+lane%32] >> (g*2+lane/32))&1)*16;
    return h(w,block)*float(sc)*float(q)-h(w,block+2)*float(mn);
}
inline float q6(device const uchar *w, uint block, uint i) {
    uint seg=i/128, r=i%128, cat=r/32, lane=r%32;
    uint qb=block+seg*64+lane+(cat&1)*32;
    uint lo=(cat<2)?(w[qb]&15):(w[qb]>>4);
    uint hi=(w[block+128+seg*32+lane]>>(cat*2))&3;
    int sc=int(as_type<char>(w[block+192+seg*8+lane/16+cat*2]));
    return h(w,block+208)*float(sc)*(float((lo|(hi<<4)))-32.0f);
}
kernel void metal_embedding(device const uchar *w [[buffer(0)]], device float *out [[buffer(1)]], constant Params &p [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i>=p.width) return;
    uint row=p.row*p.width+i;
    if (p.format==0) out[i]=float(((device const half*)w)[row]);
    else { uint b=row/256, q=row%256; out[i]=p.format==1?q4(w,b*144,q,false):p.format==2?q4(w,b*176,q,true):q6(w,b*210,q); }
}
