/*
 * graph_orizon_engine — SIMD variant of the fused Q5_K CPU kernel
 * The AVX2+FMA fused dequant+MAC for Q5_K, the last quantized format to leave the
 * generic materialize-then-dot path. Q5_K is the Q4_K super-block layout (d, dmin,
 * scales[12], qs[128]) plus a 5th bit per quant taken from qh[32]: sub-block pair `g`
 * (0..3) uses qh bit 2g for the even sub-block (low nibble) and 2g+1 for the odd
 * (high nibble). This kernel reuses that bit-exact unpack from `dequant_row_q5_k_avx2`
 * (low nibble + the qh 5th bit via the per-lane `srlv`) but FMAs each decoded 8-wide
 * weight vector straight into the accumulator — never materializes the f32 row.
 *
 * Numerics: per-row accumulation is reordered (chunk lane-accumulators / per-token
 * lanes), and `d*q - m` is a fused `fmsub` (the dequant keeps it as separate mul+sub
 * for scalar bit-identity), so the result is within the quantized tolerance of
 * dequant_row, NOT bit-identical — the Q4_K/Q5_K/Q6_K fused contract. SAFETY: every
 * entry is reached only behind a runtime AVX2+FMA check; all loads/stores stay within
 * `a`/`out`/`acc` (sized to `n`) and `bytes` — the load-validated weight tensor slice
 * (SEC-INV: `gguf::loader` rejects any tensor whose `offset+byte_len` overruns the file
 * or whose `byte_len` is incoherent with `dims × block`, before a byte reaches here), so
 * `base = row * (in_dim/256) * 176` stays within `bytes` for every `row < out_dim`.
*/

// AGENTS deroga K: kernel matmul Q5_K SIMD (AVX2), una sola operazione.

#![cfg(target_arch = "x86_64")]

use core::arch::x86_64::*;

use super::q4k::f16_at;
use crate::backend::cpu::dequant::scale_min;

// The two 8-wide weight vectors of one l-chunk of a sub-block PAIR: `wlo` for the
// even sub-block (low nibble + qh bit `su0`), `whi` for the odd (high nibble + bit
// `su1`). `su0`/`su1` are broadcast bit indices (2g, 2g+1); `dl*`/`ml*` the pair's
// scales. Shared by the decode/batched/2-row kernels — one unpack transcription.
#[target_feature(enable = "avx2,fma")]
#[inline]
#[allow(clippy::too_many_arguments)]
unsafe fn unpack_chunk(
    bytes: &[u8],
    qsp: usize,
    qhp: usize,
    dl0: __m256,
    ml0: __m256,
    dl1: __m256,
    ml1: __m256,
    su0: __m256i,
    su1: __m256i,
) -> [__m256; 2] {
    unsafe {
        let m4 = _mm256_set1_epi32(0xF);
        let c1 = _mm256_set1_epi32(1);
        let qv = _mm256_cvtepu8_epi32(_mm_loadl_epi64(bytes.as_ptr().add(qsp) as *const __m128i));
        let qh = _mm256_cvtepu8_epi32(_mm_loadl_epi64(bytes.as_ptr().add(qhp) as *const __m128i));
        // 5th bit → +16: ((qh >> bit) & 1) << 4, per-lane runtime shift (srlv).
        let hb0 = _mm256_slli_epi32(_mm256_and_si256(_mm256_srlv_epi32(qh, su0), c1), 4);
        let hb1 = _mm256_slli_epi32(_mm256_and_si256(_mm256_srlv_epi32(qh, su1), c1), 4);
        let lo = _mm256_add_epi32(_mm256_and_si256(qv, m4), hb0);
        let hi = _mm256_add_epi32(_mm256_srli_epi32(qv, 4), hb1);
        [
            _mm256_fmsub_ps(dl0, _mm256_cvtepi32_ps(lo), ml0),
            _mm256_fmsub_ps(dl1, _mm256_cvtepi32_ps(hi), ml1),
        ]
    }
}

// Single-token fused Q5_K dot (decode GEMV / lm_head). Four chunk accumulators for
// ILP. `in_dim` is a multiple of 256.
#[target_feature(enable = "avx2,fma")]
pub(super) unsafe fn row_dot_q5k_avx2(a: &[f32], bytes: &[u8], row: usize, in_dim: usize) -> f32 {
    unsafe {
        let nsb = in_dim / 256;
        let base = row * nsb * 176;
        let mut acc = [_mm256_setzero_ps(); 4];
        for s in 0..nsb {
            let blk = base + s * 176;
            let d = f16_at(bytes, blk);
            let dmin = f16_at(bytes, blk + 2);
            let (sco, qho, qso) = (blk + 4, blk + 16, blk + 48);
            let abase = s * 256;
            for g in 0..4 {
                let (sc0, mn0) = scale_min(bytes, sco, 2 * g);
                let (sc1, mn1) = scale_min(bytes, sco, 2 * g + 1);
                let dl0 = _mm256_set1_ps(d * sc0 as f32);
                let ml0 = _mm256_set1_ps(dmin * mn0 as f32);
                let dl1 = _mm256_set1_ps(d * sc1 as f32);
                let ml1 = _mm256_set1_ps(dmin * mn1 as f32);
                let su0 = _mm256_set1_epi32(2 * g as i32);
                let su1 = _mm256_set1_epi32(2 * g as i32 + 1);
                let qb = g * 32;
                let in_lo = abase + g * 64;
                let in_hi = abase + g * 64 + 32;
                for (c, sum) in acc.iter_mut().enumerate() {
                    let w = unpack_chunk(
                        bytes,
                        qso + qb + c * 8,
                        qho + c * 8,
                        dl0,
                        ml0,
                        dl1,
                        ml1,
                        su0,
                        su1,
                    );
                    *sum =
                        _mm256_fmadd_ps(_mm256_loadu_ps(a.as_ptr().add(in_lo + c * 8)), w[0], *sum);
                    *sum =
                        _mm256_fmadd_ps(_mm256_loadu_ps(a.as_ptr().add(in_hi + c * 8)), w[1], *sum);
                }
            }
        }
        let lo = _mm256_add_ps(acc[0], acc[1]);
        let hi = _mm256_add_ps(acc[2], acc[3]);
        hsum256(_mm256_add_ps(lo, hi))
    }
}

// Batched fused Q5_K dot: decode each chunk's two weight vectors ONCE and FMA them
// into every token's 8-lane accumulator (`acc[i*8]`). `acc` is `n*8` f32.
#[target_feature(enable = "avx2,fma")]
pub(super) unsafe fn row_dot_q5k_avx2_batched(
    a: &[f32],
    bytes: &[u8],
    row: usize,
    in_dim: usize,
    out: &mut [f32],
    acc: &mut [f32],
) {
    unsafe {
        let n = out.len();
        let nsb = in_dim / 256;
        let base = row * nsb * 176;
        for v in acc[..n * 8].iter_mut() {
            *v = 0.0;
        }
        for s in 0..nsb {
            let blk = base + s * 176;
            let d = f16_at(bytes, blk);
            let dmin = f16_at(bytes, blk + 2);
            let (sco, qho, qso) = (blk + 4, blk + 16, blk + 48);
            let abase = s * 256;
            for g in 0..4 {
                let (sc0, mn0) = scale_min(bytes, sco, 2 * g);
                let (sc1, mn1) = scale_min(bytes, sco, 2 * g + 1);
                let dl0 = _mm256_set1_ps(d * sc0 as f32);
                let ml0 = _mm256_set1_ps(dmin * mn0 as f32);
                let dl1 = _mm256_set1_ps(d * sc1 as f32);
                let ml1 = _mm256_set1_ps(dmin * mn1 as f32);
                let su0 = _mm256_set1_epi32(2 * g as i32);
                let su1 = _mm256_set1_epi32(2 * g as i32 + 1);
                let qb = g * 32;
                let (lo0, hi0) = (abase + g * 64, abase + g * 64 + 32);
                for c in 0..4 {
                    let w = unpack_chunk(
                        bytes,
                        qso + qb + c * 8,
                        qho + c * 8,
                        dl0,
                        ml0,
                        dl1,
                        ml1,
                        su0,
                        su1,
                    );
                    for i in 0..n {
                        let ap = acc.as_mut_ptr().add(i * 8);
                        let base_i = i * in_dim;
                        let mut ai = _mm256_loadu_ps(ap);
                        ai = _mm256_fmadd_ps(
                            _mm256_loadu_ps(a.as_ptr().add(base_i + lo0 + c * 8)),
                            w[0],
                            ai,
                        );
                        ai = _mm256_fmadd_ps(
                            _mm256_loadu_ps(a.as_ptr().add(base_i + hi0 + c * 8)),
                            w[1],
                            ai,
                        );
                        _mm256_storeu_ps(ap, ai);
                    }
                }
            }
        }
        for (i, o) in out.iter_mut().enumerate() {
            *o = hsum256(_mm256_loadu_ps(acc.as_ptr().add(i * 8)));
        }
    }
}

// Two-output-row batched fused Q5_K dot: rows `row0` and `row0+1` together, reusing
// each token's two activation loads (even/odd sub-block) across both rows. Per-token
// order matches the single-row batched kernel → bit-identical to two of those calls.
// `acc` is `2*n*8` f32 (row0 in `[0..n*8]`, row1 in `[n*8..]`).
#[target_feature(enable = "avx2,fma")]
pub(super) unsafe fn row2_dot_q5k_avx2_batched(
    a: &[f32],
    bytes: &[u8],
    row0: usize,
    in_dim: usize,
    out0: &mut [f32],
    out1: &mut [f32],
    acc: &mut [f32],
) {
    unsafe {
        let n = out0.len();
        let nsb = in_dim / 256;
        let base0 = row0 * nsb * 176;
        let base1 = (row0 + 1) * nsb * 176;
        for v in acc[..2 * n * 8].iter_mut() {
            *v = 0.0;
        }
        for s in 0..nsb {
            let blk0 = base0 + s * 176;
            let blk1 = base1 + s * 176;
            let d0 = f16_at(bytes, blk0);
            let dmin0 = f16_at(bytes, blk0 + 2);
            let d1 = f16_at(bytes, blk1);
            let dmin1 = f16_at(bytes, blk1 + 2);
            let abase = s * 256;
            for g in 0..4 {
                let (s00, m00) = scale_min(bytes, blk0 + 4, 2 * g);
                let (s01, m01) = scale_min(bytes, blk0 + 4, 2 * g + 1);
                let (s10, m10) = scale_min(bytes, blk1 + 4, 2 * g);
                let (s11, m11) = scale_min(bytes, blk1 + 4, 2 * g + 1);
                let dl00 = _mm256_set1_ps(d0 * s00 as f32);
                let ml00 = _mm256_set1_ps(dmin0 * m00 as f32);
                let dl01 = _mm256_set1_ps(d0 * s01 as f32);
                let ml01 = _mm256_set1_ps(dmin0 * m01 as f32);
                let dl10 = _mm256_set1_ps(d1 * s10 as f32);
                let ml10 = _mm256_set1_ps(dmin1 * m10 as f32);
                let dl11 = _mm256_set1_ps(d1 * s11 as f32);
                let ml11 = _mm256_set1_ps(dmin1 * m11 as f32);
                let su0 = _mm256_set1_epi32(2 * g as i32);
                let su1 = _mm256_set1_epi32(2 * g as i32 + 1);
                let qb = g * 32;
                let (lo0, hi0) = (abase + g * 64, abase + g * 64 + 32);
                for c in 0..4 {
                    let qsp = c * 8;
                    let w0 = unpack_chunk(
                        bytes,
                        blk0 + 48 + qb + qsp,
                        blk0 + 16 + qsp,
                        dl00,
                        ml00,
                        dl01,
                        ml01,
                        su0,
                        su1,
                    );
                    let w1 = unpack_chunk(
                        bytes,
                        blk1 + 48 + qb + qsp,
                        blk1 + 16 + qsp,
                        dl10,
                        ml10,
                        dl11,
                        ml11,
                        su0,
                        su1,
                    );
                    for i in 0..n {
                        let base_i = i * in_dim;
                        let avl = _mm256_loadu_ps(a.as_ptr().add(base_i + lo0 + c * 8));
                        let avh = _mm256_loadu_ps(a.as_ptr().add(base_i + hi0 + c * 8));
                        let p0 = acc.as_mut_ptr().add(i * 8);
                        let p1 = acc.as_mut_ptr().add((n + i) * 8);
                        let mut a0 = _mm256_loadu_ps(p0);
                        let mut a1 = _mm256_loadu_ps(p1);
                        a0 = _mm256_fmadd_ps(avl, w0[0], a0);
                        a1 = _mm256_fmadd_ps(avl, w1[0], a1);
                        a0 = _mm256_fmadd_ps(avh, w0[1], a0);
                        a1 = _mm256_fmadd_ps(avh, w1[1], a1);
                        _mm256_storeu_ps(p0, a0);
                        _mm256_storeu_ps(p1, a1);
                    }
                }
            }
        }
        for i in 0..n {
            out0[i] = hsum256(_mm256_loadu_ps(acc.as_ptr().add(i * 8)));
            out1[i] = hsum256(_mm256_loadu_ps(acc.as_ptr().add((n + i) * 8)));
        }
    }
}

// Horizontal sum of the 8 f32 lanes.
#[target_feature(enable = "avx2")]
unsafe fn hsum256(v: __m256) -> f32 {
    let lo = _mm256_castps256_ps128(v);
    let hi = _mm256_extractf128_ps(v, 1);
    let s = _mm_add_ps(lo, hi);
    let s = _mm_hadd_ps(s, s);
    let s = _mm_hadd_ps(s, s);
    _mm_cvtss_f32(s)
}
