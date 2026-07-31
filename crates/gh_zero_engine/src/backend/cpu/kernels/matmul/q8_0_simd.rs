/*
 * gh_zero_engine — SIMD variant of the fused Q8_0 CPU kernel
 * The AVX2+FMA fused dequant+MAC for Q8_0, the sibling of matmul_q4k_simd /
 * matmul_q6k_simd. The generic path was especially bad for Q8_0: `dequant_row_q8_0`
 * is SCALAR (no AVX2 dequant exists) and materializes the row to f32, so each 1-byte
 * weight became an int8→f32 scalar convert plus a 4-byte f32 spill+reload — the
 * decode GEMV ran at ~31% of the RAM roofline, the worst of any format. This kernel
 * never materializes the row: it loads the 8 int8 quants of a chunk, sign-extends and
 * widens them with one `cvtepi8_epi32`+`cvtepi32_ps`, scales by the block's `d`, and
 * FMAs straight into the accumulator.
 *
 * Numerics: per-row accumulation is reordered (four chunk lane-accumulators / per-token
 * lanes, summed at the end), so the result is within the quantized tolerance of
 * dequant_row, NOT bit-identical — same contract as the Q4_K/Q6_K fused kernels. The
 * int8 sign-extension and the `d*q` product match the scalar reference. SAFETY: every
 * entry is reached only behind a runtime AVX2+FMA check; all loads/stores stay within
 * `a`/`out`/`acc` (sized to `n`) and `bytes` — the load-validated weight tensor slice
 * (SEC-INV: `gguf::loader` rejects any tensor whose `offset+byte_len` overruns the file
 * or whose `byte_len` is incoherent with `dims × block`, before a byte reaches here), so
 * the per-`row` block offset stays within `bytes` (`in_dim` a multiple of 32, block =
 * d(f16) | 32×int8).
*/

// AGENTS deroga K: variante SIMD del solo kernel matmul Q8_0.

#![cfg(target_arch = "x86_64")]

use core::arch::x86_64::*;

use super::q4k::f16_at;

// The four 8-wide weight vectors of one 32-value Q8_0 block: w[c] = d * q[c*8..],
// the int8 quants sign-extended and scaled. Shared by the decode/batched/2-row
// kernels so the unpack has one transcription.
#[target_feature(enable = "avx2")]
unsafe fn unpack_block(bytes: &[u8], blk: usize, d: __m256) -> [__m256; 4] {
    unsafe {
        let mut w = [_mm256_setzero_ps(); 4];
        for (c, wc) in w.iter_mut().enumerate() {
            // 8 int8 quants → 8 SIGN-extended i32 → f32, then × d.
            let q8 = _mm_loadl_epi64(bytes.as_ptr().add(blk + 2 + c * 8) as *const __m128i);
            *wc = _mm256_mul_ps(d, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q8)));
        }
        w
    }
}

// Single-token fused Q8_0 dot (decode GEMV / lm_head). Four chunk accumulators for
// ILP (independent FMA chains, the latency hiding the single accumulator lacks at
// n==1). `in_dim` is a multiple of 32.
#[target_feature(enable = "avx2,fma")]
pub(super) unsafe fn row_dot_q8_0_avx2(a: &[f32], bytes: &[u8], row: usize, in_dim: usize) -> f32 {
    unsafe {
        let nb = in_dim / 32;
        let base = row * nb * 34;
        let mut acc = [_mm256_setzero_ps(); 4];
        for s in 0..nb {
            let blk = base + s * 34;
            let d = _mm256_set1_ps(f16_at(bytes, blk));
            let abase = s * 32;
            let w = unpack_block(bytes, blk, d);
            for c in 0..4 {
                acc[c] =
                    _mm256_fmadd_ps(_mm256_loadu_ps(a.as_ptr().add(abase + c * 8)), w[c], acc[c]);
            }
        }
        let lo = _mm256_add_ps(acc[0], acc[1]);
        let hi = _mm256_add_ps(acc[2], acc[3]);
        hsum256(_mm256_add_ps(lo, hi))
    }
}

// Batched fused Q8_0 dot: decode each block's four weight vectors ONCE and FMA them
// into every token's 8-lane accumulator (`acc[i*8]`) — the prefill amortization.
// Per-token order: chunks 0..4 of block 0, then block 1, ... so `out[i]` is
// bit-identical to a single-token run of THIS structure. `acc` is `n*8` f32.
#[target_feature(enable = "avx2,fma")]
pub(super) unsafe fn row_dot_q8_0_avx2_batched(
    a: &[f32],
    bytes: &[u8],
    row: usize,
    in_dim: usize,
    out: &mut [f32],
    acc: &mut [f32],
) {
    unsafe {
        let n = out.len();
        let nb = in_dim / 32;
        let base = row * nb * 34;
        for v in acc[..n * 8].iter_mut() {
            *v = 0.0;
        }
        for s in 0..nb {
            let blk = base + s * 34;
            let d = _mm256_set1_ps(f16_at(bytes, blk));
            let abase = s * 32;
            let w = unpack_block(bytes, blk, d);
            for i in 0..n {
                let ap = acc.as_mut_ptr().add(i * 8);
                let mut ai = _mm256_loadu_ps(ap);
                let ar = a.as_ptr().add(i * in_dim + abase);
                ai = _mm256_fmadd_ps(_mm256_loadu_ps(ar), w[0], ai);
                ai = _mm256_fmadd_ps(_mm256_loadu_ps(ar.add(8)), w[1], ai);
                ai = _mm256_fmadd_ps(_mm256_loadu_ps(ar.add(16)), w[2], ai);
                ai = _mm256_fmadd_ps(_mm256_loadu_ps(ar.add(24)), w[3], ai);
                _mm256_storeu_ps(ap, ai);
            }
        }
        for (i, o) in out.iter_mut().enumerate() {
            *o = hsum256(_mm256_loadu_ps(acc.as_ptr().add(i * 8)));
        }
    }
}

// Two-output-row batched fused Q8_0 dot: rows `row0` and `row0+1` together, reusing
// each token's four activation loads across both rows — the prefill register-blocking
// win. Per-token order matches the single-row batched kernel → bit-identical to two
// of those calls. `acc` is `2*n*8` f32 (row0 in `[0..n*8]`, row1 in `[n*8..]`).
#[target_feature(enable = "avx2,fma")]
pub(super) unsafe fn row2_dot_q8_0_avx2_batched(
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
        let nb = in_dim / 32;
        let base0 = row0 * nb * 34;
        let base1 = (row0 + 1) * nb * 34;
        for v in acc[..2 * n * 8].iter_mut() {
            *v = 0.0;
        }
        for s in 0..nb {
            let blk0 = base0 + s * 34;
            let blk1 = base1 + s * 34;
            let d0 = _mm256_set1_ps(f16_at(bytes, blk0));
            let d1 = _mm256_set1_ps(f16_at(bytes, blk1));
            let abase = s * 32;
            let w0 = unpack_block(bytes, blk0, d0);
            let w1 = unpack_block(bytes, blk1, d1);
            for i in 0..n {
                let ar = a.as_ptr().add(i * in_dim + abase);
                let av0 = _mm256_loadu_ps(ar);
                let av1 = _mm256_loadu_ps(ar.add(8));
                let av2 = _mm256_loadu_ps(ar.add(16));
                let av3 = _mm256_loadu_ps(ar.add(24));
                let p0 = acc.as_mut_ptr().add(i * 8);
                let p1 = acc.as_mut_ptr().add((n + i) * 8);
                let mut a0 = _mm256_loadu_ps(p0);
                let mut a1 = _mm256_loadu_ps(p1);
                a0 = _mm256_fmadd_ps(av0, w0[0], a0);
                a1 = _mm256_fmadd_ps(av0, w1[0], a1);
                a0 = _mm256_fmadd_ps(av1, w0[1], a0);
                a1 = _mm256_fmadd_ps(av1, w1[1], a1);
                a0 = _mm256_fmadd_ps(av2, w0[2], a0);
                a1 = _mm256_fmadd_ps(av2, w1[2], a1);
                a0 = _mm256_fmadd_ps(av3, w0[3], a0);
                a1 = _mm256_fmadd_ps(av3, w1[3], a1);
                _mm256_storeu_ps(p0, a0);
                _mm256_storeu_ps(p1, a1);
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
