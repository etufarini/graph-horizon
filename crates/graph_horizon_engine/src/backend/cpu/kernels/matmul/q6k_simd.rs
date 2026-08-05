/*
 * graph_horizon_engine — SIMD variant of the fused Q6_K CPU kernel
 * The AVX2+FMA fused dequant+MAC for Q6_K, the sibling of matmul_q4k_simd. The
 * generic path (kernels::matmul) decodes a whole Q6_K output row into an `in_dim`
 * f32 Vec (dequant::dequant_row_q6_k) and only then dots it; this kernel reuses the
 * SAME bit-exact integer unpack as `dequant_row_q6_k_avx2` (ql low/high nibble | the
 * two qh high bits, recentered by -32) but FMAs each decoded 8-wide weight vector
 * straight into an accumulator — it never materializes the f32 row. That removes the
 * row's L1/L2 round-trip: the decode win for prefill (cache-resident sub-block reuse
 * across tokens) and for the Q6_K lm_head decode GEMV (no per-row f32 spill).
 *
 * Numerics: the per-row accumulation is reordered into the four quant streams
 * (q1..q4), summed at the end, so the result is within the quantized tolerance of
 * dequant_row (not bit-identical) — same contract as the Q4_K fused kernel. The
 * unpack itself is byte-for-byte the validated `dequant_row_q6_k_avx2`. Dispatch is
 * in matmul_q6k::row_dot_q6k* behind a runtime AVX2+FMA check; other targets use the
 * scalar fused kernel. SAFETY: every entry point is reached only after that check, and
 * all loads/stores stay within `a`/`out`/`acc` (sized to `n`) and `bytes` — the
 * load-validated weight tensor slice (SEC-INV: `gguf::loader` rejects any tensor whose
 * `offset+byte_len` overruns the file or whose `byte_len` is incoherent with
 * `dims × block`, before a byte reaches here), so the per-`row` block offset stays
 * within `bytes` for every `row < out_dim` (`in_dim` a multiple of 256).
*/

// AGENTS deroga K: kernel matmul Q6_K SIMD (AVX2), una sola operazione.

#![cfg(target_arch = "x86_64")]

use core::arch::x86_64::*;

use super::q4k::f16_at;

// The four 8-wide weight vectors of one l-chunk: w[k] = d*scale_k * (q_k - 32) for
// the four quant streams q1..q4 (offsets +0,+32,+64,+96 within the 128-quant
// segment). Factored so the decode/batched/2-row kernels share ONE transcription of
// the unpack — identical to `dequant_row_q6_k_avx2`'s body. `consts` carries the
// loop-invariant masks/bias; the caller supplies the chunk's byte offsets and `d`.
#[target_feature(enable = "avx2")]
unsafe fn unpack_chunk(
    bytes: &[u8],
    qlb: usize,
    qhb: usize,
    scb: usize,
    is: usize,
    l: usize,
    d: f32,
) -> [__m256; 4] {
    unsafe {
        let m4 = _mm256_set1_epi32(0xF);
        let m3 = _mm256_set1_epi32(3);
        let c32 = _mm256_set1_epi32(32);
        let load8 = |off: usize| {
            _mm256_cvtepu8_epi32(_mm_loadl_epi64(bytes.as_ptr().add(off) as *const __m128i))
        };
        let s8 = |b: usize| bytes[b] as i8 as f32;
        let dl1 = _mm256_set1_ps(d * s8(scb + is));
        let dl2 = _mm256_set1_ps(d * s8(scb + is + 2));
        let dl3 = _mm256_set1_ps(d * s8(scb + is + 4));
        let dl4 = _mm256_set1_ps(d * s8(scb + is + 6));
        let lo0 = load8(qlb + l);
        let lo1 = load8(qlb + l + 32);
        let h = load8(qhb + l);
        let q1 = _mm256_sub_epi32(
            _mm256_or_si256(
                _mm256_and_si256(lo0, m4),
                _mm256_slli_epi32(_mm256_and_si256(h, m3), 4),
            ),
            c32,
        );
        let q2 = _mm256_sub_epi32(
            _mm256_or_si256(
                _mm256_and_si256(lo1, m4),
                _mm256_slli_epi32(_mm256_and_si256(_mm256_srli_epi32(h, 2), m3), 4),
            ),
            c32,
        );
        let q3 = _mm256_sub_epi32(
            _mm256_or_si256(
                _mm256_srli_epi32(lo0, 4),
                _mm256_slli_epi32(_mm256_and_si256(_mm256_srli_epi32(h, 4), m3), 4),
            ),
            c32,
        );
        let q4 = _mm256_sub_epi32(
            _mm256_or_si256(
                _mm256_srli_epi32(lo1, 4),
                _mm256_slli_epi32(_mm256_and_si256(_mm256_srli_epi32(h, 6), m3), 4),
            ),
            c32,
        );
        [
            _mm256_mul_ps(dl1, _mm256_cvtepi32_ps(q1)),
            _mm256_mul_ps(dl2, _mm256_cvtepi32_ps(q2)),
            _mm256_mul_ps(dl3, _mm256_cvtepi32_ps(q3)),
            _mm256_mul_ps(dl4, _mm256_cvtepi32_ps(q4)),
        ]
    }
}

// Single-token fused Q6_K dot (decode GEMV / lm_head). Four accumulators, one per
// quant stream, so the four FMA chains run independently on the FMA ports (the
// latency-hiding the single accumulator lacks at n==1, mirroring the Q4_K decode
// kernel). `in_dim` is a multiple of 256.
#[target_feature(enable = "avx2,fma")]
pub(super) unsafe fn row_dot_q6k_avx2(a: &[f32], bytes: &[u8], row: usize, in_dim: usize) -> f32 {
    unsafe {
        let nsb = in_dim / 256;
        let base = row * nsb * 210;
        let mut acc = [_mm256_setzero_ps(); 4];
        for s in 0..nsb {
            let blk = base + s * 210;
            let (qlo, qho, sco) = (blk, blk + 128, blk + 192);
            let d = f16_at(bytes, blk + 208);
            let abase = s * 256;
            let mut n = 0usize;
            while n < 256 {
                let seg = n / 128;
                let (qlb, qhb, scb) = (qlo + seg * 64, qho + seg * 32, sco + seg * 8);
                let mut l = 0usize;
                while l < 32 {
                    let w = unpack_chunk(bytes, qlb, qhb, scb, l / 16, l, d);
                    let o = abase + n + l;
                    acc[0] = _mm256_fmadd_ps(_mm256_loadu_ps(a.as_ptr().add(o)), w[0], acc[0]);
                    acc[1] = _mm256_fmadd_ps(_mm256_loadu_ps(a.as_ptr().add(o + 32)), w[1], acc[1]);
                    acc[2] = _mm256_fmadd_ps(_mm256_loadu_ps(a.as_ptr().add(o + 64)), w[2], acc[2]);
                    acc[3] = _mm256_fmadd_ps(_mm256_loadu_ps(a.as_ptr().add(o + 96)), w[3], acc[3]);
                    l += 8;
                }
                n += 128;
            }
        }
        let lo = _mm256_add_ps(acc[0], acc[1]);
        let hi = _mm256_add_ps(acc[2], acc[3]);
        hsum256(_mm256_add_ps(lo, hi))
    }
}

// Batched fused Q6_K dot: decode each l-chunk's four weight vectors ONCE and FMA
// them into every token's 8-lane accumulator (`acc[i*8]`), so the dequant is paid
// once for the whole batch (the prefill amortization). Per-token order: stream q1,
// q2, q3, q4 into the one accumulator, l-chunk by l-chunk — so `out[i]` is
// bit-identical to a single-token run of THIS kernel structure (not to the decode
// kernel above, which uses four accumulators). `acc` is caller scratch of `n*8` f32.
#[target_feature(enable = "avx2,fma")]
pub(super) unsafe fn row_dot_q6k_avx2_batched(
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
        let base = row * nsb * 210;
        for v in acc[..n * 8].iter_mut() {
            *v = 0.0;
        }
        for s in 0..nsb {
            let blk = base + s * 210;
            let (qlo, qho, sco) = (blk, blk + 128, blk + 192);
            let d = f16_at(bytes, blk + 208);
            let abase = s * 256;
            let mut n2 = 0usize;
            while n2 < 256 {
                let seg = n2 / 128;
                let (qlb, qhb, scb) = (qlo + seg * 64, qho + seg * 32, sco + seg * 8);
                let mut l = 0usize;
                while l < 32 {
                    let w = unpack_chunk(bytes, qlb, qhb, scb, l / 16, l, d);
                    let o = abase + n2 + l;
                    for i in 0..n {
                        let ap = acc.as_mut_ptr().add(i * 8);
                        let mut ai = _mm256_loadu_ps(ap);
                        let ar = a.as_ptr().add(i * in_dim + o);
                        ai = _mm256_fmadd_ps(_mm256_loadu_ps(ar), w[0], ai);
                        ai = _mm256_fmadd_ps(_mm256_loadu_ps(ar.add(32)), w[1], ai);
                        ai = _mm256_fmadd_ps(_mm256_loadu_ps(ar.add(64)), w[2], ai);
                        ai = _mm256_fmadd_ps(_mm256_loadu_ps(ar.add(96)), w[3], ai);
                        _mm256_storeu_ps(ap, ai);
                    }
                    l += 8;
                }
                n2 += 128;
            }
        }
        for (i, o) in out.iter_mut().enumerate() {
            *o = hsum256(_mm256_loadu_ps(acc.as_ptr().add(i * 8)));
        }
    }
}

// Two-output-row batched fused Q6_K dot: the register-blocked counterpart that
// processes rows `row0` and `row0+1` together so each token's four activation loads
// are REUSED across both rows (one load → two FMAs) — the prefill win, identical in
// spirit to row2_dot_q4k_avx2_batched. Per-token order matches the single-row batched
// kernel, so `out0[i]`/`out1[i]` are bit-identical to two `row_dot_q6k_avx2_batched`
// calls. `acc` is `2*n*8` f32: row0 partials in `[0..n*8]`, row1 in `[n*8..2*n*8]`.
#[target_feature(enable = "avx2,fma")]
pub(super) unsafe fn row2_dot_q6k_avx2_batched(
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
        let base0 = row0 * nsb * 210;
        let base1 = (row0 + 1) * nsb * 210;
        for v in acc[..2 * n * 8].iter_mut() {
            *v = 0.0;
        }
        for s in 0..nsb {
            let blk0 = base0 + s * 210;
            let blk1 = base1 + s * 210;
            let d0 = f16_at(bytes, blk0 + 208);
            let d1 = f16_at(bytes, blk1 + 208);
            let abase = s * 256;
            let mut n2 = 0usize;
            while n2 < 256 {
                let seg = n2 / 128;
                let (qlb0, qhb0, scb0) =
                    (blk0 + seg * 64, blk0 + 128 + seg * 32, blk0 + 192 + seg * 8);
                let (qlb1, qhb1, scb1) =
                    (blk1 + seg * 64, blk1 + 128 + seg * 32, blk1 + 192 + seg * 8);
                let mut l = 0usize;
                while l < 32 {
                    let is = l / 16;
                    let w0 = unpack_chunk(bytes, qlb0, qhb0, scb0, is, l, d0);
                    let w1 = unpack_chunk(bytes, qlb1, qhb1, scb1, is, l, d1);
                    let o = abase + n2 + l;
                    for i in 0..n {
                        let ar = a.as_ptr().add(i * in_dim + o);
                        let av0 = _mm256_loadu_ps(ar);
                        let av1 = _mm256_loadu_ps(ar.add(32));
                        let av2 = _mm256_loadu_ps(ar.add(64));
                        let av3 = _mm256_loadu_ps(ar.add(96));
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
                    l += 8;
                }
                n2 += 128;
            }
        }
        for i in 0..n {
            out0[i] = hsum256(_mm256_loadu_ps(acc.as_ptr().add(i * 8)));
            out1[i] = hsum256(_mm256_loadu_ps(acc.as_ptr().add((n + i) * 8)));
        }
    }
}

// Horizontal sum of the 8 f32 lanes (same as the Q4_K SIMD module's).
#[target_feature(enable = "avx2")]
unsafe fn hsum256(v: __m256) -> f32 {
    let lo = _mm256_castps256_ps128(v);
    let hi = _mm256_extractf128_ps(v, 1);
    let s = _mm_add_ps(lo, hi);
    let s = _mm_hadd_ps(s, s);
    let s = _mm_hadd_ps(s, s);
    _mm_cvtss_f32(s)
}
