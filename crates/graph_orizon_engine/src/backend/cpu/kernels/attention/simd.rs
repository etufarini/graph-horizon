/*
 * graph_orizon_engine — SIMD variant of the attention inner kernels (AVX2 + F16C)
 * Carved out of `attention` so that file keeps the orchestration + the portable
 * scalar inner loops, and this one holds the architecture-specific hot path. The
 * `attend` inner work is two reductions over `head_dim` against the FP16 KV cache,
 * run once per (query, key) — i.e. O(n²) over the prompt. The scalar path converts
 * each FP16 element with the software `f16_to_f32` (a branchy bit-twiddle) and does
 * a scalar MAC; this module replaces both with one `_mm256_cvtph_ps` (8 FP16→f32 in
 * a single F16C instruction) feeding an 8-wide FMA.
 *
 * Dispatch is in `attention::{dot_f16,axpy_f16}`: the caller resolves the
 * avx2+fma+f16c support ONCE per `attention_*` call (never in the inner loop) and
 * passes the bool down, so a binary built here stays correct on a CPU without those
 * features (it uses the scalar fallback). Other architectures use the scalar path
 * unchanged.
 *
 * Numerics: `_mm256_cvtph_ps` is the exact IEEE FP16→FP32 widening, identical to the
 * software `f16_to_f32`, so the converted values match bit-for-bit. Only the
 * reduction order differs — the dot product sums eight lane-partials before the
 * horizontal add (and the V accumulate uses a fused multiply-add) — so the result is
 * within the quantized tolerance of the scalar kernel, not bit-identical, exactly
 * like the Q4_K SIMD kernel. Both `attend` callers (prefill and decode) go through
 * the same dispatch, so they stay numerically consistent at n == 1.
*/

// AGENTS deroga K: variante SIMD del solo kernel numerico attention.

#[cfg(target_arch = "x86_64")]
use crate::backend::cpu::buffer::f16_to_f32;

// AVX2+F16C dot product of an f32 query head with an FP16 key row: Σ_d q[d]·f16(k[d]).
// `kbase` is the element offset of the key row in `kc` (FP16, 2 bytes each). The
// 8-wide tail (head_dim % 8) falls back to scalar — head_dim is 128 here, so the
// tail is empty, but keeping it makes the kernel correct for any head_dim. SAFETY:
// reached only when the caller resolved avx2+fma+f16c; every load stays within `q`
// (head_dim f32) and `kc` (the key row is head_dim FP16 wide).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
pub(super) unsafe fn dot_f16_avx2(q: &[f32], kc: &[u8], kbase: usize, head_dim: usize) -> f32 {
    use core::arch::x86_64::*;
    unsafe {
        let mut acc = _mm256_setzero_ps();
        let mut d = 0;
        while d + 8 <= head_dim {
            let qv = _mm256_loadu_ps(q.as_ptr().add(d));
            // 8 FP16 (16 bytes) → 8 f32 in one F16C conversion.
            let kh = _mm_loadu_si128(kc.as_ptr().add((kbase + d) * 2) as *const __m128i);
            let kf = _mm256_cvtph_ps(kh);
            acc = _mm256_fmadd_ps(qv, kf, acc);
            d += 8;
        }
        let mut s = hsum256(acc);
        while d < head_dim {
            s += q[d]
                * f16_to_f32(u16::from_le_bytes([
                    kc[(kbase + d) * 2],
                    kc[(kbase + d) * 2 + 1],
                ]));
            d += 1;
        }
        s
    }
}

// AVX2+F16C scaled accumulate of an FP16 value row into an f32 output:
// out[d] += w·f16(v[d]). `vbase` is the element offset of the value row in `vc`.
// SAFETY: as `dot_f16_avx2`; `out` and the value row are both head_dim wide.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
pub(super) unsafe fn axpy_f16_avx2(
    out: &mut [f32],
    w: f32,
    vc: &[u8],
    vbase: usize,
    head_dim: usize,
) {
    use core::arch::x86_64::*;
    unsafe {
        let wv = _mm256_set1_ps(w);
        let mut d = 0;
        while d + 8 <= head_dim {
            let ov = _mm256_loadu_ps(out.as_ptr().add(d));
            let vh = _mm_loadu_si128(vc.as_ptr().add((vbase + d) * 2) as *const __m128i);
            let vf = _mm256_cvtph_ps(vh);
            _mm256_storeu_ps(out.as_mut_ptr().add(d), _mm256_fmadd_ps(wv, vf, ov));
            d += 8;
        }
        while d < head_dim {
            out[d] += w * f16_to_f32(u16::from_le_bytes([
                vc[(vbase + d) * 2],
                vc[(vbase + d) * 2 + 1],
            ]));
            d += 1;
        }
    }
}

// Two-query GQA-group variant of `dot_f16_avx2`: dots ONE FP16 key row against two
// f32 query heads that share the same kv head, widening the key once (`_mm256_cvtph_ps`)
// and feeding both FMA chains. The per-query FMA order, cvtph result, hsum and scalar
// tail are byte-for-byte those of `dot_f16_avx2(q0)` / `(q1)`, so `(s0,s1)` are
// BIT-IDENTICAL to two separate calls — the win is halving the key load+widen (the
// inner-loop cost) across the GQA group. SAFETY: as `dot_f16_avx2`; `q0`/`q1` are
// head_dim f32, the key row is head_dim FP16 wide.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
pub(super) unsafe fn dot_f16_x2_avx2(
    q0: &[f32],
    q1: &[f32],
    kc: &[u8],
    kbase: usize,
    head_dim: usize,
) -> (f32, f32) {
    use core::arch::x86_64::*;
    unsafe {
        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();
        let mut d = 0;
        while d + 8 <= head_dim {
            let kh = _mm_loadu_si128(kc.as_ptr().add((kbase + d) * 2) as *const __m128i);
            let kf = _mm256_cvtph_ps(kh); // widen the key ONCE, reuse for both queries
            acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(q0.as_ptr().add(d)), kf, acc0);
            acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(q1.as_ptr().add(d)), kf, acc1);
            d += 8;
        }
        let mut s0 = hsum256(acc0);
        let mut s1 = hsum256(acc1);
        while d < head_dim {
            let kf = f16_to_f32(u16::from_le_bytes([
                kc[(kbase + d) * 2],
                kc[(kbase + d) * 2 + 1],
            ]));
            s0 += q0[d] * kf;
            s1 += q1[d] * kf;
            d += 1;
        }
        (s0, s1)
    }
}

// Two-output GQA-group variant of `axpy_f16_avx2`: accumulates ONE FP16 value row into
// two outputs with per-head weights, widening the value once. Byte-for-byte two
// separate `axpy_f16_avx2` calls. SAFETY: as `axpy_f16_avx2`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
pub(super) unsafe fn axpy_f16_x2_avx2(
    out0: &mut [f32],
    out1: &mut [f32],
    w0: f32,
    w1: f32,
    vc: &[u8],
    vbase: usize,
    head_dim: usize,
) {
    use core::arch::x86_64::*;
    unsafe {
        let wv0 = _mm256_set1_ps(w0);
        let wv1 = _mm256_set1_ps(w1);
        let mut d = 0;
        while d + 8 <= head_dim {
            let vh = _mm_loadu_si128(vc.as_ptr().add((vbase + d) * 2) as *const __m128i);
            let vf = _mm256_cvtph_ps(vh); // widen the value ONCE, reuse for both outputs
            let o0 = _mm256_loadu_ps(out0.as_ptr().add(d));
            _mm256_storeu_ps(out0.as_mut_ptr().add(d), _mm256_fmadd_ps(wv0, vf, o0));
            let o1 = _mm256_loadu_ps(out1.as_ptr().add(d));
            _mm256_storeu_ps(out1.as_mut_ptr().add(d), _mm256_fmadd_ps(wv1, vf, o1));
            d += 8;
        }
        while d < head_dim {
            let vf = f16_to_f32(u16::from_le_bytes([
                vc[(vbase + d) * 2],
                vc[(vbase + d) * 2 + 1],
            ]));
            out0[d] += w0 * vf;
            out1[d] += w1 * vf;
            d += 1;
        }
    }
}

// Horizontal sum of the 8 f32 lanes.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn hsum256(v: core::arch::x86_64::__m256) -> f32 {
    use core::arch::x86_64::*;
    let lo = _mm256_castps256_ps128(v);
    let hi = _mm256_extractf128_ps(v, 1);
    let s = _mm_add_ps(lo, hi);
    let s = _mm_hadd_ps(s, s);
    let s = _mm_hadd_ps(s, s);
    _mm_cvtss_f32(s)
}

#[cfg(all(test, target_arch = "x86_64"))]
mod tests {
    use super::*;
    use crate::backend::cpu::buffer::f32_to_f16;

    // Build a head_dim-wide FP16 row from f32 values (little-endian, as the cache).
    fn f16_row(vals: &[f32]) -> Vec<u8> {
        vals.iter()
            .flat_map(|&v| f32_to_f16(v).to_le_bytes())
            .collect()
    }

    // The SIMD dot and axpy must match their scalar references within the quantized
    // tolerance (rel. 8e-2). Skipped if the host lacks avx2+fma+f16c.
    #[test]
    fn simd_inner_matches_scalar_within_tolerance() {
        if !(is_x86_feature_detected!("avx2")
            && is_x86_feature_detected!("fma")
            && is_x86_feature_detected!("f16c"))
        {
            return;
        }
        let head_dim = 128;
        let q: Vec<f32> = (0..head_dim)
            .map(|i| (i as f32 * 0.021).sin() * 0.8)
            .collect();
        let kv_f: Vec<f32> = (0..head_dim)
            .map(|i| (i as f32 * 0.013).cos() * 0.5)
            .collect();
        let kv = f16_row(&kv_f);
        // Scalar references (mirror the attend inner loops over a single row at base 0).
        let dot_scalar: f32 = (0..head_dim)
            .map(|d| q[d] * f16_to_f32(u16::from_le_bytes([kv[d * 2], kv[d * 2 + 1]])))
            .sum();
        // SAFETY: guarded by the feature check above.
        let dot_simd = unsafe { dot_f16_avx2(&q, &kv, 0, head_dim) };
        let tol = 8e-2 * dot_scalar.abs().max(1e-3);
        assert!(
            (dot_simd - dot_scalar).abs() <= tol,
            "dot {dot_simd} vs {dot_scalar}"
        );

        let w = 0.37f32;
        let mut out_scalar = vec![1.0f32; head_dim];
        let mut out_simd = vec![1.0f32; head_dim];
        for d in 0..head_dim {
            out_scalar[d] += w * f16_to_f32(u16::from_le_bytes([kv[d * 2], kv[d * 2 + 1]]));
        }
        unsafe { axpy_f16_avx2(&mut out_simd, w, &kv, 0, head_dim) };
        for d in 0..head_dim {
            let tol = 8e-2 * out_scalar[d].abs().max(1e-3);
            assert!((out_simd[d] - out_scalar[d]).abs() <= tol, "axpy[{d}]");
        }
    }
}
