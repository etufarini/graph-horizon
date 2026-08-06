/*
 * graph_horizon_engine — CPU f16 <-> f32 conversion
 * This file owns only CPU slice conversion and its F16C/AVX2 acceleration.
 * Scalar IEEE conversion lives in `backend::f16`; both SIMD tails and tests use
 * those primitives so CPU and Vulkan share one rounding implementation. Buffer
 * storage re-exports this narrow surface for existing CPU kernel call sites.
*/

pub(crate) use crate::backend::f16::{f16_to_f32, f32_to_f16, f32_to_f16_bytes};

// Widens a little-endian FP16 byte slice (length a multiple of 2) to a fresh f32
// Vec. On x86_64 with F16C the bulk runs 8 elements per `_mm256_cvtph_ps` (the
// hardware FP16→FP32 widening): FP16→FP32 is EXACT (every f16 is representable in
// f32), so the SIMD result is BIT-IDENTICAL to the scalar `f16_to_f32`, not merely
// within tolerance. Any tail (and every non-x86 target) uses the scalar path. This
// is the hot read on every CPU op's activation input, so vectorizing it removes a
// serial scalar pass that the parallel kernels could not hide.
pub(crate) fn f16_slice_to_f32(src: &[u8]) -> Vec<f32> {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("f16c") && is_x86_feature_detected!("avx2") {
            let n = src.len() / 2;
            // SAFETY: guarded by the runtime F16C+AVX2 detection just above.
            return unsafe { f16_slice_to_f32_f16c(src, n) };
        }
    }
    src.chunks_exact(2)
        .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
        .collect()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "f16c,avx2")]
unsafe fn f16_slice_to_f32_f16c(src: &[u8], n: usize) -> Vec<f32> {
    use core::arch::x86_64::*;
    unsafe {
        let mut out = Vec::<f32>::with_capacity(n);
        let dst = out.as_mut_ptr();
        let mut i = 0;
        // 8 FP16 (16 bytes) → 8 f32 per iteration.
        while i + 8 <= n {
            let raw = _mm_loadu_si128(src.as_ptr().add(i * 2) as *const __m128i);
            _mm256_storeu_ps(dst.add(i), _mm256_cvtph_ps(raw));
            i += 8;
        }
        // Scalar tail for the last < 8 elements; same bits as the SIMD widening.
        while i < n {
            let h = u16::from_le_bytes([src[i * 2], src[i * 2 + 1]]);
            *dst.add(i) = f16_to_f32(h);
            i += 1;
        }
        out.set_len(n);
        out
    }
}

// Narrows `v` (f32) into `dst` as little-endian FP16 bytes (`dst.len() >= v.len()*2`),
// dispatching to the F16C SIMD path when present, else the scalar fold — bit-identical
// (both round-to-nearest-even). The single narrow entry point: `write_f16_from_f32` and
// the fused parallel transpose+narrow (`kernels::matmul`) both go through it, so the
// rounding has one transcription. Operates on a caller-provided slice so a worker can
// narrow straight into its disjoint window region with no temporary Vec.
pub(crate) fn narrow_f32_to_f16(v: &[f32], dst: &mut [u8]) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("f16c") && is_x86_feature_detected!("avx2") {
            // SAFETY: guarded by the runtime F16C+AVX2 detection just above.
            unsafe { f32_slice_to_f16(v, dst) };
            return;
        }
    }
    for (d, &x) in dst.chunks_exact_mut(2).zip(v) {
        d.copy_from_slice(&f32_to_f16(x).to_le_bytes());
    }
}

// Narrows `v` (f32) into `dst` as little-endian FP16 bytes (dst length ≥ v.len()*2).
// On x86_64 with F16C the bulk runs 8 elements per `_mm256_cvtps_ph` using
// round-to-nearest-even — the SAME rounding the scalar `f32_to_f16` implements, so
// the bytes are bit-identical for all finite inputs (verified against the scalar
// path over a wide value sweep by `simd_f16_write_matches_scalar`). The tail and
// non-x86 targets use the scalar path. This is the hot write on every CPU op's
// output; vectorizing it removes a serial scalar pass.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "f16c,avx2")]
unsafe fn f32_slice_to_f16(v: &[f32], dst: &mut [u8]) {
    use core::arch::x86_64::*;
    unsafe {
        let n = v.len().min(dst.len() / 2);
        let mut i = 0;
        while i + 8 <= n {
            let f = _mm256_loadu_ps(v.as_ptr().add(i));
            // 8 f32 → 8 f16 (round to nearest even), stored as 16 bytes.
            let h = _mm256_cvtps_ph::<_MM_FROUND_TO_NEAREST_INT>(f);
            _mm_storeu_si128(dst.as_mut_ptr().add(i * 2) as *mut __m128i, h);
            i += 8;
        }
        while i < n {
            let bytes = f32_to_f16(v[i]).to_le_bytes();
            dst[i * 2] = bytes[0];
            dst[i * 2 + 1] = bytes[1];
            i += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::{CpuBuffer, CpuFormat};
    use super::f32_to_f16;

    // The SIMD F16C write path must be BIT-IDENTICAL to the scalar `f32_to_f16` over
    // a dense value sweep (normals, subnormals near the f16 min, values straddling
    // each rounding boundary, negatives, zeros). Skipped if the host lacks F16C.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn simd_f16_write_matches_scalar() {
        if !(is_x86_feature_detected!("f16c") && is_x86_feature_detected!("avx2")) {
            return;
        }
        let mut vals: Vec<f32> = Vec::new();
        // Dense sweep across magnitudes, both signs, plus exact-half rounding cases.
        let mut x = -70000.0f32;
        while x < 70000.0 {
            vals.push(x);
            x += 0.013;
        }
        for k in -260..260 {
            vals.push(2f32.powi(k / 10) * 1.000_976_6); // near rounding boundaries
            vals.push(6.103_515_6e-5 * k as f32); // around the f16 subnormal range
        }
        vals.extend_from_slice(&[0.0, -0.0, 1.0, -1.0, 65504.0, -65504.0, 1e-8, -1e-8]);

        let buf = CpuBuffer::zeroed(vals.len() * 2, CpuFormat::F16);
        buf.write_f16_from_f32(&vals); // SIMD path on this host
        let got = buf.bytes().clone();
        for (i, &v) in vals.iter().enumerate() {
            let want = f32_to_f16(v).to_le_bytes();
            assert_eq!(
                [got[i * 2], got[i * 2 + 1]],
                want,
                "value {v} (idx {i}): simd != scalar"
            );
        }
    }
}
