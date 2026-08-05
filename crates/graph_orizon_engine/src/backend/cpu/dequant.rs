/*
 * graph_orizon_engine — CPU weight dequantization
 * Scalar transcription of the on-the-fly dequantization the Vulkan shaders do
 * for the retained Q4_K/Q5_K/Q6_K formats. Holds their ggml block layouts —
 * Q4_K (144 B / 256 values, with get_scale_min_k4), Q5_K (176 B / 256
 * values, Q4_K layout plus a 5th bit per quant from qh[32]), Q6_K (210 B / 256
 * values) — and the F16 widening. Length validation lives HERE and only runs at
 * `load`: `dequant_row` returns `()` and assumes valid blocks, exactly like the
 * shaders. Values are produced in natural in-dimension order, so weight `i` pairs
 * with activation `a[i]`; the kernels (m1/m2) consume one row at a time, never
 * materializing the whole tensor in f32.
*/

// AGENTS deroga K: kernel dequant per-formato di una sola operazione (dequantizzazione righe pesi), nessun I/O né ownership.

use color_eyre::eyre::{Result, bail};

use super::buffer::{CpuFormat, f16_to_f32};

// Bytes per ggml block for each weight format.
pub(crate) fn block_len(format: CpuFormat) -> usize {
    match format {
        CpuFormat::F16 => 2,
        CpuFormat::Q4_K => 144,
        CpuFormat::Q5_K => 176,
        CpuFormat::Q6_K => 210,
        CpuFormat::F32 => 4,
    }
}

// Validates a quantized weight's byte length at load: it must be a whole number
// of blocks. This is the only place the block sizes are checked; downstream
// kernels then assume valid blocks. GGUF data is untrusted, so this is an
// explicit guard shared by every retained quantized CPU format.
pub(crate) fn validate(format: CpuFormat, byte_len: usize) -> Result<()> {
    let bl = block_len(format);
    if !byte_len.is_multiple_of(bl) {
        bail!("cpu: malformed quantized weight (length not a multiple of the block size)");
    }
    Ok(())
}

// Dequantizes output row `row` (length `in_dim`) into `out[..in_dim]`, in natural
// in-dimension order. A row is a whole number of contiguous blocks; the
// exactness of that division is guaranteed by `validate` at load. Cannot fail:
// block validity is established once, at load. Consumed by the kernels in m1.
pub(crate) fn dequant_row(
    format: CpuFormat,
    bytes: &[u8],
    row: usize,
    in_dim: usize,
    out: &mut [f32],
) {
    match format {
        CpuFormat::F16 => {
            let base = row * in_dim * 2;
            for (i, o) in out[..in_dim].iter_mut().enumerate() {
                *o = f16_at(bytes, base + i * 2);
            }
        }
        CpuFormat::Q4_K => dequant_row_q4_k(bytes, row, in_dim, out),
        CpuFormat::Q5_K => dequant_row_q5_k(bytes, row, in_dim, out),
        CpuFormat::Q6_K => dequant_row_q6_k(bytes, row, in_dim, out),
        // F32 is not a weight format; weights are F16/Q4_K/Q5_K/Q6_K.
        CpuFormat::F32 => unreachable!("cpu: F32 is not a quantized weight format"),
    }
}

// FP16 at byte address `b` (two little-endian bytes) widened to f32.
fn f16_at(bytes: &[u8], b: usize) -> f32 {
    f16_to_f32(u16::from_le_bytes([bytes[b], bytes[b + 1]]))
}

// 6-bit scale/min codes of sub-block `j` (0..7) from scales[12] at byte `sco`.
// Exact port of get_scale_min_k4 / scale_min in matmul_q4_k.comp. Shared with the
// fused Q4_K kernel (kernels::matmul::q4k) so the easy-to-get-wrong j>=4 6-bit
// recomposition has a single transcription.
pub(crate) fn scale_min(bytes: &[u8], sco: usize, j: usize) -> (u32, u32) {
    let gb = |b: usize| bytes[b] as u32;
    if j < 4 {
        (gb(sco + j) & 63, gb(sco + j + 4) & 63)
    } else {
        let hi = gb(sco + j + 4);
        let lo = gb(sco + j - 4);
        let sc = (hi & 0xF) | ((lo >> 6) << 4);
        let mn = (hi >> 4) | ((gb(sco + j) >> 6) << 4);
        (sc, mn)
    }
}

// Q4_K: block = d(f16) | dmin(f16) | scales[12] | qs[128]; 8 sub-blocks of 32,
// w = d*scale*q4 - dmin*min. Mirror of matmul_q4_k.comp.
fn dequant_row_q4_k(bytes: &[u8], row: usize, in_dim: usize, out: &mut [f32]) {
    let nsb = in_dim / 256;
    let base = row * nsb * 144;
    for s in 0..nsb {
        let blk = base + s * 144;
        let d = f16_at(bytes, blk);
        let dmin = f16_at(bytes, blk + 2);
        let sco = blk + 4;
        let qso = blk + 16;
        let abase = s * 256;
        let mut is = 0usize;
        let mut qb = 0usize;
        let mut j = 0usize;
        while j < 256 {
            let (sc1, mn1) = scale_min(bytes, sco, is);
            let (sc2, mn2) = scale_min(bytes, sco, is + 1);
            let d1 = d * sc1 as f32;
            let m1 = dmin * mn1 as f32;
            let d2 = d * sc2 as f32;
            let m2 = dmin * mn2 as f32;
            for l in 0..32 {
                let qv = bytes[qso + qb + l] as u32;
                out[abase + j + l] = d1 * (qv & 0xF) as f32 - m1;
                out[abase + j + 32 + l] = d2 * (qv >> 4) as f32 - m2;
            }
            qb += 32;
            is += 2;
            j += 64;
        }
    }
}

// Q5_K: block = d(f16) | dmin(f16) | scales[12] | qh[32] | qs[128]; 8 sub-blocks
// of 32, w = d*sc*(q4 + 5th_bit*16) - dmin*mn. Identical to Q4_K (same scales via
// get_scale_min_k4, same low-4-bit qs nibbles) plus one high bit per quant taken
// from qh[32]: sub-block pair `p` (0..3) uses bit 2*p for the even sub-block and
// bit 2*p+1 for the odd one. Mirror of dequantize_row_q5_K.
// Q5_K row dequant. AVX2 path (bit-identical, validated by the q5_k golden test):
// same nibble + 5th-bit unpack and the same `d*q - m` order (kept as separate
// mul+sub, NOT an FMA, so the two roundings match the scalar exactly).
fn dequant_row_q5_k(bytes: &[u8], row: usize, in_dim: usize, out: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            // SAFETY: guarded by the runtime AVX2 detection just above.
            return unsafe { dequant_row_q5_k_avx2(bytes, row, in_dim, out) };
        }
    }
    dequant_row_q5_k_scalar(bytes, row, in_dim, out);
}

fn dequant_row_q5_k_scalar(bytes: &[u8], row: usize, in_dim: usize, out: &mut [f32]) {
    let nsb = in_dim / 256;
    let base = row * nsb * 176;
    for s in 0..nsb {
        let blk = base + s * 176;
        let d = f16_at(bytes, blk);
        let dmin = f16_at(bytes, blk + 2);
        let sco = blk + 4;
        let qho = blk + 16; // qh[32]: the 5th bit of every quant
        let qso = blk + 48; // qs[128]: the low 4 bits, two nibbles per byte
        let abase = s * 256;
        let mut is = 0usize;
        let mut qb = 0usize; // advances 32 per sub-block pair (one nibble each)
        let mut u = 0u32; // bit index of the even sub-block's 5th bit
        let mut j = 0usize;
        while j < 256 {
            let (sc1, mn1) = scale_min(bytes, sco, is);
            let (sc2, mn2) = scale_min(bytes, sco, is + 1);
            let d1 = d * sc1 as f32;
            let m1 = dmin * mn1 as f32;
            let d2 = d * sc2 as f32;
            let m2 = dmin * mn2 as f32;
            let u1 = 1u32 << u; // even sub-block high bit
            let u2 = 1u32 << (u + 1); // odd sub-block high bit
            for l in 0..32 {
                let qv = bytes[qso + qb + l] as u32;
                let qh = bytes[qho + l] as u32;
                let lo = (qv & 0xF) + if qh & u1 != 0 { 16 } else { 0 };
                let hi = (qv >> 4) + if qh & u2 != 0 { 16 } else { 0 };
                out[abase + j + l] = d1 * lo as f32 - m1;
                out[abase + j + 32 + l] = d2 * hi as f32 - m2;
            }
            qb += 32;
            is += 2;
            u += 2;
            j += 64;
        }
    }
}

// AVX2 transcription of `dequant_row_q5_k_scalar`: the `for l in 0..32` inner loop
// runs as four 8-wide chunks. The per-pair scales (d1/m1/d2/m2) stay scalar (one
// broadcast each), matching the scalar; the low nibble + the 5th bit (selected from
// qh at the runtime bit index `u`/`u+1` via the AVX2 variable shift `srlv`) are the
// SAME integer ops, vectorized; and `d*q - m` is kept as a separate mul then sub
// (NOT fused) so the two roundings match the scalar exactly → bit-identical. `in_dim`
// is a multiple of 256, so the 32-quant sub-blocks split cleanly into 8s, no tail.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dequant_row_q5_k_avx2(bytes: &[u8], row: usize, in_dim: usize, out: &mut [f32]) {
    use core::arch::x86_64::*;
    // SAFETY: AVX2 gated by the caller's runtime check. `bytes` is the load-validated
    // weight tensor slice (SEC-INV: gguf::loader rejects offset+byte_len overrun and any
    // byte_len incoherent with dims×block before a byte reaches here), so `base = row *
    // (in_dim/256) * 176` and the 176-byte block read at it stay within `bytes` for row <
    // out_dim; `out` is sized to `in_dim` by the caller.
    unsafe {
        let nsb = in_dim / 256;
        let base = row * nsb * 176;
        let m4 = _mm256_set1_epi32(0xF);
        let c1 = _mm256_set1_epi32(1);
        let load8 = |off: usize| {
            _mm256_cvtepu8_epi32(_mm_loadl_epi64(bytes.as_ptr().add(off) as *const __m128i))
        };
        for s in 0..nsb {
            let blk = base + s * 176;
            let d = f16_at(bytes, blk);
            let dmin = f16_at(bytes, blk + 2);
            let sco = blk + 4;
            let qho = blk + 16; // qh[32]: the 5th bit of every quant
            let qso = blk + 48; // qs[128]: the low 4 bits
            let abase = s * 256;
            let mut is = 0usize;
            let mut qb = 0usize;
            let mut u = 0u32;
            let mut j = 0usize;
            while j < 256 {
                let (sc1, mn1) = scale_min(bytes, sco, is);
                let (sc2, mn2) = scale_min(bytes, sco, is + 1);
                let dl1 = _mm256_set1_ps(d * sc1 as f32);
                let ml1 = _mm256_set1_ps(dmin * mn1 as f32);
                let dl2 = _mm256_set1_ps(d * sc2 as f32);
                let ml2 = _mm256_set1_ps(dmin * mn2 as f32);
                let su1 = _mm256_set1_epi32(u as i32); // even sub-block 5th-bit index
                let su2 = _mm256_set1_epi32((u + 1) as i32);
                let mut l = 0usize;
                while l < 32 {
                    let qv = load8(qso + qb + l);
                    let qh = load8(qho + l);
                    // 5th bit → +16: ((qh >> u) & 1) << 4 (srlv: per-lane runtime shift).
                    let hb1 =
                        _mm256_slli_epi32(_mm256_and_si256(_mm256_srlv_epi32(qh, su1), c1), 4);
                    let hb2 =
                        _mm256_slli_epi32(_mm256_and_si256(_mm256_srlv_epi32(qh, su2), c1), 4);
                    let lo = _mm256_add_epi32(_mm256_and_si256(qv, m4), hb1);
                    let hi = _mm256_add_epi32(_mm256_srli_epi32(qv, 4), hb2);
                    // d*q - m, separate mul then sub (two roundings, matching scalar).
                    let o_lo = _mm256_sub_ps(_mm256_mul_ps(dl1, _mm256_cvtepi32_ps(lo)), ml1);
                    let o_hi = _mm256_sub_ps(_mm256_mul_ps(dl2, _mm256_cvtepi32_ps(hi)), ml2);
                    _mm256_storeu_ps(out.as_mut_ptr().add(abase + j + l), o_lo);
                    _mm256_storeu_ps(out.as_mut_ptr().add(abase + j + 32 + l), o_hi);
                    l += 8;
                }
                qb += 32;
                is += 2;
                u += 2;
                j += 64;
            }
        }
    }
}

// Q6_K: block = ql[128] | qh[64] | scales[16] (int8) | d(f16); q6 = (ql | qh<<4)
// recentered by -32, w = d*scale*q6. Mirror of matmul_q6_k.comp.
// Q6_K row dequant. On x86_64 with AVX2 the inner unpack runs 8 quants per
// iteration (`dequant_row_q6_k_avx2`); the SIMD path does the IDENTICAL integer
// unpack and the same `(d*scale)*q` f32 product order, so it is BIT-IDENTICAL to
// the scalar body — the golden-vector tests (which compare exact bytes) validate
// it directly. Only AVX2 is needed (the `d` scale is read scalar via `f16_at`).
fn dequant_row_q6_k(bytes: &[u8], row: usize, in_dim: usize, out: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            // SAFETY: guarded by the runtime AVX2 detection just above.
            return unsafe { dequant_row_q6_k_avx2(bytes, row, in_dim, out) };
        }
    }
    dequant_row_q6_k_scalar(bytes, row, in_dim, out);
}

fn dequant_row_q6_k_scalar(bytes: &[u8], row: usize, in_dim: usize, out: &mut [f32]) {
    let s8 = |b: usize| bytes[b] as i8 as f32;
    let nsb = in_dim / 256;
    let base = row * nsb * 210;
    for s in 0..nsb {
        let blk = base + s * 210;
        let qlo = blk;
        let qho = blk + 128;
        let sco = blk + 192;
        let d = f16_at(bytes, blk + 208);
        let abase = s * 256;
        let mut n = 0usize;
        while n < 256 {
            let seg = n / 128;
            let qlb = qlo + seg * 64;
            let qhb = qho + seg * 32;
            let scb = sco + seg * 8;
            for l in 0..32 {
                let is = l / 16;
                let lo0 = bytes[qlb + l] as u32;
                let lo1 = bytes[qlb + l + 32] as u32;
                let h = bytes[qhb + l] as u32;
                let q1 = ((lo0 & 0xF) | ((h & 3) << 4)) as i32 - 32;
                let q2 = ((lo1 & 0xF) | (((h >> 2) & 3) << 4)) as i32 - 32;
                let q3 = ((lo0 >> 4) | (((h >> 4) & 3) << 4)) as i32 - 32;
                let q4 = ((lo1 >> 4) | (((h >> 6) & 3) << 4)) as i32 - 32;
                out[abase + n + l] = d * s8(scb + is) * q1 as f32;
                out[abase + n + l + 32] = d * s8(scb + is + 2) * q2 as f32;
                out[abase + n + l + 64] = d * s8(scb + is + 4) * q3 as f32;
                out[abase + n + l + 96] = d * s8(scb + is + 6) * q4 as f32;
            }
            n += 128;
        }
    }
}

// AVX2 transcription of `dequant_row_q6_k_scalar`: the `for l in 0..32` inner loop
// runs as four 8-wide chunks (l = 0,8,16,24). `is = l/16` is constant within each
// 8-chunk (8 divides 16), so the four per-quant scales stay scalar (one broadcast
// each), matching the scalar `s8(scb+is..)` reads; the nibble/high-bit unpack and
// the `- 32` bias are the SAME integer ops, vectorized, and `dl = d*scale` then
// `dl*q` keeps the scalar product order — so every f32 is bit-identical. `in_dim`
// is a multiple of 256, so the 32-quant segments split cleanly into 8s, no tail.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dequant_row_q6_k_avx2(bytes: &[u8], row: usize, in_dim: usize, out: &mut [f32]) {
    use core::arch::x86_64::*;
    // SAFETY: AVX2 gated by the caller's runtime check. `bytes` is the load-validated
    // weight tensor slice (SEC-INV: gguf::loader rejects offset+byte_len overrun and any
    // byte_len incoherent with dims×block before a byte reaches here), so `base = row *
    // (in_dim/256) * 210` and the 210-byte block read at it stay within `bytes` for row <
    // out_dim; `out` is sized to `in_dim` by the caller.
    unsafe {
        let s8 = |b: usize| bytes[b] as i8 as f32;
        let nsb = in_dim / 256;
        let base = row * nsb * 210;
        let m4 = _mm256_set1_epi32(0xF);
        let m3 = _mm256_set1_epi32(3);
        let c32 = _mm256_set1_epi32(32);
        // Loads 8 consecutive bytes at `off` as 8 zero-extended i32 lanes.
        let load8 = |off: usize| {
            _mm256_cvtepu8_epi32(_mm_loadl_epi64(bytes.as_ptr().add(off) as *const __m128i))
        };
        for s in 0..nsb {
            let blk = base + s * 210;
            let (qlo, qho, sco) = (blk, blk + 128, blk + 192);
            let d = f16_at(bytes, blk + 208);
            let abase = s * 256;
            let mut n = 0usize;
            while n < 256 {
                let seg = n / 128;
                let qlb = qlo + seg * 64;
                let qhb = qho + seg * 32;
                let scb = sco + seg * 8;
                let mut l = 0usize;
                while l < 32 {
                    let is = l / 16;
                    let dl1 = _mm256_set1_ps(d * s8(scb + is));
                    let dl2 = _mm256_set1_ps(d * s8(scb + is + 2));
                    let dl3 = _mm256_set1_ps(d * s8(scb + is + 4));
                    let dl4 = _mm256_set1_ps(d * s8(scb + is + 6));
                    let lo0 = load8(qlb + l);
                    let lo1 = load8(qlb + l + 32);
                    let h = load8(qhb + l);
                    // q = ((low nibble) | ((2 high bits) << 4)) - 32, per the scalar.
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
                    let base_o = abase + n + l;
                    _mm256_storeu_ps(
                        out.as_mut_ptr().add(base_o),
                        _mm256_mul_ps(dl1, _mm256_cvtepi32_ps(q1)),
                    );
                    _mm256_storeu_ps(
                        out.as_mut_ptr().add(base_o + 32),
                        _mm256_mul_ps(dl2, _mm256_cvtepi32_ps(q2)),
                    );
                    _mm256_storeu_ps(
                        out.as_mut_ptr().add(base_o + 64),
                        _mm256_mul_ps(dl3, _mm256_cvtepi32_ps(q3)),
                    );
                    _mm256_storeu_ps(
                        out.as_mut_ptr().add(base_o + 96),
                        _mm256_mul_ps(dl4, _mm256_cvtepi32_ps(q4)),
                    );
                    l += 8;
                }
                n += 128;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    // Golden-vector tests: the expected vectors are FIXED reference data built
    // from known input bytes and the ggml block formula written out by hand
    // (explicit scale/quant tables + literal spot-checks). They are NEVER
    // produced by `dequant_row` — a reference generated by the code under test
    // would only confirm itself.
    use super::*;
    use crate::backend::cpu::buffer::f32_to_f16;

    #[test]
    fn validate_rejects_partial_blocks() {
        assert!(validate(CpuFormat::Q4_K, 144).is_ok());
        assert!(validate(CpuFormat::Q4_K, 143).is_err());
        assert!(validate(CpuFormat::Q6_K, 210 * 2).is_ok());
        assert!(validate(CpuFormat::Q6_K, 209).is_err());
        assert!(validate(CpuFormat::Q5_K, 176).is_ok());
        assert!(validate(CpuFormat::Q5_K, 175).is_err());
    }

    #[test]
    fn f16_row_widening() {
        let vals = [1.0f32, -2.0, 0.5, 0.0];
        let mut bytes = Vec::new();
        for &v in &vals {
            bytes.extend_from_slice(&f32_to_f16(v).to_le_bytes());
        }
        let mut out = vec![0f32; 4];
        dequant_row(CpuFormat::F16, &bytes, 0, 4, &mut out);
        assert_eq!(out, vals);
    }

    // FP16-scale rounding tolerance: scales are FP16, so compare in f32 with a
    // small relative tolerance (C-2). The golden inputs below use scales/quants
    // chosen so every expected value is an exact small integer, but the assertion
    // stays tolerant per the spec.
    fn approx_eq(got: f32, want: f32) {
        let tol = 1e-3 * want.abs().max(1.0);
        assert!(
            (got - want).abs() <= tol,
            "golden mismatch: got {got}, want {want}"
        );
    }

    // Q4_K golden: block = d(f16)|dmin(f16)|scales[12]|qs[128]; 8 sub-blocks of 32,
    // w = d*sc_j*q4 - dmin*mn_j. Inputs chosen so each sub-block is a single
    // constant: d=1, dmin=1, qs all 0x21 (low nibble=1 for even sub-blocks, high
    // nibble=2 for odd ones). The scales bytes set bit 6/7 on scales[0] and
    // scales[4] so the j>=4 branch recomposes the 6-bit sc/mn from NON-adjacent
    // bytes (the part of get_scale_min_k4 that is easiest to get wrong).
    #[test]
    fn q4_k_golden_vector() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&f32_to_f16(1.0).to_le_bytes()); // d
        bytes.extend_from_slice(&f32_to_f16(1.0).to_le_bytes()); // dmin
        // scales[12]: low byte holds sc (j<4) / mn (j<4) in bits 0..6; bit 6/7 of
        // scales[0] and scales[4] feed the high nibble of sc_4 / mn_4 (j>=4 branch).
        let scales: [u8; 12] = [0x45, 6, 7, 8, 0x81, 2, 3, 4, 0x39, 0x21, 0x42, 0x53];
        bytes.extend_from_slice(&scales);
        bytes.extend(std::iter::repeat_n(0x21u8, 128)); // qs

        let mut out = vec![0f32; 256];
        dequant_row(CpuFormat::Q4_K, &bytes, 0, 256, &mut out);

        // Independently derived (sc_j, mn_j) and per-sub-block value:
        //   sb0 even q4=1: sc0=5 mn0=1 ->  5-1 =  4
        //   sb1 odd  q4=2: sc1=6 mn1=2 -> 12-2 = 10
        //   sb2 even      : sc2=7 mn2=3 ->  7-3 =  4
        //   sb3 odd       : sc3=8 mn3=4 -> 16-4 = 12
        //   sb4 even      : sc4=25 mn4=35 -> 25-35 = -10  (j>=4 recomposition)
        //   sb5 odd       : sc5=1 mn5=2 ->  2-2 =  0
        //   sb6 even      : sc6=2 mn6=4 ->  2-4 = -2
        //   sb7 odd       : sc7=3 mn7=5 ->  6-5 =  1
        let per_sub = [4.0f32, 10.0, 4.0, 12.0, -10.0, 0.0, -2.0, 1.0];
        for (sb, &want) in per_sub.iter().enumerate() {
            for l in 0..32 {
                approx_eq(out[sb * 32 + l], want);
            }
        }
        // Literal anchors, incl. the j>=4 sub-block 4.
        approx_eq(out[0], 4.0);
        approx_eq(out[128], -10.0);
        approx_eq(out[255], 1.0);
    }

    // Q5_K golden: block = d(f16)|dmin(f16)|scales[12]|qh[32]|qs[128]; 8 sub-blocks
    // of 32, w = d*sc_j*(q4 + 16*high_bit) - dmin*mn_j. Reuses the SAME scales[12]
    // as the Q4_K golden (so the (sc_j,mn_j) below are independently known) and the
    // same qs=0x21 nibbles (low=1 even sub-blocks, high=2 odd). The new part is
    // qh=0x55 (bits 0,2,4,6 set, 1,3,5,7 clear): every EVEN sub-block gets +16, every
    // ODD sub-block gets +0 — so the test pins the per-sub-block 5th-bit selection.
    #[test]
    fn q5_k_golden_vector() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&f32_to_f16(1.0).to_le_bytes()); // d
        bytes.extend_from_slice(&f32_to_f16(1.0).to_le_bytes()); // dmin
        let scales: [u8; 12] = [0x45, 6, 7, 8, 0x81, 2, 3, 4, 0x39, 0x21, 0x42, 0x53];
        bytes.extend_from_slice(&scales);
        bytes.extend(std::iter::repeat_n(0x55u8, 32)); // qh
        bytes.extend(std::iter::repeat_n(0x21u8, 128)); // qs

        let mut out = vec![0f32; 256];
        dequant_row(CpuFormat::Q5_K, &bytes, 0, 256, &mut out);

        // (sc_j,mn_j) as in the Q4_K golden. Even sub-blocks: q=1+16=17. Odd: q=2+0=2.
        //   sb0 even: 5*17 - 1 = 84
        //   sb1 odd : 6*2  - 2 = 10
        //   sb2 even: 7*17 - 3 = 116
        //   sb3 odd : 8*2  - 4 = 12
        //   sb4 even: 25*17 - 35 = 390   (j>=4 scale recomposition)
        //   sb5 odd : 1*2  - 2 = 0
        //   sb6 even: 2*17 - 4 = 30
        //   sb7 odd : 3*2  - 5 = 1
        let per_sub = [84.0f32, 10.0, 116.0, 12.0, 390.0, 0.0, 30.0, 1.0];
        for (sb, &want) in per_sub.iter().enumerate() {
            for l in 0..32 {
                approx_eq(out[sb * 32 + l], want);
            }
        }
        // Literal anchors, incl. the j>=4 sub-block 4 with its +16 high bit.
        approx_eq(out[0], 84.0);
        approx_eq(out[128], 390.0);
        approx_eq(out[255], 1.0);
    }

    // Q6_K golden: block = ql[128]|qh[64]|scales[16](int8)|d(f16); q6 = (ql_nibble |
    // qh_bits<<4) - 32, w = d*scale*q6. Inputs: d=1, ql all 0x0F (low nibble=15,
    // high nibble=0), qh all 0x00. So the two values taken from the LOW nibble are
    // 15-32 = -17 and the two from the HIGH nibble are 0-32 = -32 (this checks both
    // the nibble split and the -32 recentering). scales = 1..16 so each of the 8
    // scale slots per segment is distinct, exercising the scb+is / +2 / +4 / +6
    // indexing.
    #[test]
    fn q6_k_golden_vector() {
        let mut bytes = Vec::new();
        bytes.extend(std::iter::repeat_n(0x0Fu8, 128)); // ql
        bytes.extend(std::iter::repeat_n(0x00u8, 64)); // qh
        for s in 0..16i32 {
            bytes.push((s + 1) as i8 as u8); // scales 1..16
        }
        bytes.extend_from_slice(&f32_to_f16(1.0).to_le_bytes()); // d

        let mut out = vec![0f32; 256];
        dequant_row(CpuFormat::Q6_K, &bytes, 0, 256, &mut out);

        // Reference built from the ggml formula directly (NOT dequant_row): for each
        // of the two 128-wide segments, scb = seg*8; the four 32-wide quarters use
        // scales scb+is, scb+is+2, scb+is+4, scb+is+6 with is=l/16, and q = -17 for
        // the low-nibble quarters (0,1) and -32 for the high-nibble quarters (2,3).
        let q_for_quarter = [-17.0f32, -17.0, -32.0, -32.0];
        for seg in 0..2usize {
            let scb = seg * 8;
            for (quarter, &quant) in q_for_quarter.iter().enumerate() {
                for l in 0..32usize {
                    let is = l / 16;
                    let scale = (scb + is + quarter * 2 + 1) as f32; // scales are 1..16
                    let want = scale * quant;
                    let idx = seg * 128 + quarter * 32 + l;
                    approx_eq(out[idx], want);
                }
            }
        }
        // Literal anchors.
        approx_eq(out[0], -17.0); // seg0, quarter0, is0: scale 1, q -17
        approx_eq(out[16], -34.0); // seg0, quarter0, is1: scale 2, q -17
        approx_eq(out[64], -160.0); // seg0, quarter2, is0: scale 5, q -32
        approx_eq(out[255], -512.0); // seg1, quarter3, is1: scale 16, q -32
    }
}
