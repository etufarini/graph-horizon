/*
 * graph_horizon_engine — SIMD variant of the fused Q4_K CPU kernel
 * Holds the architecture-specific fused dequant+MAC implementation: on x86_64,
 * an AVX2+FMA per-row dot product; other targets retain the scalar kernel.
 *
 * Dispatch is in matmul_q4k::row_dot_q4k: it runtime-detects AVX2+FMA, so a binary
 * built here stays correct on a CPU without them (it falls back to scalar). Other
 * architectures (e.g. aarch64/NEON) use the scalar fallback unchanged: NEON is
 * deliberately NOT hand-written here because it cannot be validated on this host,
 * and an unvalidated SIMD transcription would risk a silent numeric bug; the
 * scalar path keeps those targets correct, just not accelerated.
 *
 * Numerics: the dequant per sub-block (d, dmin, sc, mn) stays scalar and reuses
 * dequant::scale_min (single source of truth for the j>=4 6-bit recomposition);
 * only the 32-wide nibble→f32→FMA inner loop is vectorized. Accumulation is
 * reordered (8 lane-parallel partials, summed at the end), so the result is within
 * the quantized tolerance of the scalar kernel, not bit-identical. This remains a
 * per-row function, so rows stay independent.
*/

// AGENTS deroga K: kernel matmul Q4_K SIMD (AVX2), una sola operazione.

#[cfg(target_arch = "x86_64")]
use super::q4k::f16_at;
#[cfg(target_arch = "x86_64")]
use crate::backend::cpu::dequant::scale_min;

// AVX2+FMA fused dot of activation `a` with Q4_K weight row `row`. Same result as
// matmul_q4k::row_dot_q4k_scalar within tolerance. `in_dim` is a multiple of 256,
// so every 32-value sub-block splits cleanly into four 8-wide AVX2 chunks with no
// remainder. SAFETY: callers reach this (and the sibling AVX2 entries below) only behind
// a runtime AVX2+FMA check; `bytes` is the load-validated weight tensor slice (SEC-INV:
// gguf::loader rejects offset+byte_len overrun and any byte_len incoherent with dims×block
// before a byte reaches here), so `base = row * (in_dim/256) * 144` and its 144-byte block
// read stay within `bytes` for every `row < out_dim`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub(super) unsafe fn row_dot_q4k_avx2(a: &[f32], bytes: &[u8], row: usize, in_dim: usize) -> f32 {
    use core::arch::x86_64::*;
    unsafe {
        let nb = in_dim / 256;
        let base = row * nb * 144;
        let mask = _mm256_set1_epi32(0xF);
        // FOUR independent lane-parallel accumulators, one per 8-wide chunk `c`. At
        // n==1 (decode) there is no other token to overlap, so a single accumulator
        // serialises every `fmadd` into one ~128-deep dependency chain (4-cycle FMA
        // latency ⇒ latency-bound, FMA units idle). Splitting chunk `c` into its own
        // accumulator gives four independent chains that the two FMA ports pipeline,
        // hiding the latency — the decode GEMV win. The four are reduced at the end;
        // this REASSOCIATES the sum vs the single-acc order — still within the quant
        // tolerance the kernel already carries vs the scalar/dequant_row reference.
        // The batched kernel keeps its single-acc-per-token form (it is throughput-
        // bound — extra accumulators would only add L1 traffic), so it is now within
        // tolerance of, not bit-identical to, this per-token path.
        let mut acc = [_mm256_setzero_ps(); 4];
        for s in 0..nb {
            let blk = base + s * 144;
            let d = f16_at(bytes, blk);
            let dmin = f16_at(bytes, blk + 2);
            let sco = blk + 4;
            let qso = blk + 16;
            let abase = s * 256;
            // Process the 8 sub-blocks as 4 GROUPS of two: the even (low-nibble) and
            // odd (high-nibble) sub-block of a group read the SAME 32 qs bytes, so we
            // load each byte ONCE and extract both nibbles — halving the qs byte loads
            // and cvtepu8 ops (the int-unpack pressure that co-limits decode with
            // memory). `acc[c]` accumulates chunk-`c` of sub-block 2g then 2g+1, the
            // identical order to the per-sub-block loop → BIT-IDENTICAL result.
            for g in 0..4 {
                let (sc0, mn0) = scale_min(bytes, sco, 2 * g);
                let (sc1, mn1) = scale_min(bytes, sco, 2 * g + 1);
                let dl0 = _mm256_set1_ps(d * sc0 as f32);
                let ml0 = _mm256_set1_ps(dmin * mn0 as f32);
                let dl1 = _mm256_set1_ps(d * sc1 as f32);
                let ml1 = _mm256_set1_ps(dmin * mn1 as f32);
                let group = g * 32;
                let in_lo = abase + 2 * g * 32;
                let in_hi = abase + (2 * g + 1) * 32;
                for (c, sum) in acc.iter_mut().enumerate() {
                    let qp = qso + group + c * 8;
                    // 8 qs bytes → 8 zero-extended i32, loaded ONCE for both nibbles.
                    let v = _mm256_cvtepu8_epi32(_mm_loadl_epi64(
                        bytes.as_ptr().add(qp) as *const __m128i
                    ));
                    let nlo = _mm256_and_si256(v, mask);
                    let nhi = _mm256_and_si256(_mm256_srli_epi32(v, 4), mask);
                    let wlo = _mm256_fmsub_ps(dl0, _mm256_cvtepi32_ps(nlo), ml0);
                    let whi = _mm256_fmsub_ps(dl1, _mm256_cvtepi32_ps(nhi), ml1);
                    // Even sub-block (low) then odd (high), preserving order.
                    *sum =
                        _mm256_fmadd_ps(_mm256_loadu_ps(a.as_ptr().add(in_lo + c * 8)), wlo, *sum);
                    *sum =
                        _mm256_fmadd_ps(_mm256_loadu_ps(a.as_ptr().add(in_hi + c * 8)), whi, *sum);
                }
            }
        }
        // Reduce the four chunk accumulators (tree order: (0+1)+(2+3)).
        let lo = _mm256_add_ps(acc[0], acc[1]);
        let hi = _mm256_add_ps(acc[2], acc[3]);
        hsum256(_mm256_add_ps(lo, hi))
    }
}

// Batched AVX2+FMA Q4_K row dot: decode each 32-value sub-block ONCE (the four
// 8-wide weight vectors `w[c] = dl*q - ml`) and FMA it into EVERY token's
// lane-parallel accumulator, instead of re-decoding the whole row per token (the
// prefill amortization — the dequant cost is paid once for the batch). `a` is
// token-major `[n][in_dim]`; `out[0..n]` receives one dot per token. The per-token
// accumulation order is sub-block by sub-block, chunk 0..4 into one 8-lane
// accumulator, matching the two-row batched kernel exactly. The decode kernel
// instead uses four accumulators and has a bounded numeric parity gate. `acc` is caller-owned
// scratch of `n*8` f32 (the 8 lane partials per token); it lives in L1/L2 because
// `n` accumulators cannot stay in registers. SAFETY: reached only behind a runtime
// AVX2+FMA check; every load/store stays within `a`/`out`/`acc` (all sized to `n`).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub(super) unsafe fn row_dot_q4k_avx2_batched(
    a: &[f32],
    bytes: &[u8],
    row: usize,
    in_dim: usize,
    out: &mut [f32],
    acc: &mut [f32],
) {
    use core::arch::x86_64::*;
    unsafe {
        let n = out.len();
        let nb = in_dim / 256;
        let base = row * nb * 144;
        let mask = _mm256_set1_epi32(0xF);
        // Reset this row's per-token partials (acc is reused across rows by the caller).
        for v in acc[..n * 8].iter_mut() {
            *v = 0.0;
        }
        for s in 0..nb {
            let blk = base + s * 144;
            let d = f16_at(bytes, blk);
            let dmin = f16_at(bytes, blk + 2);
            let sco = blk + 4;
            let qso = blk + 16;
            let abase = s * 256;
            for sb in 0..8 {
                let (sc, mn) = scale_min(bytes, sco, sb);
                let dl = _mm256_set1_ps(d * sc as f32);
                let ml = _mm256_set1_ps(dmin * mn as f32);
                let group = (sb / 2) * 32;
                let hi = sb & 1 == 1;
                let in0 = abase + sb * 32;
                // Decode the four 8-wide weight vectors for this sub-block ONCE.
                let mut w = [_mm256_setzero_ps(); 4];
                for (c, wc) in w.iter_mut().enumerate() {
                    let qp = qso + group + c * 8;
                    let raw = _mm_loadl_epi64(bytes.as_ptr().add(qp) as *const __m128i);
                    let v = _mm256_cvtepu8_epi32(raw);
                    let nib = if hi {
                        _mm256_and_si256(_mm256_srli_epi32(v, 4), mask)
                    } else {
                        _mm256_and_si256(v, mask)
                    };
                    let qf = _mm256_cvtepi32_ps(nib);
                    *wc = _mm256_fmsub_ps(dl, qf, ml);
                }
                // FMA the decoded weights into every token's 8-lane accumulator,
                // chunks 0..4 folded in order (matching the per-token kernel).
                for i in 0..n {
                    let ap = acc.as_mut_ptr().add(i * 8);
                    let mut ai = _mm256_loadu_ps(ap);
                    let arow = a.as_ptr().add(i * in_dim + in0);
                    ai = _mm256_fmadd_ps(_mm256_loadu_ps(arow), w[0], ai);
                    ai = _mm256_fmadd_ps(_mm256_loadu_ps(arow.add(8)), w[1], ai);
                    ai = _mm256_fmadd_ps(_mm256_loadu_ps(arow.add(16)), w[2], ai);
                    ai = _mm256_fmadd_ps(_mm256_loadu_ps(arow.add(24)), w[3], ai);
                    _mm256_storeu_ps(ap, ai);
                }
            }
        }
        for (i, o) in out.iter_mut().enumerate() {
            *o = hsum256(_mm256_loadu_ps(acc.as_ptr().add(i * 8)));
        }
    }
}

// Two-output-row batched AVX2+FMA Q4_K dot: the register-blocked counterpart of
// `row_dot_q4k_avx2_batched`. The single-row kernel re-loads every token's four
// 8-wide activation vectors once per output row; here two adjacent rows (`row0`,
// `row0+1`) are processed together so each activation load is REUSED across both
// rows (one load → two FMAs). The eight decoded weight vectors (w0[0..4] for row0,
// w1[0..4] for row1) stay live in registers across the whole token loop, so the
// inner loop shifts from load-bound (5 reads / 4 FMA per token) to FMA-bound
// (4 reads / 8 FMA) — the prefill win on the FFN/projection GEMMs. The per-token
// accumulation order is byte-for-byte the single-row kernel's (sub-block by
// sub-block, chunk 0..4 into one 8-lane accumulator), so `out0[i]`/`out1[i]` are
// BIT-IDENTICAL to two separate `row_dot_q4k_avx2_batched` calls. `acc` is
// caller-owned scratch of `2*n*8` f32: row0's per-token partials in `[0..n*8]`,
// row1's in `[n*8..2*n*8]`. SAFETY: reached only behind a runtime AVX2+FMA check;
// Activations are [in_dim/32][batch][32], with the slice starting at the
// current token tile's offset. Each tile has n <= batch token windows. Every
// load/store stays within the validated packed activation/output/scratch slices, and
// `row0+1` is a valid row (the caller pairs rows only when two remain).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn row2_dot_q4k_avx2_batched(
    a: &[f32],
    bytes: &[u8],
    row0: usize,
    in_dim: usize,
    out0: &mut [f32],
    out1: &mut [f32],
    acc: &mut [f32],
    batch: usize,
) {
    use core::arch::x86_64::*;
    unsafe {
        let n = out0.len();
        let nb = in_dim / 256;
        let base0 = row0 * nb * 144;
        let base1 = (row0 + 1) * nb * 144;
        let mask = _mm256_set1_epi32(0xF);
        // Reset both rows' per-token partials (acc is reused across row pairs).
        for v in acc[..2 * n * 8].iter_mut() {
            *v = 0.0;
        }
        for s in 0..nb {
            let blk0 = base0 + s * 144;
            let blk1 = base1 + s * 144;
            let d0 = f16_at(bytes, blk0);
            let dmin0 = f16_at(bytes, blk0 + 2);
            let d1 = f16_at(bytes, blk1);
            let dmin1 = f16_at(bytes, blk1 + 2);
            let (sco0, qso0) = (blk0 + 4, blk0 + 16);
            let (sco1, qso1) = (blk1 + 4, blk1 + 16);
            let abase = s * 256;
            for sb in 0..8 {
                let (sc0, mn0) = scale_min(bytes, sco0, sb);
                let (sc1, mn1) = scale_min(bytes, sco1, sb);
                let dl0 = _mm256_set1_ps(d0 * sc0 as f32);
                let ml0 = _mm256_set1_ps(dmin0 * mn0 as f32);
                let dl1 = _mm256_set1_ps(d1 * sc1 as f32);
                let ml1 = _mm256_set1_ps(dmin1 * mn1 as f32);
                let group = (sb / 2) * 32;
                let hi = sb & 1 == 1;
                let in0 = abase + sb * 32;
                // Decode both rows' four 8-wide weight vectors ONCE; they stay in
                // registers across every token below.
                let mut w0 = [_mm256_setzero_ps(); 4];
                let mut w1 = [_mm256_setzero_ps(); 4];
                for c in 0..4 {
                    let raw0 =
                        _mm_loadl_epi64(bytes.as_ptr().add(qso0 + group + c * 8) as *const __m128i);
                    let raw1 =
                        _mm_loadl_epi64(bytes.as_ptr().add(qso1 + group + c * 8) as *const __m128i);
                    let v0 = _mm256_cvtepu8_epi32(raw0);
                    let v1 = _mm256_cvtepu8_epi32(raw1);
                    let (n0, n1) = if hi {
                        (
                            _mm256_and_si256(_mm256_srli_epi32(v0, 4), mask),
                            _mm256_and_si256(_mm256_srli_epi32(v1, 4), mask),
                        )
                    } else {
                        (_mm256_and_si256(v0, mask), _mm256_and_si256(v1, mask))
                    };
                    w0[c] = _mm256_fmsub_ps(dl0, _mm256_cvtepi32_ps(n0), ml0);
                    w1[c] = _mm256_fmsub_ps(dl1, _mm256_cvtepi32_ps(n1), ml1);
                }
                // One activation load per chunk, reused by both rows' FMAs.
                for i in 0..n {
                    let arow = a.as_ptr().add(in0 * batch + i * 32);
                    let p0 = acc.as_mut_ptr().add(i * 8);
                    let p1 = acc.as_mut_ptr().add((n + i) * 8);
                    let mut a0 = _mm256_loadu_ps(p0);
                    let mut a1 = _mm256_loadu_ps(p1);
                    let av0 = _mm256_loadu_ps(arow);
                    let av1 = _mm256_loadu_ps(arow.add(8));
                    let av2 = _mm256_loadu_ps(arow.add(16));
                    let av3 = _mm256_loadu_ps(arow.add(24));
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
        }
        for i in 0..n {
            out0[i] = hsum256(_mm256_loadu_ps(acc.as_ptr().add(i * 8)));
            out1[i] = hsum256(_mm256_loadu_ps(acc.as_ptr().add((n + i) * 8)));
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
    use super::super::q4k::row_dot_q4k_scalar;
    use super::*;
    use crate::backend::cpu::buffer::{CpuBuffer, CpuFormat, f32_to_f16};

    // Same synthetic, valid Q4_K weight as the matmul_q4k tests: a deterministic
    // byte pattern with finite FP16 d/dmin.
    fn q4k_bytes(in_dim: usize, out_dim: usize) -> Vec<u8> {
        let nb = in_dim / 256;
        let mut bytes = vec![0u8; out_dim * nb * 144];
        for (i, b) in bytes.iter_mut().enumerate() {
            *b = ((i * 31 + 7) % 251) as u8;
        }
        for blk in 0..out_dim * nb {
            let base = blk * 144;
            bytes[base..base + 2].copy_from_slice(&f32_to_f16(0.05).to_le_bytes());
            bytes[base + 2..base + 4].copy_from_slice(&f32_to_f16(0.01).to_le_bytes());
        }
        bytes
    }

    // The AVX2 variant must match the scalar reference within the quantized
    // tolerance (rel. 8e-2). Spans two 256-blocks (j>=4 sub-block + multi-block
    // accumulation) over several rows. Skipped if the host lacks AVX2+FMA.
    #[test]
    fn avx2_matches_scalar_within_tolerance() {
        if !(is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")) {
            return;
        }
        let in_dim = 512;
        let out_dim = 9;
        let a: Vec<f32> = (0..in_dim)
            .map(|i| (i as f32 * 0.017).sin() * 0.7)
            .collect();
        let bytes = q4k_bytes(in_dim, out_dim);
        // Activations must round-trip through FP16 like the real path so the inputs
        // are identical for both kernels.
        let abuf = CpuBuffer::zeroed(in_dim * 2, CpuFormat::F16);
        abuf.write_f16_from_f32(&a);
        let a = abuf.read_f16_as_f32();

        for row in 0..out_dim {
            let scalar = row_dot_q4k_scalar(&a, &bytes, row, in_dim);
            // SAFETY: guarded by the feature check at the top of the test.
            let simd = unsafe { row_dot_q4k_avx2(&a, &bytes, row, in_dim) };
            let tol = 8e-2 * scalar.abs().max(1e-3);
            assert!(
                (simd - scalar).abs() <= tol,
                "row {row}: simd {simd} vs scalar {scalar} (tol {tol})"
            );
        }
    }
}
