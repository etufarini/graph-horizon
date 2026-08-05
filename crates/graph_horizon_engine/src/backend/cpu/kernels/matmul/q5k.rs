/*
 * graph_horizon_engine — fused Q5_K CPU matmul kernel
 * Fused dequant+MAC for CpuFormat::Q5_K, the last quantized format moved off the
 * generic materialize-then-dot path (the sibling of matmul_q4k/q6k). Q5_K is the
 * Q4_K super-block (d, dmin, scales[12], qs[128]) plus a 5th bit per quant from
 * qh[32]: sub-block pair `g` (0..3) uses qh bit 2g for the even sub-block (low nibble)
 * and 2g+1 for the odd (high nibble). This kernel never materializes the f32 row — it
 * unpacks one sub-block-pair chunk at a time and accumulates straight away.
 *
 * Numerics: per-row accumulation is reordered (intra-row), so the result is within the
 * quantized tolerance of dequant_row, NOT bit-identical — the fused-kernel contract.
 * Rows stay independent, split by `parallel::for_units` (stride 1 GEMV / `n` batched).
 * Block validity is guaranteed once at load; `in_dim` is a multiple of 256.
*/

// AGENTS deroga K: kernel matmul Q5_K denso, una sola operazione.

use super::q4k::{f16_at, token_tile};
use crate::backend::cpu::buffer::CpuBuffer;
use crate::backend::cpu::dequant::scale_min;
use crate::backend::cpu::parallel;

// Scalar fused single-token dot of activation `a` with Q5_K weight row `row`,
// accumulated in unpack order. Mirror of dequant_row_q5_k_scalar's decode, consuming
// each value immediately. Portable fallback and the parity reference for the SIMD path.
pub(super) fn row_dot_q5k_scalar(a: &[f32], bytes: &[u8], row: usize, in_dim: usize) -> f32 {
    let nsb = in_dim / 256;
    let base = row * nsb * 176;
    let mut acc = 0f32;
    for s in 0..nsb {
        let blk = base + s * 176;
        let d = f16_at(bytes, blk);
        let dmin = f16_at(bytes, blk + 2);
        let (sco, qho, qso) = (blk + 4, blk + 16, blk + 48);
        let abase = s * 256;
        for g in 0..4 {
            let (sc0, mn0) = scale_min(bytes, sco, 2 * g);
            let (sc1, mn1) = scale_min(bytes, sco, 2 * g + 1);
            let d0 = d * sc0 as f32;
            let m0 = dmin * mn0 as f32;
            let d1 = d * sc1 as f32;
            let m1 = dmin * mn1 as f32;
            let (u0, u1) = (1u32 << (2 * g), 1u32 << (2 * g + 1));
            let qb = g * 32;
            let in_lo = abase + g * 64;
            let in_hi = abase + g * 64 + 32;
            for l in 0..32 {
                let qv = bytes[qso + qb + l] as u32;
                let qh = bytes[qho + l] as u32;
                let lo = (qv & 0xF) + if qh & u0 != 0 { 16 } else { 0 };
                let hi = (qv >> 4) + if qh & u1 != 0 { 16 } else { 0 };
                acc += (d0 * lo as f32 - m0) * a[in_lo + l];
                acc += (d1 * hi as f32 - m1) * a[in_hi + l];
            }
        }
    }
    acc
}

// Per-row dot dispatcher (decode GEMV / lm_head): AVX2+FMA when available, else scalar.
#[inline]
pub(crate) fn row_dot_q5k(a: &[f32], bytes: &[u8], row: usize, in_dim: usize) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: guarded by the runtime AVX2+FMA detection just above.
            return unsafe { super::q5k_simd::row_dot_q5k_avx2(a, bytes, row, in_dim) };
        }
    }
    row_dot_q5k_scalar(a, bytes, row, in_dim)
}

// Two-output-row batched dispatcher: AVX2 register-blocked kernel, else two scalar passes.
#[inline]
fn row2_dot_q5k_batched(
    a: &[f32],
    bytes: &[u8],
    row0: usize,
    in_dim: usize,
    out0: &mut [f32],
    out1: &mut [f32],
    acc: &mut [f32],
) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: reached only after the runtime AVX2+FMA check just above.
            return unsafe {
                super::q5k_simd::row2_dot_q5k_avx2_batched(a, bytes, row0, in_dim, out0, out1, acc)
            };
        }
    }
    let _ = acc;
    for (i, o) in out0.iter_mut().enumerate() {
        *o = row_dot_q5k_scalar(&a[i * in_dim..], bytes, row0, in_dim);
    }
    for (i, o) in out1.iter_mut().enumerate() {
        *o = row_dot_q5k_scalar(&a[i * in_dim..], bytes, row0 + 1, in_dim);
    }
}

// Single-row batched dispatcher (the odd trailing row).
#[inline]
fn row_dot_q5k_batched(
    a: &[f32],
    bytes: &[u8],
    row: usize,
    in_dim: usize,
    out: &mut [f32],
    acc: &mut [f32],
) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: reached only after the runtime AVX2+FMA check just above.
            return unsafe {
                super::q5k_simd::row_dot_q5k_avx2_batched(a, bytes, row, in_dim, out, acc)
            };
        }
    }
    let _ = acc;
    for (i, o) in out.iter_mut().enumerate() {
        *o = row_dot_q5k_scalar(&a[i * in_dim..], bytes, row, in_dim);
    }
}

// y = W·a for a Q5_K weight, stored FP16. Same contract as `kernels::matmul`.
pub(crate) fn matmul(out: &CpuBuffer, a: &CpuBuffer, w: &CpuBuffer, in_dim: usize, out_dim: usize) {
    let a = a.read_f16_as_f32();
    let w_bytes = w.bytes();
    let w_bytes: &[u8] = &w_bytes[w.window()];
    let mut y = vec![0f32; out_dim];
    parallel::for_units(&mut y, 1, |o0, chunk| {
        for (k, dst) in chunk.iter_mut().enumerate() {
            *dst = row_dot_q5k(&a, w_bytes, o0 + k, in_dim);
        }
    });
    out.write_f16_from_f32(&y);
}

// Batched Q5_K matmul: y[n][out_dim] = A[n][in_dim] · Wᵀ, token-major. Output rows
// decoded and dotted in PAIRS (two-row register blocking), token-tiled — same
// structure and contract as `matmul_q4k::matmul_batched`.
pub(crate) fn matmul_batched(
    out: &CpuBuffer,
    a: &CpuBuffer,
    w: &CpuBuffer,
    in_dim: usize,
    out_dim: usize,
    n: usize,
) {
    let a = a.read_f16_as_f32();
    let w_bytes = w.bytes();
    let w_bytes: &[u8] = &w_bytes[w.window()];
    let tile = token_tile(in_dim);
    let mut yt = vec![0f32; out_dim * n];
    parallel::for_units(&mut yt, n, |o0, chunk| {
        let mut acc = vec![0f32; 2 * tile.min(n) * 8];
        let rows = chunk.len() / n;
        let mut t0 = 0;
        while t0 < n {
            let tb = tile.min(n - t0);
            let mut k = 0;
            while k + 1 < rows {
                let (lo, hi) = chunk.split_at_mut((k + 1) * n);
                row2_dot_q5k_batched(
                    &a[t0 * in_dim..],
                    w_bytes,
                    o0 + k,
                    in_dim,
                    &mut lo[k * n + t0..k * n + t0 + tb],
                    &mut hi[t0..t0 + tb],
                    &mut acc[..2 * tb * 8],
                );
                k += 2;
            }
            if k < rows {
                row_dot_q5k_batched(
                    &a[t0 * in_dim..],
                    w_bytes,
                    o0 + k,
                    in_dim,
                    &mut chunk[k * n + t0..k * n + t0 + tb],
                    &mut acc[..tb * 8],
                );
            }
            t0 += tb;
        }
    });
    super::write_transposed_f16(out, &yt, out_dim, n);
}

// Same as `matmul` but FP32 vocab logits (no FP16 narrowing) — the decode lm_head.
pub(crate) fn logits(out: &CpuBuffer, x: &CpuBuffer, w: &CpuBuffer, in_dim: usize, out_dim: usize) {
    let x = x.read_f16_as_f32();
    let w_bytes = w.bytes();
    let w_bytes: &[u8] = &w_bytes[w.window()];
    let mut y = vec![0f32; out_dim];
    parallel::for_units(&mut y, 1, |o0, chunk| {
        for (k, dst) in chunk.iter_mut().enumerate() {
            *dst = row_dot_q5k(&x, w_bytes, o0 + k, in_dim);
        }
    });
    out.write_f32(&y);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::cpu::buffer::{CpuBuffer, CpuFormat, f32_to_f16};
    use crate::backend::cpu::dequant;

    fn f16_buf(values: &[f32]) -> CpuBuffer {
        let buf = CpuBuffer::zeroed(values.len() * 2, CpuFormat::F16);
        buf.write_f16_from_f32(values);
        buf
    }

    // Synthetic, valid Q5_K weight: `out_dim` rows of `in_dim/256` 176-byte blocks,
    // deterministic bytes and a finite FP16 d/dmin.
    fn q5k_weight(in_dim: usize, out_dim: usize) -> CpuBuffer {
        let nsb = in_dim / 256;
        let mut bytes = vec![0u8; out_dim * nsb * 176];
        for (i, b) in bytes.iter_mut().enumerate() {
            *b = ((i * 41 + 13) % 251) as u8;
        }
        for blk in 0..out_dim * nsb {
            let base = blk * 176;
            bytes[base..base + 2].copy_from_slice(&f32_to_f16(0.04).to_le_bytes());
            bytes[base + 2..base + 4].copy_from_slice(&f32_to_f16(0.012).to_le_bytes());
        }
        CpuBuffer::from_bytes(bytes, CpuFormat::Q5_K)
    }

    fn reference_row(a: &[f32], w: &CpuBuffer, row: usize, in_dim: usize) -> f32 {
        let wb = w.bytes();
        let mut r = vec![0f32; in_dim];
        dequant::dequant_row(CpuFormat::Q5_K, &wb, row, in_dim, &mut r);
        (0..in_dim).map(|i| a[i] * r[i]).sum()
    }

    #[test]
    fn fused_matmul_matches_dequant_row_within_tolerance() {
        let (in_dim, out_dim) = (512usize, 17usize);
        let a_vals: Vec<f32> = (0..in_dim).map(|i| (i as f32 * 0.013).sin()).collect();
        let a = f16_buf(&a_vals);
        let aw = a.read_f16_as_f32();
        let w = q5k_weight(in_dim, out_dim);
        let out = CpuBuffer::zeroed(out_dim * 2, CpuFormat::F16);
        matmul(&out, &a, &w, in_dim, out_dim);
        let got = out.read_f16_as_f32();
        for (o, &value) in got.iter().enumerate().take(out_dim) {
            let want = reference_row(&aw, &w, o, in_dim);
            let tol = 8e-2 * want.abs().max(1e-3);
            assert!(
                (value - want).abs() <= tol,
                "row {o}: fused {} vs ref {} (tol {tol})",
                value,
                want
            );
        }
    }

    // out_dim odd so both the row-pair path and the trailing single row run.
    #[test]
    fn batched_matches_per_token() {
        let (in_dim, out_dim, n) = (512usize, 17usize, 5usize);
        let w = q5k_weight(in_dim, out_dim);
        let a_vals: Vec<f32> = (0..n * in_dim).map(|k| (k as f32 * 0.013).sin()).collect();
        let a = f16_buf(&a_vals);
        let batched = CpuBuffer::zeroed(n * out_dim * 2, CpuFormat::F16);
        matmul_batched(&batched, &a, &w, in_dim, out_dim, n);
        let got = batched.read_f16_as_f32();
        for i in 0..n {
            let ai = f16_buf(&a_vals[i * in_dim..(i + 1) * in_dim]);
            let oi = CpuBuffer::zeroed(out_dim * 2, CpuFormat::F16);
            matmul(&oi, &ai, &w, in_dim, out_dim);
            let want = oi.read_f16_as_f32();
            for o in 0..out_dim {
                let tol = 8e-2 * want[o].abs().max(1e-3);
                assert!(
                    (got[i * out_dim + o] - want[o]).abs() <= tol,
                    "token {i} row {o}: batched {} vs per-token {} (tol {tol})",
                    got[i * out_dim + o],
                    want[o]
                );
            }
        }
    }

    #[test]
    fn fused_logits_matches_dequant_row_within_tolerance() {
        let (in_dim, out_dim) = (256usize, 13usize);
        let x_vals: Vec<f32> = (0..in_dim)
            .map(|i| (i as f32 * 0.021).cos() * 0.5)
            .collect();
        let x = f16_buf(&x_vals);
        let xw = x.read_f16_as_f32();
        let w = q5k_weight(in_dim, out_dim);
        let out = CpuBuffer::zeroed(out_dim * 4, CpuFormat::F32);
        logits(&out, &x, &w, in_dim, out_dim);
        let got = out.read_f32();
        for (o, &value) in got.iter().enumerate().take(out_dim) {
            let want = reference_row(&xw, &w, o, in_dim);
            let tol = 8e-2 * want.abs().max(1e-3);
            assert!(
                (value - want).abs() <= tol,
                "logit {o}: {} vs {}",
                value,
                want
            );
        }
    }
}
