/*
 * gh_zero_engine — fused Q6_K CPU matmul kernel
 * Fused dequant+MAC for CpuFormat::Q6_K, the sibling of matmul_q4k. The generic
 * path (kernels::matmul) decodes a whole Q6_K output row into an `in_dim` f32 Vec
 * (dequant::dequant_row_q6_k) and only then dots it; this kernel never materializes
 * that row — it unpacks one 8-quant chunk at a time and accumulates straight away.
 * Why Q6_K specifically: in a Q4_K_M model the per-layer `ffn_down` and the `output`
 * lm_head are Q6_K (the lm_head is the single biggest decode GEMV), and they were the
 * only heavy weights still going through the materialize-then-dot generic path.
 *
 * Numerics: the per-row accumulation is reordered (intra-row, into the four quant
 * streams / per-token lane accumulators), so the result is within the quantized
 * tolerance of dequant_row, NOT bit-identical — exactly the Q4_K fused contract. The
 * integer unpack is the byte-for-byte transcription of dequant_row_q6_k. Rows stay
 * independent and are split across cores by `parallel::for_units` (stride 1 for the
 * GEMV paths, stride `n` for the batched path), like the Q4_K kernel. Block validity
 * is guaranteed once at load, so this kernel cannot fail; `in_dim` is a multiple of
 * 256 for any validated Q6_K weight.
*/

// AGENTS deroga K: kernel matmul Q6_K denso, una sola operazione.

use super::q4k::{f16_at, token_tile};
use crate::backend::cpu::buffer::CpuBuffer;
use crate::backend::cpu::parallel;

// Scalar fused single-token dot of activation `a` with Q6_K weight row `row`,
// accumulated in unpack order (the four quant streams interleaved). Mirror of
// dequant_row_q6_k_scalar's block decode, consuming each value immediately. Portable
// fallback and the parity reference for the SIMD variant.
pub(super) fn row_dot_q6k_scalar(a: &[f32], bytes: &[u8], row: usize, in_dim: usize) -> f32 {
    let s8 = |b: usize| bytes[b] as i8 as f32;
    let nsb = in_dim / 256;
    let base = row * nsb * 210;
    let mut acc = 0f32;
    for s in 0..nsb {
        let blk = base + s * 210;
        let (qlo, qho, sco) = (blk, blk + 128, blk + 192);
        let d = f16_at(bytes, blk + 208);
        let abase = s * 256;
        let mut n = 0usize;
        while n < 256 {
            let seg = n / 128;
            let (qlb, qhb, scb) = (qlo + seg * 64, qho + seg * 32, sco + seg * 8);
            for l in 0..32 {
                let is = l / 16;
                let lo0 = bytes[qlb + l] as u32;
                let lo1 = bytes[qlb + l + 32] as u32;
                let h = bytes[qhb + l] as u32;
                let q1 = ((lo0 & 0xF) | ((h & 3) << 4)) as i32 - 32;
                let q2 = ((lo1 & 0xF) | (((h >> 2) & 3) << 4)) as i32 - 32;
                let q3 = ((lo0 >> 4) | (((h >> 4) & 3) << 4)) as i32 - 32;
                let q4 = ((lo1 >> 4) | (((h >> 6) & 3) << 4)) as i32 - 32;
                let o = abase + n + l;
                acc += d * s8(scb + is) * q1 as f32 * a[o];
                acc += d * s8(scb + is + 2) * q2 as f32 * a[o + 32];
                acc += d * s8(scb + is + 4) * q3 as f32 * a[o + 64];
                acc += d * s8(scb + is + 6) * q4 as f32 * a[o + 96];
            }
            n += 128;
        }
    }
    acc
}

// Per-row dot dispatcher (decode GEMV / lm_head): AVX2+FMA when available, else scalar.
#[inline]
pub(crate) fn row_dot_q6k(a: &[f32], bytes: &[u8], row: usize, in_dim: usize) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: guarded by the runtime AVX2+FMA detection just above.
            return unsafe { super::q6k_simd::row_dot_q6k_avx2(a, bytes, row, in_dim) };
        }
    }
    row_dot_q6k_scalar(a, bytes, row, in_dim)
}

// Two-output-row batched dispatcher: AVX2 register-blocked kernel (reuses each
// activation load across both rows), else two scalar single-row batched passes.
#[inline]
fn row2_dot_q6k_batched(
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
                super::q6k_simd::row2_dot_q6k_avx2_batched(a, bytes, row0, in_dim, out0, out1, acc)
            };
        }
    }
    let _ = acc;
    for (i, o) in out0.iter_mut().enumerate() {
        *o = row_dot_q6k_scalar(&a[i * in_dim..], bytes, row0, in_dim);
    }
    for (i, o) in out1.iter_mut().enumerate() {
        *o = row_dot_q6k_scalar(&a[i * in_dim..], bytes, row0 + 1, in_dim);
    }
}

// Single-row batched dispatcher (for the odd trailing row).
#[inline]
fn row_dot_q6k_batched(
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
                super::q6k_simd::row_dot_q6k_avx2_batched(a, bytes, row, in_dim, out, acc)
            };
        }
    }
    let _ = acc;
    for (i, o) in out.iter_mut().enumerate() {
        *o = row_dot_q6k_scalar(&a[i * in_dim..], bytes, row, in_dim);
    }
}

// y = W·a for a Q6_K weight, stored FP16. Same contract as `kernels::matmul`.
pub(crate) fn matmul(out: &CpuBuffer, a: &CpuBuffer, w: &CpuBuffer, in_dim: usize, out_dim: usize) {
    let a = a.read_f16_as_f32();
    let w_bytes = w.bytes();
    let w_bytes: &[u8] = &w_bytes[w.window()];
    let mut y = vec![0f32; out_dim];
    parallel::for_units(&mut y, 1, |o0, chunk| {
        for (k, dst) in chunk.iter_mut().enumerate() {
            *dst = row_dot_q6k(&a, w_bytes, o0 + k, in_dim);
        }
    });
    out.write_f16_from_f32(&y);
}

// Batched Q6_K matmul: y[n][out_dim] = A[n][in_dim] · Wᵀ, token-major. Output rows
// are decoded and dotted in PAIRS (two-row register blocking, reusing each
// activation load), token-tiled to keep the working set cache-resident — the same
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
    let mut yt = vec![0f32; out_dim * n]; // output-row-major: row o at o*n
    parallel::for_units(&mut yt, n, |o0, chunk| {
        // Per-worker scratch for two token tiles (both rows' lane partials), L1-resident.
        let mut acc = vec![0f32; 2 * tile.min(n) * 8];
        let rows = chunk.len() / n;
        let mut t0 = 0;
        while t0 < n {
            let tb = tile.min(n - t0);
            let mut k = 0;
            while k + 1 < rows {
                let (lo, hi) = chunk.split_at_mut((k + 1) * n);
                row2_dot_q6k_batched(
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
                row_dot_q6k_batched(
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
            *dst = row_dot_q6k(&x, w_bytes, o0 + k, in_dim);
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

    // Synthetic, valid Q6_K weight: `out_dim` rows of `in_dim/256` 210-byte blocks,
    // filled with a deterministic byte pattern and finite FP16 `d`. dequant_row and
    // the fused kernel read the bytes identically, so the parity check is meaningful.
    fn q6k_weight(in_dim: usize, out_dim: usize) -> CpuBuffer {
        let nsb = in_dim / 256;
        let mut bytes = vec![0u8; out_dim * nsb * 210];
        for (i, b) in bytes.iter_mut().enumerate() {
            *b = ((i * 37 + 11) % 251) as u8;
        }
        for blk in 0..out_dim * nsb {
            let base = blk * 210;
            bytes[base + 208..base + 210].copy_from_slice(&f32_to_f16(0.03).to_le_bytes());
        }
        CpuBuffer::from_bytes(bytes, CpuFormat::Q6_K)
    }

    fn reference_row(a: &[f32], w: &CpuBuffer, row: usize, in_dim: usize) -> f32 {
        let wb = w.bytes();
        let mut r = vec![0f32; in_dim];
        dequant::dequant_row(CpuFormat::Q6_K, &wb, row, in_dim, &mut r);
        (0..in_dim).map(|i| a[i] * r[i]).sum()
    }

    // Fused matmul must match the generic dequant_row path within the quantized
    // tolerance (rel. 8e-2). in_dim spans two 256-blocks; out_dim > a core count.
    #[test]
    fn fused_matmul_matches_dequant_row_within_tolerance() {
        let (in_dim, out_dim) = (512usize, 17usize);
        let a_vals: Vec<f32> = (0..in_dim).map(|i| (i as f32 * 0.013).sin()).collect();
        let a = f16_buf(&a_vals);
        let aw = a.read_f16_as_f32();
        let w = q6k_weight(in_dim, out_dim);

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

    // Batched Q6_K matmul must match, per token, the single-token `matmul` within the
    // quant tolerance (the batched lane order differs from the 4-acc decode order).
    // out_dim odd so both the row-pair path and the trailing single row run.
    #[test]
    fn batched_matches_per_token() {
        let (in_dim, out_dim, n) = (512usize, 17usize, 5usize);
        let w = q6k_weight(in_dim, out_dim);
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

    // logits path: same parity, FP32 output.
    #[test]
    fn fused_logits_matches_dequant_row_within_tolerance() {
        let (in_dim, out_dim) = (256usize, 13usize);
        let x_vals: Vec<f32> = (0..in_dim)
            .map(|i| (i as f32 * 0.021).cos() * 0.5)
            .collect();
        let x = f16_buf(&x_vals);
        let xw = x.read_f16_as_f32();
        let w = q6k_weight(in_dim, out_dim);

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
