/*
 * graph_horizon_engine — fused Q4_K CPU matmul kernel
 * Fused dequant+MAC variant of `kernels::matmul`, specialized to CpuFormat::Q4_K.
 * Why it exists: the generic path dequantizes a whole output row into an f32 Vec
 * of length `in_dim` (via dequant::dequant_row) and only then takes the dot
 * product. This kernel never materializes that row: it walks the
 * 256-value blocks, dequantizes one 32-value sub-block into a stack buffer, and
 * accumulates `Σ a[i]·w[i]` straight away, advancing along `in_dim`.
 *
 * Numerics: accumulation is reordered INTRA-row — each 32-value sub-block sums
 * into a local accumulator, added to the row total — so the result is NOT
 * bit-identical to the generic path, but stays within the quantized tolerance
 * used by `validate` (rel. 8e-2). Rows stay independent
 * and are distributed by `parallel::for_units` with stride 1, exactly like
 * `matmul`, so each row's value is independent of the worker count (no cross-row
 * reordering). Block validity is guaranteed once at load (dequant::validate), so
 * this kernel cannot fail; `in_dim` is a multiple of 256 for any validated Q4_K
 * weight (documented assumption, as in the generic path).
*/

// AGENTS deroga K: kernel matmul Q4_K denso (fused dequant+MAC), una sola operazione.

use crate::backend::cpu::buffer::{CpuBuffer, f16_to_f32};
use crate::backend::cpu::dequant::scale_min;
use crate::backend::cpu::parallel;

// FP16 at byte address `b` (two little-endian bytes) widened to f32. Shared with
// the SIMD variant (matmul_q4k_simd) so the scale read has one transcription.
pub(super) fn f16_at(bytes: &[u8], b: usize) -> f32 {
    f16_to_f32(u16::from_le_bytes([bytes[b], bytes[b + 1]]))
}

// Per-row dot dispatcher: on x86_64 with AVX2+FMA the SIMD variant runs
// (runtime-detected, so a binary built on one host stays correct on a CPU without
// AVX2); every other target uses the scalar kernel. Both satisfy the same
// tolerance gate; neither is bit-identical to `dequant_row`.
#[inline]
pub(crate) fn row_dot_q4k(a: &[f32], bytes: &[u8], row: usize, in_dim: usize) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: the AVX2/FMA intrinsics inside are reachable only here, and
            // only after the runtime feature check just above confirms the CPU
            // supports them.
            return unsafe { super::q4k_simd::row_dot_q4k_avx2(a, bytes, row, in_dim) };
        }
    }
    row_dot_q4k_scalar(a, bytes, row, in_dim)
}

// Scalar fused dot product of activation `a` (f32) with Q4_K weight row `row`,
// summed in natural in-dimension order. Mirror of dequant_row_q4_k's block decode,
// but the dequantized values are consumed immediately instead of being stored. The
// stack buffer `wbuf` holds at most one 32-value sub-block, not a `Vec` of
// length `in_dim`). The fixed `0..32` loops have known bounds so they
// auto-vectorize. This is the portable fallback and the parity reference for the
// SIMD variant.
pub(super) fn row_dot_q4k_scalar(a: &[f32], bytes: &[u8], row: usize, in_dim: usize) -> f32 {
    let nb = in_dim / 256;
    let base = row * nb * 144;
    let mut acc = 0f32;
    for s in 0..nb {
        let blk = base + s * 144;
        let d = f16_at(bytes, blk);
        let dmin = f16_at(bytes, blk + 2);
        let sco = blk + 4;
        let qso = blk + 16;
        let abase = s * 256;
        for sb in 0..8 {
            // Sub-block `sb` reads the low (even) or high (odd) nibble of the
            // 32-byte qs group that advances every two sub-blocks — identical
            // mapping to dequant_row_q4_k. scale_min is shared so the j>=4 6-bit
            // recomposition has a single source of truth.
            let (sc, mn) = scale_min(bytes, sco, sb);
            let dl = d * sc as f32;
            let ml = dmin * mn as f32;
            let group = (sb / 2) * 32;
            let hi = sb & 1 == 1;
            let in0 = abase + sb * 32;
            let mut wbuf = [0f32; 32];
            for l in 0..32 {
                let qv = bytes[qso + group + l] as u32;
                let q4 = if hi { qv >> 4 } else { qv & 0xF };
                wbuf[l] = dl * q4 as f32 - ml;
            }
            // Fixed-length dot of the sub-block; a separate accumulator per
            // sub-block is the documented intra-row reordering.
            let mut partial = 0f32;
            for l in 0..32 {
                partial += a[in0 + l] * wbuf[l];
            }
            acc += partial;
        }
    }
    acc
}

// Per-row batched dot dispatcher: on x86_64 with AVX2+FMA the SIMD batched kernel
// runs (decode each sub-block once, FMA into every token's lane-parallel
// accumulator — same per-token order as `row_dot_q4k_avx2`, so bit-identical to it);
// every other target uses the scalar batched kernel. `acc` is a per-worker scratch
// of `n*8` f32 used only by the SIMD path (the eight lane partials per token).
#[inline]
fn row_dot_q4k_batched(
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
                super::q4k_simd::row_dot_q4k_avx2_batched(a, bytes, row, in_dim, out, acc)
            };
        }
    }
    let _ = acc;
    row_dot_q4k_batched_scalar(a, bytes, row, in_dim, out);
}

// Two-output-row batched dot dispatcher: on x86_64 with AVX2+FMA the register-blocked
// SIMD kernel processes rows `row0` and `row0+1` together, reusing each activation load
// across both rows (bit-identical to two single-row calls); every other target falls
// back to two scalar single-row dots. `acc` is per-worker scratch of `2*n*8` f32 (both
// rows' lane partials), used only by the SIMD path.
#[inline]
fn row2_dot_q4k_batched(
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
                super::q4k_simd::row2_dot_q4k_avx2_batched(a, bytes, row0, in_dim, out0, out1, acc)
            };
        }
    }
    let _ = acc;
    row_dot_q4k_batched_scalar(a, bytes, row0, in_dim, out0);
    row_dot_q4k_batched_scalar(a, bytes, row0 + 1, in_dim, out1);
}

// Fused Q4_K row dot against `n` activation vectors at once: decode each 32-value
// sub-block ONCE into the stack buffer `wbuf`, then accumulate its partial dot into
// each of the `n` token accumulators in `out[0..n]`. This is the prefill
// amortization — the per-token kernel re-decodes the whole row for every token. The
// per-token accumulation order matches `row_dot_q4k_scalar` (sub-block by sub-block,
// `partial` then `acc += partial`), so `out[i]` is bit-identical to the single-token
// scalar kernel for every token. `a` is token-major `[n][in_dim]`; `out` is the `n`
// contiguous accumulators for this output row.
pub(super) fn row_dot_q4k_batched_scalar(
    a: &[f32],
    bytes: &[u8],
    row: usize,
    in_dim: usize,
    out: &mut [f32],
) {
    let nb = in_dim / 256;
    let base = row * nb * 144;
    for o in out.iter_mut() {
        *o = 0.0;
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
            let dl = d * sc as f32;
            let ml = dmin * mn as f32;
            let group = (sb / 2) * 32;
            let hi = sb & 1 == 1;
            let in0 = abase + sb * 32;
            // Decode the 32-value sub-block ONCE, then reuse across all n tokens.
            let mut wbuf = [0f32; 32];
            for l in 0..32 {
                let qv = bytes[qso + group + l] as u32;
                let q4 = if hi { qv >> 4 } else { qv & 0xF };
                wbuf[l] = dl * q4 as f32 - ml;
            }
            for (i, acc) in out.iter_mut().enumerate() {
                let row_off = i * in_dim + in0;
                let mut partial = 0f32;
                for l in 0..32 {
                    partial += a[row_off + l] * wbuf[l];
                }
                *acc += partial;
            }
        }
    }
}

// Token-tile width for the batched kernel, ADAPTIVE to `in_dim`. Within a token tile,
// the kernel re-reads each token's `in_dim`-float activation row across the 8 sub-blocks
// of every 256-block, so the live activation working set is `tile × in_dim × 4`
// bytes. Keep it near ¾ of the private L2, leaving room for decoded weights and the
// small lane-accumulator scratch. The [16, 64] bounds cap both repeated weight decode
// and scratch size; changing the tile does not change per-token accumulation order.
pub(super) fn token_tile(in_dim: usize) -> usize {
    #[cfg(target_arch = "x86_64")]
    #[allow(unused_unsafe)] // `__cpuid` became safe after the repository's minimum Rust.
    let l2_bytes = unsafe {
        use core::arch::x86_64::__cpuid;

        // SAFETY: CPUID is always available in x86_64 user mode. Extended leaf
        // 0x80000006 reports per-core L2 size in KiB; zero or an absent leaf keeps
        // the historical 256 KiB policy instead of guessing a larger cache.
        let max_extended = __cpuid(0x8000_0000).eax;
        if max_extended >= 0x8000_0006 {
            let kib = (__cpuid(0x8000_0006).ecx >> 16) as usize;
            kib.checked_mul(1024)
                .filter(|&bytes| bytes > 0)
                .unwrap_or(256 * 1024)
        } else {
            256 * 1024
        }
    };
    #[cfg(not(target_arch = "x86_64"))]
    let l2_bytes = 256 * 1024;

    token_tile_for_l2(in_dim, l2_bytes)
}

fn token_tile_for_l2(in_dim: usize, l2_bytes: usize) -> usize {
    (l2_bytes.saturating_mul(3) / 16 / in_dim).clamp(16, 64)
}

// Batched Q4_K matmul: y[n][out_dim] = A[n][in_dim] · Wᵀ, token-major. Each output
// row's weight blocks are decoded once per token tile and dotted against all `n`
// activations (`row_dot_q4k_batched`); output-row-major scratch, transposed to
// token-major FP16 like the generic `matmul_batched`. Same contract as
// `kernels::matmul_batched`.
pub(crate) fn matmul_batched(
    out: &CpuBuffer,
    a: &CpuBuffer,
    w: &CpuBuffer,
    in_dim: usize,
    out_dim: usize,
    n: usize,
) {
    // The single-token kernel has four independent accumulators to hide FMA
    // latency. The batched kernel deliberately uses one accumulator per token
    // for throughput, so its n=1 case is the wrong execution shape for decode.
    if n == 1 {
        matmul(out, a, w, in_dim, out_dim);
        return;
    }
    let a = a.read_f16_as_f32();
    let w_bytes = w.bytes();
    let w_bytes: &[u8] = &w_bytes[w.window()];
    let tile = token_tile(in_dim);
    let mut yt = vec![0f32; out_dim * n]; // output-row-major: row o at o*n
    parallel::for_units(&mut yt, n, |o0, chunk| {
        // Per-worker AVX2 lane-partials scratch, sized to TWO token tiles (2 × tile ×
        // 8 lanes) so the register-blocked two-row kernel keeps both rows' partials
        // L1-resident; the scalar fallback ignores it. Allocated once per worker.
        let mut acc = vec![0f32; 2 * tile.min(n) * 8];
        let rows = chunk.len() / n;
        // Token tile OUTER, output row inner: the tile's activations (`tb × in_dim × 4`)
        // are read once and reused across ALL of this worker's output rows while
        // resident in cache, instead of being re-streamed per output row. This matters
        // for wide-in_dim projections whose full activation exceeds L3 (`down`: act is
        // [n][3072] ≈ 12 MB > 9 MB L3). The adaptive `tile` keeps `tb × in_dim × 4` near
        // L2. Per-token accumulation order is unchanged → bit-identical.
        let mut t0 = 0;
        while t0 < n {
            let tb = tile.min(n - t0);
            // Output rows in pairs: the two-row kernel reuses each activation load
            // across both rows. `split_at_mut` hands out the two disjoint row slices
            // (row k+1 starts at (k+1)*n > k*n+tb, so the windows never overlap).
            let mut k = 0;
            while k + 1 < rows {
                let (lo, hi) = chunk.split_at_mut((k + 1) * n);
                row2_dot_q4k_batched(
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
            // Odd trailing row (if any) keeps the single-row kernel.
            if k < rows {
                row_dot_q4k_batched(
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

// y = W·a for a Q4_K weight W, stored FP16. Same contract as `kernels::matmul`;
// rows are independent and split across cores by `for_units` with stride 1.
pub(crate) fn matmul(out: &CpuBuffer, a: &CpuBuffer, w: &CpuBuffer, in_dim: usize, out_dim: usize) {
    let a = a.read_f16_as_f32();
    let w_bytes = w.bytes();
    let w_bytes: &[u8] = &w_bytes[w.window()];
    let mut y = vec![0f32; out_dim];
    parallel::for_units(&mut y, 1, |o0, chunk| {
        for (k, dst) in chunk.iter_mut().enumerate() {
            *dst = row_dot_q4k(&a, w_bytes, o0 + k, in_dim);
        }
    });
    out.write_f16_from_f32(&y);
}

// Same as `matmul` but the output is the FP32 vocab logits (no FP16 narrowing).
// Same contract as `kernels::logits`.
pub(crate) fn logits(out: &CpuBuffer, x: &CpuBuffer, w: &CpuBuffer, in_dim: usize, out_dim: usize) {
    let x = x.read_f16_as_f32();
    let w_bytes = w.bytes();
    let w_bytes: &[u8] = &w_bytes[w.window()];
    let mut y = vec![0f32; out_dim];
    parallel::for_units(&mut y, 1, |o0, chunk| {
        for (k, dst) in chunk.iter_mut().enumerate() {
            *dst = row_dot_q4k(&x, w_bytes, o0 + k, in_dim);
        }
    });
    out.write_f32(&y);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::cpu::buffer::{CpuBuffer, CpuFormat};
    use crate::backend::cpu::dequant;

    fn f16_buf(values: &[f32]) -> CpuBuffer {
        let buf = CpuBuffer::zeroed(values.len() * 2, CpuFormat::F16);
        buf.write_f16_from_f32(values);
        buf
    }

    // Builds a synthetic, valid Q4_K weight: `out_dim` rows of `in_dim/256` blocks
    // (144 B each), filled with a deterministic byte pattern. The bytes need not be
    // "realistic" quants — dequant_row and the fused kernel read them identically,
    // so the parity check is meaningful for any well-formed block.
    fn q4k_weight(in_dim: usize, out_dim: usize) -> CpuBuffer {
        let nb = in_dim / 256;
        let mut bytes = vec![0u8; out_dim * nb * 144];
        for (i, b) in bytes.iter_mut().enumerate() {
            *b = ((i * 31 + 7) % 251) as u8;
        }
        // d/dmin are FP16 scales; force small finite values so products stay in a
        // sane range (raw bytes could otherwise decode to inf/NaN FP16).
        for blk in 0..out_dim * nb {
            let base = blk * 144;
            bytes[base..base + 2]
                .copy_from_slice(&crate::backend::cpu::buffer::f32_to_f16(0.05).to_le_bytes());
            bytes[base + 2..base + 4]
                .copy_from_slice(&crate::backend::cpu::buffer::f32_to_f16(0.01).to_le_bytes());
        }
        CpuBuffer::from_bytes(bytes, CpuFormat::Q4_K)
    }

    // Reference: the generic dequant_row path (materialize the row, then dot).
    fn reference_row(a: &[f32], w: &CpuBuffer, row: usize, in_dim: usize) -> f32 {
        let wb = w.bytes();
        let mut r = vec![0f32; in_dim];
        dequant::dequant_row(CpuFormat::Q4_K, &wb, row, in_dim, &mut r);
        (0..in_dim).map(|i| a[i] * r[i]).sum()
    }

    // Fused matmul must match the generic dequant_row path within the quantized
    // tolerance (rel. 8e-2). out_dim > a typical core count so the parallel
    // path actually splits; in_dim spans two 256-blocks so the j>=4 sub-block and
    // multi-block accumulation are both exercised.
    #[test]
    fn fused_matmul_matches_dequant_row_within_tolerance() {
        let in_dim = 512;
        let out_dim = 17;
        let a_vals: Vec<f32> = (0..in_dim).map(|i| (i as f32 * 0.013).sin()).collect();
        let a = f16_buf(&a_vals);
        let aw = a.read_f16_as_f32();
        let w = q4k_weight(in_dim, out_dim);

        let out = CpuBuffer::zeroed(out_dim * 2, CpuFormat::F16);
        matmul(&out, &a, &w, in_dim, out_dim);
        let got = out.read_f16_as_f32();

        for (o, &value) in got.iter().enumerate().take(out_dim) {
            let want = reference_row(&aw, &w, o, in_dim);
            let tol = 8e-2 * want.abs().max(1e-3);
            assert!(
                (value - want).abs() <= tol,
                "row {o}: fused {value} vs ref {want} (tol {tol})"
            );
        }
    }

    // Batched Q4_K matmul must match, per token, the single-token `matmul` on each
    // token within the quant tolerance. They share the dequant but NOT the exact FP
    // reduction order: the single-token AVX2 kernel uses four chunk accumulators (to
    // break the latency chain at n==1, the decode win), while the batched kernel keeps
    // one accumulator per token (it is throughput-bound, so extra accumulators would
    // only add L1 traffic). The reassociation difference is far below the quant
    // tolerance the kernel already carries. out_dim > a core count so the parallel
    // split runs; in_dim spans two 256-blocks.
    #[test]
    fn batched_matches_per_token() {
        let (in_dim, out_dim, n) = (512usize, 17usize, 5usize);
        let w = q4k_weight(in_dim, out_dim);
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

    // logits path: same parity, FP32 output (no narrowing).
    #[test]
    fn fused_logits_matches_dequant_row_within_tolerance() {
        let in_dim = 256;
        let out_dim = 13;
        let x_vals: Vec<f32> = (0..in_dim)
            .map(|i| (i as f32 * 0.021).cos() * 0.5)
            .collect();
        let x = f16_buf(&x_vals);
        let xw = x.read_f16_as_f32();
        let w = q4k_weight(in_dim, out_dim);

        let out = CpuBuffer::zeroed(out_dim * 4, CpuFormat::F32);
        logits(&out, &x, &w, in_dim, out_dim);
        let got = out.read_f32();

        for (o, &value) in got.iter().enumerate().take(out_dim) {
            let want = reference_row(&xw, &w, o, in_dim);
            let tol = 8e-2 * want.abs().max(1e-3);
            assert!((value - want).abs() <= tol, "logit {o}: {value} vs {want}");
        }
    }

    #[test]
    fn token_tile_tracks_l2_without_leaving_bounds() {
        assert_eq!(token_tile_for_l2(3072, 256 * 1024), 16);
        assert_eq!(token_tile_for_l2(3072, 1024 * 1024), 64);
        assert_eq!(token_tile_for_l2(9216, 1024 * 1024), 21);
        assert_eq!(token_tile_for_l2(256, 1024 * 1024), 64);
    }
}
