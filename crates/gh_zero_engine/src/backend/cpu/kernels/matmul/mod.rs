/*
 * gh_zero_engine — CPU matmul kernels
 * Scalar transcription of matmul_*.comp / logits_*.comp: one output row per
 * iteration. The retained weight format (read off CpuBuffer.format) selects the dequant
 * (dequant::dequant_row), which yields the row in natural in-dimension order so
 * weight `i` pairs with activation `a[i]`. Accumulation is FP32; the result is
 * FP16 for matmul and FP32 for logits. The activation is FP16, widened to f32 on
 * read. These kernels cannot fail: block validity is guaranteed once, at load.
 * Output rows are independent and distributed across the cores via
 * `parallel::for_units` (stride 1); each row's inner accumulation stays in
 * `i = 0..in_dim` order, so the result is bit-identical to the single-thread
 * path. The `row` scratch is per-thread (one Vec per chunk).
*/

// AGENTS deroga K: kernel matmul/logits denso + selezione per-formato della stessa operazione GEMM, nessun dispatch cross-operazione.

use crate::backend::cpu::buffer::{self, CpuBuffer, CpuFormat};
use crate::backend::cpu::dequant;
use crate::backend::cpu::parallel;

pub(crate) mod q4k;
pub(crate) mod q4k_simd;
pub(crate) mod q5k;
pub(crate) mod q5k_simd;
pub(crate) mod q6k;
pub(crate) mod q6k_simd;

// Dot product of two equal-length f32 slices, used by the generic (non-Q4_K) matmul
// paths once the weight row has been dequantized to f32 — i.e. Q5_K/Q6_K/F16
// for both prefill (per token) and decode (incl. the Q6_K lm_head, the biggest single
// decode GEMV). The scalar `.sum()` it replaces is a sequential fold: latency-bound
// (each add waits on the previous) and not auto-vectorizable (f32 add is not
// associative). On x86_64 with AVX2+FMA the bulk runs four independent 8-wide FMA
// accumulators (32 elements/iter) that the FMA ports pipeline; the < 8 tail is the
// SAME sequential fold, so for the small-`in_dim` unit tests (all tail) the result is
// bit-identical to `.sum()`, and only real (large) dims reassociate — within the quant
// tolerance the dequant already carries.
#[inline]
fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: guarded by the runtime AVX2+FMA detection just above.
            return unsafe { dot_f32_avx2(a, b) };
        }
    }
    a.iter().zip(b).map(|(&x, &y)| x * y).sum()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_f32_avx2(a: &[f32], b: &[f32]) -> f32 {
    use core::arch::x86_64::*;
    unsafe {
        let n = a.len().min(b.len());
        let mut acc = [_mm256_setzero_ps(); 4];
        let mut i = 0;
        while i + 32 <= n {
            acc[0] = _mm256_fmadd_ps(
                _mm256_loadu_ps(a.as_ptr().add(i)),
                _mm256_loadu_ps(b.as_ptr().add(i)),
                acc[0],
            );
            acc[1] = _mm256_fmadd_ps(
                _mm256_loadu_ps(a.as_ptr().add(i + 8)),
                _mm256_loadu_ps(b.as_ptr().add(i + 8)),
                acc[1],
            );
            acc[2] = _mm256_fmadd_ps(
                _mm256_loadu_ps(a.as_ptr().add(i + 16)),
                _mm256_loadu_ps(b.as_ptr().add(i + 16)),
                acc[2],
            );
            acc[3] = _mm256_fmadd_ps(
                _mm256_loadu_ps(a.as_ptr().add(i + 24)),
                _mm256_loadu_ps(b.as_ptr().add(i + 24)),
                acc[3],
            );
            i += 32;
        }
        while i + 8 <= n {
            acc[0] = _mm256_fmadd_ps(
                _mm256_loadu_ps(a.as_ptr().add(i)),
                _mm256_loadu_ps(b.as_ptr().add(i)),
                acc[0],
            );
            i += 8;
        }
        // Same reduction tree as the dual-row kernel (shared `reduce4`).
        let mut r = reduce4(acc);
        // Sequential < 8 tail, identical to the scalar fold (keeps small-dim tests exact).
        while i < n {
            r += a[i] * b[i];
            i += 1;
        }
        r
    }
}

// Dual-row dot: `(a·b0, a·b1)` reusing each activation load across BOTH weight rows
// (one load → two FMAs). The batched generic matmul decodes two output rows at a
// time, so this halves the activation L1 traffic vs two separate `dot_f32` calls,
// shifting the inner loop from load-bound toward FMA-bound — the prefill win on the
// non-Q4_K formats (Q6_K/Q5_K/F16). Each row uses the SAME 4-accumulator /
// 32-elem-per-iter reduction (and the same < 8 sequential tail) as `dot_f32`, so the
// results are BIT-IDENTICAL to `dot_f32(a,b0)` and `dot_f32(a,b1)` — no numeric
// change, the per-token parity test still holds exactly.
#[inline]
fn dot2_f32(a: &[f32], b0: &[f32], b1: &[f32]) -> (f32, f32) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: guarded by the runtime AVX2+FMA detection just above.
            return unsafe { dot2_f32_avx2(a, b0, b1) };
        }
    }
    (
        a.iter().zip(b0).map(|(&x, &y)| x * y).sum(),
        a.iter().zip(b1).map(|(&x, &y)| x * y).sum(),
    )
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot2_f32_avx2(a: &[f32], b0: &[f32], b1: &[f32]) -> (f32, f32) {
    use core::arch::x86_64::*;
    unsafe {
        let n = a.len().min(b0.len()).min(b1.len());
        let mut acc0 = [_mm256_setzero_ps(); 4];
        let mut acc1 = [_mm256_setzero_ps(); 4];
        let mut i = 0;
        // 32 elems/iter: load the four activation vectors ONCE, FMA into both rows.
        while i + 32 <= n {
            let a0 = _mm256_loadu_ps(a.as_ptr().add(i));
            let a1 = _mm256_loadu_ps(a.as_ptr().add(i + 8));
            let a2 = _mm256_loadu_ps(a.as_ptr().add(i + 16));
            let a3 = _mm256_loadu_ps(a.as_ptr().add(i + 24));
            acc0[0] = _mm256_fmadd_ps(a0, _mm256_loadu_ps(b0.as_ptr().add(i)), acc0[0]);
            acc0[1] = _mm256_fmadd_ps(a1, _mm256_loadu_ps(b0.as_ptr().add(i + 8)), acc0[1]);
            acc0[2] = _mm256_fmadd_ps(a2, _mm256_loadu_ps(b0.as_ptr().add(i + 16)), acc0[2]);
            acc0[3] = _mm256_fmadd_ps(a3, _mm256_loadu_ps(b0.as_ptr().add(i + 24)), acc0[3]);
            acc1[0] = _mm256_fmadd_ps(a0, _mm256_loadu_ps(b1.as_ptr().add(i)), acc1[0]);
            acc1[1] = _mm256_fmadd_ps(a1, _mm256_loadu_ps(b1.as_ptr().add(i + 8)), acc1[1]);
            acc1[2] = _mm256_fmadd_ps(a2, _mm256_loadu_ps(b1.as_ptr().add(i + 16)), acc1[2]);
            acc1[3] = _mm256_fmadd_ps(a3, _mm256_loadu_ps(b1.as_ptr().add(i + 24)), acc1[3]);
            i += 32;
        }
        while i + 8 <= n {
            let av = _mm256_loadu_ps(a.as_ptr().add(i));
            acc0[0] = _mm256_fmadd_ps(av, _mm256_loadu_ps(b0.as_ptr().add(i)), acc0[0]);
            acc1[0] = _mm256_fmadd_ps(av, _mm256_loadu_ps(b1.as_ptr().add(i)), acc1[0]);
            i += 8;
        }
        // Same reduction tree as `dot_f32`, per row, so each result matches it exactly.
        let mut r0 = reduce4(acc0);
        let mut r1 = reduce4(acc1);
        while i < n {
            r0 += a[i] * b0[i];
            r1 += a[i] * b1[i];
            i += 1;
        }
        (r0, r1)
    }
}

// The horizontal reduction `dot_f32_avx2` uses, factored so the dual-row kernel is
// bit-identical to it: tree-sum (0+1)+(2+3) then a 128-bit hadd fold.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn reduce4(acc: [core::arch::x86_64::__m256; 4]) -> f32 {
    use core::arch::x86_64::*;
    let lo = _mm256_add_ps(acc[0], acc[1]);
    let hi = _mm256_add_ps(acc[2], acc[3]);
    let v = _mm256_add_ps(lo, hi);
    let q = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    let q = _mm_hadd_ps(q, q);
    let q = _mm_hadd_ps(q, q);
    _mm_cvtss_f32(q)
}

// y = W·a, W row-major [out_dim, in_dim]: per output row `o`, acc(f32) =
// Σ_i dequant(w, row o, i) * f32(a[i]); stored as FP16. Mirror of matmul_*.comp.
// Q4_K — the heavy quant for offloaded weights — is dispatched to the fused
// dequant+MAC kernel (matmul_q4k) that skips the intermediate f32 row; every
// other format keeps this scalar dequant_row path, byte-identical to before.
pub(crate) fn matmul(out: &CpuBuffer, a: &CpuBuffer, w: &CpuBuffer, in_dim: usize, out_dim: usize) {
    if w.format == CpuFormat::Q4_K {
        return q4k::matmul(out, a, w, in_dim, out_dim);
    }
    if w.format == CpuFormat::Q6_K {
        return q6k::matmul(out, a, w, in_dim, out_dim);
    }
    if w.format == CpuFormat::Q5_K {
        return q5k::matmul(out, a, w, in_dim, out_dim);
    }
    let a = a.read_f16_as_f32();
    // One read-guard held by the parent; workers share the weight bytes read-only.
    // Sliced to the buffer's window so a sub-view reads only its own rows.
    let w_bytes = w.bytes();
    let w_bytes: &[u8] = &w_bytes[w.window()];
    let mut y = vec![0f32; out_dim];
    parallel::for_units(&mut y, 1, |o0, chunk| {
        let mut row = vec![0f32; in_dim]; // per-thread scratch, never shared
        for (k, dst) in chunk.iter_mut().enumerate() {
            dequant::dequant_row(w.format, w_bytes, o0 + k, in_dim, &mut row);
            // AVX2 multi-accumulator dot of the activation with the decoded row (see
            // `dot_f32`); rows stay independent, distributed by `for_units`.
            *dst = dot_f32(&a, &row);
        }
    });
    out.write_f16_from_f32(&y);
}

// Batched y[n][out_dim] = A[n][in_dim] · Wᵀ, token-major (row i = token i). Unlike
// `matmul` (which re-reads/re-dequantizes the whole weight per token), each output
// row `o` is dequantized ONCE and dotted against all `n` activation vectors — the
// prefill amortization of the weight read + dequant over the batch. The result is
// computed into OUTPUT-row-major scratch `yt[o*n + i]` so `for_units` (stride `n`)
// hands each worker a contiguous block of whole output rows (same disjoint split as
// `matmul`), then transposed into the token-major FP16 `out`. Each token's fold
// stays in `i = 0..in_dim` order, so column `i` is bit-identical to the single-token
// `matmul`. Q4_K dispatches to its fused batched kernel.
pub(crate) fn matmul_batched(
    out: &CpuBuffer,
    a: &CpuBuffer,
    w: &CpuBuffer,
    in_dim: usize,
    out_dim: usize,
    n: usize,
) {
    if w.format == CpuFormat::Q4_K {
        return q4k::matmul_batched(out, a, w, in_dim, out_dim, n);
    }
    if w.format == CpuFormat::Q6_K {
        return q6k::matmul_batched(out, a, w, in_dim, out_dim, n);
    }
    if w.format == CpuFormat::Q5_K {
        return q5k::matmul_batched(out, a, w, in_dim, out_dim, n);
    }
    let a = a.read_f16_as_f32(); // n * in_dim, token-major
    let w_bytes = w.bytes();
    let w_bytes: &[u8] = &w_bytes[w.window()];
    let mut yt = vec![0f32; out_dim * n]; // output-row-major: row o at o*n
    parallel::for_units(&mut yt, n, |o0, chunk| {
        // Two per-thread row scratches: output rows are decoded and dotted in PAIRS so
        // each token's activation is loaded once and reused across both rows (`dot2_f32`).
        let mut row0 = vec![0f32; in_dim];
        let mut row1 = vec![0f32; in_dim];
        let rows = chunk.len() / n;
        let mut k = 0;
        while k + 1 < rows {
            dequant::dequant_row(w.format, w_bytes, o0 + k, in_dim, &mut row0);
            dequant::dequant_row(w.format, w_bytes, o0 + k + 1, in_dim, &mut row1);
            // Row k+1 starts at (k+1)*n, past row k's window, so the two are disjoint.
            let (lo, hi) = chunk.split_at_mut((k + 1) * n);
            let (dst0, dst1) = (&mut lo[k * n..k * n + n], &mut hi[..n]);
            for i in 0..n {
                let ai = &a[i * in_dim..(i + 1) * in_dim];
                let (d0, d1) = dot2_f32(ai, &row0, &row1);
                dst0[i] = d0;
                dst1[i] = d1;
            }
            k += 2;
        }
        // Odd trailing row (if any) keeps the single-row dot.
        if k < rows {
            dequant::dequant_row(w.format, w_bytes, o0 + k, in_dim, &mut row0);
            let dst = &mut chunk[k * n..k * n + n];
            for (i, d) in dst.iter_mut().enumerate() {
                *d = dot_f32(&a[i * in_dim..(i + 1) * in_dim], &row0);
            }
        }
    });
    write_transposed_f16(out, &yt, out_dim, n);
}

// Writes the GEMM result straight into `out` as token-major FP16, fusing the
// transpose (output-row-major `yt[o*n+i]` → token-major `[n][out_dim]`) with the
// f16 narrow in ONE parallel pass — no intermediate `[n][out_dim]` f32 Vec (it was a
// multi-MB allocation per batched matmul, ~7 per layer). Parallelized over whole
// output tokens (stride `out_dim` f16 = `out_dim*2` bytes): each worker owns a
// disjoint block of tokens, gathers that token's column of `yt` (the strided part the
// SIMD+multi-core GEMM otherwise left serial) into a small reused scratch, then
// narrows it into its contiguous byte region. Bit-identical to `transpose` then
// `write_f16_from_f32` (same values, same per-element RNE narrow; only the work split
// changes). Shared by every batched kernel so the layout contract has one transcription.
pub(super) fn write_transposed_f16(out: &CpuBuffer, yt: &[f32], out_dim: usize, n: usize) {
    out.with_bytes_mut(|dst| {
        parallel::for_units(dst, out_dim * 2, |t0, chunk| {
            let mut scratch = vec![0f32; out_dim]; // per-worker, reused across its tokens
            for jt in 0..chunk.len() / (out_dim * 2) {
                let i = t0 + jt;
                for o in 0..out_dim {
                    scratch[o] = yt[o * n + i];
                }
                buffer::narrow_f32_to_f16(
                    &scratch,
                    &mut chunk[jt * out_dim * 2..(jt + 1) * out_dim * 2],
                );
            }
        });
    });
}

// Same as matmul but the output is the FP32 vocab logits: the accumulator is
// stored without the FP16 narrowing (compatible with read_logits). Mirror of
// logits_*.comp.
pub(crate) fn logits(out: &CpuBuffer, x: &CpuBuffer, w: &CpuBuffer, in_dim: usize, out_dim: usize) {
    if w.format == CpuFormat::Q4_K {
        return q4k::logits(out, x, w, in_dim, out_dim);
    }
    if w.format == CpuFormat::Q6_K {
        return q6k::logits(out, x, w, in_dim, out_dim);
    }
    if w.format == CpuFormat::Q5_K {
        return q5k::logits(out, x, w, in_dim, out_dim);
    }
    let x = x.read_f16_as_f32();
    let w_bytes = w.bytes();
    let w_bytes: &[u8] = &w_bytes[w.window()];
    let mut y = vec![0f32; out_dim];
    parallel::for_units(&mut y, 1, |o0, chunk| {
        let mut row = vec![0f32; in_dim]; // per-thread scratch, never shared
        for (k, dst) in chunk.iter_mut().enumerate() {
            dequant::dequant_row(w.format, w_bytes, o0 + k, in_dim, &mut row);
            // Same vectorization-friendly form as `matmul`; bit-identical fold over
            // `i = 0..in_dim`, stored without the FP16 narrowing.
            *dst = dot_f32(&x, &row);
        }
    });
    out.write_f32(&y);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::cpu::buffer::{CpuBuffer, CpuFormat};

    fn f16_buf(values: &[f32]) -> CpuBuffer {
        let buf = CpuBuffer::zeroed(values.len() * 2, CpuFormat::F16);
        buf.write_f16_from_f32(values);
        buf
    }

    // Bit-identical parity: the chunked matmul must equal a plain serial loop over
    // the rows, byte for byte. out_dim is larger than a typical core count so the
    // parallel path actually splits when more than one worker is available.
    #[test]
    fn matmul_parallel_matches_serial_reference() {
        let in_dim = 6;
        let out_dim = 17;
        let a_vals: Vec<f32> = (0..in_dim).map(|i| i as f32 * 0.5 - 1.0).collect();
        let w_vals: Vec<f32> = (0..out_dim * in_dim)
            .map(|k| (k % 7) as f32 * 0.25 - 0.75)
            .collect();
        let a = f16_buf(&a_vals);
        let w = f16_buf(&w_vals);
        let out = CpuBuffer::zeroed(out_dim * 2, CpuFormat::F16);
        matmul(&out, &a, &w, in_dim, out_dim);

        let aw = a.read_f16_as_f32();
        let wb = w.bytes();
        let mut row = vec![0f32; in_dim];
        let mut expected = vec![0f32; out_dim];
        for (o, dst) in expected.iter_mut().enumerate() {
            dequant::dequant_row(CpuFormat::F16, &wb, o, in_dim, &mut row);
            *dst = (0..in_dim).map(|i| aw[i] * row[i]).sum();
        }
        let ref_buf = CpuBuffer::zeroed(out_dim * 2, CpuFormat::F16);
        ref_buf.write_f16_from_f32(&expected);
        assert_eq!(out.read_f16_as_f32(), ref_buf.read_f16_as_f32());
    }

    // Batched matmul must be bit-identical, per token, to running the single-token
    // `matmul` on each token's row — token-major output, several tokens and an
    // out_dim larger than the core count so the parallel split is exercised.
    #[test]
    fn matmul_batched_matches_per_token() {
        let (in_dim, out_dim, n) = (6usize, 17usize, 4usize);
        let w_vals: Vec<f32> = (0..out_dim * in_dim)
            .map(|k| (k % 7) as f32 * 0.25 - 0.75)
            .collect();
        let w = f16_buf(&w_vals);
        let a_vals: Vec<f32> = (0..n * in_dim)
            .map(|k| (k % 11) as f32 * 0.3 - 1.2)
            .collect();
        let a = f16_buf(&a_vals);

        let batched = CpuBuffer::zeroed(n * out_dim * 2, CpuFormat::F16);
        matmul_batched(&batched, &a, &w, in_dim, out_dim, n);

        // Per-token reference: run `matmul` on each token row into a token-major buffer.
        let mut expected = vec![0f32; n * out_dim];
        for i in 0..n {
            let ai = f16_buf(&a_vals[i * in_dim..(i + 1) * in_dim]);
            let oi = CpuBuffer::zeroed(out_dim * 2, CpuFormat::F16);
            matmul(&oi, &ai, &w, in_dim, out_dim);
            expected[i * out_dim..(i + 1) * out_dim].copy_from_slice(&oi.read_f16_as_f32());
        }
        assert_eq!(batched.read_f16_as_f32(), expected);
    }

    #[test]
    fn logits_parallel_matches_serial_reference() {
        let in_dim = 5;
        let out_dim = 13;
        let x_vals: Vec<f32> = (0..in_dim).map(|i| i as f32 * 0.3 - 0.6).collect();
        let w_vals: Vec<f32> = (0..out_dim * in_dim)
            .map(|k| (k % 5) as f32 * 0.2 - 0.4)
            .collect();
        let x = f16_buf(&x_vals);
        let w = f16_buf(&w_vals);
        let out = CpuBuffer::zeroed(out_dim * 4, CpuFormat::F32);
        logits(&out, &x, &w, in_dim, out_dim);

        let xw = x.read_f16_as_f32();
        let wb = w.bytes();
        let mut row = vec![0f32; in_dim];
        let mut expected = vec![0f32; out_dim];
        for (o, dst) in expected.iter_mut().enumerate() {
            dequant::dequant_row(CpuFormat::F16, &wb, o, in_dim, &mut row);
            *dst = (0..in_dim).map(|i| xw[i] * row[i]).sum();
        }
        assert_eq!(out.read_f32(), expected);
    }
}
