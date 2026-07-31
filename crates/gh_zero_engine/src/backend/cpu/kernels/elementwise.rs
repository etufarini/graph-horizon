/*
 * gh_zero_engine — CPU elementwise kernels
 * Scalar CPU implementations for embedding, residual-stream RMSNorm, RoPE,
 * residual add, and fused SiLU-mul. Every kernel computes in f32 and writes the
 * destination format; the fused operation rounds its intermediate to FP16 so
 * its test-only sequential `silu`/`mul` oracle remains byte-identical. This file
 * owns no dispatch or storage lifecycle.
*/

use super::super::buffer::{CpuBuffer, f16_to_f32, f32_to_f16};
use super::super::dequant;
use super::super::parallel;

// Round-trips a value through FP16, exactly as a write-then-read of an FP16 buffer
// would. The fused kernels apply it to the intermediate so the result matches the
// two-kernel path that materialises that intermediate in FP16 memory.
fn round_f16(v: f32) -> f32 {
    f16_to_f32(f32_to_f16(v))
}

// Dequantizes row `token` of token_embd (any of the 4 formats) into the FP32
// residual stream `x`. Mirror of embed_{f16,q8_0,q4_k}.comp (extended to Q6_K via
// dequant_row, which already handles all four formats).
pub(crate) fn embed(x: &CpuBuffer, token_embd: &CpuBuffer, token: usize, embd: usize) {
    let mut row = vec![0f32; embd];
    // Slice the guard to the buffer's window so a sub-view reads only its bytes.
    let eb = token_embd.bytes();
    dequant::dequant_row(
        token_embd.format,
        &eb[token_embd.window()],
        token,
        embd,
        &mut row,
    );
    x.write_f32(&row);
}

// As rmsnorm but the source is the FP32 residual stream (FP16 weight, FP16 out).
// Mirror of rmsnorm_x.comp.
pub(crate) fn rmsnorm_x(
    out: &CpuBuffer,
    x: &CpuBuffer,
    w: &CpuBuffer,
    dim: usize,
    eps: f32,
    rows: usize,
) {
    let x = x.read_f32();
    let w = w.read_f16_as_f32();
    rmsnorm_rows(out, &x, &w, dim, eps, rows);
}

// Shared per-row body: the only difference between the two kernels above is how
// the input `x` is read (FP16 vs FP32), already resolved into an f32 slice here.
fn rmsnorm_rows(out: &CpuBuffer, x: &[f32], w: &[f32], dim: usize, eps: f32, rows: usize) {
    let mut o = vec![0f32; rows * dim];
    // Rows are independent reductions; distribute them across cores (stride `dim`).
    // For prefill `rows` = n (≈1k) this was a serial pass while the matmuls used all
    // cores. `u0` is the absolute ROW index of the chunk's first row (for_units passes
    // unit indices, and a unit here is one `dim`-long row), so input row `u0 + r` (at
    // element base `(u0+r)*dim`) pairs with output row `r*dim` inside the chunk — same
    // per-row order, so the result is bit-identical to the serial path.
    parallel::for_units(&mut o, dim, |u0, chunk| {
        for r in 0..chunk.len() / dim {
            let xbase = (u0 + r) * dim;
            let obase = r * dim;
            let mut sum = 0f32;
            for i in 0..dim {
                let v = x[xbase + i];
                sum += v * v;
            }
            let inv = 1.0 / (sum / dim as f32 + eps).sqrt();
            for i in 0..dim {
                chunk[obase + i] = x[xbase + i] * inv * w[i];
            }
        }
    });
    out.write_f16_from_f32(&o);
}

// In-place YaRN RoPE with Mistral's adjacent-pair (NORM) layout and the same
// role-specific post-scale as the Vulkan shader. Config validation guarantees
// an even rope dimension within head_dim.
pub(crate) fn rope_yarn(
    x: &CpuBuffer,
    heads: usize,
    head_dim: usize,
    pos: usize,
    yarn: &crate::backend::rope::Yarn,
    role: crate::backend::rope::RopeRole,
) -> color_eyre::eyre::Result<()> {
    let mut values = x.read_f16_as_f32();
    let half = yarn.rope_dim / 2;
    let scale = yarn.post_scale(role, pos);
    for head in 0..heads {
        let base = head * head_dim;
        for pair in 0..half {
            let p = yarn.pair(role, pair, pos)?;
            let first = base + pair * 2;
            let second = first + 1;
            let a = values[first];
            let b = values[second];
            values[first] = (a * p.cos - b * p.sin) * scale;
            values[second] = (a * p.sin + b * p.cos) * scale;
        }
    }
    x.write_f16_from_f32(&values);
    Ok(())
}

// SiLU: out[i] = x[i] * sigmoid(x[i]). FP16 in and out. Mirror of silu.comp.
#[cfg(test)]
pub(crate) fn silu(out: &CpuBuffer, x: &CpuBuffer, n: usize) {
    let x = x.read_f16_as_f32();
    let mut o = vec![0f32; n];
    for i in 0..n {
        let v = x[i];
        o[i] = v / (1.0 + (-v).exp());
    }
    out.write_f16_from_f32(&o);
}

// Elementwise product (gate ⊙ up). FP16 in and out. Mirror of mul.comp.
#[cfg(test)]
pub(crate) fn mul(out: &CpuBuffer, a: &CpuBuffer, b: &CpuBuffer, n: usize) {
    let a = a.read_f16_as_f32();
    let b = b.read_f16_as_f32();
    let mut o = vec![0f32; n];
    for i in 0..n {
        o[i] = a[i] * b[i];
    }
    out.write_f16_from_f32(&o);
}

// Fused SiLU(gate) ⊙ up. FP16 in and out. Equals silu(act,gate) then mul(out,
// act,up): the silu intermediate is rounded to FP16 (round_f16) before the
// product, exactly as the two-kernel path materialises `act` in FP16. Mirror of
// silu_mul.comp.
pub(crate) fn silu_mul(out: &CpuBuffer, gate: &CpuBuffer, up: &CpuBuffer, n: usize) {
    let gate = gate.read_f16_as_f32();
    let up = up.read_f16_as_f32();
    let mut o = vec![0f32; n];
    // The per-element SiLU (an `exp` each) over n = rows×ffn (≈3M at prefill) was a
    // serial pass; split it across cores. `i0` is the absolute element index of the
    // chunk, so each element keeps its own `gate[i]`/`up[i]` — bit-identical.
    parallel::for_units(&mut o, 1, |i0, chunk| {
        for (j, oj) in chunk.iter_mut().enumerate() {
            let v = gate[i0 + j];
            let silu = v / (1.0 + (-v).exp());
            *oj = round_f16(silu) * up[i0 + j];
        }
    });
    out.write_f16_from_f32(&o);
}

// Residual add in place: x[i] += y[i], with the residual `x` FP32 and `y` FP16
// (widened to f32 before the add). Mirror of residual.comp.
pub(crate) fn residual_add(x: &CpuBuffer, y: &CpuBuffer, n: usize) {
    let mut xf = x.read_f32();
    let yf = y.read_f16_as_f32();
    // Independent per element; split the first `n` across cores (`i0` = absolute
    // element index). Only the first `n` elements are touched, as in the serial path.
    parallel::for_units(&mut xf[..n], 1, |i0, chunk| {
        for (j, xj) in chunk.iter_mut().enumerate() {
            *xj += yf[i0 + j];
        }
    });
    x.write_f32(&xf);
}

#[cfg(test)]
mod tests {
    use super::super::super::buffer::CpuFormat;
    use super::*;

    // Deterministic, non-FP16-exact f32 inputs (24-bit fraction): the silu / normed
    // intermediates are generally not representable in FP16, so the FP16 rounding of
    // the intermediate is observable and the fused/sequential equality is meaningful.
    fn vals(n: usize, seed: u32) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                (s >> 8) as f32 / (1u32 << 24) as f32 * 4.0 - 2.0
            })
            .collect()
    }

    fn f16_buf(v: &[f32]) -> CpuBuffer {
        let b = CpuBuffer::zeroed(v.len() * 2, CpuFormat::F16);
        b.write_f16_from_f32(v);
        b
    }

    fn raw(b: &CpuBuffer) -> Vec<u8> {
        b.bytes()[b.window()].to_vec()
    }

    // silu_mul ≡ silu then mul, byte-for-byte in FP16.
    #[test]
    fn silu_mul_matches_silu_then_mul() {
        let n = 64;
        let gate = f16_buf(&vals(n, 1));
        let up = f16_buf(&vals(n, 2));

        let act = CpuBuffer::zeroed(n * 2, CpuFormat::F16);
        let out_seq = CpuBuffer::zeroed(n * 2, CpuFormat::F16);
        silu(&act, &gate, n);
        mul(&out_seq, &act, &up, n);

        let out_fused = CpuBuffer::zeroed(n * 2, CpuFormat::F16);
        silu_mul(&out_fused, &gate, &up, n);

        assert_eq!(raw(&out_seq), raw(&out_fused));
    }

    #[test]
    fn rope_uses_mistral_adjacent_pairs() {
        let yarn = crate::backend::rope::Yarn {
            rope_dim: 4,
            original_context: 128,
            freq_base: 10_000.0,
            factor: 1.0,
            beta_fast: 32.0,
            beta_slow: 1.0,
            log_multiplier: 1.0,
            q_temperature_scale: 1.0,
        };
        let input = [1.0, 2.0, 3.0, 4.0];
        let buffer = f16_buf(&input);

        rope_yarn(&buffer, 1, 4, 1, &yarn, crate::backend::rope::RopeRole::Key).unwrap();

        let got = buffer.read_f16_as_f32();
        for pair in 0..2 {
            let rotation = yarn
                .pair(crate::backend::rope::RopeRole::Key, pair, 1)
                .unwrap();
            let first = pair * 2;
            let expected = [
                input[first] * rotation.cos - input[first + 1] * rotation.sin,
                input[first] * rotation.sin + input[first + 1] * rotation.cos,
            ];
            for (actual, expected) in got[first..first + 2].iter().zip(expected) {
                assert!((*actual - expected).abs() <= 3e-3);
            }
        }
    }
}
