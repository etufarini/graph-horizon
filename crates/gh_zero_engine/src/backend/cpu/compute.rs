/*
 * gh_zero_engine — generic f32 CPU compute primitives
 * Test-only numeric oracles for Vulkan matmul parity and scalar activation math.
 * The hybrid Vulkan tests dequantize CPU weight rows directly; CPU-only tests
 * retain RMSNorm, L2 norm, SiLU, softplus, and sigmoid references. This module
 * owns no runtime dispatch, resource state, or production entry point.
*/

#[cfg(feature = "vulkan")]
use super::dequant;
#[cfg(feature = "vulkan")]
use super::parallel;
#[cfg(feature = "vulkan")]
use super::row_dot_q4k;
use super::{CpuBuffer, CpuFormat};

// Reads weight row `row` (length `in_dim`) from the already-locked weight `bytes`
// into `out`: quantized/F16 rows are dequantized on the fly; an F32 row is read
// directly (little-endian). Takes the byte slice (not the `CpuBuffer`) so callers
// lock the storage once and reuse it across many rows / across worker threads.
#[cfg(feature = "vulkan")]
fn weight_row(format: CpuFormat, bytes: &[u8], row: usize, in_dim: usize, out: &mut [f32]) {
    if matches!(format, CpuFormat::F32) {
        let base = row * in_dim * 4;
        for (i, o) in out[..in_dim].iter_mut().enumerate() {
            let b = base + i * 4;
            *o = f32::from_le_bytes([bytes[b], bytes[b + 1], bytes[b + 2], bytes[b + 3]]);
        }
    } else {
        dequant::dequant_row(format, bytes, row, in_dim, out);
    }
}

// y = W·a, with W a [out_dim, in_dim] matrix (ggml order: in_dim is the row length)
// and `a` an f32 vector of length `in_dim`. Output rows are independent, so they
// are split across cores by `parallel::for_units` (stride 1): the per-row value is
// unchanged by the worker count (no cross-row reordering), so the result matches
// the serial path. Q4_K weights take the fused dequant+MAC kernel (AVX2 when the
// CPU has it, scalar otherwise — runtime-detected) and never materialize a
// dequantized row; every other format dequantizes one row into a per-worker scratch
// buffer and dots it. The weight storage is locked once here and shared as `&[u8]`.
#[cfg(feature = "vulkan")]
pub(crate) fn matmul(w: &CpuBuffer, a: &[f32], in_dim: usize, out_dim: usize) -> Vec<f32> {
    let guard = w.bytes();
    let bytes: &[u8] = &guard[w.window()];
    let mut y = vec![0.0f32; out_dim];
    if matches!(w.format, CpuFormat::Q4_K) {
        parallel::for_units(&mut y, 1, |o0, chunk| {
            for (k, dst) in chunk.iter_mut().enumerate() {
                *dst = row_dot_q4k(a, bytes, o0 + k, in_dim);
            }
        });
    } else {
        let format = w.format;
        parallel::for_units(&mut y, 1, |o0, chunk| {
            let mut row = vec![0.0f32; in_dim];
            for (k, dst) in chunk.iter_mut().enumerate() {
                weight_row(format, bytes, o0 + k, in_dim, &mut row);
                *dst = a.iter().zip(&row).map(|(&ai, &ri)| ai * ri).sum();
            }
        });
    }
    y
}

// RMSNorm: x_i * w_i / sqrt(mean(x^2) + eps). `w` is an F32 weight buffer of length
// `dim`. Returns a fresh vector.
pub(crate) fn rmsnorm(x: &[f32], w: &CpuBuffer, eps: f32) -> Vec<f32> {
    let dim = x.len();
    let ms = x.iter().map(|&v| v * v).sum::<f32>() / dim as f32;
    let inv = 1.0 / (ms + eps).sqrt();
    let wv = w.read_f32();
    x.iter().zip(&wv).map(|(&xi, &wi)| xi * inv * wi).collect()
}

// L2-norm over a slice in place: x_i / sqrt(Σ x_j^2 + eps). Mirrors ggml_l2_norm.
pub(crate) fn l2norm(x: &mut [f32], eps: f32) {
    let ss = x.iter().map(|&v| v * v).sum::<f32>();
    let inv = 1.0 / (ss + eps).sqrt();
    for v in x.iter_mut() {
        *v *= inv;
    }
}

// SiLU: x * sigmoid(x).
pub(crate) fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

// softplus: ln(1 + e^x), numerically stable for large x.
pub(crate) fn softplus(x: f32) -> f32 {
    if x > 20.0 { x } else { (1.0 + x.exp()).ln() }
}

// sigmoid: 1 / (1 + e^-x).
pub(crate) fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rmsnorm_unit_weight_matches_formula() {
        // x = [3,4]; ms = (9+16)/2 = 12.5; inv = 1/sqrt(12.5); w = 1.
        let w = CpuBuffer::from_bytes(
            [1.0f32, 1.0].iter().flat_map(|v| v.to_le_bytes()).collect(),
            CpuFormat::F32,
        );
        let out = rmsnorm(&[3.0, 4.0], &w, 0.0);
        let inv = 1.0 / 12.5f32.sqrt();
        assert!((out[0] - 3.0 * inv).abs() < 1e-6);
        assert!((out[1] - 4.0 * inv).abs() < 1e-6);
    }

    #[test]
    fn l2norm_makes_unit_vector() {
        let mut x = vec![3.0f32, 4.0];
        l2norm(&mut x, 0.0);
        // 3/5, 4/5
        assert!((x[0] - 0.6).abs() < 1e-6);
        assert!((x[1] - 0.8).abs() < 1e-6);
        assert!((x.iter().map(|v| v * v).sum::<f32>() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn silu_softplus_sigmoid_known_values() {
        assert!((silu(0.0) - 0.0).abs() < 1e-6);
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-6);
        assert!((softplus(0.0) - 2.0f32.ln()).abs() < 1e-6);
        assert!((softplus(100.0) - 100.0).abs() < 1e-3);
    }
}
