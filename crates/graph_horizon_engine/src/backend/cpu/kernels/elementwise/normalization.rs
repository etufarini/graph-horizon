/*
 * graph_horizon_engine — CPU RMSNorm kernel
 * Normalizes FP32 residual rows with FP16 weights and writes FP16 output; it
 * owns no dispatch or buffer lifecycle.
 */

// AGENTS deroga K: singolo kernel numerico RMSNorm per righe indipendenti.

use crate::backend::cpu::buffer::CpuBuffer;
use crate::backend::cpu::parallel;

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
    let mut values = vec![0f32; rows * dim];
    // Each chunk contains whole rows; `row0` is its absolute first row.
    parallel::for_units(&mut values, dim, |row0, chunk| {
        for row in 0..chunk.len() / dim {
            let input = (row0 + row) * dim;
            let output = row * dim;
            let mut sum = 0f32;
            for index in 0..dim {
                let value = x[input + index];
                sum += value * value;
            }
            let inverse = 1.0 / (sum / dim as f32 + eps).sqrt();
            for index in 0..dim {
                chunk[output + index] = x[input + index] * inverse * w[index];
            }
        }
    });
    out.write_f16_from_f32(&values);
}
