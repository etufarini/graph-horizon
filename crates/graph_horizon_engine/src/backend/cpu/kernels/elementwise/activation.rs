/*
 * graph_horizon_engine — CPU fused activation kernel
 * Computes SiLU(gate) multiplied by up in FP16, including the intermediate FP16
 * rounding required for byte identity with the sequential reference.
 */

// AGENTS deroga K: singolo kernel numerico fused SiLU-mul.

use crate::backend::cpu::buffer::{CpuBuffer, f16_to_f32, f32_to_f16};
use crate::backend::cpu::parallel;

fn round_f16(value: f32) -> f32 {
    f16_to_f32(f32_to_f16(value))
}

pub(crate) fn silu_mul(out: &CpuBuffer, gate: &CpuBuffer, up: &CpuBuffer, n: usize) {
    let gate = gate.read_f16_as_f32();
    let up = up.read_f16_as_f32();
    let mut values = vec![0f32; n];
    // `start` is the absolute element index, preserving independent element order.
    parallel::for_units(&mut values, 1, |start, chunk| {
        for (offset, output) in chunk.iter_mut().enumerate() {
            let value = gate[start + offset];
            let silu = value / (1.0 + (-value).exp());
            *output = round_f16(silu) * up[start + offset];
        }
    });
    out.write_f16_from_f32(&values);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::cpu::buffer::CpuFormat;

    fn values(n: usize, seed: u32) -> Vec<f32> {
        let mut state = seed;
        (0..n)
            .map(|_| {
                state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                (state >> 8) as f32 / (1u32 << 24) as f32 * 4.0 - 2.0
            })
            .collect()
    }

    fn buffer(values: &[f32]) -> CpuBuffer {
        let buffer = CpuBuffer::zeroed(values.len() * 2, CpuFormat::F16);
        buffer.write_f16_from_f32(values);
        buffer
    }

    #[test]
    fn matches_sequential_fp16_rounding() {
        let n = 64;
        let gate = buffer(&values(n, 1));
        let up = buffer(&values(n, 2));
        let expected = gate
            .read_f16_as_f32()
            .into_iter()
            .zip(up.read_f16_as_f32())
            .map(|(gate, up)| {
                let silu = gate / (1.0 + (-gate).exp());
                round_f16(silu) * up
            })
            .collect::<Vec<_>>();
        let out = CpuBuffer::zeroed(n * 2, CpuFormat::F16);

        silu_mul(&out, &gate, &up, n);

        let expected = buffer(&expected);
        assert_eq!(
            out.bytes()[out.window()],
            expected.bytes()[expected.window()]
        );
    }
}
