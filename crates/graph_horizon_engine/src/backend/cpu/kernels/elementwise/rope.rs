/*
 * graph_horizon_engine — CPU YaRN RoPE kernel
 * Applies Mistral adjacent-pair rotations in place with the role-specific YaRN
 * post-scale and no dispatch or resource ownership.
 */

// AGENTS deroga K: singolo kernel numerico YaRN RoPE.

use color_eyre::eyre::Result;

use crate::backend::cpu::buffer::CpuBuffer;
use crate::backend::rope::{RopeRole, Yarn};

pub(crate) fn rope_yarn(
    x: &CpuBuffer,
    heads: usize,
    head_dim: usize,
    pos: usize,
    yarn: &Yarn,
    role: RopeRole,
) -> Result<()> {
    let mut values = x.read_f16_as_f32();
    let half = yarn.rope_dim / 2;
    let scale = yarn.post_scale(role, pos);
    for head in 0..heads {
        let base = head * head_dim;
        for pair in 0..half {
            let rotation = yarn.pair(role, pair, pos)?;
            let first = base + pair * 2;
            let second = first + 1;
            let a = values[first];
            let b = values[second];
            values[first] = (a * rotation.cos - b * rotation.sin) * scale;
            values[second] = (a * rotation.sin + b * rotation.cos) * scale;
        }
    }
    x.write_f16_from_f32(&values);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::cpu::buffer::CpuFormat;

    #[test]
    fn uses_mistral_adjacent_pairs() {
        let yarn = Yarn {
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
        let buffer = CpuBuffer::zeroed(input.len() * 2, CpuFormat::F16);
        buffer.write_f16_from_f32(&input);

        rope_yarn(&buffer, 1, 4, 1, &yarn, RopeRole::Key).unwrap();

        let got = buffer.read_f16_as_f32();
        for pair in 0..2 {
            let rotation = yarn.pair(RopeRole::Key, pair, 1).unwrap();
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
