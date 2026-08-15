/*
 * graph_horizon_engine — Vulkan execution-weight predecode
 * Converts one validated Q4_K matrix from canonical GGUF blocks to the exact
 * row-major FP16 values consumed by the experimental Matrix2 prefill path.
 * It performs no I/O, allocation ownership, routing, or model selection.
 */

// AGENTS deroga K: kernel numerico della sola predecodifica Q4_K in FP16.

use color_eyre::eyre::{Result, bail};

use crate::backend::f16::{f16_to_f32, f32_to_f16};

const BLOCK_VALUES: usize = 256;
const BLOCK_BYTES: usize = 144;

pub(crate) fn q4_f16(bytes: &[u8], in_dim: usize, out_dim: usize) -> Result<Vec<u8>> {
    let blocks_per_row = in_dim / BLOCK_VALUES;
    let expected = out_dim
        .checked_mul(blocks_per_row)
        .and_then(|blocks| blocks.checked_mul(BLOCK_BYTES));
    let output_bytes = in_dim
        .checked_mul(out_dim)
        .and_then(|values| values.checked_mul(2));
    if !in_dim.is_multiple_of(BLOCK_VALUES)
        || expected != Some(bytes.len())
        || output_bytes.is_none()
    {
        bail!("vulkan: invalid predecoded weight");
    }

    let mut output = vec![0u8; output_bytes.unwrap()];
    for row in 0..out_dim {
        for block_index in 0..blocks_per_row {
            let block = (row * blocks_per_row + block_index) * BLOCK_BYTES;
            let d = half(bytes, block);
            let dmin = half(bytes, block + 2);
            for pair in 0..4 {
                let (scale_low, min_low) = scale_min(bytes, block + 4, pair * 2);
                let (scale_high, min_high) = scale_min(bytes, block + 4, pair * 2 + 1);
                let scale_low = d * scale_low as f32;
                let min_low = dmin * min_low as f32;
                let scale_high = d * scale_high as f32;
                let min_high = dmin * min_high as f32;
                for value in 0..32 {
                    let packed = bytes[block + 16 + pair * 32 + value];
                    let index = block_index * BLOCK_VALUES + pair * 64 + value;
                    write_half(
                        &mut output,
                        row * in_dim + index,
                        scale_low * f32::from(packed & 0x0f) - min_low,
                    );
                    write_half(
                        &mut output,
                        row * in_dim + index + 32,
                        scale_high * f32::from(packed >> 4) - min_high,
                    );
                }
            }
        }
    }
    Ok(output)
}

fn half(bytes: &[u8], offset: usize) -> f32 {
    f16_to_f32(u16::from_le_bytes([bytes[offset], bytes[offset + 1]]))
}

fn scale_min(bytes: &[u8], offset: usize, group: usize) -> (u32, u32) {
    if group < 4 {
        (
            u32::from(bytes[offset + group] & 63),
            u32::from(bytes[offset + group + 4] & 63),
        )
    } else {
        let high = bytes[offset + group + 4];
        let low = bytes[offset + group - 4];
        (
            u32::from((high & 0x0f) | ((low >> 6) << 4)),
            u32::from((high >> 4) | ((bytes[offset + group] >> 6) << 4)),
        )
    }
}

fn write_half(output: &mut [u8], index: usize, value: f32) {
    let offset = index * 2;
    output[offset..offset + 2].copy_from_slice(&f32_to_f16(value).to_le_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_block_predecodes_to_zeroes() {
        assert_eq!(
            q4_f16(&[0; BLOCK_BYTES], BLOCK_VALUES, 1).unwrap(),
            [0; 512]
        );
    }

    #[test]
    fn malformed_shape_is_rejected() {
        assert!(q4_f16(&[0; BLOCK_BYTES], BLOCK_VALUES - 1, 1).is_err());
    }

    #[cfg(feature = "vulkan-hybrid")]
    #[test]
    fn conversion_matches_cpu_q4_dequant() {
        use crate::backend::cpu::buffer::{CpuFormat, f16_to_f32};
        use crate::backend::cpu::dequant;

        let mut bytes = [0u8; BLOCK_BYTES];
        bytes[..2].copy_from_slice(&f32_to_f16(0.03).to_le_bytes());
        bytes[2..4].copy_from_slice(&f32_to_f16(0.01).to_le_bytes());
        for (index, byte) in bytes[4..].iter_mut().enumerate() {
            *byte = index.wrapping_mul(37) as u8;
        }
        let got = q4_f16(&bytes, BLOCK_VALUES, 1).unwrap();
        let mut expected = [0.0f32; BLOCK_VALUES];
        dequant::dequant_row(CpuFormat::Q4_K, &bytes, 0, BLOCK_VALUES, &mut expected);
        for (index, raw) in got.chunks_exact(2).enumerate() {
            let actual = f16_to_f32(u16::from_le_bytes([raw[0], raw[1]]));
            assert_eq!(
                actual,
                f16_to_f32(f32_to_f16(expected[index])),
                "value {index}"
            );
        }
    }
}
