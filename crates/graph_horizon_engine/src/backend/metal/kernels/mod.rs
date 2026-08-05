/*
 * graph_horizon_engine — Metal operation dispatch namespace
 * Exports one bounded host wrapper per Backend operation; numeric work stays MSL.
 */
pub(crate) mod argmax;
pub(crate) mod attention;
pub(crate) mod embedding;
pub(crate) mod kv_write;
pub(crate) mod matmul;
pub(crate) mod normalization;
pub(crate) mod residual_add;
pub(crate) mod rope;
pub(crate) mod silu_mul;
pub(crate) mod topk;

pub(super) fn u32s(values: &[u32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect()
}

#[cfg(test)]
mod tests {
    use color_eyre::eyre::Result;

    use super::*;
    use crate::backend::f16::{f16_to_f32, f32_to_f16};
    use crate::backend::metal::pipeline::{Kernel, PipelineRegistry};
    use crate::backend::metal::{Device, MetalBuffer, MetalEncoder, MetalFormat};
    use crate::kv_cache::{
        Kv, layout,
        scheme::{KvQuant, KvRole},
    };

    fn buffer(device: &Device, values: &[f32], format: MetalFormat) -> Result<MetalBuffer> {
        let bytes: Vec<u8> = match format {
            MetalFormat::F32 => values.iter().flat_map(|v| v.to_ne_bytes()).collect(),
            _ => values
                .iter()
                .flat_map(|v| f32_to_f16(*v).to_ne_bytes())
                .collect(),
        };
        let buffer = MetalBuffer::allocate(device, bytes.len() as u64, format)?;
        buffer.write(&bytes)?;
        Ok(buffer)
    }

    fn halfs(buffer: &MetalBuffer, count: usize) -> Result<Vec<f32>> {
        Ok(buffer
            .read(count * 2)?
            .chunks_exact(2)
            .map(|b| f16_to_f32(u16::from_ne_bytes([b[0], b[1]])))
            .collect())
    }

    fn half_at(bytes: &[u8], offset: usize) -> f32 {
        f16_to_f32(u16::from_ne_bytes([bytes[offset], bytes[offset + 1]]))
    }

    fn set_half(bytes: &mut [u8], offset: usize, value: f32) {
        bytes[offset..offset + 2].copy_from_slice(&f32_to_f16(value).to_ne_bytes());
    }

    fn scale_min(bytes: &[u8], base: usize, index: usize) -> (u32, u32) {
        if index < 4 {
            (
                u32::from(bytes[base + index] & 63),
                u32::from(bytes[base + index + 4] & 63),
            )
        } else {
            let high = bytes[base + index + 4];
            let low = bytes[base + index - 4];
            (
                u32::from((high & 15) | ((low >> 6) << 4)),
                u32::from((high >> 4) | ((bytes[base + index] >> 6) << 4)),
            )
        }
    }

    fn quant_value(
        format: MetalFormat,
        bytes: &[u8],
        row: usize,
        index: usize,
        input: usize,
    ) -> f32 {
        let blocks = input / 256;
        let block_index = index / 256;
        let local = index % 256;
        match format {
            MetalFormat::Q4K => {
                let block = (row * blocks + block_index) * 144;
                let group = local / 32;
                let lane = local % 32;
                let (scale, min) = scale_min(bytes, block + 4, group);
                let packed = bytes[block + 16 + (group / 2) * 32 + lane];
                let q = if group & 1 == 0 {
                    packed & 15
                } else {
                    packed >> 4
                };
                half_at(bytes, block) * scale as f32 * f32::from(q)
                    - half_at(bytes, block + 2) * min as f32
            }
            MetalFormat::Q6K => {
                let block = (row * blocks + block_index) * 210;
                let segment = local / 128;
                let category = (local % 128) / 32;
                let lane = local % 32;
                let packed = bytes[block + segment * 64 + (category & 1) * 32 + lane];
                let low = if category < 2 {
                    packed & 15
                } else {
                    packed >> 4
                };
                let high = (bytes[block + 128 + segment * 32 + lane] >> (category * 2)) & 3;
                let scale =
                    bytes[block + 192 + segment * 8 + lane / 16 + category * 2] as i8 as f32;
                half_at(bytes, block + 208) * scale * (f32::from(low | (high << 4)) - 32.)
            }
            _ => unreachable!("test oracle accepts only Q4_K and Q6_K"),
        }
    }

    fn quant_fixture(format: MetalFormat, rows: usize, input: usize) -> Vec<u8> {
        let block_bytes = if format == MetalFormat::Q4K { 144 } else { 210 };
        let mut bytes: Vec<u8> = (0..rows * (input / 256) * block_bytes)
            .map(|index| ((index * 37 + 13) % 251) as u8)
            .collect();
        for block in 0..rows * (input / 256) {
            let base = block * block_bytes;
            if format == MetalFormat::Q4K {
                set_half(&mut bytes, base, 0.0005);
                set_half(&mut bytes, base + 2, 0.0002);
            } else {
                for scale in 0..16 {
                    bytes[base + 192 + scale] = (((block + scale) % 9) as i8 - 4) as u8;
                }
                set_half(&mut bytes, base + 208, 0.0005);
            }
        }
        bytes
    }

    #[test]
    fn f16_operations_run_on_non_multiple_shapes() -> Result<()> {
        let device = Device::acquire()?;
        let pipelines = PipelineRegistry::load(&device)?;
        let weights = buffer(
            &device,
            &[1., 2., 3., 4., 5., 6., 7., 8., 9., 10.],
            MetalFormat::F16,
        )?;
        let embedded = MetalBuffer::allocate(&device, 20, MetalFormat::F32)?;
        let encoder = MetalEncoder::begin(&device)?;
        embedding::encode(&encoder, &pipelines, &embedded, &weights, 1, 5)?;
        encoder.submit()?;
        assert_eq!(
            crate::backend::metal::exec::readback::logits(&embedded, 5)?,
            vec![6., 7., 8., 9., 10.]
        );

        let input = buffer(&device, &[1., 2., 3., 4., 5.], MetalFormat::F16)?;
        let matrix = buffer(&device, &[1.; 15], MetalFormat::F16)?;
        let output = MetalBuffer::allocate(&device, 6, MetalFormat::F16)?;
        let encoder = MetalEncoder::begin(&device)?;
        matmul::encode(&encoder, &pipelines, &output, &input, &matrix, 5, 3, false)?;
        encoder.submit()?;
        assert_eq!(halfs(&output, 3)?, vec![15.; 3]);

        let residual = buffer(&device, &[3., 4., 0., 1., 2., 2.], MetalFormat::F32)?;
        let scale = buffer(&device, &[1., 1., 1.], MetalFormat::F16)?;
        let norm = MetalBuffer::allocate(&device, 12, MetalFormat::F16)?;
        let encoder = MetalEncoder::begin(&device)?;
        normalization::encode(&encoder, &pipelines, &norm, &residual, &scale, 3, 1e-5, 2)?;
        encoder.submit()?;
        assert!(halfs(&norm, 6)?.into_iter().all(f32::is_finite));
        Ok(())
    }

    #[test]
    fn every_quant_projection_handles_two_blocks_and_odd_output() -> Result<()> {
        let device = Device::acquire()?;
        let pipelines = PipelineRegistry::load(&device)?;
        let input = buffer(&device, &[1.; 512], MetalFormat::F16)?;
        for (format, block) in [
            (MetalFormat::Q4K, 144),
            (MetalFormat::Q5K, 176),
            (MetalFormat::Q6K, 210),
        ] {
            let weights = MetalBuffer::allocate(&device, (block * 2 * 3) as u64, format)?;
            weights.write(&vec![0; block * 2 * 3])?;
            let output = MetalBuffer::allocate(&device, 6, MetalFormat::F16)?;
            let encoder = MetalEncoder::begin(&device)?;
            matmul::encode(
                &encoder, &pipelines, &output, &input, &weights, 512, 3, false,
            )?;
            encoder.submit()?;
            assert_eq!(halfs(&output, 3)?, vec![0.; 3]);
        }
        Ok(())
    }

    #[test]
    fn q4_and_q6_projection_match_nonzero_scalar_oracles() -> Result<()> {
        const INPUT: usize = 512;
        const OUTPUT: usize = 3;
        let device = Device::acquire()?;
        let pipelines = PipelineRegistry::load(&device)?;
        let source: Vec<f32> = (0..INPUT)
            .map(|index| (index % 31) as f32 / 16. - 15. / 16.)
            .collect();
        let rounded: Vec<f32> = source
            .iter()
            .map(|value| f16_to_f32(f32_to_f16(*value)))
            .collect();
        let input = buffer(&device, &source, MetalFormat::F16)?;

        for format in [MetalFormat::Q4K, MetalFormat::Q6K] {
            let bytes = quant_fixture(format, OUTPUT, INPUT);
            let weights = MetalBuffer::allocate(&device, bytes.len() as u64, format)?;
            weights.write(&bytes)?;
            let output = MetalBuffer::allocate(&device, (OUTPUT * 4) as u64, MetalFormat::F32)?;
            let encoder = MetalEncoder::begin(&device)?;
            matmul::encode(
                &encoder,
                &pipelines,
                &output,
                &input,
                &weights,
                INPUT as u32,
                OUTPUT as u32,
                true,
            )?;
            encoder.submit()?;

            let actual = crate::backend::metal::exec::readback::logits(&output, OUTPUT)?;
            for (row, got) in actual.into_iter().enumerate() {
                let want = rounded.iter().enumerate().fold(0., |sum, (index, value)| {
                    sum + value * quant_value(format, &bytes, row, index, INPUT)
                });
                let tolerance = 1e-2_f32.max(want.abs() * 1e-3);
                assert!(got.is_finite());
                assert!(
                    (got - want).abs() <= tolerance,
                    "{format:?} row {row}: got {got}, want {want}, tolerance {tolerance}"
                );
            }
        }
        Ok(())
    }

    #[test]
    fn q4_and_q6_batched_projection_matches_sequential_rows() -> Result<()> {
        const INPUT: usize = 512;
        const OUTPUT: usize = 32;
        const ROWS: usize = 3;
        let device = Device::acquire()?;
        let pipelines = PipelineRegistry::load(&device)?;
        let source: Vec<f32> = (0..ROWS * INPUT)
            .map(|index| ((index * 7) % 41) as f32 / 16. - 20. / 16.)
            .collect();
        let input = buffer(&device, &source, MetalFormat::F16)?;

        for format in [MetalFormat::Q4K, MetalFormat::Q6K] {
            let bytes = quant_fixture(format, OUTPUT, INPUT);
            let weights = MetalBuffer::allocate(&device, bytes.len() as u64, format)?;
            weights.write(&bytes)?;
            let batched =
                MetalBuffer::allocate(&device, (ROWS * OUTPUT * 2) as u64, MetalFormat::F16)?;
            let sequential =
                MetalBuffer::allocate(&device, (ROWS * OUTPUT * 2) as u64, MetalFormat::F16)?;

            let encoder = MetalEncoder::begin(&device)?;
            matmul::encode_batched(
                &encoder,
                &pipelines,
                &batched,
                &input,
                &weights,
                INPUT as u32,
                OUTPUT as u32,
                ROWS as u32,
            )?;
            encoder.submit()?;

            let encoder = MetalEncoder::begin(&device)?;
            for row in 0..ROWS {
                matmul::encode(
                    &encoder,
                    &pipelines,
                    &sequential.view((row * OUTPUT * 2) as u64, (OUTPUT * 2) as u64)?,
                    &input.view((row * INPUT * 2) as u64, (INPUT * 2) as u64)?,
                    &weights,
                    INPUT as u32,
                    OUTPUT as u32,
                    false,
                )?;
            }
            encoder.submit()?;

            let batched = halfs(&batched, ROWS * OUTPUT)?;
            let sequential = halfs(&sequential, ROWS * OUTPUT)?;
            for (index, (got, want)) in batched.into_iter().zip(sequential).enumerate() {
                assert!(got.is_finite());
                assert!(
                    (got - want).abs() <= 0.05,
                    "{format:?} value {index}: got {got}, want {want}"
                );
            }
        }
        Ok(())
    }

    #[test]
    fn kv_attention_and_gpu_reductions_are_deterministic() -> Result<()> {
        let device = Device::acquire()?;
        let pipelines = PipelineRegistry::load(&device)?;
        let cache_bytes = layout::buffer_bytes(KvQuant::F16, KvRole::Key, 1, 2, 1, 3);
        let kv = Kv {
            k: MetalBuffer::allocate(&device, cache_bytes, MetalFormat::Raw)?,
            v: MetalBuffer::allocate(&device, cache_bytes, MetalFormat::Raw)?,
            scheme: KvQuant::F16,
            block_count: 1,
            context: 2,
            kv_heads: 1,
            head_dim: 3,
            value_dim: 3,
        };
        let key = buffer(&device, &[1., 0., 0.], MetalFormat::F16)?;
        let value = buffer(&device, &[2., 4., 6.], MetalFormat::F16)?;
        let query = buffer(&device, &[1., 0., 0.], MetalFormat::F16)?;
        let attended = MetalBuffer::allocate(&device, 6, MetalFormat::F16)?;
        let encoder = MetalEncoder::begin(&device)?;
        kv_write::encode(
            &encoder,
            &pipelines,
            &kv,
            &key,
            &value,
            0,
            0,
            cache_bytes,
            cache_bytes,
            1,
        )?;
        attention::encode(&encoder, &pipelines, &attended, &query, &kv, 1, 0, 1, 0)?;
        encoder.submit()?;
        assert_eq!(halfs(&attended, 3)?, vec![2., 4., 6.]);

        let key = buffer(&device, &[0., 1., 0.], MetalFormat::F16)?;
        let value = buffer(&device, &[8., 10., 12.], MetalFormat::F16)?;
        let payload = layout::payload_offset(KvQuant::F16, 0, 1, 1, 3, 2);
        let encoder = MetalEncoder::begin(&device)?;
        kv_write::encode(
            &encoder,
            &pipelines,
            &kv,
            &key,
            &value,
            payload,
            payload,
            cache_bytes,
            cache_bytes,
            1,
        )?;
        attention::encode(&encoder, &pipelines, &attended, &query, &kv, 1, 1, 1, 0)?;
        encoder.submit()?;
        let first_weight = (1.0_f32 / 3.0_f32.sqrt()).exp();
        let denominator = first_weight + 1.0;
        let expected = [
            (first_weight * 2.0 + 8.0) / denominator,
            (first_weight * 4.0 + 10.0) / denominator,
            (first_weight * 6.0 + 12.0) / denominator,
        ];
        for (got, want) in halfs(&attended, 3)?.into_iter().zip(expected) {
            assert!((got - want).abs() <= 0.01, "got {got}, want {want}");
        }

        let reduce = MetalBuffer::allocate(&device, 128, MetalFormat::Raw)?;
        let width = pipelines.get(Kernel::Argmax).width;
        let mut values = vec![-2.; width * 3 + 5];
        values[width - 1] = 3.;
        values[width * 2 + 1] = 3.;
        let logits = buffer(&device, &values, MetalFormat::F32)?;
        assert_eq!(
            argmax::read(&device, &pipelines, &logits, &reduce, values.len())?,
            (width - 1) as u32
        );

        let negative_infinity = vec![f32::NEG_INFINITY; width + 3];
        let logits = buffer(&device, &negative_infinity, MetalFormat::F32)?;
        assert_eq!(
            argmax::read(
                &device,
                &pipelines,
                &logits,
                &reduce,
                negative_infinity.len()
            )?,
            0
        );

        let mut non_finite = vec![f32::NAN; width + 3];
        non_finite[width + 1] = 1.;
        let logits = buffer(&device, &non_finite, MetalFormat::F32)?;
        assert_eq!(
            argmax::read(&device, &pipelines, &logits, &reduce, non_finite.len())?,
            (width + 1) as u32
        );

        let logits = buffer(&device, &[1., 3., 3., -2.], MetalFormat::F32)?;
        assert_eq!(argmax::read(&device, &pipelines, &logits, &reduce, 4)?, 1);
        assert_eq!(
            topk::read(&device, &pipelines, &logits, &reduce, 4, 3)?,
            vec![(1, 3.), (2, 3.), (0, 1.)]
        );
        Ok(())
    }

    #[test]
    fn elementwise_rope_and_int8_zero_vector_edges_are_finite() -> Result<()> {
        use crate::backend::rope::{RopeRole, Yarn};

        let device = Device::acquire()?;
        let pipelines = PipelineRegistry::load(&device)?;
        let gate = buffer(&device, &[0., 1., -1.], MetalFormat::F16)?;
        let up = buffer(&device, &[2., 2., 2.], MetalFormat::F16)?;
        let act = MetalBuffer::allocate(&device, 6, MetalFormat::F16)?;
        let residual = buffer(&device, &[1., 2., 3.], MetalFormat::F32)?;
        let encoder = MetalEncoder::begin(&device)?;
        silu_mul::encode(&encoder, &pipelines, &act, &gate, &up, 3)?;
        residual_add::encode(&encoder, &pipelines, &residual, &act, 3)?;
        encoder.submit()?;
        let result = crate::backend::metal::exec::readback::logits(&residual, 3)?;
        assert!(result.iter().all(|value| value.is_finite()));
        assert!((result[0] - 1.).abs() < 1e-3);

        let yarn = Yarn {
            rope_dim: 4,
            original_context: 32,
            freq_base: 10_000.,
            factor: 1.,
            beta_fast: 32.,
            beta_slow: 1.,
            log_multiplier: 1.,
            q_temperature_scale: 1.,
        };
        let rotated = buffer(&device, &[1., 0., 0., 1.], MetalFormat::F16)?;
        let encoder = MetalEncoder::begin(&device)?;
        rope::encode(
            &encoder,
            &pipelines,
            &rotated,
            1,
            4,
            1,
            &yarn,
            RopeRole::Key,
        )?;
        encoder.submit()?;
        assert!(halfs(&rotated, 4)?.into_iter().all(f32::is_finite));

        let bytes = layout::buffer_bytes(KvQuant::Int8, KvRole::Key, 1, 1, 1, 4);
        let kv = Kv {
            k: MetalBuffer::allocate(&device, bytes, MetalFormat::Raw)?,
            v: MetalBuffer::allocate(&device, bytes, MetalFormat::Raw)?,
            scheme: KvQuant::Int8,
            block_count: 1,
            context: 1,
            kv_heads: 1,
            head_dim: 4,
            value_dim: 4,
        };
        let zero = buffer(&device, &[0.; 4], MetalFormat::F16)?;
        let encoder = MetalEncoder::begin(&device)?;
        kv_write::encode(&encoder, &pipelines, &kv, &zero, &zero, 0, 0, 4, 4, 1)?;
        encoder.submit()?;
        assert_eq!(kv.k.read(bytes as usize)?, vec![0; bytes as usize]);
        Ok(())
    }
}
