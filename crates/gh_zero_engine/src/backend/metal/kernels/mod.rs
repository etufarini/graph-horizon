/*
 * gh_zero_engine — Metal operation dispatch namespace
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
    use crate::backend::metal::pipeline::PipelineRegistry;
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

        let logits = buffer(&device, &[1., 3., 3., -2.], MetalFormat::F32)?;
        let reduce = MetalBuffer::allocate(&device, 128, MetalFormat::Raw)?;
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
