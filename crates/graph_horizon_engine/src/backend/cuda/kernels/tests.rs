use color_eyre::eyre::Result;

use super::super::exec::encoder::CudaEncoder;
use super::super::module::Module;
use super::super::{CudaBuffer, CudaFormat, Device};
use crate::backend::f16::{f16_to_f32, f32_to_f16};
use crate::backend::rope::{RopeRole, Yarn};
use crate::kv_cache::Kv;
use crate::kv_cache::scheme::{KvQuant, KvRole};

fn f16_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| f32_to_f16(*value).to_le_bytes())
        .collect()
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn upload_f16(device: &Device, values: &[f32]) -> Result<CudaBuffer> {
    CudaBuffer::upload(device, &f16_bytes(values), CudaFormat::F16)
}

fn upload_f32(device: &Device, values: &[f32]) -> Result<CudaBuffer> {
    CudaBuffer::upload(device, &f32_bytes(values), CudaFormat::F32)
}

fn read_f16(device: &Device, buffer: &CudaBuffer, count: usize) -> Result<Vec<f32>> {
    Ok(buffer
        .read(device, count * 2)?
        .chunks_exact(2)
        .map(|bytes| f16_to_f32(u16::from_le_bytes([bytes[0], bytes[1]])))
        .collect())
}

fn close(actual: f32, expected: f32) {
    let tolerance = 0.02 + 0.02 * expected.abs();
    assert!(
        actual.is_finite() && (actual - expected).abs() <= tolerance,
        "{actual} != {expected} (tol {tolerance})"
    );
}

fn run(device: &Device, encoder: CudaEncoder) -> Result<()> {
    encoder.submit()?;
    device.context.check_err().map_err(Into::into)
}

fn constant_weight(format: CudaFormat, rows: usize) -> Vec<u8> {
    match format {
        CudaFormat::F16 => f16_bytes(&vec![1.0; rows * 256]),
        CudaFormat::Q4K | CudaFormat::Q5K => {
            let block_bytes = if format == CudaFormat::Q4K { 144 } else { 176 };
            let mut output = Vec::with_capacity(rows * block_bytes);
            for _ in 0..rows {
                output.extend_from_slice(&f32_to_f16(1.0).to_le_bytes());
                output.extend_from_slice(&f32_to_f16(0.0).to_le_bytes());
                output.extend_from_slice(&[1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1]);
                if format == CudaFormat::Q5K {
                    output.extend_from_slice(&[0; 32]);
                }
                output.extend_from_slice(&[0x11; 128]);
            }
            output
        }
        CudaFormat::Q6K => {
            let mut output = Vec::with_capacity(rows * 210);
            for _ in 0..rows {
                output.extend_from_slice(&[0x11; 128]);
                output.extend_from_slice(&[0xaa; 64]);
                output.extend_from_slice(&[1; 16]);
                output.extend_from_slice(&f32_to_f16(1.0).to_le_bytes());
            }
            output
        }
        _ => unreachable!(),
    }
}

#[test]
fn dense_operations_cover_f16_q4_q5_q6_and_batch_tails() -> Result<()> {
    let device = Device::acquire()?;
    let module = Module::load(&device.context)?;
    let input = upload_f16(&device, &[1.0; 256])?;
    let batched = upload_f16(&device, &[1.0; 3 * 256])?;
    for format in [
        CudaFormat::F16,
        CudaFormat::Q4K,
        CudaFormat::Q5K,
        CudaFormat::Q6K,
    ] {
        let weight = CudaBuffer::upload(&device, &constant_weight(format, 2), format)?;
        let out = CudaBuffer::allocate(&device, 4, CudaFormat::F16)?;
        let logits = CudaBuffer::allocate(&device, 8, CudaFormat::F32)?;
        let batch_out = CudaBuffer::allocate(&device, 12, CudaFormat::F16)?;
        let embedding = CudaBuffer::allocate(&device, 256 * 4, CudaFormat::F32)?;
        let encoder = CudaEncoder::begin(&device);
        super::matmul::encode(&encoder, &module, &out, &input, &weight, 256, 2, false)?;
        super::matmul::encode(&encoder, &module, &logits, &input, &weight, 256, 2, true)?;
        super::matmul::encode_batched(&encoder, &module, &batch_out, &batched, &weight, 256, 2, 3)?;
        super::embedding::encode(&encoder, &module, &embedding, &weight, 1, 256)?;
        run(&device, encoder)?;
        for value in read_f16(&device, &out, 2)? {
            close(value, 256.0);
        }
        for value in super::super::exec::readback::logits(&device, &logits, 2)? {
            close(value, 256.0);
        }
        for value in read_f16(&device, &batch_out, 6)? {
            close(value, 256.0);
        }
        for value in super::super::exec::readback::logits(&device, &embedding, 256)? {
            close(value, 1.0);
        }
    }
    Ok(())
}

#[test]
fn dense_operations_cover_normalization_rope_and_elementwise() -> Result<()> {
    let device = Device::acquire()?;
    let module = Module::load(&device.context)?;
    let x = upload_f32(&device, &[1.0, 2.0, 3.0, 4.0])?;
    let weight = upload_f16(&device, &[1.0; 4])?;
    let norm = CudaBuffer::allocate(&device, 8, CudaFormat::F16)?;
    let gate = upload_f16(&device, &[-1.0, 0.0, 1.0, 2.0])?;
    let up = upload_f16(&device, &[2.0, 2.0, 2.0, 2.0])?;
    let activated = CudaBuffer::allocate(&device, 8, CudaFormat::F16)?;
    let rotary = upload_f16(&device, &[1.0, 0.0, 0.5, -0.5])?;
    let yarn = Yarn {
        rope_dim: 4,
        original_context: 128,
        freq_base: 10_000.0,
        factor: 2.0,
        beta_fast: 32.0,
        beta_slow: 1.0,
        log_multiplier: 0.1,
        q_temperature_scale: 1.0,
    };
    let expected_pair = yarn.pair(RopeRole::Query, 0, 128)?;
    let post_scale = yarn.post_scale(RopeRole::Query, 128);
    let encoder = CudaEncoder::begin(&device);
    super::normalization::encode(&encoder, &module, &norm, &x, &weight, 4, 1e-5, 1)?;
    super::silu_mul::encode(&encoder, &module, &activated, &gate, &up, 4)?;
    super::residual_add::encode(&encoder, &module, &x, &up, 4)?;
    super::rope::encode(
        &encoder,
        &module,
        &rotary,
        1,
        4,
        128,
        &yarn,
        RopeRole::Query,
    )?;
    run(&device, encoder)?;

    let inverse = (7.5f32 + 1e-5).sqrt().recip();
    for (actual, expected) in read_f16(&device, &norm, 4)?
        .into_iter()
        .zip([1.0, 2.0, 3.0, 4.0])
    {
        close(actual, expected * inverse);
    }
    for (actual, gate) in read_f16(&device, &activated, 4)?
        .into_iter()
        .zip([-1.0f32, 0.0, 1.0, 2.0])
    {
        let rounded = f16_to_f32(f32_to_f16(gate / (1.0 + (-gate).exp())));
        close(actual, rounded * 2.0);
    }
    assert_eq!(
        super::super::exec::readback::logits(&device, &x, 4)?,
        [3.0, 4.0, 5.0, 6.0]
    );
    let rotated = read_f16(&device, &rotary, 4)?;
    close(rotated[0], expected_pair.cos * post_scale);
    close(rotated[1], expected_pair.sin * post_scale);
    Ok(())
}

fn kv(device: &Device, scheme: KvQuant, context: usize, dim: usize) -> Result<Kv<CudaBuffer>> {
    let bytes = crate::kv_cache::layout::buffer_bytes(scheme, KvRole::Key, 1, context, 1, dim);
    Ok(Kv {
        k: CudaBuffer::allocate(device, bytes, CudaFormat::Raw)?,
        v: CudaBuffer::allocate(device, bytes, CudaFormat::Raw)?,
        scheme,
        block_count: 1,
        context,
        kv_heads: 1,
        head_dim: dim,
        value_dim: dim,
    })
}

#[test]
fn kv_f16_and_int8_layouts_are_exact() -> Result<()> {
    let device = Device::acquire()?;
    let module = Module::load(&device.context)?;
    let values = [1.0, -1.0, 0.5, -0.5];
    let input = upload_f16(&device, &values)?;
    for scheme in [KvQuant::F16, KvQuant::Int8] {
        let cache = kv(&device, scheme, 1, 4)?;
        let encoder = CudaEncoder::begin(&device);
        super::kv_write::encode(
            &encoder,
            &module,
            &cache,
            &input,
            &input,
            0,
            0,
            cache.meta_base_for(KvRole::Key),
            cache.meta_base_for(KvRole::Value),
            1,
        )?;
        run(&device, encoder)?;
        match scheme {
            KvQuant::F16 => assert_eq!(cache.k.read(&device, 8)?, f16_bytes(&values)),
            KvQuant::Int8 => {
                let mut expected = [0u8; 4];
                let metadata = crate::kv_cache::int8::quantize_group(&values, &mut expected);
                assert_eq!(cache.k.read(&device, 4)?, expected);
                let view = cache.k.view(cache.meta_base_for(KvRole::Key), 4)?;
                assert_eq!(view.read(&device, 4)?, metadata);
            }
        }
    }
    Ok(())
}

#[test]
fn attention_prefill_is_causal_and_one_row_matches_decode() -> Result<()> {
    let device = Device::acquire()?;
    let module = Module::load(&device.context)?;
    let keys = upload_f16(&device, &[0.0; 6])?;
    let values = upload_f16(&device, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])?;
    let queries = upload_f16(&device, &[0.0; 6])?;
    for scheme in [KvQuant::F16, KvQuant::Int8] {
        let cache = kv(&device, scheme, 3, 2)?;
        let encoder = CudaEncoder::begin(&device);
        super::kv_write::encode(
            &encoder,
            &module,
            &cache,
            &keys,
            &values,
            0,
            0,
            cache.meta_base_for(KvRole::Key),
            cache.meta_base_for(KvRole::Value),
            3,
        )?;
        run(&device, encoder)?;

        let prefill = CudaBuffer::allocate(&device, 12, CudaFormat::F16)?;
        let encoder = CudaEncoder::begin(&device);
        super::attention::prefill(&encoder, &module, &prefill, &queries, &cache, 1, 0, 3, 0)?;
        run(&device, encoder)?;
        for (actual, expected) in read_f16(&device, &prefill, 6)?
            .into_iter()
            .zip([1.0, 2.0, 2.0, 3.0, 3.0, 4.0])
        {
            close(actual, expected);
        }

        let decode = CudaBuffer::allocate(&device, 4, CudaFormat::F16)?;
        let query = queries.view(8, 4)?;
        let encoder = CudaEncoder::begin(&device);
        super::attention::decode(&encoder, &module, &decode, &query, &cache, 1, 2, 0)?;
        run(&device, encoder)?;
        assert_eq!(
            read_f16(&device, &decode, 2)?,
            read_f16(&device, &prefill.view(8, 4)?, 2)?
        );
    }
    Ok(())
}

#[test]
fn reduction_order_matches_rust_total_cmp() -> Result<()> {
    let device = Device::acquire()?;
    let module = Module::load(&device.context)?;
    let values = [
        1.0,
        f32::from_bits(0x7fc0_0001),
        f32::INFINITY,
        f32::INFINITY,
        -0.0,
        0.0,
    ];
    let logits = upload_f32(&device, &values)?;
    let output = CudaBuffer::allocate(&device, 64, CudaFormat::Raw)?;
    assert_eq!(
        super::argmax::read(&device, &module, &logits, &output, values.len())?,
        1
    );
    assert!(super::topk::read(&device, &module, &logits, &output, values.len(), 0)?.is_empty());
    let actual = super::topk::read(&device, &module, &logits, &output, values.len(), 99)?;
    let mut expected = values.into_iter().enumerate().collect::<Vec<_>>();
    expected.sort_by(|(left_i, left), (right_i, right)| {
        right.total_cmp(left).then_with(|| left_i.cmp(right_i))
    });
    for ((actual_index, actual_value), (expected_index, expected_value)) in
        actual.into_iter().zip(expected)
    {
        assert_eq!(actual_index as usize, expected_index);
        assert_eq!(actual_value.to_bits(), expected_value.to_bits());
    }
    Ok(())
}
