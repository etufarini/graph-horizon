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

fn patterned_weight(format: CudaFormat, rows: usize) -> Vec<u8> {
    let block_bytes = match format {
        CudaFormat::Q4K => 144,
        CudaFormat::Q5K => 176,
        CudaFormat::Q6K => 210,
        _ => unreachable!(),
    };
    let mut output = vec![0u8; rows * block_bytes];
    for row in 0..rows {
        let block = &mut output[row * block_bytes..(row + 1) * block_bytes];
        match format {
            CudaFormat::Q4K | CudaFormat::Q5K => {
                block[..2].copy_from_slice(&f32_to_f16(0.03125).to_le_bytes());
                block[2..4].copy_from_slice(&f32_to_f16(0.015625).to_le_bytes());
                block[4..16].copy_from_slice(&[
                    0x41, 0x82, 0xc3, 0x04, 0x35, 0x76, 0xb7, 0xf8, 0x19, 0x2a, 0x3b, 0x4c,
                ]);
                let (high, quants) = if format == CudaFormat::Q5K {
                    (Some(16..48), 48)
                } else {
                    (None, 16)
                };
                if let Some(high) = high {
                    for (index, byte) in block[high].iter_mut().enumerate() {
                        *byte = (index as u8).wrapping_mul(29).wrapping_add(row as u8);
                    }
                }
                for (index, byte) in block[quants..].iter_mut().enumerate() {
                    *byte = (index as u8)
                        .wrapping_mul(37)
                        .wrapping_add((row * 11) as u8);
                }
            }
            CudaFormat::Q6K => {
                for (index, byte) in block[..192].iter_mut().enumerate() {
                    *byte = (index as u8).wrapping_mul(23).wrapping_add((row * 7) as u8);
                }
                for (index, byte) in block[192..208].iter_mut().enumerate() {
                    *byte = (index as i8 - 8 + row as i8) as u8;
                }
                block[208..].copy_from_slice(&f32_to_f16(0.03125).to_le_bytes());
            }
            _ => unreachable!(),
        }
    }
    output
}

fn reference_weight(format: CudaFormat, bytes: &[u8], row: usize, index: usize) -> f32 {
    let half = |offset: usize| f16_to_f32(u16::from_le_bytes([bytes[offset], bytes[offset + 1]]));
    match format {
        CudaFormat::Q4K | CudaFormat::Q5K => {
            let block_bytes = if format == CudaFormat::Q4K { 144 } else { 176 };
            let block = row * block_bytes;
            let group = index / 64;
            let lane = index % 64;
            let scale_index = group * 2 + lane / 32;
            let scales = &bytes[block + 4..block + 16];
            let (scale, minimum) = if scale_index < 4 {
                (scales[scale_index] & 63, scales[scale_index + 4] & 63)
            } else {
                (
                    (scales[scale_index + 4] & 15) | ((scales[scale_index - 4] >> 6) << 4),
                    (scales[scale_index + 4] >> 4) | ((scales[scale_index] >> 6) << 4),
                )
            };
            let quant_base = block + if format == CudaFormat::Q4K { 16 } else { 48 };
            let packed = bytes[quant_base + group * 32 + lane % 32];
            let mut quant = if lane < 32 { packed & 15 } else { packed >> 4 };
            if format == CudaFormat::Q5K {
                quant += ((bytes[block + 16 + lane % 32] >> (group * 2 + lane / 32)) & 1) * 16;
            }
            half(block) * f32::from(scale) * f32::from(quant) - half(block + 2) * f32::from(minimum)
        }
        CudaFormat::Q6K => {
            let block = row * 210;
            let segment = index / 128;
            let category = (index % 128) / 32;
            let lane = index % 32;
            let packed = bytes[block + segment * 64 + (category & 1) * 32 + lane];
            let low = if category < 2 {
                packed & 15
            } else {
                packed >> 4
            };
            let high = (bytes[block + 128 + segment * 32 + lane] >> (category * 2)) & 3;
            let scale = bytes[block + 192 + segment * 8 + lane / 16 + category * 2] as i8;
            half(block + 208) * f32::from(scale) * (f32::from(low | (high << 4)) - 32.0)
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
fn dense_operations_match_packed_quant_reference_with_signed_values() -> Result<()> {
    let device = Device::acquire()?;
    let module = Module::load(&device.context)?;
    let input_values = (0..256)
        .map(|index| [0.0, -0.5, 0.25, 1.0, -1.0][index % 5])
        .collect::<Vec<_>>();
    let input = upload_f16(&device, &input_values)?;
    for format in [CudaFormat::Q4K, CudaFormat::Q5K, CudaFormat::Q6K] {
        let raw = patterned_weight(format, 2);
        let weight = CudaBuffer::upload(&device, &raw, format)?;
        let output = CudaBuffer::allocate(&device, 4, CudaFormat::F16)?;
        let encoder = CudaEncoder::begin(&device);
        super::matmul::encode(&encoder, &module, &output, &input, &weight, 256, 2, false)?;
        run(&device, encoder)?;
        for (row, actual) in read_f16(&device, &output, 2)?.into_iter().enumerate() {
            let expected = input_values
                .iter()
                .enumerate()
                .map(|(index, input)| input * reference_weight(format, &raw, row, index))
                .sum();
            close(actual, expected);
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
    let rotary_key = upload_f16(&device, &[1.0, 0.0, 0.5, -0.5])?;
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
    let expected_key_pair = yarn.pair(RopeRole::Key, 0, 128)?;
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
    super::rope::encode(
        &encoder,
        &module,
        &rotary_key,
        1,
        4,
        128,
        &yarn,
        RopeRole::Key,
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
    let rotated_key = read_f16(&device, &rotary_key, 4)?;
    close(rotated_key[0], expected_key_pair.cos);
    close(rotated_key[1], expected_key_pair.sin);
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
fn kv_int8_varied_vectors_match_the_shared_reference_exactly() -> Result<()> {
    let device = Device::acquire()?;
    let module = Module::load(&device.context)?;
    let values = (0..4 * 128)
        .map(|index| {
            let value = ((index * 73 + 19) % 257) as f32 / 17.0 - 7.5;
            f16_to_f32(f32_to_f16(value))
        })
        .collect::<Vec<_>>();
    let input = upload_f16(&device, &values)?;
    let cache = kv(&device, KvQuant::Int8, 4, 128)?;
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
        4,
    )?;
    run(&device, encoder)?;

    let mut expected_payload = vec![0u8; values.len()];
    let mut expected_metadata = Vec::with_capacity(16);
    for (source, payload) in values
        .chunks_exact(128)
        .zip(expected_payload.chunks_exact_mut(128))
    {
        expected_metadata.extend(crate::kv_cache::int8::quantize_group(source, payload));
    }
    assert_eq!(cache.k.read(&device, values.len())?, expected_payload);
    assert_eq!(
        cache
            .k
            .view(cache.meta_base_for(KvRole::Key), 16)?
            .read(&device, 16)?,
        expected_metadata
    );
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

fn stored_kv_vector(scheme: KvQuant, values: &[f32]) -> Vec<f32> {
    if scheme == KvQuant::F16 {
        return values.to_vec();
    }
    let mut payload = vec![0; values.len()];
    let metadata = crate::kv_cache::int8::quantize_group(values, &mut payload);
    let (minimum, scale) = crate::kv_cache::int8::meta_decode(&metadata);
    payload
        .into_iter()
        .map(|code| crate::kv_cache::int8::dequant(code, minimum, scale))
        .collect()
}

#[test]
fn attention_gqa_scores_match_reference_for_both_kv_schemes() -> Result<()> {
    const DIM: usize = 128;
    const Q_HEADS: usize = 2;
    const ROWS: usize = 2;
    let device = Device::acquire()?;
    let module = Module::load(&device.context)?;
    let mut keys = vec![0.0f32; ROWS * DIM];
    keys[0] = 1.0;
    keys[DIM + 1] = 2.0;
    let values = (0..ROWS * DIM)
        .map(|index| ((index * 11 + 3) % 31) as f32 / 8.0 - 2.0)
        .collect::<Vec<_>>();
    let mut queries = vec![0.0f32; ROWS * Q_HEADS * DIM];
    for row in 0..ROWS {
        queries[(row * Q_HEADS) * DIM] = 1.0;
        queries[(row * Q_HEADS + 1) * DIM + 1] = 1.0;
    }
    let key_input = upload_f16(&device, &keys)?;
    let value_input = upload_f16(&device, &values)?;
    let query_input = upload_f16(&device, &queries)?;

    for scheme in [KvQuant::F16, KvQuant::Int8] {
        let cache = kv(&device, scheme, ROWS, DIM)?;
        let encoder = CudaEncoder::begin(&device);
        super::kv_write::encode(
            &encoder,
            &module,
            &cache,
            &key_input,
            &value_input,
            0,
            0,
            cache.meta_base_for(KvRole::Key),
            cache.meta_base_for(KvRole::Value),
            ROWS as u32,
        )?;
        run(&device, encoder)?;

        let output =
            CudaBuffer::allocate(&device, (ROWS * Q_HEADS * DIM * 2) as u64, CudaFormat::F16)?;
        let encoder = CudaEncoder::begin(&device);
        super::attention::prefill(
            &encoder,
            &module,
            &output,
            &query_input,
            &cache,
            Q_HEADS as u32,
            0,
            ROWS as u32,
            0,
        )?;
        run(&device, encoder)?;

        let stored_keys = keys
            .chunks_exact(DIM)
            .map(|vector| stored_kv_vector(scheme, vector))
            .collect::<Vec<_>>();
        let stored_values = values
            .chunks_exact(DIM)
            .map(|vector| stored_kv_vector(scheme, vector))
            .collect::<Vec<_>>();
        let scale = (DIM as f32).sqrt().recip();
        let mut expected = Vec::with_capacity(ROWS * Q_HEADS * DIM);
        for row in 0..ROWS {
            for head in 0..Q_HEADS {
                let query = &queries[(row * Q_HEADS + head) * DIM..][..DIM];
                let scores = stored_keys[..=row]
                    .iter()
                    .map(|key| {
                        query
                            .iter()
                            .zip(key)
                            .map(|(left, right)| left * right)
                            .sum::<f32>()
                            * scale
                    })
                    .collect::<Vec<_>>();
                let maximum = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let weights = scores
                    .iter()
                    .map(|score| (score - maximum).exp())
                    .collect::<Vec<_>>();
                let denominator = weights.iter().sum::<f32>();
                for dimension in 0..DIM {
                    expected.push(
                        weights
                            .iter()
                            .zip(&stored_values)
                            .map(|(weight, value)| weight * value[dimension])
                            .sum::<f32>()
                            / denominator,
                    );
                }
            }
        }
        for (actual, expected) in read_f16(&device, &output, expected.len())?
            .into_iter()
            .zip(expected)
        {
            close(actual, expected);
        }

        let row_bytes = (Q_HEADS * DIM * 2) as u64;
        let decode = CudaBuffer::allocate(&device, row_bytes, CudaFormat::F16)?;
        let query = query_input.view(row_bytes, row_bytes)?;
        let encoder = CudaEncoder::begin(&device);
        super::attention::decode(
            &encoder,
            &module,
            &decode,
            &query,
            &cache,
            Q_HEADS as u32,
            1,
            0,
        )?;
        run(&device, encoder)?;
        assert_eq!(
            read_f16(&device, &decode, Q_HEADS * DIM)?,
            read_f16(&device, &output.view(row_bytes, row_bytes)?, Q_HEADS * DIM)?
        );
    }
    Ok(())
}

#[test]
fn attention_rejects_empty_overflow_gqa_and_context_before_submission() -> Result<()> {
    let device = Device::acquire()?;
    let module = Module::load(&device.context)?;
    let query = CudaBuffer::allocate(&device, 4, CudaFormat::F16)?;
    let output = CudaBuffer::allocate(&device, 4, CudaFormat::F16)?;
    let cache = kv(&device, KvQuant::F16, 1, 2)?;
    let invalid_gqa = Kv {
        k: CudaBuffer::allocate(&device, 8, CudaFormat::Raw)?,
        v: CudaBuffer::allocate(&device, 8, CudaFormat::Raw)?,
        scheme: KvQuant::F16,
        block_count: 1,
        context: 1,
        kv_heads: 2,
        head_dim: 2,
        value_dim: 2,
    };
    let encoder = CudaEncoder::begin(&device);
    for result in [
        super::attention::prefill(&encoder, &module, &output, &query, &cache, 1, 0, 0, 0),
        super::attention::prefill(
            &encoder,
            &module,
            &output,
            &query,
            &cache,
            1,
            u32::MAX,
            2,
            0,
        ),
        super::attention::decode(&encoder, &module, &output, &query, &cache, 1, 1, 0),
        super::attention::decode(&encoder, &module, &output, &query, &invalid_gqa, 3, 0, 0),
    ] {
        assert_eq!(
            result.unwrap_err().to_string(),
            "cuda: buffer arithmetic overflow"
        );
    }
    encoder.submit()
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
    let first = super::topk::read(&device, &module, &logits, &output, values.len(), 1)?;
    assert_eq!(first.len(), 1);
    assert_eq!(first[0].0, 1);
    assert_eq!(first[0].1.to_bits(), values[1].to_bits());
    let at_vocab = super::topk::read(
        &device,
        &module,
        &logits,
        &output,
        values.len(),
        values.len(),
    )?;
    let over_vocab = super::topk::read(&device, &module, &logits, &output, values.len(), 99)?;
    assert_eq!(
        at_vocab
            .into_iter()
            .map(|(index, value)| (index, value.to_bits()))
            .collect::<Vec<_>>(),
        over_vocab
            .iter()
            .map(|(index, value)| (*index, value.to_bits()))
            .collect::<Vec<_>>(),
    );
    let actual = over_vocab;
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
    let tied = upload_f32(&device, &[5.0, 5.0, -1.0])?;
    assert_eq!(super::argmax::read(&device, &module, &tied, &output, 3)?, 0);
    Ok(())
}
