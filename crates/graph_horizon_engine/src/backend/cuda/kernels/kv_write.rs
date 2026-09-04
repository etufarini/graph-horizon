/*
 * graph_horizon_engine — checked CUDA f16/int8 KV write dispatch.
 */

use color_eyre::eyre::Result;

use super::super::exec::dispatch::{self, Arg};
use super::super::module::{Kernel, Module};
use super::super::{CudaBuffer, CudaEncoder, CudaFormat};
use crate::kv_cache::Kv;
use crate::kv_cache::scheme::KvQuant;

#[allow(clippy::too_many_arguments)]
pub(crate) fn encode(
    encoder: &CudaEncoder,
    module: &Module,
    kv: &Kv<CudaBuffer>,
    k: &CudaBuffer,
    v: &CudaBuffer,
    k_payload: u64,
    v_payload: u64,
    k_metadata: u64,
    v_metadata: u64,
    vectors: u32,
) -> Result<()> {
    let dim = u32::try_from(kv.head_dim).map_err(|_| super::arithmetic())?;
    if vectors == 0 || dim == 0 || kv.head_dim != kv.value_dim {
        return Err(super::arithmetic());
    }
    let input_items = u64::from(vectors)
        .checked_mul(u64::from(dim))
        .ok_or_else(super::arithmetic)?;
    let input_bytes = super::bytes(input_items, 2)?;
    super::span(k, CudaFormat::F16, input_bytes)?;
    super::span(v, CudaFormat::F16, input_bytes)?;
    super::span(&kv.k, CudaFormat::Raw, kv.k.len())?;
    super::span(&kv.v, CudaFormat::Raw, kv.v.len())?;

    let (payload_bytes, metadata_bytes) = match kv.scheme {
        KvQuant::F16 => (super::bytes(input_items, 2)?, 0),
        KvQuant::Int8 => (
            super::bytes(input_items, 1)?,
            super::bytes(u64::from(vectors), 4)?,
        ),
    };
    checked_window(k_payload, payload_bytes, kv.k.len())?;
    checked_window(v_payload, payload_bytes, kv.v.len())?;
    if kv.scheme == KvQuant::Int8 {
        checked_window(k_metadata, metadata_bytes, kv.k.len())?;
        checked_window(v_metadata, metadata_bytes, kv.v.len())?;
    }
    let (grid, block) = dispatch::one_dim(u64::from(vectors))?;
    let mut args = vec![
        Arg::Buffer(k, input_bytes),
        Arg::Buffer(v, input_bytes),
        Arg::Buffer(&kv.k, kv.k.len()),
        Arg::Buffer(&kv.v, kv.v.len()),
        Arg::U64(k_payload),
        Arg::U64(v_payload),
    ];
    let kernel = match kv.scheme {
        KvQuant::F16 => Kernel::KvWriteF16,
        KvQuant::Int8 => {
            args.extend([Arg::U64(k_metadata), Arg::U64(v_metadata)]);
            Kernel::KvWriteInt8
        }
    };
    args.extend([Arg::U32(vectors), Arg::U32(dim)]);
    dispatch::launch(encoder, module, kernel, &args, grid, block)
}

fn checked_window(offset: u64, bytes: usize, total: usize) -> Result<()> {
    let end = offset
        .checked_add(u64::try_from(bytes).map_err(|_| super::arithmetic())?)
        .ok_or_else(super::arithmetic)?;
    if end <= u64::try_from(total).map_err(|_| super::arithmetic())? {
        Ok(())
    } else {
        Err(super::arithmetic())
    }
}
