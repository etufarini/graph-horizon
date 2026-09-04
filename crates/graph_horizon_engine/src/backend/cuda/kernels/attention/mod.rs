/*
 * graph_horizon_engine — shared CUDA causal-attention validation and ABI.
 */

mod decode;
mod prefill;

pub(crate) use decode::encode as decode;
pub(crate) use prefill::encode as prefill;

use color_eyre::eyre::Result;

use super::super::exec::dispatch::Arg;
use super::super::{CudaBuffer, CudaFormat};
use crate::kv_cache::Kv;
use crate::kv_cache::scheme::{KvQuant, KvRole};

fn validate<'a>(
    out: &'a CudaBuffer,
    query: &'a CudaBuffer,
    kv: &'a Kv<CudaBuffer>,
    q_heads: u32,
    base: u32,
    rows: u32,
    layer: u32,
) -> Result<Vec<Arg<'a>>> {
    let dim = u32::try_from(kv.head_dim).map_err(|_| super::arithmetic())?;
    let kv_heads = u32::try_from(kv.kv_heads).map_err(|_| super::arithmetic())?;
    let context = u32::try_from(kv.context).map_err(|_| super::arithmetic())?;
    let blocks = u32::try_from(kv.block_count).map_err(|_| super::arithmetic())?;
    let end = base.checked_add(rows).ok_or_else(super::arithmetic)?;
    if rows == 0
        || dim == 0
        || dim > 256
        || kv.head_dim != kv.value_dim
        || kv_heads == 0
        || q_heads == 0
        || !q_heads.is_multiple_of(kv_heads)
        || layer >= blocks
        || end > context
    {
        return Err(super::arithmetic());
    }
    let query_items = u64::from(rows)
        .checked_mul(u64::from(q_heads))
        .and_then(|value| value.checked_mul(u64::from(dim)))
        .ok_or_else(super::arithmetic)?;
    let query_bytes = super::bytes(query_items, 2)?;
    super::span(query, CudaFormat::F16, query_bytes)?;
    super::span(out, CudaFormat::F16, query_bytes)?;
    let k_bytes = crate::kv_cache::layout::buffer_bytes(
        kv.scheme,
        KvRole::Key,
        kv.block_count,
        kv.context,
        kv.kv_heads,
        kv.head_dim,
    );
    let v_bytes = crate::kv_cache::layout::buffer_bytes(
        kv.scheme,
        KvRole::Value,
        kv.block_count,
        kv.context,
        kv.kv_heads,
        kv.value_dim,
    );
    let k_bytes = usize::try_from(k_bytes).map_err(|_| super::arithmetic())?;
    let v_bytes = usize::try_from(v_bytes).map_err(|_| super::arithmetic())?;
    super::span(&kv.k, CudaFormat::Raw, k_bytes)?;
    super::span(&kv.v, CudaFormat::Raw, v_bytes)?;
    let mut args = vec![
        Arg::Buffer(query, query_bytes),
        Arg::Buffer(&kv.k, k_bytes),
        Arg::Buffer(&kv.v, v_bytes),
        Arg::Buffer(out, query_bytes),
        Arg::U32(dim),
        Arg::U32(kv_heads),
        Arg::U32(q_heads),
        Arg::U32(base),
    ];
    if rows > 1 {
        args.push(Arg::U32(rows));
    }
    args.extend([Arg::U32(layer), Arg::U32(context)]);
    if kv.scheme == KvQuant::Int8 {
        args.extend([
            Arg::U64(kv.meta_base_for(KvRole::Key)),
            Arg::U64(kv.meta_base_for(KvRole::Value)),
        ]);
    }
    Ok(args)
}
