/*
 * graph_horizon_engine — checked CUDA YaRN rotary dispatch.
 */

use color_eyre::eyre::Result;

use super::super::exec::dispatch::{self, Arg};
use super::super::module::{Kernel, Module};
use super::super::{CudaBuffer, CudaEncoder, CudaFormat};
use crate::backend::rope::{RopeRole, Yarn};

#[allow(clippy::too_many_arguments)]
pub(crate) fn encode(
    encoder: &CudaEncoder,
    module: &Module,
    values: &CudaBuffer,
    heads: u32,
    head_dim: u32,
    position: u32,
    yarn: &Yarn,
    role: RopeRole,
) -> Result<()> {
    encode_rows(
        encoder, module, values, heads, head_dim, position, 1, yarn, role,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_batched(
    encoder: &CudaEncoder,
    module: &Module,
    q: &CudaBuffer,
    k: &CudaBuffer,
    q_heads: u32,
    kv_heads: u32,
    head_dim: u32,
    base: u32,
    rows: u32,
    yarn: &Yarn,
) -> Result<()> {
    let original = u32::try_from(yarn.original_context).map_err(|_| super::arithmetic())?;
    if rows == 0 || original == 0 || base.checked_add(rows - 1).is_none() {
        return Err(super::arithmetic());
    }
    let q_stride = u64::from(q_heads)
        .checked_mul(u64::from(head_dim))
        .and_then(|items| items.checked_mul(2))
        .ok_or_else(super::arithmetic)?;
    let k_stride = u64::from(kv_heads)
        .checked_mul(u64::from(head_dim))
        .and_then(|items| items.checked_mul(2))
        .ok_or_else(super::arithmetic)?;

    let mut first = 0u32;
    while first < rows {
        let position = base.checked_add(first).ok_or_else(super::arithmetic)?;
        // Query post-scale changes only when the original-context bucket changes.
        let segment = (rows - first).min(original - position % original);
        let q_offset = u64::from(first)
            .checked_mul(q_stride)
            .ok_or_else(super::arithmetic)?;
        let k_offset = u64::from(first)
            .checked_mul(k_stride)
            .ok_or_else(super::arithmetic)?;
        let q_bytes = u64::from(segment)
            .checked_mul(q_stride)
            .ok_or_else(super::arithmetic)?;
        let k_bytes = u64::from(segment)
            .checked_mul(k_stride)
            .ok_or_else(super::arithmetic)?;
        let q_rows = q.view(q_offset, q_bytes)?;
        let k_rows = k.view(k_offset, k_bytes)?;
        encode_rows(
            encoder,
            module,
            &q_rows,
            q_heads,
            head_dim,
            position,
            segment,
            yarn,
            RopeRole::Query,
        )?;
        encode_rows(
            encoder,
            module,
            &k_rows,
            kv_heads,
            head_dim,
            position,
            segment,
            yarn,
            RopeRole::Key,
        )?;
        first += segment;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn encode_rows(
    encoder: &CudaEncoder,
    module: &Module,
    values: &CudaBuffer,
    heads: u32,
    head_dim: u32,
    position: u32,
    rows: u32,
    yarn: &Yarn,
    role: RopeRole,
) -> Result<()> {
    let rope_dim = u32::try_from(yarn.rope_dim).map_err(|_| super::arithmetic())?;
    let original = u32::try_from(yarn.original_context).map_err(|_| super::arithmetic())?;
    if heads == 0
        || rows == 0
        || rope_dim == 0
        || !rope_dim.is_multiple_of(2)
        || rope_dim > head_dim
        || original == 0
        || [yarn.freq_base, yarn.factor, yarn.beta_fast, yarn.beta_slow]
            .into_iter()
            .any(|value| !value.is_finite() || value <= 0.0)
    {
        return Err(super::arithmetic());
    }
    if position.checked_add(rows - 1).is_none() {
        return Err(super::arithmetic());
    }
    let items = u64::from(rows)
        .checked_mul(u64::from(heads))
        .and_then(|items| items.checked_mul(u64::from(head_dim)))
        .ok_or_else(super::arithmetic)?;
    let value_bytes = super::bytes(items, 2)?;
    super::span(values, CudaFormat::F16, value_bytes)?;
    let pairs = u64::from(rows)
        .checked_mul(u64::from(heads))
        .and_then(|pairs| pairs.checked_mul(u64::from(rope_dim / 2)))
        .ok_or_else(super::arithmetic)?;
    let (grid, block) = dispatch::one_dim(pairs)?;
    dispatch::launch(
        encoder,
        module,
        Kernel::Rope,
        &[
            Arg::Buffer(values, value_bytes),
            Arg::U32(heads),
            Arg::U32(head_dim),
            Arg::U32(rope_dim),
            Arg::U32(position),
            Arg::U32(rows),
            Arg::F32(yarn.freq_base),
            Arg::F32(yarn.factor),
            Arg::F32(yarn.beta_fast),
            Arg::F32(yarn.beta_slow),
            Arg::F32(original as f32),
            Arg::F32(yarn.post_scale(role, position as usize)),
        ],
        grid,
        block,
    )
}
