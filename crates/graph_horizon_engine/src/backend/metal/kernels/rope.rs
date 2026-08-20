/*
 * graph_horizon_engine — Metal YaRN RoPE dispatch
 * Binds validated YaRN scalars and the explicit Q/K role for in-place rotation.
 */
use super::super::exec::dispatch;
use super::super::{
    MetalBuffer, MetalEncoder,
    pipeline::{Kernel, PipelineRegistry},
};
use crate::backend::rope::{RopeRole, Yarn};
use color_eyre::eyre::{Result, eyre};
// Mirrors the in-place kernel shape, YaRN contract, role, and dispatch owners.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode(
    e: &MetalEncoder,
    p: &PipelineRegistry,
    x: &MetalBuffer,
    heads: u32,
    dim: u32,
    pos: u32,
    rows: u32,
    y: &Yarn,
    role: RopeRole,
) -> Result<()> {
    if rows == 0 {
        return Err(eyre!("rope: empty batch"));
    }
    pos.checked_add(rows - 1)
        .ok_or_else(|| eyre!("rope: position overflow"))?;
    let rope = u32::try_from(y.rope_dim).map_err(|_| eyre!("metal: buffer arithmetic overflow"))?;
    let mut c = super::u32s(&[heads, dim, rope, pos, rows]);
    for value in [
        y.freq_base,
        y.factor,
        y.beta_fast,
        y.beta_slow,
        y.original_context as f32,
        y.post_scale(role, pos as usize),
    ] {
        c.extend(value.to_ne_bytes());
    }
    let grid = rows
        .checked_mul(heads)
        .and_then(|value| value.checked_mul(rope / 2))
        .ok_or_else(|| eyre!("metal: buffer arithmetic overflow"))?;
    dispatch::encode(e, p, Kernel::Rope, &[x], &c, [grid as usize, 1, 1])
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_batched(
    e: &MetalEncoder,
    p: &PipelineRegistry,
    q: &MetalBuffer,
    k: &MetalBuffer,
    q_heads: u32,
    kv_heads: u32,
    dim: u32,
    base: u32,
    rows: u32,
    y: &Yarn,
) -> Result<()> {
    if rows == 0 {
        return Err(eyre!("rope: empty batch"));
    }
    let original = u32::try_from(y.original_context)
        .ok()
        .filter(|value| *value > 0)
        .ok_or_else(|| eyre!("rope: invalid original context"))?;
    base.checked_add(rows - 1)
        .ok_or_else(|| eyre!("rope: position overflow"))?;
    let q_stride = u64::from(q_heads)
        .checked_mul(u64::from(dim))
        .and_then(|elements| elements.checked_mul(2))
        .ok_or_else(|| eyre!("rope: buffer size overflow"))?;
    let k_stride = u64::from(kv_heads)
        .checked_mul(u64::from(dim))
        .and_then(|elements| elements.checked_mul(2))
        .ok_or_else(|| eyre!("rope: buffer size overflow"))?;

    let mut first = 0u32;
    while first < rows {
        let position = base + first;
        // Query post-scale changes only at original-context bucket boundaries.
        let segment = (rows - first).min(original - position % original);
        let q_offset = u64::from(first)
            .checked_mul(q_stride)
            .ok_or_else(|| eyre!("rope: buffer size overflow"))?;
        let k_offset = u64::from(first)
            .checked_mul(k_stride)
            .ok_or_else(|| eyre!("rope: buffer size overflow"))?;
        let q_bytes = u64::from(segment)
            .checked_mul(q_stride)
            .ok_or_else(|| eyre!("rope: buffer size overflow"))?;
        let k_bytes = u64::from(segment)
            .checked_mul(k_stride)
            .ok_or_else(|| eyre!("rope: buffer size overflow"))?;
        let q_rows = q.view(q_offset, q_bytes)?;
        let k_rows = k.view(k_offset, k_bytes)?;

        encode(
            e,
            p,
            &q_rows,
            q_heads,
            dim,
            position,
            segment,
            y,
            RopeRole::Query,
        )?;
        encode(
            e,
            p,
            &k_rows,
            kv_heads,
            dim,
            position,
            segment,
            y,
            RopeRole::Key,
        )?;
        first += segment;
    }
    Ok(())
}
