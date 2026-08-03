/*
 * gh_zero_engine — Metal YaRN RoPE dispatch
 * Binds validated YaRN scalars and the explicit Q/K role for in-place rotation.
 */
use super::super::exec::dispatch;
use super::super::{
    MetalBuffer, MetalEncoder,
    pipeline::{Kernel, PipelineRegistry},
};
use crate::backend::rope::{RopeRole, Yarn};
use color_eyre::eyre::{Result, eyre};
pub(crate) fn encode(
    e: &MetalEncoder,
    p: &PipelineRegistry,
    x: &MetalBuffer,
    heads: u32,
    dim: u32,
    pos: u32,
    y: &Yarn,
    role: RopeRole,
) -> Result<()> {
    let rope = u32::try_from(y.rope_dim).map_err(|_| eyre!("metal: buffer arithmetic overflow"))?;
    let mut c = super::u32s(&[heads, dim, rope, pos]);
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
    let grid = heads
        .checked_mul(rope / 2)
        .ok_or_else(|| eyre!("metal: buffer arithmetic overflow"))?;
    dispatch::encode(e, p, Kernel::Rope, &[x], &c, [grid as usize, 1, 1])
}
