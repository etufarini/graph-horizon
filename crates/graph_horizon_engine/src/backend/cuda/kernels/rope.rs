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
    let rope_dim = u32::try_from(yarn.rope_dim).map_err(|_| super::arithmetic())?;
    let original = u32::try_from(yarn.original_context).map_err(|_| super::arithmetic())?;
    if heads == 0
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
    let items = u64::from(heads)
        .checked_mul(u64::from(head_dim))
        .ok_or_else(super::arithmetic)?;
    let value_bytes = super::bytes(items, 2)?;
    super::span(values, CudaFormat::F16, value_bytes)?;
    let pairs = u64::from(heads)
        .checked_mul(u64::from(rope_dim / 2))
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
