/*
 * graph_horizon_engine — checked CUDA RMS normalization dispatch.
 */

use color_eyre::eyre::Result;

use super::super::exec::dispatch::{self, Arg};
use super::super::module::{Kernel, Module};
use super::super::{CudaBuffer, CudaEncoder, CudaFormat};

#[allow(clippy::too_many_arguments)]
pub(crate) fn encode(
    encoder: &CudaEncoder,
    module: &Module,
    out: &CudaBuffer,
    input: &CudaBuffer,
    weight: &CudaBuffer,
    width: u32,
    epsilon: f32,
    rows: u32,
) -> Result<()> {
    if rows == 0 || !epsilon.is_finite() || epsilon < 0.0 {
        return Err(super::arithmetic());
    }
    let items = u64::from(width)
        .checked_mul(u64::from(rows))
        .ok_or_else(super::arithmetic)?;
    let input_bytes = super::bytes(items, 4)?;
    let output_bytes = super::bytes(items, 2)?;
    let weight_bytes = super::bytes(u64::from(width), 2)?;
    super::span(input, CudaFormat::F32, input_bytes)?;
    super::span(out, CudaFormat::F16, output_bytes)?;
    super::span(weight, CudaFormat::F16, weight_bytes)?;
    let (grid, block) = dispatch::one_dim(u64::from(rows))?;
    dispatch::launch(
        encoder,
        module,
        Kernel::RmsNorm,
        &[
            Arg::Buffer(input, input_bytes),
            Arg::Buffer(weight, weight_bytes),
            Arg::Buffer(out, output_bytes),
            Arg::U32(width),
            Arg::F32(epsilon),
            Arg::U32(rows),
        ],
        grid,
        block,
    )
}
