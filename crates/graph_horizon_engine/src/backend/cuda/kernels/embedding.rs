/*
 * graph_horizon_engine — checked CUDA embedding dispatch.
 */

use color_eyre::eyre::Result;

use super::super::exec::dispatch::{self, Arg};
use super::super::module::{Kernel, Module};
use super::super::{CudaBuffer, CudaEncoder, CudaFormat};

pub(crate) fn encode(
    encoder: &CudaEncoder,
    module: &Module,
    out: &CudaBuffer,
    weight: &CudaBuffer,
    token: u32,
    width: u32,
) -> Result<()> {
    let out_bytes = super::bytes(u64::from(width), 4)?;
    let rows = token.checked_add(1).ok_or_else(super::arithmetic)?;
    let weight_bytes = super::weight_bytes(weight.format(), width, rows)?;
    super::span(out, CudaFormat::F32, out_bytes)?;
    let format = super::format_code(weight.format())?;
    let (grid, block) = dispatch::one_dim(u64::from(width))?;
    dispatch::launch(
        encoder,
        module,
        Kernel::Embedding,
        &[
            Arg::Buffer(weight, weight_bytes),
            Arg::Buffer(out, out_bytes),
            Arg::U32(token),
            Arg::U32(width),
            Arg::U32(format),
        ],
        grid,
        block,
    )
}
