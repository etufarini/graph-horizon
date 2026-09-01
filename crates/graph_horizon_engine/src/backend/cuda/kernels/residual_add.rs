/*
 * graph_horizon_engine — checked CUDA residual-add dispatch.
 */

use color_eyre::eyre::Result;

use super::super::exec::dispatch::{self, Arg};
use super::super::module::{Kernel, Module};
use super::super::{CudaBuffer, CudaEncoder, CudaFormat};

pub(crate) fn encode(
    encoder: &CudaEncoder,
    module: &Module,
    x: &CudaBuffer,
    y: &CudaBuffer,
    length: u32,
) -> Result<()> {
    let x_bytes = super::bytes(u64::from(length), 4)?;
    let y_bytes = super::bytes(u64::from(length), 2)?;
    super::span(x, CudaFormat::F32, x_bytes)?;
    super::span(y, CudaFormat::F16, y_bytes)?;
    let (grid, block) = dispatch::one_dim(u64::from(length))?;
    dispatch::launch(
        encoder,
        module,
        Kernel::ResidualAdd,
        &[
            Arg::Buffer(x, x_bytes),
            Arg::Buffer(y, y_bytes),
            Arg::U32(length),
        ],
        grid,
        block,
    )
}
