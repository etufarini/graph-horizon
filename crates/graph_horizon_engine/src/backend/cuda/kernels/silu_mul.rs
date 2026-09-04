/*
 * graph_horizon_engine — checked CUDA SiLU-multiply dispatch.
 */

use color_eyre::eyre::Result;

use super::super::exec::dispatch::{self, Arg};
use super::super::module::{Kernel, Module};
use super::super::{CudaBuffer, CudaEncoder, CudaFormat};

pub(crate) fn encode(
    encoder: &CudaEncoder,
    module: &Module,
    out: &CudaBuffer,
    gate: &CudaBuffer,
    up: &CudaBuffer,
    length: u32,
) -> Result<()> {
    let bytes = super::bytes(u64::from(length), 2)?;
    for buffer in [out, gate, up] {
        super::span(buffer, CudaFormat::F16, bytes)?;
    }
    let (grid, block) = dispatch::one_dim(u64::from(length))?;
    dispatch::launch(
        encoder,
        module,
        Kernel::SiluMul,
        &[
            Arg::Buffer(gate, bytes),
            Arg::Buffer(up, bytes),
            Arg::Buffer(out, bytes),
            Arg::U32(length),
        ],
        grid,
        block,
    )
}
