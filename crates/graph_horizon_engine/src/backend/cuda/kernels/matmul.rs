/*
 * graph_horizon_engine — checked CUDA projection and logits dispatch.
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
    input_width: u32,
    output_width: u32,
    logits: bool,
) -> Result<()> {
    let input_bytes = super::bytes(u64::from(input_width), 2)?;
    let output_bytes = super::bytes(u64::from(output_width), if logits { 4 } else { 2 })?;
    super::span(input, CudaFormat::F16, input_bytes)?;
    super::span(
        out,
        if logits {
            CudaFormat::F32
        } else {
            CudaFormat::F16
        },
        output_bytes,
    )?;
    let weight_bytes = super::weight_bytes(weight.format(), input_width, output_width)?;
    let format = super::format_code(weight.format())?;
    if output_width == 0 {
        return Err(super::arithmetic());
    }
    dispatch::launch(
        encoder,
        module,
        if logits {
            Kernel::Logits
        } else {
            Kernel::Matmul
        },
        &[
            Arg::Buffer(input, input_bytes),
            Arg::Buffer(weight, weight_bytes),
            Arg::Buffer(out, output_bytes),
            Arg::U32(input_width),
            Arg::U32(output_width),
            Arg::U32(format),
        ],
        (output_width, 1, 1),
        (if format == 0 { 128 } else { 64 }, 1, 1),
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_batched(
    encoder: &CudaEncoder,
    module: &Module,
    out: &CudaBuffer,
    input: &CudaBuffer,
    weight: &CudaBuffer,
    input_width: u32,
    output_width: u32,
    rows: u32,
) -> Result<()> {
    let input_items = u64::from(input_width)
        .checked_mul(u64::from(rows))
        .ok_or_else(super::arithmetic)?;
    let output_items = u64::from(output_width)
        .checked_mul(u64::from(rows))
        .ok_or_else(super::arithmetic)?;
    let input_bytes = super::bytes(input_items, 2)?;
    let output_bytes = super::bytes(output_items, 2)?;
    super::span(input, CudaFormat::F16, input_bytes)?;
    super::span(out, CudaFormat::F16, output_bytes)?;
    let weight_bytes = super::weight_bytes(weight.format(), input_width, output_width)?;
    let format = super::format_code(weight.format())?;
    if output_width == 0 || rows == 0 {
        return Err(super::arithmetic());
    }
    if rows == 1 {
        return encode(
            encoder,
            module,
            out,
            input,
            weight,
            input_width,
            output_width,
            false,
        );
    }
    let token_groups = rows.checked_add(3).ok_or_else(super::arithmetic)? / 4;
    // Large output grids amortize M32 staging; smaller grids retain M16 parallelism.
    let (kernel, grid) = if format != 0 && rows >= 32 && output_width >= 8192 {
        (
            Kernel::MatmulTensorWide,
            (output_width.div_ceil(64), rows.div_ceil(32), 1),
        )
    } else if format != 0 && rows >= 16 {
        (
            Kernel::MatmulTensor,
            (output_width.div_ceil(64), rows.div_ceil(16), 1),
        )
    } else {
        (Kernel::MatmulBatched, (output_width, token_groups, 1))
    };
    dispatch::launch(
        encoder,
        module,
        kernel,
        &[
            Arg::Buffer(input, input_bytes),
            Arg::Buffer(weight, weight_bytes),
            Arg::Buffer(out, output_bytes),
            Arg::U32(input_width),
            Arg::U32(output_width),
            Arg::U32(rows),
            Arg::U32(format),
        ],
        grid,
        (128, 1, 1),
    )
}
