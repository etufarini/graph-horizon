/*
 * graph_horizon_engine — deterministic CUDA top-k dispatch and readback.
 */

use color_eyre::eyre::{Result, eyre};

use super::super::exec::dispatch::{self, Arg};
use super::super::module::{Kernel, Module};
use super::super::{CudaBuffer, CudaEncoder, CudaFormat, Device};

pub(crate) fn read(
    device: &Device,
    module: &Module,
    logits: &CudaBuffer,
    output: &CudaBuffer,
    vocab: usize,
    requested: usize,
) -> Result<Vec<(u32, f32)>> {
    let vocab = u32::try_from(vocab).map_err(|_| failed())?;
    let requested = u32::try_from(requested).map_err(|_| failed())?;
    if vocab == 0 || requested == 0 {
        return Ok(Vec::new());
    }
    let count = requested.min(vocab);
    let input_bytes = super::bytes(u64::from(vocab), 4)?;
    let output_bytes = super::bytes(u64::from(count), 8)?;
    super::span(logits, CudaFormat::F32, input_bytes)?;
    super::span(output, CudaFormat::Raw, output_bytes)?;
    let encoder = CudaEncoder::begin(device);
    dispatch::launch(
        &encoder,
        module,
        Kernel::TopK,
        &[
            Arg::Buffer(logits, input_bytes),
            Arg::Buffer(output, output_bytes),
            Arg::U32(vocab),
            Arg::U32(requested),
        ],
        (1, 1, 1),
        (1, 1, 1),
    )?;
    encoder.submit()?;
    let raw = output.read(device, output_bytes)?;
    let count = usize::try_from(count).map_err(|_| failed())?;
    let (indices, values) = raw.split_at(count.checked_mul(4).ok_or_else(failed)?);
    (0..count)
        .map(|index| {
            let start = index * 4;
            let id =
                u32::from_le_bytes(indices[start..start + 4].try_into().map_err(|_| failed())?);
            let value =
                f32::from_le_bytes(values[start..start + 4].try_into().map_err(|_| failed())?);
            Ok((id, value))
        })
        .collect()
}

fn failed() -> color_eyre::Report {
    eyre!("cuda: readback failed")
}
