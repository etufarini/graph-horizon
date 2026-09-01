/*
 * graph_horizon_engine — deterministic CUDA argmax dispatch and readback.
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
) -> Result<u32> {
    let length = u32::try_from(vocab).map_err(|_| failed())?;
    if length == 0 {
        return Err(failed());
    }
    let input_bytes = super::bytes(u64::from(length), 4)?;
    super::span(logits, CudaFormat::F32, input_bytes)?;
    super::span(output, CudaFormat::Raw, 4)?;
    let encoder = CudaEncoder::begin(device);
    dispatch::launch(
        &encoder,
        module,
        Kernel::Argmax,
        &[
            Arg::Buffer(logits, input_bytes),
            Arg::Buffer(output, 4),
            Arg::U32(length),
        ],
        (1, 1, 1),
        (1, 1, 1),
    )?;
    encoder.submit()?;
    let bytes = output.read(device, 4)?;
    Ok(u32::from_le_bytes(bytes.try_into().map_err(|_| failed())?))
}

fn failed() -> color_eyre::Report {
    eyre!("cuda: readback failed")
}
