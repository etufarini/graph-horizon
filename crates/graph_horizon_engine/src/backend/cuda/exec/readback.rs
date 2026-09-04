/*
 * graph_horizon_engine — synchronized, range-checked CUDA logits readback.
 */

use color_eyre::eyre::{Result, bail, eyre};

use super::super::{CudaBuffer, Device};

pub(crate) fn logits(device: &Device, buffer: &CudaBuffer, vocab: usize) -> Result<Vec<f32>> {
    if vocab == 0 {
        bail!("cuda: readback failed");
    }
    let bytes = vocab.checked_mul(4).ok_or_else(failed)?;
    let raw = buffer.read(device, bytes)?;
    if raw.len() != bytes {
        bail!("cuda: readback failed");
    }
    Ok(raw
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("four-byte CUDA logit")))
        .collect())
}

fn failed() -> color_eyre::Report {
    eyre!("cuda: readback failed")
}
