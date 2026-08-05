/*
 * graph_horizon_engine — Metal argmax dispatch
 * Runs the deterministic GPU reduction synchronously and decodes one bounded index.
 */
use super::super::{
    Device, MetalBuffer,
    exec::{dispatch, encoder::MetalEncoder},
    pipeline::{Kernel, PipelineRegistry},
};
use color_eyre::eyre::{Result, eyre};
pub(crate) fn read(
    d: &Device,
    p: &PipelineRegistry,
    logits: &MetalBuffer,
    out: &MetalBuffer,
    vocab: usize,
) -> Result<u32> {
    let n = u32::try_from(vocab).map_err(|_| eyre!("metal: buffer arithmetic overflow"))?;
    if n == 0 {
        return Err(eyre!("metal: invalid readback size"));
    }
    let width = p.get(Kernel::Argmax).width;
    let e = MetalEncoder::begin(d)?;
    dispatch::encode(
        &e,
        p,
        Kernel::Argmax,
        &[logits, out],
        &n.to_ne_bytes(),
        // Exactly one complete SIMD group cooperates on the deterministic reduction.
        [width, 1, 1],
    )?;
    e.submit()?;
    let b = out.read(4)?;
    Ok(u32::from_ne_bytes(
        b.try_into().expect("four-byte reduction"),
    ))
}
