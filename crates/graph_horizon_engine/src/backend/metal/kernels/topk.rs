/*
 * graph_horizon_engine — Metal top-k dispatch
 * Runs the bounded GPU ordering synchronously and decodes only finite result pairs.
 */
use super::super::{
    Device, MetalBuffer,
    exec::{dispatch, encoder::MetalEncoder},
    pipeline::{Kernel, PipelineRegistry},
};
use color_eyre::eyre::{Result, bail, eyre};
pub(crate) fn read(
    d: &Device,
    p: &PipelineRegistry,
    logits: &MetalBuffer,
    out: &MetalBuffer,
    vocab: usize,
    k: usize,
) -> Result<Vec<(u32, f32)>> {
    if vocab == 0 || k == 0 {
        bail!("metal: invalid readback size");
    }
    let count = k.min(vocab);
    let bytes = count
        .checked_mul(8)
        .ok_or_else(|| eyre!("metal: buffer arithmetic overflow"))?;
    if bytes > out.len() {
        bail!("metal: readback past buffer window");
    }
    let v = u32::try_from(vocab).map_err(|_| eyre!("metal: buffer arithmetic overflow"))?;
    let n = u32::try_from(count).map_err(|_| eyre!("metal: buffer arithmetic overflow"))?;
    let e = MetalEncoder::begin(d)?;
    dispatch::encode(
        &e,
        p,
        Kernel::Topk,
        &[logits, out],
        &super::u32s(&[v, n]),
        [1, 1, 1],
    )?;
    e.submit()?;
    let raw = out.read(bytes)?;
    raw.chunks_exact(8)
        .map(|b| {
            let i = u32::from_ne_bytes(b[..4].try_into().unwrap());
            let v = f32::from_ne_bytes(b[4..].try_into().unwrap());
            if !v.is_finite() {
                bail!("metal: non-finite logits");
            }
            Ok((i, v))
        })
        .collect()
}
