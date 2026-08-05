/*
 * gh_zero_engine — Metal embedding dispatch
 * Selects the retained weight format and binds one checked token row. It owns no resources.
 */
use super::super::exec::dispatch;
use super::super::{
    MetalBuffer, MetalEncoder,
    pipeline::{Kernel, PipelineRegistry},
};
use color_eyre::eyre::{Result, eyre};
pub(crate) fn encode(
    e: &MetalEncoder,
    p: &PipelineRegistry,
    out: &MetalBuffer,
    w: &MetalBuffer,
    token: u32,
    width: u32,
) -> Result<()> {
    let format = match w.format() {
        super::super::MetalFormat::F16 => 0,
        super::super::MetalFormat::Q4K => 1,
        super::super::MetalFormat::Q5K => 2,
        super::super::MetalFormat::Q6K => 3,
        _ => return Err(eyre!("metal: unsupported weight format")),
    };
    dispatch::encode(
        e,
        p,
        Kernel::Embedding,
        &[w, out],
        &super::u32s(&[token, width, format]),
        [width as usize, 1, 1],
    )
}
