/*
 * gh_zero_engine — Metal projection dispatch
 * Selects one retained weight/output format and binds a checked projection grid.
 */
use super::super::exec::dispatch;
use super::super::{
    MetalBuffer, MetalEncoder, MetalFormat,
    pipeline::{Kernel, PipelineRegistry},
};
use color_eyre::eyre::{Result, eyre};
pub(crate) fn encode(
    e: &MetalEncoder,
    p: &PipelineRegistry,
    out: &MetalBuffer,
    a: &MetalBuffer,
    w: &MetalBuffer,
    input: u32,
    output: u32,
    fp32: bool,
) -> Result<()> {
    let format = match w.format() {
        MetalFormat::F16 => 0,
        MetalFormat::Q4K => 1,
        MetalFormat::Q5K => 2,
        MetalFormat::Q6K => 3,
        other => return Err(eyre!("metal: unsupported weight format '{other:?}'")),
    };
    dispatch::encode(
        e,
        p,
        Kernel::Matmul,
        &[a, w, out],
        &super::u32s(&[input, output, format, u32::from(fp32)]),
        [output as usize, 1, 1],
    )
}
