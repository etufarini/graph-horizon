/*
 * graph_horizon_engine — Metal residual-add dispatch
 * Binds one FP16 projection into an in-place FP32 residual update.
 */
use super::super::exec::dispatch;
use super::super::{
    MetalBuffer, MetalEncoder,
    pipeline::{Kernel, PipelineRegistry},
};
use color_eyre::eyre::Result;
pub(crate) fn encode(
    e: &MetalEncoder,
    p: &PipelineRegistry,
    x: &MetalBuffer,
    y: &MetalBuffer,
    n: u32,
) -> Result<()> {
    let fp32 = y.len() == (n as usize).checked_mul(4).unwrap_or(usize::MAX);
    dispatch::encode(
        e,
        p,
        Kernel::ResidualAdd,
        &[x, y],
        &super::u32s(&[n, u32::from(fp32)]),
        [n as usize, 1, 1],
    )
}
