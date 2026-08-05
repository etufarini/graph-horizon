/*
 * graph_orizon_engine — Metal residual-add dispatch
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
    dispatch::encode(
        e,
        p,
        Kernel::ResidualAdd,
        &[x, y],
        &n.to_ne_bytes(),
        [n as usize, 1, 1],
    )
}
