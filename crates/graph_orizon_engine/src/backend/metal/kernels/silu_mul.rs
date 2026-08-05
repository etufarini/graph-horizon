/*
 * graph_orizon_engine — Metal SiLU-multiply dispatch
 * Binds gate, up, and output windows for the FP16-rounded fused operation.
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
    out: &MetalBuffer,
    g: &MetalBuffer,
    u: &MetalBuffer,
    n: u32,
) -> Result<()> {
    dispatch::encode(
        e,
        p,
        Kernel::SiluMul,
        &[g, u, out],
        &n.to_ne_bytes(),
        [n as usize, 1, 1],
    )
}
