/*
 * graph_horizon_engine — Metal SiLU-multiply dispatch
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
    let fp32 = g.len() == (n as usize).checked_mul(4).unwrap_or(usize::MAX);
    dispatch::encode(
        e,
        p,
        Kernel::SiluMul,
        &[g, u, out],
        &super::u32s(&[n, u32::from(fp32)]),
        [n as usize, 1, 1],
    )
}
