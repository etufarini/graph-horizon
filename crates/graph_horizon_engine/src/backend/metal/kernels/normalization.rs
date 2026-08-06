/*
 * graph_horizon_engine — Metal RMS normalization dispatch
 * Binds FP32 residual, FP16 scale/output, row shape, and epsilon. It owns no buffers.
 */
use super::super::exec::dispatch;
use super::super::{
    MetalBuffer, MetalEncoder,
    pipeline::{Kernel, PipelineRegistry},
};
use color_eyre::eyre::Result;
// Mirrors the kernel's three buffers, row shape, epsilon, and dispatch owners.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode(
    e: &MetalEncoder,
    p: &PipelineRegistry,
    out: &MetalBuffer,
    x: &MetalBuffer,
    w: &MetalBuffer,
    dim: u32,
    eps: f32,
    rows: u32,
) -> Result<()> {
    let mut c = super::u32s(&[dim]);
    c.extend(eps.to_ne_bytes());
    c.extend(rows.to_ne_bytes());
    dispatch::encode(
        e,
        p,
        Kernel::Rmsnorm,
        &[x, w, out],
        &c,
        [rows as usize, 1, 1],
    )
}
