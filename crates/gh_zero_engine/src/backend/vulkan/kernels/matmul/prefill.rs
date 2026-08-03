/*
 * gh_zero_engine — Vulkan batched matmul dispatch
 * Selects coopmat only under the preserved opt-in capability gate and otherwise
 * records the Q4_K batched fallback. Decode and diagnostics stay outside.
 */

use ash::vk;

use super::trace;
use crate::backend::vulkan::buffers::{GpuBuffer, WeightFormat};
use crate::backend::vulkan::device::Device;
use crate::backend::vulkan::pipeline::{Kernel, PipelineRegistry, dispatch_2d};

fn coopmat_enabled() -> bool {
    use std::sync::OnceLock;
    static FLAG: OnceLock<bool> = OnceLock::new();
    *FLAG.get_or_init(|| {
        matches!(
            std::env::var("GH_ZERO_PREFILL_COOPMAT").ok().as_deref(),
            Some("1") | Some("true") | Some("yes")
        )
    })
}

pub(crate) fn matmul_batched_q4k(
    dev: &Device,
    reg: &PipelineRegistry,
    cmd: vk::CommandBuffer,
    out: &GpuBuffer,
    a: &GpuBuffer,
    w: &GpuBuffer,
    in_dim: u32,
    out_dim: u32,
    n: u32,
) {
    let caps = dev.coopmat;
    if coopmat_enabled()
        && w.quant == WeightFormat::Q4K
        && caps.available
        && caps.m == 16
        && caps.n == 16
        && caps.k == 16
        && in_dim.is_multiple_of(256)
    {
        trace::log_batched_path_once(true);
        super::super::coopmat::dispatch_coopmat(dev, reg, cmd, out, a, w, in_dim, out_dim, n, caps);
        return;
    }
    trace::log_batched_path_once(false);
    let mut push = Vec::with_capacity(12);
    push.extend_from_slice(&in_dim.to_le_bytes());
    push.extend_from_slice(&out_dim.to_le_bytes());
    push.extend_from_slice(&n.to_le_bytes());
    dispatch_2d(
        dev,
        reg,
        cmd,
        Kernel::MatmulQ4KBatchF16Out,
        &[
            (a.buffer, a.offset, a.size),
            (w.buffer, w.offset, w.size),
            (out.buffer, out.offset, out.size),
        ],
        &push,
        out_dim.div_ceil(64),
        n.div_ceil(64),
    );
}
