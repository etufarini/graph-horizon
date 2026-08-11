/*
 * graph_horizon_engine — host-side dispatch of the Q4_K coopmat prefill GEMM
 * Records the FP16-output cooperative-matrix Q4_K prefill GEMM: binds the FP16
 * activations, Q4_K weights, and FP16 output, then pushes the
 * (in_dim, out_dim, n) shape and dispatches a 2D grid of one workgroup per MMA tile sized
 * from the detected CoopmatCaps. It owns no selection logic; routing decides when a coopmat
 * shape and aligned dimensions make this kernel usable, and only then calls here.
*/

#![allow(clippy::too_many_arguments)]

use ash::vk;

use crate::backend::vulkan::buffers::GpuBuffer;
use crate::backend::vulkan::coopmat::CoopmatCaps;
use crate::backend::vulkan::device::Device;
use crate::backend::vulkan::pipeline::{Kernel, PipelineRegistry, dispatch_2d};

// Conservative floor of Vulkan maxComputeWorkGroupCount[0].
const MAX_GROUPS_X: u32 = 65535;

// Dispatches a coopmat GEMM Y[n][out_dim] = A[n][in_dim] · Wᵀ over a 2D grid of MMA tiles:
// x over positions (M = `n`, two `caps.m` tiles sharing one weight tile), y over output
// rows (N = `out_dim`, tile `caps.n`). The retained pipeline uses three bindings
// (A FP16, W Q4_K, Y FP16) and the
// (in_dim,out_dim,n) push. Edge tiles are zero-padded inside the shader, so `n`/`out_dim`
// need not be multiples of the tile; the caller has already checked that the weight is Q4_K,
// in_dim % 256 == 0, and the caps shape matches the kernel's 16×16×16 MMA.
pub(crate) fn dispatch_coopmat(
    dev: &Device,
    reg: &PipelineRegistry,
    cmd: vk::CommandBuffer,
    out: &GpuBuffer,
    a: &GpuBuffer,
    w: &GpuBuffer,
    in_dim: u32,
    out_dim: u32,
    n: u32,
    caps: CoopmatCaps,
) {
    let mut push = Vec::with_capacity(12);
    push.extend_from_slice(&in_dim.to_le_bytes());
    push.extend_from_slice(&out_dim.to_le_bytes());
    push.extend_from_slice(&n.to_le_bytes());
    dispatch_2d(
        dev,
        reg,
        cmd,
        Kernel::MatmulQ4KCoopmatF16Out,
        &[
            (a.buffer, a.offset, a.size),
            (w.buffer, w.offset, w.size),
            (out.buffer, out.offset, out.size),
        ],
        &push,
        n.div_ceil(caps.m * 2).min(MAX_GROUPS_X),
        out_dim.div_ceil(caps.n).min(MAX_GROUPS_X),
    );
}
