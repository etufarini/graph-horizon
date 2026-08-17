/*
 * graph_horizon_engine — host-side dispatch of cooperative prefill GEMMs
 * Records an FP16-output cooperative-matrix prefill GEMM: binds the FP16
 * activations, retained quantized weights, and FP16 output, then pushes the
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
// (A FP16, W quantized, Y FP16) and the
// (in_dim,out_dim,n) push. Edge tiles are zero-padded inside the shader, so `n`/`out_dim`
// need not be multiples of the tile; the caller has already checked the weight format,
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
    kernel: Kernel,
) {
    debug_assert!(matches!(
        kernel,
        Kernel::MatmulQ4KCoopmatF16Out
            | Kernel::MatmulQ4KCoopmatMetadataF16Out
            | Kernel::MatmulQ6KCoopmatF16Out
    ));
    let mut push = Vec::with_capacity(12);
    push.extend_from_slice(&in_dim.to_le_bytes());
    push.extend_from_slice(&out_dim.to_le_bytes());
    push.extend_from_slice(&n.to_le_bytes());
    dispatch_2d(
        dev,
        reg,
        cmd,
        kernel,
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

// Dispatches a fixed matrix2 matmul. Q4 owns 128 prompt rows; Q6 and direct
// FP16 own 64. Every workgroup owns 32 output rows, with shader-guarded tails.
pub(crate) fn dispatch_matrix2(
    dev: &Device,
    reg: &PipelineRegistry,
    cmd: vk::CommandBuffer,
    out: &GpuBuffer,
    a: &GpuBuffer,
    w: &GpuBuffer,
    in_dim: u32,
    out_dim: u32,
    n: u32,
    kernel: Kernel,
) {
    debug_assert!(matches!(
        kernel,
        Kernel::MatmulQ4KMatrix2F16Out
            | Kernel::MatmulQ6KMatrix2F16Out
            | Kernel::MatmulF16Matrix2F16Out
    ));
    let mut push = Vec::with_capacity(12);
    push.extend_from_slice(&in_dim.to_le_bytes());
    push.extend_from_slice(&out_dim.to_le_bytes());
    push.extend_from_slice(&n.to_le_bytes());
    let (weight_offset, weight_size) = if kernel == Kernel::MatmulF16Matrix2F16Out {
        let native = w.native_offset.expect("native Matrix2 weight offset");
        (w.offset + native, w.size - native)
    } else {
        (w.offset, w.size)
    };
    if kernel == Kernel::MatmulQ4KMatrix2F16Out {
        dispatch_2d(
            dev,
            reg,
            cmd,
            kernel,
            &[
                (w.buffer, weight_offset, weight_size),
                (a.buffer, a.offset, a.size),
                (out.buffer, out.offset, out.size),
            ],
            &push,
            out_dim.div_ceil(256).min(MAX_GROUPS_X),
            n.div_ceil(128).min(MAX_GROUPS_X),
        );
        return;
    }
    dispatch_2d(
        dev,
        reg,
        cmd,
        kernel,
        &[
            (a.buffer, a.offset, a.size),
            (w.buffer, weight_offset, weight_size),
            (out.buffer, out.offset, out.size),
        ],
        &push,
        n.div_ceil(64).min(MAX_GROUPS_X),
        out_dim.div_ceil(32).min(MAX_GROUPS_X),
    );
}
