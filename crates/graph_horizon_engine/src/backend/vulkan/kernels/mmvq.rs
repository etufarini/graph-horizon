/*
 * graph_horizon_engine — host-side Q4_K MMVQ dispatch
 * Records decode and batched FP16-interface paths: quantize activations into
 * persistent per-8 Q8 scratch, then execute DP4A Q4_K multiplication. This
 * module owns the shared applicability and scratch-capacity gates.
 *
 * Each entry returns `true` only after recording its complete route. Decode
 * accepts DP4A Q4_K devices; batch additionally requires the AMD architecture
 * family and Q4_K. Out-of-scope callers retain their existing kernels.
 * Scratch follows the project's explicit persistent-scratch convention.
*/

#![allow(clippy::too_many_arguments)]

use ash::vk;

use crate::backend::vulkan::buffers::{GpuBuffer, WeightFormat};
use crate::backend::vulkan::device::{AMD_VENDOR_ID, Device};
use crate::backend::vulkan::pipeline::{Kernel, PipelineRegistry, dispatch, dispatch_2d};
use crate::backend::vulkan::{MMVQ_SCRATCH_ELEMENTS, MMVQ_SCRATCH_IN_DIM};

// Conservative floor of Vulkan maxComputeWorkGroupCount[0].
const MAX_GROUPS_X: u32 = 65535;
// Pure form of the complete route gate, kept testable without global state or a
// Vulkan device.
fn applies(dp4a: bool, format: WeightFormat, in_dim: u32) -> bool {
    dp4a && format == WeightFormat::Q4K
        && in_dim.is_multiple_of(256)
        && u64::from(in_dim) <= MMVQ_SCRATCH_IN_DIM
}

fn batch_applies(vendor_id: u32, dp4a: bool, format: WeightFormat, in_dim: u32, n: u32) -> bool {
    vendor_id == AMD_VENDOR_ID
        && dp4a
        && format == WeightFormat::Q4K
        && in_dim.is_multiple_of(256)
        && u64::from(in_dim) <= MMVQ_SCRATCH_IN_DIM
        && n > 1
}

// y[out_dim] = W[out_dim,in_dim] · a[in_dim] via the FP16-interface MMVQ path.
// Out of scope returns false without recording a command.
pub(crate) fn dispatch_mmvq(
    dev: &Device,
    reg: &PipelineRegistry,
    cmd: vk::CommandBuffer,
    out: &GpuBuffer,
    a: &GpuBuffer,
    qs: &GpuBuffer,
    ds: &GpuBuffer,
    w: &GpuBuffer,
    in_dim: u32,
    out_dim: u32,
) -> bool {
    if !applies(dev.dp4a, w.quant, in_dim) {
        return false; // out of scope: caller keeps the float GEMV
    }
    // Quantize A → int8 Q8 (one invocation per 8-wide block). The trailing compute
    // barrier (in `record`) orders the quant ahead of the GEMV that reads its scratch.
    let nblocks = in_dim / 8;
    let qpush = in_dim.to_le_bytes();
    dispatch(
        dev,
        reg,
        cmd,
        Kernel::QuantAQ8F16,
        &[
            (a.buffer, a.offset, a.size),
            (qs.buffer, qs.offset, qs.size),
            (ds.buffer, ds.offset, ds.size),
        ],
        &qpush,
        nblocks.div_ceil(64).min(MAX_GROUPS_X),
    );
    // DP4A Q4_K GEMV: eight lanes per output row, 32 rows per workgroup.
    let mut mpush = Vec::with_capacity(8);
    mpush.extend_from_slice(&in_dim.to_le_bytes());
    mpush.extend_from_slice(&out_dim.to_le_bytes());
    dispatch(
        dev,
        reg,
        cmd,
        Kernel::MatmulQ4KMmvqF16Out,
        &[
            (qs.buffer, qs.offset, qs.size),
            (ds.buffer, ds.offset, ds.size),
            (w.buffer, w.offset, w.size),
            (out.buffer, out.offset, out.size),
        ],
        &mpush,
        out_dim.div_ceil(32).min(MAX_GROUPS_X),
    );
    true
}

// Batched y[n,out_dim] = W[out_dim,in_dim] * a[n,in_dim]. Q8 scratch is
// overwritten for every projection and consumed before the next dispatch.
#[allow(clippy::too_many_arguments)]
pub(crate) fn dispatch_mmq_batched(
    dev: &Device,
    reg: &PipelineRegistry,
    cmd: vk::CommandBuffer,
    out: &GpuBuffer,
    a: &GpuBuffer,
    qs: &GpuBuffer,
    ds: &GpuBuffer,
    w: &GpuBuffer,
    in_dim: u32,
    out_dim: u32,
    n: u32,
) -> bool {
    let Some(elements) = u64::from(in_dim).checked_mul(u64::from(n)) else {
        return false;
    };
    if !batch_applies(dev.vendor_id, dev.dp4a, w.quant, in_dim, n)
        || elements > MMVQ_SCRATCH_ELEMENTS
        || elements > qs.size
        || elements > ds.size
    {
        return false;
    }

    let total = u32::try_from(elements).expect("bounded MMVQ scratch fits u32");
    dispatch(
        dev,
        reg,
        cmd,
        Kernel::QuantAQ8F16,
        &[
            (a.buffer, a.offset, a.size),
            (qs.buffer, qs.offset, elements),
            (ds.buffer, ds.offset, elements),
        ],
        &total.to_le_bytes(),
        (total / 8).div_ceil(64).min(MAX_GROUPS_X),
    );
    let mut push = Vec::with_capacity(12);
    push.extend_from_slice(&in_dim.to_le_bytes());
    push.extend_from_slice(&out_dim.to_le_bytes());
    push.extend_from_slice(&n.to_le_bytes());
    dispatch_2d(
        dev,
        reg,
        cmd,
        Kernel::MatmulQ4KMmqBatchF16Out,
        &[
            (qs.buffer, qs.offset, elements),
            (ds.buffer, ds.offset, elements),
            (w.buffer, w.offset, w.size),
            (out.buffer, out.offset, out.size),
        ],
        &push,
        out_dim.div_ceil(32),
        n.div_ceil(4),
    );
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn complete_route_gate_requires_device_format_shape_and_scratch() {
        assert!(applies(true, WeightFormat::Q4K, 256));
        assert!(!applies(false, WeightFormat::Q4K, 256));
        assert!(!applies(true, WeightFormat::Q5K, 256));
        assert!(!applies(true, WeightFormat::Q4K, 255));
        assert!(!applies(
            true,
            WeightFormat::Q4K,
            MMVQ_SCRATCH_IN_DIM as u32 + 256
        ));
        assert!(batch_applies(
            AMD_VENDOR_ID,
            true,
            WeightFormat::Q4K,
            256,
            2
        ));
        assert!(!batch_applies(0x10de, true, WeightFormat::Q4K, 256, 2));
    }
}
