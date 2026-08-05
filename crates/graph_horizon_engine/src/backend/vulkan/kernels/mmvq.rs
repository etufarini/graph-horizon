/*
 * graph_horizon_engine — host-side dispatch of the mmvq Q4_K decode GEMV
 * Records the retained two-step FP16 decode path: quantize FP16 activations to
 * Q8_1 scratch, then execute the DP4A Q4_K GEMV into FP16 output. The caller
 * supplies persistent scratch; this module owns the complete applicability gate.
 *
 * Returns `true` when it dispatched the mmvq path, `false` when out of scope (no dp4a
 * feature, weight ≠ Q4_K, or in_dim not a multiple of 256) so the caller keeps the existing
 * float GEMV. The scratch follows the project's explicit-scratch convention (cf.
 * `dispatch_tiled_f16`'s f16 A scratch); a no-scratch signature could not source the
 * persistent buffers required by the kernel.
*/

#![allow(clippy::too_many_arguments)]

use ash::vk;

use crate::backend::vulkan::MMVQ_SCRATCH_IN_DIM;
use crate::backend::vulkan::buffers::{GpuBuffer, WeightFormat};
use crate::backend::vulkan::device::Device;
use crate::backend::vulkan::pipeline::{Kernel, PipelineRegistry, dispatch};

// Conservative floor of Vulkan maxComputeWorkGroupCount[0].
const MAX_GROUPS_X: u32 = 65535;

// Whether to route dense decode GEMV through the MMVQ int8/DP4A path. Opt-in via
// `GRAPH_HORIZON_DECODE_MMVQ=1` (read once) while it is measured against the float GEMV: the
// mmvq path changes the decode numerics (int8 Q8_1 activations), so — like the prefill f16
// paths — it stays off by default until greedy-parity + tok/s measurements confirm it
// on real hardware. The kernel itself is validated by `mmvq_q4k_matches_cpu_oracle`.
pub(crate) fn decode_mmvq_enabled() -> bool {
    use std::sync::OnceLock;
    static FLAG: OnceLock<bool> = OnceLock::new();
    *FLAG.get_or_init(|| {
        matches!(
            std::env::var("GRAPH_HORIZON_DECODE_MMVQ").ok().as_deref(),
            Some("1") | Some("true") | Some("yes")
        )
    })
}

// Pure form of the complete route gate, kept testable without global state or a
// Vulkan device.
fn applies(enabled: bool, dp4a: bool, format: WeightFormat, in_dim: u32) -> bool {
    enabled
        && dp4a
        && format == WeightFormat::Q4K
        && in_dim.is_multiple_of(256)
        && u64::from(in_dim) <= MMVQ_SCRATCH_IN_DIM
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
    if !applies(decode_mmvq_enabled(), dev.dp4a, w.quant, in_dim) {
        return false; // out of scope: caller keeps the float GEMV
    }
    // Quantize A → int8 Q8_1 (one invocation per 32-wide block). The trailing compute
    // barrier (in `record`) orders the quant ahead of the GEMV that reads its scratch.
    let nblocks = in_dim / 32;
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
    // dp4a Q4_K GEMV: one invocation per output row, reads the Q4_K weights directly.
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
        out_dim.div_ceil(64).min(MAX_GROUPS_X),
    );
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn complete_route_gate_requires_flag_device_format_shape_and_scratch() {
        assert!(applies(true, true, WeightFormat::Q4K, 256));
        assert!(!applies(false, true, WeightFormat::Q4K, 256));
        assert!(!applies(true, false, WeightFormat::Q4K, 256));
        assert!(!applies(true, true, WeightFormat::Q5K, 256));
        assert!(!applies(true, true, WeightFormat::Q4K, 255));
        assert!(!applies(
            true,
            true,
            WeightFormat::Q4K,
            MMVQ_SCRATCH_IN_DIM as u32 + 256
        ));
    }
}
