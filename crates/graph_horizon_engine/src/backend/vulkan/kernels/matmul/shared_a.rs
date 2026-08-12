/*
 * graph_horizon_engine — shared-A MLP gate/up dispatch
 * Selects and records the measured Vulkan-only Q4_K cooperative gate/up
 * operation. It owns the narrow shape/capability/runtime gate and five-buffer
 * dispatch ABI; generic fallback remains in the Backend implementation.
 */

#![allow(clippy::too_many_arguments)]

use ash::vk;

use crate::backend::vulkan::buffers::{GpuBuffer, WeightFormat};
use crate::backend::vulkan::coopmat::CoopmatCaps;
use crate::backend::vulkan::device::Device;
use crate::backend::vulkan::pipeline::{Kernel, PipelineRegistry, dispatch_2d};

fn enabled() -> bool {
    use std::sync::OnceLock;
    static FLAG: OnceLock<bool> = OnceLock::new();
    *FLAG.get_or_init(|| {
        match std::env::var("GRAPH_HORIZON_PREFILL_SHARED_A")
            .ok()
            .as_deref()
        {
            None | Some("1" | "true" | "yes") => true,
            Some(_) => false,
        }
    })
}

fn supported(
    gate_format: WeightFormat,
    up_format: WeightFormat,
    in_dim: u32,
    out_dim: u32,
    n: u32,
    caps: CoopmatCaps,
) -> bool {
    gate_format == WeightFormat::Q4K
        && up_format == WeightFormat::Q4K
        && in_dim == 3072
        && out_dim == 9216
        && n == 32
        && caps.available
        && (caps.m, caps.n, caps.k) == (16, 16, 16)
}

pub(crate) fn dispatch(
    dev: &Device,
    reg: &PipelineRegistry,
    cmd: vk::CommandBuffer,
    gate: &GpuBuffer,
    up: &GpuBuffer,
    a: &GpuBuffer,
    gate_w: &GpuBuffer,
    up_w: &GpuBuffer,
    in_dim: u32,
    out_dim: u32,
    n: u32,
) -> bool {
    if !enabled() || !supported(gate_w.quant, up_w.quant, in_dim, out_dim, n, dev.coopmat) {
        return false;
    }

    let mut push = Vec::with_capacity(12);
    push.extend_from_slice(&in_dim.to_le_bytes());
    push.extend_from_slice(&out_dim.to_le_bytes());
    push.extend_from_slice(&n.to_le_bytes());
    dispatch_2d(
        dev,
        reg,
        cmd,
        Kernel::MlpGateUpQ4KCoopmat,
        &[
            (a.buffer, a.offset, a.size),
            (gate_w.buffer, gate_w.offset, gate_w.size),
            (up_w.buffer, up_w.offset, up_w.size),
            (gate.buffer, gate.offset, gate.size),
            (up.buffer, up.offset, up.size),
        ],
        &push,
        n.div_ceil(32),
        out_dim.div_ceil(16),
    );
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn route_is_limited_to_measured_gate_up_shape() {
        let caps = CoopmatCaps {
            available: true,
            m: 16,
            n: 16,
            k: 16,
        };
        assert!(supported(
            WeightFormat::Q4K,
            WeightFormat::Q4K,
            3072,
            9216,
            32,
            caps
        ));
        assert!(!supported(
            WeightFormat::Q6K,
            WeightFormat::Q4K,
            3072,
            9216,
            32,
            caps
        ));
        assert!(!supported(
            WeightFormat::Q4K,
            WeightFormat::Q4K,
            3072,
            9216,
            31,
            caps
        ));
        assert!(!supported(
            WeightFormat::Q4K,
            WeightFormat::Q4K,
            4096,
            9216,
            32,
            caps
        ));
    }
}
