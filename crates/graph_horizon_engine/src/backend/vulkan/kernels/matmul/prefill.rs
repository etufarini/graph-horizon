/*
 * graph_horizon_engine — Vulkan batched matmul dispatch
 * Records Q4_K and Q6_K batched projections, selecting their measured
 * cooperative paths only for supported devices and prefill shapes. Decode and
 * diagnostics stay outside.
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
        match std::env::var("GRAPH_HORIZON_PREFILL_COOPMAT")
            .ok()
            .as_deref()
        {
            None | Some("1" | "true" | "yes") => true,
            Some(_) => false,
        }
    })
}

fn coopmat_shape(in_dim: u32) -> bool {
    matches!(in_dim, 3072 | 4096 | 9216)
}

fn q4_metadata_shape(out_dim: u32) -> bool {
    matches!(out_dim, 3072 | 4096 | 9216)
}

fn q4_metadata_enabled() -> bool {
    use std::sync::OnceLock;
    static FLAG: OnceLock<bool> = OnceLock::new();
    *FLAG.get_or_init(|| {
        match std::env::var("GRAPH_HORIZON_PREFILL_Q4_METADATA")
            .ok()
            .as_deref()
        {
            None | Some("1" | "true" | "yes") => true,
            Some(_) => false,
        }
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
        // Only measured 3B projection/MLP K dimensions select coopmat. K=4096
        // is viable with the retained two-subgroup B reuse; the earlier
        // one-subgroup candidate for that shape had regressed.
        && coopmat_shape(in_dim)
    {
        trace::log_batched_path_once(true);
        let kernel = if q4_metadata_shape(out_dim)
            && q4_metadata_enabled()
            && reg.contains(Kernel::MatmulQ4KCoopmatMetadataF16Out)
        {
            Kernel::MatmulQ4KCoopmatMetadataF16Out
        } else {
            Kernel::MatmulQ4KCoopmatF16Out
        };
        super::super::coopmat::dispatch_coopmat(
            dev, reg, cmd, out, a, w, in_dim, out_dim, n, caps, kernel,
        );
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
        n.div_ceil(32),
    );
}

pub(crate) fn matmul_batched_q6k(
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
        && w.quant == WeightFormat::Q6K
        && caps.available
        && caps.m == 16
        && caps.n == 16
        && caps.k == 16
        && coopmat_shape(in_dim)
    {
        super::super::coopmat::dispatch_coopmat(
            dev,
            reg,
            cmd,
            out,
            a,
            w,
            in_dim,
            out_dim,
            n,
            caps,
            Kernel::MatmulQ6KCoopmatF16Out,
        );
        return;
    }
    let mut push = Vec::with_capacity(12);
    push.extend_from_slice(&in_dim.to_le_bytes());
    push.extend_from_slice(&out_dim.to_le_bytes());
    push.extend_from_slice(&n.to_le_bytes());
    dispatch_2d(
        dev,
        reg,
        cmd,
        Kernel::MatmulQ6KBatchF16Out,
        &[
            (a.buffer, a.offset, a.size),
            (w.buffer, w.offset, w.size),
            (out.buffer, out.offset, out.size),
        ],
        &push,
        out_dim.div_ceil(64),
        n.div_ceil(32),
    );
}

#[cfg(all(test, feature = "vulkan-hybrid"))]
mod tests {
    use crate::backend::Backend;
    use crate::backend::cpu::buffer::{f16_to_f32, f32_to_f16_bytes};
    use crate::backend::cpu::compute;
    use crate::backend::cpu::{CpuBuffer, CpuFormat};
    use crate::backend::vulkan::VulkanBackend;
    use crate::backend::vulkan::buffers::{GpuBuffer, WeightFormat};

    #[test]
    fn batched_q4k_metadata_matches_cpu_oracle() {
        q4k_cpu_oracle(3072, 3072, 3, "q4_metadata");
    }

    #[test]
    fn batched_q4k_small_output_matches_cpu_oracle() {
        q4k_cpu_oracle(4096, 70, 37, "q4_small_output");
    }

    fn q4k_cpu_oracle(in_dim: usize, out_dim: usize, rows: usize, label: &str) {
        let backend = match VulkanBackend::bare() {
            Ok(backend) => backend,
            Err(_) => {
                eprintln!("batched Q4_K parity skipped: no Vulkan device");
                return;
            }
        };
        let mut seed = 0x9e3779b9u32;
        let mut rng = || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            seed
        };
        let mut qbytes: Vec<u8> = (0..out_dim * (in_dim / 256) * 144)
            .map(|_| (rng() >> 16) as u8)
            .collect();
        let f16le = |value: f32| {
            let bytes = f32_to_f16_bytes(&value.to_le_bytes());
            [bytes[0], bytes[1]]
        };
        for block in qbytes.chunks_exact_mut(144) {
            block[..2].copy_from_slice(&f16le(0.03));
            block[2..4].copy_from_slice(&f16le(0.01));
        }
        let activations: Vec<f32> = (0..rows * in_dim)
            .map(|_| ((rng() >> 8) as f32 / u32::MAX as f32) - 0.5)
            .collect();
        let cpu_weights = CpuBuffer::from_bytes(qbytes.clone(), CpuFormat::Q4_K);
        let expected: Vec<f32> = activations
            .chunks_exact(in_dim)
            .flat_map(|row| compute::matmul(&cpu_weights, row, in_dim, out_dim))
            .collect();

        let activation_bytes = f32_to_f16_bytes(
            &activations
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<_>>(),
        );
        let input = backend
            .alloc_buffer(activation_bytes.len() as u64)
            .expect("input");
        backend
            .upload_bytes(&input, &activation_bytes)
            .expect("upload input");
        let mut weights =
            GpuBuffer::alloc(&backend.dev, qbytes.len() as u64, false).expect("weights");
        weights.quant = WeightFormat::Q4K;
        backend
            .upload_bytes(&weights, &qbytes)
            .expect("upload weights");
        let output = backend
            .alloc_buffer((rows * out_dim * 2) as u64)
            .expect("output");
        let encoder = backend.begin().expect("begin");
        backend.matmul_batched(
            &encoder,
            &output,
            &input,
            &weights,
            in_dim as u32,
            out_dim as u32,
            rows as u32,
        );
        backend.submit(encoder).expect("submit");
        let raw = backend
            .read_bytes(&output, rows * out_dim * 2)
            .expect("read output");
        let mut max_abs = 0.0f32;
        let mut mean_abs = 0.0f64;
        let mut max_relative = 0.0f32;
        let mut mean_relative = 0.0f64;
        for (index, bytes) in raw.chunks_exact(2).enumerate() {
            let got = f16_to_f32(u16::from_le_bytes([bytes[0], bytes[1]]));
            let reference = expected[index];
            let absolute = (got - reference).abs();
            let relative = absolute / reference.abs().max(1.0);
            max_abs = max_abs.max(absolute);
            mean_abs += f64::from(absolute);
            max_relative = max_relative.max(relative);
            mean_relative += f64::from(relative);
            assert!(got.is_finite(), "value {index}: non-finite result");
            assert!(
                absolute <= 5e-2 || absolute <= 5e-3 * reference.abs().max(1.0),
                "value {index}: got={got} want={reference}"
            );
        }
        let count = raw.len() as f64 / 2.0;
        println!(
            "{label} max_abs={max_abs} mean_abs={} max_relative={max_relative} mean_relative={}",
            mean_abs / count,
            mean_relative / count
        );
        input.destroy(&backend.dev);
        weights.destroy(&backend.dev);
        output.destroy(&backend.dev);
    }

    #[test]
    fn cooperative_route_is_limited_to_measured_k_dimensions() {
        assert!(super::coopmat_shape(3072));
        assert!(super::coopmat_shape(9216));
        assert!(super::coopmat_shape(4096));
        assert!(!super::coopmat_shape(256));
    }

    #[test]
    fn metadata_route_is_limited_to_measured_output_dimensions() {
        assert!(super::q4_metadata_shape(3072));
        assert!(super::q4_metadata_shape(4096));
        assert!(super::q4_metadata_shape(9216));
        assert!(!super::q4_metadata_shape(1024));
        assert!(!super::q4_metadata_shape(5000));
    }

    #[test]
    fn batched_q6k_matches_cpu_oracle() {
        let backend = match VulkanBackend::bare() {
            Ok(backend) => backend,
            Err(_) => {
                eprintln!("batched Q6_K parity skipped: no Vulkan device");
                return;
            }
        };
        let in_dim = 3072usize;
        let out_dim = 70usize;
        let rows = 5usize;
        let mut qbytes = vec![0u8; out_dim * (in_dim / 256) * 210];
        for (block_index, block) in qbytes.chunks_exact_mut(210).enumerate() {
            for (index, byte) in block[..192].iter_mut().enumerate() {
                *byte = (index as u8)
                    .wrapping_mul(17)
                    .wrapping_add(block_index as u8);
            }
            for (index, scale) in block[192..208].iter_mut().enumerate() {
                *scale = (index % 7 + 1) as u8;
            }
            let scale = f32_to_f16_bytes(&0.01f32.to_le_bytes());
            block[208..210].copy_from_slice(&scale[..2]);
        }
        let activations: Vec<f32> = (0..rows * in_dim)
            .map(|index| (index % 31) as f32 / 32.0 - 0.5)
            .collect();
        let cpu_weights = CpuBuffer::from_bytes(qbytes.clone(), CpuFormat::Q6_K);
        let expected: Vec<f32> = activations
            .chunks_exact(in_dim)
            .flat_map(|row| compute::matmul(&cpu_weights, row, in_dim, out_dim))
            .collect();

        let activation_bytes = f32_to_f16_bytes(
            &activations
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<_>>(),
        );
        let input = backend
            .alloc_buffer(activation_bytes.len() as u64)
            .expect("input");
        backend
            .upload_bytes(&input, &activation_bytes)
            .expect("upload input");
        let mut weights =
            GpuBuffer::alloc(&backend.dev, qbytes.len() as u64, false).expect("weights");
        weights.quant = WeightFormat::Q6K;
        backend
            .upload_bytes(&weights, &qbytes)
            .expect("upload weights");
        let output = backend
            .alloc_buffer((rows * out_dim * 2) as u64)
            .expect("output");
        let encoder = backend.begin().expect("begin");
        backend.matmul_batched(
            &encoder,
            &output,
            &input,
            &weights,
            in_dim as u32,
            out_dim as u32,
            rows as u32,
        );
        backend.submit(encoder).expect("submit");
        let raw = backend
            .read_bytes(&output, rows * out_dim * 2)
            .expect("read output");
        let mut max_abs = 0.0f32;
        let mut mean_abs = 0.0f64;
        let mut max_relative = 0.0f32;
        let mut mean_relative = 0.0f64;
        for (index, bytes) in raw.chunks_exact(2).enumerate() {
            let got = f16_to_f32(u16::from_le_bytes([bytes[0], bytes[1]]));
            let reference = expected[index];
            let absolute = (got - reference).abs();
            let relative = absolute / reference.abs().max(1.0);
            max_abs = max_abs.max(absolute);
            mean_abs += f64::from(absolute);
            max_relative = max_relative.max(relative);
            mean_relative += f64::from(relative);
            assert!(got.is_finite(), "value {index}: non-finite result");
            assert!(
                absolute <= 5e-2 || absolute <= 5e-3 * reference.abs().max(1.0),
                "value {index}: got={got} want={reference}"
            );
        }
        let count = raw.len() as f64 / 2.0;
        println!(
            "q6_control max_abs={max_abs} mean_abs={} max_relative={max_relative} mean_relative={}",
            mean_abs / count,
            mean_relative / count
        );
        input.destroy(&backend.dev);
        weights.destroy(&backend.dev);
        output.destroy(&backend.dev);
    }
}
