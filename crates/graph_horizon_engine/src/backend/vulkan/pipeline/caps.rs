/*
 * graph_horizon_engine — Vulkan pipeline capability gates
 * Queries immutable device limits once and decides which optional compute
 * pipeline families are safe to build. It owns no Vulkan resources.
 */

use ash::vk;
use color_eyre::eyre::{Result, eyre};

use super::Device;
use crate::backend::vulkan::coopmat::CoopmatCaps;

const WIDE_ATTENTION_SHARED_BYTES: u32 = 32 * 128 * 4 + 32 * 4 * 2;
const TILED_ATTENTION_SHARED_BYTES: u32 = 64 * 128 * 2 + 8 * 128 * 2 + 8 * 64 * 4 + 8 * 3 * 4;
const GQA_PREFILL_SHARED_BYTES: u32 =
    64 * 128 * 2 + 8 * 4 * 128 * 2 + 8 * 4 * 64 * 4 + 8 * 4 * 3 * 4;
const COOP_QK_ATTENTION_SHARED_BYTES: u32 =
    64 * 128 * 2 + 16 * 128 * 2 + 16 * 64 * 2 + 16 * 64 * 4 + 16 * 3 * 4;
const MATRIX2_ATTENTION_SHARED_BYTES: u32 = 32 * 128 * 4 + 32 * 64 * 2 + 32 * 3 * 4;
const MATRIX2_MATMUL_SHARED_BYTES: u32 = 128 * 32 * 2 + 64 * 32 * 4;
const ATTENTION_1024_SHARED_BYTES: u32 = 64 * 128 * 4 + 64 * 4 * 2;
const GQA_DECODE_SHARED_BYTES: u32 = 8 * 2 * 4 + 8 * 32 * 4 * 4;

fn supports_wide_attention(invocations: u32, size_x: u32, shared_bytes: u32) -> bool {
    invocations >= 512 && size_x >= 512 && shared_bytes >= WIDE_ATTENTION_SHARED_BYTES
}

fn supports_attention_1024(invocations: u32, size_x: u32, shared_bytes: u32) -> bool {
    invocations >= 1024 && size_x >= 1024 && shared_bytes >= ATTENTION_1024_SHARED_BYTES
}

fn supports_gqa_prefill(
    invocations: u32,
    size_x: u32,
    shared_bytes: u32,
    subgroup_size: u32,
    fixed_wave32: bool,
    integer_q4_batch: bool,
) -> bool {
    invocations >= 1024
        && size_x >= 1024
        && shared_bytes >= GQA_PREFILL_SHARED_BYTES
        && subgroup_size == 64
        && fixed_wave32
        && integer_q4_batch
}

fn supports_gqa_decode_resources(invocations: u32, size_x: u32, shared_bytes: u32) -> bool {
    invocations >= 256 && size_x >= 256 && shared_bytes >= GQA_DECODE_SHARED_BYTES
}

fn supports_gqa_decode(resources: bool, subgroup_size: u32) -> bool {
    resources && subgroup_size == 32
}

fn supports_gqa_decode_required_wave32(
    resources: bool,
    subgroup_size: u32,
    fixed_wave32: bool,
) -> bool {
    resources && subgroup_size == 64 && fixed_wave32
}

fn supports_gqa_decode_wave64(resources: bool, subgroup_size: u32) -> bool {
    resources && subgroup_size == 64
}

fn supports_tiled_attention(
    invocations: u32,
    size_x: u32,
    shared_bytes: u32,
    subgroup_size: u32,
) -> bool {
    invocations >= 512
        && size_x >= 512
        && shared_bytes >= TILED_ATTENTION_SHARED_BYTES
        && matches!(subgroup_size, 16 | 32 | 64)
}

fn supports_coop_qk_attention(
    tiled: bool,
    shared_bytes: u32,
    subgroup_size: u32,
    coop: CoopmatCaps,
) -> bool {
    tiled
        && shared_bytes >= COOP_QK_ATTENTION_SHARED_BYTES
        && matches!(subgroup_size, 16 | 32)
        && coop.available
        && (coop.m, coop.n, coop.k) == (16, 16, 16)
}

fn supports_q4(subgroup_size: u32, operations: vk::SubgroupFeatureFlags) -> bool {
    subgroup_size == 32 && operations.contains(vk::SubgroupFeatureFlags::SHUFFLE)
}

fn supports_matrix2(shared_bytes: u32, reserved_bytes: u32, available: bool) -> bool {
    let shader_bytes = MATRIX2_ATTENTION_SHARED_BYTES.max(MATRIX2_MATMUL_SHARED_BYTES);
    available
        && shader_bytes
            .checked_add(reserved_bytes)
            .is_some_and(|required| shared_bytes >= required)
}

fn supports_matrix2_attention_q64(
    invocations: u32,
    size_x: u32,
    shared_bytes: u32,
    reserved_bytes: u32,
    available: bool,
) -> bool {
    available && invocations >= 128 && size_x >= 128 && shared_bytes >= reserved_bytes
}

pub(super) struct PipelineCaps {
    pub wide_attention: bool,
    pub tiled_attention: bool,
    pub coop_qk_attention: bool,
    pub matrix2: bool,
    pub matrix2_attention_q64: bool,
    pub attention_1024: bool,
    pub gqa_prefill_required_wave32: bool,
    pub gqa_decode: bool,
    pub gqa_decode_required_wave32: bool,
    pub gqa_decode_wave64: bool,
    pub q4_metadata: bool,
}

pub(super) fn check(dev: &Device) -> Result<PipelineCaps> {
    // Required kernels use up to 256 invocations; optional attention variants
    // are gated independently at 512 and 1024. Vulkan guarantees 128-byte pushes.
    let mut vulkan11 = vk::PhysicalDeviceVulkan11Properties::default();
    let mut properties = vk::PhysicalDeviceProperties2::default().push_next(&mut vulkan11);
    // SAFETY: the complete output chain outlives this read-only driver query.
    unsafe {
        dev.instance
            .get_physical_device_properties2(dev.physical, &mut properties)
    };
    let limits = properties.properties.limits;
    if limits.max_compute_work_group_invocations < 256
        || limits.max_compute_work_group_size[0] < 256
        || limits.max_push_constants_size < 36
    {
        return Err(eyre!(
            "vulkan: device workgroup/push-constant limits too small"
        ));
    }
    let available = (
        limits.max_compute_work_group_invocations,
        limits.max_compute_work_group_size[0],
        limits.max_compute_shared_memory_size,
    );
    let subgroup = vulkan11.subgroup_size;
    let tiled = supports_tiled_attention(available.0, available.1, available.2, subgroup);
    let coop_qk = supports_coop_qk_attention(tiled, available.2, subgroup, dev.coopmat);
    let gqa = supports_gqa_decode_resources(available.0, available.1, available.2);
    let matrix2 = supports_matrix2(
        available.2,
        dev.coopmat2.reserved_shared_memory,
        dev.coopmat2.available,
    );
    Ok(PipelineCaps {
        wide_attention: supports_wide_attention(available.0, available.1, available.2),
        tiled_attention: tiled,
        coop_qk_attention: coop_qk,
        matrix2,
        matrix2_attention_q64: supports_matrix2_attention_q64(
            available.0,
            available.1,
            available.2,
            dev.coopmat2.reserved_shared_memory,
            dev.coopmat2.attention_q64_wg128,
        ),
        attention_1024: supports_attention_1024(available.0, available.1, available.2),
        gqa_prefill_required_wave32: supports_gqa_prefill(
            available.0,
            available.1,
            available.2,
            subgroup,
            dev.profile.fixed_wave32(),
            dev.profile.integer_q4_batch(),
        ),
        gqa_decode: supports_gqa_decode(gqa, subgroup),
        gqa_decode_required_wave32: supports_gqa_decode_required_wave32(
            gqa,
            subgroup,
            dev.profile.fixed_wave32(),
        ),
        gqa_decode_wave64: supports_gqa_decode_wave64(gqa, subgroup),
        q4_metadata: supports_q4(subgroup, vulkan11.subgroup_supported_operations),
    })
}

#[cfg(test)]
mod tests {
    use ash::vk;

    use super::*;

    #[test]
    fn wide_attention_requires_every_resource_limit() {
        assert!(supports_wide_attention(
            512,
            512,
            WIDE_ATTENTION_SHARED_BYTES
        ));
        assert!(!supports_wide_attention(
            511,
            512,
            WIDE_ATTENTION_SHARED_BYTES
        ));
        assert!(!supports_wide_attention(
            512,
            511,
            WIDE_ATTENTION_SHARED_BYTES
        ));
        assert!(!supports_wide_attention(
            512,
            512,
            WIDE_ATTENTION_SHARED_BYTES - 1
        ));
    }

    #[test]
    fn attention_1024_requires_every_resource_limit() {
        assert!(supports_attention_1024(
            1024,
            1024,
            ATTENTION_1024_SHARED_BYTES
        ));
        assert!(!supports_attention_1024(
            1023,
            1024,
            ATTENTION_1024_SHARED_BYTES
        ));
        assert!(!supports_attention_1024(
            1024,
            1023,
            ATTENTION_1024_SHARED_BYTES
        ));
        assert!(!supports_attention_1024(
            1024,
            1024,
            ATTENTION_1024_SHARED_BYTES - 1
        ));
    }

    #[test]
    fn gqa_prefill_requires_qualified_wave32_profile() {
        assert!(supports_gqa_prefill(
            1024,
            1024,
            GQA_PREFILL_SHARED_BYTES,
            64,
            true,
            true,
        ));
        assert!(!supports_gqa_prefill(
            1024,
            1024,
            GQA_PREFILL_SHARED_BYTES,
            32,
            true,
            true,
        ));
        assert!(!supports_gqa_prefill(
            1024,
            1024,
            GQA_PREFILL_SHARED_BYTES,
            64,
            false,
            true,
        ));
        assert!(!supports_gqa_prefill(
            1024,
            1024,
            GQA_PREFILL_SHARED_BYTES,
            64,
            true,
            false,
        ));
    }

    #[test]
    fn gqa_decode_requires_its_exact_subgroup_and_resources() {
        assert!(supports_gqa_decode_resources(
            256,
            256,
            GQA_DECODE_SHARED_BYTES
        ));
        assert!(!supports_gqa_decode_resources(
            255,
            256,
            GQA_DECODE_SHARED_BYTES
        ));
        assert!(!supports_gqa_decode_resources(
            256,
            255,
            GQA_DECODE_SHARED_BYTES
        ));
        assert!(!supports_gqa_decode_resources(
            256,
            256,
            GQA_DECODE_SHARED_BYTES - 1
        ));
        assert!(supports_gqa_decode(true, 32));
        assert!(!supports_gqa_decode(true, 64));
        assert!(supports_gqa_decode_wave64(true, 64));
        assert!(!supports_gqa_decode_wave64(true, 32));
        assert!(supports_gqa_decode_required_wave32(true, 64, true));
        assert!(!supports_gqa_decode_required_wave32(true, 64, false));
        assert!(!supports_gqa_decode_required_wave32(false, 64, true));
    }

    #[test]
    fn tiled_attention_requires_every_resource_limit() {
        assert!(supports_tiled_attention(
            512,
            512,
            TILED_ATTENTION_SHARED_BYTES,
            32
        ));
        assert!(!supports_tiled_attention(
            511,
            512,
            TILED_ATTENTION_SHARED_BYTES,
            32
        ));
        assert!(!supports_tiled_attention(
            512,
            511,
            TILED_ATTENTION_SHARED_BYTES,
            32
        ));
        assert!(!supports_tiled_attention(
            512,
            512,
            TILED_ATTENTION_SHARED_BYTES - 1,
            32,
        ));
        assert!(!supports_tiled_attention(
            512,
            512,
            TILED_ATTENTION_SHARED_BYTES,
            8
        ));
    }

    #[test]
    fn coop_qk_attention_requires_exact_shape_and_shared_memory() {
        let supported = CoopmatCaps {
            available: true,
            m: 16,
            n: 16,
            k: 16,
        };
        assert!(supports_coop_qk_attention(
            true,
            COOP_QK_ATTENTION_SHARED_BYTES,
            32,
            supported,
        ));
        assert!(supports_coop_qk_attention(
            true,
            COOP_QK_ATTENTION_SHARED_BYTES,
            16,
            supported,
        ));
        assert!(!supports_coop_qk_attention(
            false,
            COOP_QK_ATTENTION_SHARED_BYTES,
            32,
            supported,
        ));
        assert!(!supports_coop_qk_attention(
            true,
            COOP_QK_ATTENTION_SHARED_BYTES - 1,
            32,
            supported,
        ));
        assert!(!supports_coop_qk_attention(
            true,
            COOP_QK_ATTENTION_SHARED_BYTES,
            64,
            supported,
        ));
        assert!(!supports_coop_qk_attention(
            true,
            COOP_QK_ATTENTION_SHARED_BYTES,
            32,
            CoopmatCaps { m: 8, ..supported },
        ));
    }

    #[test]
    fn q4_metadata_requires_subgroup_shuffle() {
        assert!(supports_q4(32, vk::SubgroupFeatureFlags::SHUFFLE));
        assert!(!supports_q4(16, vk::SubgroupFeatureFlags::SHUFFLE));
        assert!(!supports_q4(32, vk::SubgroupFeatureFlags::BASIC));
    }

    #[test]
    fn matrix2_requires_available_feature_and_checked_shared_sum() {
        assert_eq!(MATRIX2_MATMUL_SHARED_BYTES, 16_384);
        let shader = MATRIX2_ATTENTION_SHARED_BYTES.max(MATRIX2_MATMUL_SHARED_BYTES);
        assert!(supports_matrix2(shader + 8192, 8192, true));
        assert!(!supports_matrix2(shader + 8191, 8192, true));
        assert!(!supports_matrix2(u32::MAX, u32::MAX, true));
        assert!(!supports_matrix2(u32::MAX, 0, false));
    }

    #[test]
    fn q64_matrix2_requires_wg128_and_reserved_shared_memory() {
        assert!(supports_matrix2_attention_q64(128, 128, 8192, 8192, true));
        assert!(!supports_matrix2_attention_q64(127, 128, 8192, 8192, true));
        assert!(!supports_matrix2_attention_q64(128, 127, 8192, 8192, true));
        assert!(!supports_matrix2_attention_q64(128, 128, 8191, 8192, true));
        assert!(!supports_matrix2_attention_q64(128, 128, 8192, 8192, false));
    }
}
