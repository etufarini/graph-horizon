/*
 * gh_zero_engine — cooperative-matrix capability detection
 * Detects, for a chosen physical device, whether a usable VK_KHR_cooperative_matrix
 * shape exists for the prefill path: f16 inputs (A/B), f32 accumulation (C/result),
 * in the subgroup scope. This file only DESCRIBES the capability — it owns no Vulkan
 * resource, issues no dispatch, and never panics or propagates an error: an absent
 * extension, a failed enumeration, or no matching shape all yield `available = false`,
 * so the caller silently falls back to the f16×f16 tiled GEMM.
 *
 * The result feeds device bootstrap, which enables the extension only when
 * usable, and dense prefill routing, which checks the selected MMA tile.
*/

use ash::vk;

// VK_KHR_cooperative_matrix — the portable MMA (tensor-core) extension.
pub(crate) const COOPERATIVE_MATRIX: &std::ffi::CStr = ash::khr::cooperative_matrix::NAME;

// The cooperative-matrix shape chosen for the f16→f32 prefill GEMM, or `available =
// false` (all-zero) when the device exposes no usable one. `m`/`n`/`k` are the MMA tile
// the SPIR-V cooperative_matrix types are sized to; the routing requires the GEMM's
// out_dim/n to map onto these (with zero-padded tail tiles).
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct CoopmatCaps {
    pub available: bool,
    pub m: u32,
    pub n: u32,
    pub k: u32,
}

// Enumerate the device's cooperative-matrix shapes and pick, deterministically, the
// first one with f16 A/B inputs and f32 C/result accumulation in the subgroup scope.
// Returns `CoopmatCaps::default()` (available = false) when the extension is absent,
// the enumeration fails, or no f16→f32 subgroup shape exists — never an error.
pub(crate) fn detect(
    entry: &ash::Entry,
    instance: &ash::Instance,
    physical: vk::PhysicalDevice,
) -> CoopmatCaps {
    // The extension must be advertised before its instance-level query is loaded and
    // called; otherwise the function pointer is unresolved (a panic in ash's stub).
    // SAFETY: `instance` and `physical` are live handles from a successful Vulkan init;
    // the call only reads driver-reported extension properties, no buffers aliased.
    let advertised = unsafe { instance.enumerate_device_extension_properties(physical) }
        .map(|exts| {
            exts.iter().any(|e| {
                // SAFETY: extension_name is a NUL-terminated C string from the driver.
                let name = unsafe { std::ffi::CStr::from_ptr(e.extension_name.as_ptr()) };
                name == COOPERATIVE_MATRIX
            })
        })
        .unwrap_or(false);
    if !advertised {
        return CoopmatCaps::default();
    }

    let coop = ash::khr::cooperative_matrix::Instance::new(entry, instance);
    // SAFETY: the extension is advertised, so the queried function pointer is valid;
    // the driver fills a list of supported shapes (possibly empty).
    let props = match unsafe { coop.get_physical_device_cooperative_matrix_properties(physical) } {
        Ok(p) => p,
        Err(_) => return CoopmatCaps::default(),
    };

    // First f16-input / f32-accumulate subgroup shape, in driver enumeration order.
    for p in props {
        let f16_inputs =
            p.a_type == vk::ComponentTypeKHR::FLOAT16 && p.b_type == vk::ComponentTypeKHR::FLOAT16;
        let f32_accum = p.c_type == vk::ComponentTypeKHR::FLOAT32
            && p.result_type == vk::ComponentTypeKHR::FLOAT32;
        if f16_inputs && f32_accum && p.scope == vk::ScopeKHR::SUBGROUP {
            return CoopmatCaps {
                available: true,
                m: p.m_size,
                n: p.n_size,
                k: p.k_size,
            };
        }
    }
    CoopmatCaps::default()
}
