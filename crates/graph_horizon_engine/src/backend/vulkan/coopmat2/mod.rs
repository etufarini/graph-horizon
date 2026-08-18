/*
 * graph_horizon_engine — NVIDIA cooperative-matrix2 capability discovery
 * Queries one physical device and reduces the advertised KHR/NV feature and
 * property surface to the exact optional pipeline families. It owns no Vulkan
 * resource; every absent or incomplete capability becomes a clean fallback.
 */

mod abi;

use std::ffi::CStr;

use ash::vk;

use self::abi::{Features, FlexibleDimensions, Properties};

pub(crate) const COOPERATIVE_MATRIX2: &CStr = c"VK_NV_cooperative_matrix2";
const NVIDIA_VENDOR_ID: u32 = 0x10de;

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Coopmat2Caps {
    // The established WG256 Matrix2 attention/matmul family.
    pub available: bool,
    // The independent WG128/Q64 attention contract; it does not require WG256.
    pub attention_q64_wg128: bool,
    pub reserved_shared_memory: u32,
}

impl Coopmat2Caps {
    pub(crate) const fn enabled(self) -> bool {
        self.available || self.attention_q64_wg128
    }
}

type GetFlexible =
    unsafe extern "system" fn(vk::PhysicalDevice, *mut u32, *mut FlexibleDimensions) -> vk::Result;

pub(crate) fn detect(
    entry: &ash::Entry,
    instance: &ash::Instance,
    physical: vk::PhysicalDevice,
) -> Coopmat2Caps {
    // SAFETY: the live physical device is queried read-only for its extension list.
    let Ok(extensions) = (unsafe { instance.enumerate_device_extension_properties(physical) })
    else {
        return Coopmat2Caps::default();
    };
    let advertised = |wanted: &CStr| {
        extensions.iter().any(|extension| {
            // SAFETY: Vulkan guarantees a NUL-terminated extension name.
            unsafe { CStr::from_ptr(extension.extension_name.as_ptr()) == wanted }
        })
    };
    if !advertised(COOPERATIVE_MATRIX2)
        || !advertised(crate::backend::vulkan::coopmat::COOPERATIVE_MATRIX)
    {
        return Coopmat2Caps::default();
    }

    let mut matrix2_features = Features::default();
    let mut matrix_features = vk::PhysicalDeviceCooperativeMatrixFeaturesKHR::default();
    let mut feature_query = vk::PhysicalDeviceFeatures2::default()
        .push_next(&mut matrix_features)
        .push_next(&mut matrix2_features);
    // SAFETY: the output chain is stack-owned and outlives this driver query.
    unsafe { instance.get_physical_device_features2(physical, &mut feature_query) };
    if matrix_features.cooperative_matrix == 0 || !abi::common_features(&matrix2_features) {
        return Coopmat2Caps::default();
    }

    let mut matrix2_properties = Properties::default();
    let mut property_query =
        vk::PhysicalDeviceProperties2::default().push_next(&mut matrix2_properties);
    // SAFETY: the output chain is stack-owned and outlives this driver query.
    unsafe { instance.get_physical_device_properties2(physical, &mut property_query) };
    if property_query.properties.vendor_id != NVIDIA_VENDOR_ID {
        return Coopmat2Caps::default();
    }

    // SAFETY: both dependent extensions were advertised, so the function may be loaded.
    let function = unsafe {
        entry.get_instance_proc_addr(
            instance.handle(),
            c"vkGetPhysicalDeviceCooperativeMatrixFlexibleDimensionsPropertiesNV".as_ptr(),
        )
    };
    let Some(function) = function else {
        return Coopmat2Caps::default();
    };
    // SAFETY: Vulkan returned the address for the exact function name/signature above.
    let get_flexible: GetFlexible = unsafe { std::mem::transmute(function) };
    let mut count = 0;
    // SAFETY: the extension function writes only the requested count.
    if unsafe { get_flexible(physical, &mut count, std::ptr::null_mut()) } != vk::Result::SUCCESS
        || count == 0
    {
        return Coopmat2Caps::default();
    }
    let mut dimensions = vec![FlexibleDimensions::default(); count as usize];
    // SAFETY: `dimensions` has `count` initialized ABI-compatible output records.
    if unsafe { get_flexible(physical, &mut count, dimensions.as_mut_ptr()) } != vk::Result::SUCCESS
    {
        return Coopmat2Caps::default();
    }
    let dimensions = &dimensions[..count as usize];
    let available = abi::generic_supported(&matrix2_features, &matrix2_properties, dimensions);
    let attention_q64_wg128 =
        abi::q64_supported(&matrix2_features, &matrix2_properties, dimensions);
    Coopmat2Caps {
        available,
        attention_q64_wg128,
        reserved_shared_memory: matrix2_properties.reserved_shared_memory,
    }
}

pub(crate) fn enabled_features(caps: Coopmat2Caps) -> Features {
    Features {
        workgroup_scope: vk::TRUE,
        flexible_dimensions: vk::TRUE,
        reductions: u32::from(caps.attention_q64_wg128),
        conversions: u32::from(caps.attention_q64_wg128),
        per_element_operations: vk::TRUE,
        tensor_addressing: vk::TRUE,
        ..Features::default()
    }
}
