/*
 * graph_horizon_engine — NVIDIA cooperative-matrix2 capability discovery
 * Owns the small local ABI missing from ash 0.38 and reduces the advertised NV2
 * feature/property surface to the exact workgroup FP16→FP32 shapes needed by
 * prefill attention. It allocates no Vulkan resources and absence is a clean
 * `available = false` fallback.
 */

use std::ffi::{CStr, c_void};

use ash::vk;

pub(crate) const COOPERATIVE_MATRIX2: &CStr = c"VK_NV_cooperative_matrix2";

const FEATURES_TYPE: vk::StructureType = vk::StructureType::from_raw(1_000_593_000);
const FLEXIBLE_TYPE: vk::StructureType = vk::StructureType::from_raw(1_000_593_001);
const PROPERTIES_TYPE: vk::StructureType = vk::StructureType::from_raw(1_000_593_002);

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct Features {
    s_type: vk::StructureType,
    p_next: *mut c_void,
    pub workgroup_scope: vk::Bool32,
    pub flexible_dimensions: vk::Bool32,
    pub reductions: vk::Bool32,
    pub conversions: vk::Bool32,
    pub per_element_operations: vk::Bool32,
    pub tensor_addressing: vk::Bool32,
    pub block_loads: vk::Bool32,
}

impl Default for Features {
    fn default() -> Self {
        Self {
            s_type: FEATURES_TYPE,
            p_next: std::ptr::null_mut(),
            workgroup_scope: 0,
            flexible_dimensions: 0,
            reductions: 0,
            conversions: 0,
            per_element_operations: 0,
            tensor_addressing: 0,
            block_loads: 0,
        }
    }
}

// SAFETY: `Features` is the exact VkPhysicalDeviceCooperativeMatrix2FeaturesNV
// layout and is valid on both feature-query and device-create pNext chains.
unsafe impl vk::ExtendsPhysicalDeviceFeatures2 for Features {}
// SAFETY: same ABI invariant as above; every enabled bit was queried first.
unsafe impl vk::ExtendsDeviceCreateInfo for Features {}

#[repr(C)]
#[derive(Clone, Copy)]
struct Properties {
    s_type: vk::StructureType,
    p_next: *mut c_void,
    max_workgroup_size: u32,
    max_dimension: u32,
    reserved_shared_memory: u32,
}

impl Default for Properties {
    fn default() -> Self {
        Self {
            s_type: PROPERTIES_TYPE,
            p_next: std::ptr::null_mut(),
            max_workgroup_size: 0,
            max_dimension: 0,
            reserved_shared_memory: 0,
        }
    }
}

// SAFETY: `Properties` exactly matches VkPhysicalDeviceCooperativeMatrix2PropertiesNV.
unsafe impl vk::ExtendsPhysicalDeviceProperties2 for Properties {}

#[repr(C)]
#[derive(Clone, Copy)]
struct FlexibleDimensions {
    s_type: vk::StructureType,
    p_next: *mut c_void,
    m_granularity: u32,
    n_granularity: u32,
    k_granularity: u32,
    a_type: vk::ComponentTypeKHR,
    b_type: vk::ComponentTypeKHR,
    c_type: vk::ComponentTypeKHR,
    result_type: vk::ComponentTypeKHR,
    saturating_accumulation: vk::Bool32,
    scope: vk::ScopeKHR,
    workgroup_invocations: u32,
}

impl Default for FlexibleDimensions {
    fn default() -> Self {
        Self {
            s_type: FLEXIBLE_TYPE,
            p_next: std::ptr::null_mut(),
            m_granularity: 0,
            n_granularity: 0,
            k_granularity: 0,
            a_type: vk::ComponentTypeKHR::default(),
            b_type: vk::ComponentTypeKHR::default(),
            c_type: vk::ComponentTypeKHR::default(),
            result_type: vk::ComponentTypeKHR::default(),
            saturating_accumulation: 0,
            scope: vk::ScopeKHR::default(),
            workgroup_invocations: 0,
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Coopmat2Caps {
    pub available: bool,
    pub attention_q64_wg128: bool,
    pub reserved_shared_memory: u32,
}

type GetFlexible =
    unsafe extern "system" fn(vk::PhysicalDevice, *mut u32, *mut FlexibleDimensions) -> vk::Result;

pub(crate) fn detect(
    entry: &ash::Entry,
    instance: &ash::Instance,
    physical: vk::PhysicalDevice,
) -> Coopmat2Caps {
    // SAFETY: the live physical device is queried read-only for its extension list.
    let advertised = unsafe { instance.enumerate_device_extension_properties(physical) }.is_ok_and(
        |extensions| {
            extensions.iter().any(|extension| {
                // SAFETY: Vulkan guarantees a NUL-terminated extension name.
                unsafe { CStr::from_ptr(extension.extension_name.as_ptr()) == COOPERATIVE_MATRIX2 }
            })
        },
    );
    if !advertised {
        return Coopmat2Caps::default();
    }

    let mut features = Features::default();
    let mut feature_query = vk::PhysicalDeviceFeatures2::default().push_next(&mut features);
    // SAFETY: the output chain is stack-owned and outlives this driver query.
    unsafe { instance.get_physical_device_features2(physical, &mut feature_query) };
    let required_features = features.workgroup_scope != 0
        && features.flexible_dimensions != 0
        && features.per_element_operations != 0
        && features.tensor_addressing != 0;
    if !required_features {
        return Coopmat2Caps::default();
    }

    let mut properties = Properties::default();
    let mut property_query = vk::PhysicalDeviceProperties2::default().push_next(&mut properties);
    // SAFETY: the output chain is stack-owned and outlives this driver query.
    unsafe { instance.get_physical_device_properties2(physical, &mut property_query) };
    if properties.max_workgroup_size < 256 || properties.max_dimension < 128 {
        return Coopmat2Caps::default();
    }

    // SAFETY: the extension was advertised, so the instance-level function may be loaded.
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
    let compatible = |property: &FlexibleDimensions| {
        property.scope == vk::ScopeKHR::WORKGROUP
            && property.a_type == vk::ComponentTypeKHR::FLOAT16
            && property.b_type == vk::ComponentTypeKHR::FLOAT16
            && property.c_type == vk::ComponentTypeKHR::FLOAT32
            && property.result_type == vk::ComponentTypeKHR::FLOAT32
            && property.k_granularity == 16
    };
    let available = dimensions[..count as usize].iter().any(|property| {
        compatible(property)
            && property.workgroup_invocations == 256
            && property.m_granularity == 32
            && property.n_granularity == 32
    });
    let attention_q64_wg128 = available
        && features.reductions != 0
        && dimensions[..count as usize].iter().any(|property| {
            compatible(property)
                && property.workgroup_invocations == 128
                && property.m_granularity != 0
                && property.n_granularity != 0
                && property.k_granularity != 0
                && 64_u32.is_multiple_of(property.m_granularity)
                && 64_u32.is_multiple_of(property.n_granularity)
                && 64_u32.is_multiple_of(property.k_granularity)
        });
    Coopmat2Caps {
        available,
        attention_q64_wg128,
        reserved_shared_memory: properties.reserved_shared_memory,
    }
}

pub(crate) fn enabled_features(caps: Coopmat2Caps) -> Features {
    Features {
        workgroup_scope: vk::TRUE,
        flexible_dimensions: vk::TRUE,
        reductions: u32::from(caps.attention_q64_wg128),
        per_element_operations: vk::TRUE,
        tensor_addressing: vk::TRUE,
        ..Features::default()
    }
}
