/*
 * graph_horizon_engine — NVIDIA cooperative-matrix2 ABI and capability policy
 * Defines the Vulkan records missing from ash 0.38 and the pure predicates for
 * the exact generic-WG256 and attention-Q64 flexible-dimension contracts. It
 * performs no driver query, allocation, device creation, or dispatch.
 */

use std::ffi::c_void;

use ash::vk;

const FEATURES_TYPE: vk::StructureType = vk::StructureType::from_raw(1_000_593_000);
const FLEXIBLE_TYPE: vk::StructureType = vk::StructureType::from_raw(1_000_593_001);
const PROPERTIES_TYPE: vk::StructureType = vk::StructureType::from_raw(1_000_593_002);

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct Features {
    pub(super) s_type: vk::StructureType,
    pub(super) p_next: *mut c_void,
    pub(super) workgroup_scope: vk::Bool32,
    pub(super) flexible_dimensions: vk::Bool32,
    pub(super) reductions: vk::Bool32,
    pub(super) conversions: vk::Bool32,
    pub(super) per_element_operations: vk::Bool32,
    pub(super) tensor_addressing: vk::Bool32,
    pub(super) block_loads: vk::Bool32,
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

// SAFETY: `Features` exactly matches VkPhysicalDeviceCooperativeMatrix2FeaturesNV.
unsafe impl vk::ExtendsPhysicalDeviceFeatures2 for Features {}
// SAFETY: every bit placed on the device-create chain was observed first.
unsafe impl vk::ExtendsDeviceCreateInfo for Features {}

#[repr(C)]
#[derive(Clone, Copy)]
pub(super) struct Properties {
    s_type: vk::StructureType,
    p_next: *mut c_void,
    pub max_workgroup_size: u32,
    pub max_dimension: u32,
    pub reserved_shared_memory: u32,
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
pub(super) struct FlexibleDimensions {
    s_type: vk::StructureType,
    p_next: *mut c_void,
    pub m_granularity: u32,
    pub n_granularity: u32,
    pub k_granularity: u32,
    pub a_type: vk::ComponentTypeKHR,
    pub b_type: vk::ComponentTypeKHR,
    pub c_type: vk::ComponentTypeKHR,
    pub result_type: vk::ComponentTypeKHR,
    pub saturating_accumulation: vk::Bool32,
    pub scope: vk::ScopeKHR,
    pub workgroup_invocations: u32,
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

pub(super) fn common_features(features: &Features) -> bool {
    features.workgroup_scope != 0
        && features.flexible_dimensions != 0
        && features.per_element_operations != 0
        && features.tensor_addressing != 0
}

fn f16_f32(property: &FlexibleDimensions) -> bool {
    property.scope == vk::ScopeKHR::WORKGROUP
        && property.a_type == vk::ComponentTypeKHR::FLOAT16
        && property.b_type == vk::ComponentTypeKHR::FLOAT16
        && property.c_type == vk::ComponentTypeKHR::FLOAT32
        && property.result_type == vk::ComponentTypeKHR::FLOAT32
        && property.saturating_accumulation == 0
}

pub(super) fn generic_property(property: &FlexibleDimensions) -> bool {
    f16_f32(property)
        && property.workgroup_invocations == 256
        && property.m_granularity == 32
        && property.n_granularity == 32
        && property.k_granularity == 16
}

pub(super) fn q64_property(property: &FlexibleDimensions) -> bool {
    f16_f32(property)
        && property.workgroup_invocations == 128
        && [
            property.m_granularity,
            property.n_granularity,
            property.k_granularity,
        ]
        .into_iter()
        .all(|granularity| granularity != 0 && 64_u32.is_multiple_of(granularity))
}

pub(super) fn generic_supported(
    features: &Features,
    properties: &Properties,
    dimensions: &[FlexibleDimensions],
) -> bool {
    common_features(features)
        && properties.max_workgroup_size >= 256
        && properties.max_dimension >= 128
        && dimensions.iter().any(generic_property)
}

pub(super) fn q64_supported(
    features: &Features,
    properties: &Properties,
    dimensions: &[FlexibleDimensions],
) -> bool {
    common_features(features)
        && features.reductions != 0
        && features.conversions != 0
        && properties.max_workgroup_size >= 128
        && properties.max_dimension >= 128
        && dimensions.iter().any(q64_property)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn features() -> Features {
        Features {
            workgroup_scope: vk::TRUE,
            flexible_dimensions: vk::TRUE,
            reductions: vk::TRUE,
            conversions: vk::TRUE,
            per_element_operations: vk::TRUE,
            tensor_addressing: vk::TRUE,
            ..Features::default()
        }
    }

    fn property() -> FlexibleDimensions {
        FlexibleDimensions {
            m_granularity: 16,
            n_granularity: 16,
            k_granularity: 16,
            a_type: vk::ComponentTypeKHR::FLOAT16,
            b_type: vk::ComponentTypeKHR::FLOAT16,
            c_type: vk::ComponentTypeKHR::FLOAT32,
            result_type: vk::ComponentTypeKHR::FLOAT32,
            scope: vk::ScopeKHR::WORKGROUP,
            workgroup_invocations: 128,
            ..FlexibleDimensions::default()
        }
    }

    fn properties() -> Properties {
        Properties {
            max_workgroup_size: 128,
            max_dimension: 128,
            reserved_shared_memory: 8192,
            ..Properties::default()
        }
    }

    #[test]
    fn common_contract_requires_every_shared_matrix2_feature() {
        let complete = features();
        assert!(common_features(&complete));
        for missing in 0..4 {
            let mut candidate = complete;
            match missing {
                0 => candidate.workgroup_scope = 0,
                1 => candidate.flexible_dimensions = 0,
                2 => candidate.per_element_operations = 0,
                _ => candidate.tensor_addressing = 0,
            }
            assert!(!common_features(&candidate));
        }
    }

    #[test]
    fn q64_requires_exact_types_scope_wg_and_divisible_granularity() {
        let complete = property();
        assert!(q64_property(&complete));
        assert!(!q64_property(&FlexibleDimensions {
            workgroup_invocations: 256,
            ..complete
        }));
        assert!(!q64_property(&FlexibleDimensions {
            saturating_accumulation: vk::TRUE,
            ..complete
        }));
        assert!(!q64_property(&FlexibleDimensions {
            a_type: vk::ComponentTypeKHR::FLOAT32,
            ..complete
        }));
        assert!(!q64_property(&FlexibleDimensions {
            scope: vk::ScopeKHR::SUBGROUP,
            ..complete
        }));
        assert!(!q64_property(&FlexibleDimensions {
            m_granularity: 128,
            ..complete
        }));
        assert!(!q64_property(&FlexibleDimensions {
            k_granularity: 0,
            ..complete
        }));
    }

    #[test]
    fn q64_contract_rejects_missing_reduction_and_resource_properties() {
        let complete = features();
        assert!(q64_supported(&complete, &properties(), &[property()]));
        assert!(!q64_supported(
            &Features {
                reductions: 0,
                ..complete
            },
            &properties(),
            &[property()]
        ));
        assert!(!q64_supported(
            &Features {
                conversions: 0,
                ..complete
            },
            &properties(),
            &[property()]
        ));
        assert!(!q64_supported(
            &complete,
            &Properties {
                max_workgroup_size: 127,
                ..properties()
            },
            &[property()]
        ));
        assert!(!q64_supported(
            &complete,
            &Properties {
                max_dimension: 127,
                ..properties()
            },
            &[property()]
        ));
        assert!(!q64_supported(&complete, &properties(), &[]));
    }

    #[test]
    fn conversion_feature_is_not_required_by_the_generic_matrix2_family() {
        let generic = FlexibleDimensions {
            m_granularity: 32,
            n_granularity: 32,
            k_granularity: 16,
            workgroup_invocations: 256,
            ..property()
        };
        let no_conversion = Features {
            conversions: 0,
            ..features()
        };
        let resources = Properties {
            max_workgroup_size: 256,
            ..properties()
        };
        assert!(generic_supported(&no_conversion, &resources, &[generic]));
        assert!(!q64_supported(&no_conversion, &resources, &[property()]));
    }
}
