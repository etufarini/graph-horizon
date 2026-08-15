/*
 * graph_horizon_engine — Vulkan native-weight eligibility
 * Resolves whether a tensor receives an additional execution-native region and
 * reports its exact persistent byte cost to both preflight and upload.
 * It performs no conversion, allocation, or execution dispatch.
 */

#[cfg(feature = "vulkan")]
use crate::gguf::tensor_index::GgmlType;
use crate::gguf::tensor_index::TensorInfo;

#[cfg(feature = "vulkan")]
pub(super) fn bytes(info: &TensorInfo, native_matrix2: bool) -> Option<u64> {
    if !native_matrix2
        || info.ggml_type != GgmlType::Q4_K
        || info.dims != [3072, 9216]
        || !enabled(info)
    {
        return None;
    }
    info.element_count()?.checked_mul(2)
}

#[cfg(feature = "vulkan")]
fn enabled(info: &TensorInfo) -> bool {
    let flag = |name| {
        matches!(
            std::env::var(name).ok().as_deref(),
            Some("1" | "true" | "yes")
        )
    };
    if !matches!(
        std::env::var("GRAPH_HORIZON_PREFILL_MATMUL_MATRIX2")
            .ok()
            .as_deref(),
        None | Some("1" | "true" | "yes")
    ) {
        return false;
    }
    let mlp = flag("GRAPH_HORIZON_PREFILL_PREDECODE_MLP");
    (info.name.ends_with(".ffn_gate.weight")
        && (mlp || flag("GRAPH_HORIZON_PREFILL_PREDECODE_GATE")))
        || (info.name.ends_with(".ffn_up.weight")
            && (mlp || flag("GRAPH_HORIZON_PREFILL_PREDECODE_UP")))
}

#[cfg(feature = "vulkan-hybrid")]
pub(super) fn bytes(_info: &TensorInfo, _native_matrix2: bool) -> Option<u64> {
    None
}
