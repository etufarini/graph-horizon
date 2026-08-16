/*
 * Vulkan sparse-prefill policy: exposes the fixed Phase 16 mask as one opt-in
 * preset. It records no command, owns no GPU resource, and leaves dense exact
 * as the default and fallback.
 */

use std::sync::OnceLock;

const LATE_LAYERS: u32 = ((1 << 26) - 1) ^ ((1 << 19) - 1);

#[derive(Clone, Copy)]
pub(super) struct SparseConfig {
    pub(super) window: u32,
    pub(super) global_stride_blocks: u32,
    pub(super) layer_mask: u32,
    pub(super) context_threshold: u32,
}

pub(super) fn production() -> Option<SparseConfig> {
    static CONFIG: OnceLock<Option<SparseConfig>> = OnceLock::new();
    *CONFIG.get_or_init(|| {
        matches!(
            std::env::var("GRAPH_HORIZON_PREFILL_SPARSE")
                .ok()
                .as_deref(),
            Some("hybrid" | "1" | "true" | "yes")
        )
        .then_some(SparseConfig {
            window: 4_096,
            global_stride_blocks: 16,
            layer_mask: LATE_LAYERS,
            context_threshold: 16_384,
        })
    })
}
