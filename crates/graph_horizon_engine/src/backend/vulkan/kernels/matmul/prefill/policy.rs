/*
 * graph_horizon_engine — Vulkan prefill matmul selection policy
 * Owns immutable environment switches and the measured projection shapes for
 * optional cooperative matmul paths. It records no command and owns no GPU resource.
 */

use std::sync::OnceLock;

fn enabled(name: &str, flag: &'static OnceLock<bool>) -> bool {
    *flag.get_or_init(|| {
        matches!(
            std::env::var(name).ok().as_deref(),
            None | Some("1" | "true" | "yes")
        )
    })
}

pub(super) fn coopmat_enabled() -> bool {
    static FLAG: OnceLock<bool> = OnceLock::new();
    enabled("GRAPH_HORIZON_PREFILL_COOPMAT", &FLAG)
}

pub(super) fn q4_metadata_enabled() -> bool {
    static FLAG: OnceLock<bool> = OnceLock::new();
    enabled("GRAPH_HORIZON_PREFILL_Q4_METADATA", &FLAG)
}

pub(super) fn matrix2_enabled() -> bool {
    static FLAG: OnceLock<bool> = OnceLock::new();
    enabled("GRAPH_HORIZON_PREFILL_MATMUL_MATRIX2", &FLAG)
}

pub(super) fn coopmat_shape(in_dim: u32) -> bool {
    matches!(in_dim, 3072 | 4096 | 9216)
}

pub(super) fn q4_metadata_shape(out_dim: u32) -> bool {
    matches!(out_dim, 3072 | 4096 | 9216)
}

pub(super) fn matrix2_shape(in_dim: u32, out_dim: u32) -> bool {
    (coopmat_shape(in_dim) && matches!(out_dim, 1024 | 3072 | 4096 | 9216))
        // Large-K qualification is intentionally exact; unmeasured widths fall back.
        || (in_dim == 14336 && out_dim == 4096)
}
