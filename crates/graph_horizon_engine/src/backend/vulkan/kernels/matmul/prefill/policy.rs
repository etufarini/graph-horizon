/*
 * graph_horizon_engine — Vulkan prefill matmul selection policy
 * Owns the measured projection shapes for optional cooperative matmul paths.
 * It records no command and owns no GPU resource.
 */

pub(super) fn coopmat_shape(in_dim: u32) -> bool {
    matches!(in_dim, 3072 | 4096 | 9216)
}

pub(super) fn q4_metadata_shape(out_dim: u32) -> bool {
    matches!(out_dim, 3072 | 4096 | 9216)
}

pub(super) fn matrix2_shape(in_dim: u32, out_dim: u32) -> bool {
    (coopmat_shape(in_dim) && matches!(out_dim, 1024 | 3072 | 4096 | 9216))
        // Measured 4096-wide canonical Q4_K gate/up tensors tile exactly.
        || (in_dim == 4096 && out_dim == 14336)
        // Large-K qualification is intentionally exact; unmeasured widths fall back.
        || (in_dim == 14336 && out_dim == 4096)
        // The 14B family is block-aligned and uses the same runtime-dimension ABI.
        || (in_dim == 5120 && matches!(out_dim, 1024 | 4096 | 16384))
        || (in_dim == 4096 && out_dim == 5120)
        || (in_dim == 16384 && out_dim == 5120)
}
