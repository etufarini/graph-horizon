/*
 * graph_horizon_engine — Vulkan matmul diagnostics
 * Reads the preserved opt-in trace switch once and reports decode or prefill
 * path selection once. It owns no kernel selection or command recording.
 */

use std::sync::{Once, OnceLock};

use crate::backend::vulkan::pipeline::Kernel;

fn enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        matches!(
            std::env::var("GRAPH_HORIZON_LOAD_TRACE").ok().as_deref(),
            Some("1") | Some("true") | Some("yes")
        )
    })
}

pub(crate) fn log_path_once(kernel: Kernel) {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        if enabled() {
            let path = match kernel {
                Kernel::MatmulQ4KTiled => "Q4_K tiled-subgroup",
                Kernel::MatmulQ4KMmvqF16Out => "Q4_K mmvq (int8/dp4a)",
                _ => "per-format GEMV",
            };
            eprintln!("matmul: path={path}");
        }
    });
}

pub(super) fn log_batched_path_once(kernel: Kernel) {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        if enabled() {
            let path = match kernel {
                Kernel::MatmulQ4KMatrix2F16Out => "Q4_K matrix2 (f16-out)",
                Kernel::MatmulQ4KCoopmatF16Out | Kernel::MatmulQ4KCoopmatMetadataF16Out => {
                    "Q4_K coopmat (f16-out)"
                }
                _ => "Q4_K batched f16-out",
            };
            eprintln!("matmul_batched: path={path}");
        }
    });
}
