/*
 * Vulkan profiler report: formats phase-separated totals on stderr for
 * performance investigations. It owns no profiler state, Vulkan resources,
 * or kernel classification.
 */

use super::Totals;
use super::category::{Category, Phase};

pub(super) fn write(totals: &[Totals; Phase::COUNT], allocations: [u64; 3]) {
    eprintln!("vulkan_profile allocation_count_bytes_host={allocations:?}");
    for phase in Phase::ALL {
        let totals = &totals[phase as usize];
        if totals.counts[0] == 0 {
            continue;
        }
        let total_ms = totals.gpu_ms[0].max(f64::MIN_POSITIVE);
        let accounted = totals.gpu_ms[1] / total_ms * 100.0;
        eprintln!(
            "vulkan_profile phase={} counts_command_dispatch_barrier={:?} gpu_total_kernel_ms={:?} residual_ms={:.3} accounted_pct={:.2} cpu_record_submit_wait_ms={:?}",
            phase.name(),
            totals.counts,
            totals.gpu_ms,
            (totals.gpu_ms[0] - totals.gpu_ms[1]).max(0.0),
            accounted,
            totals.cpu_ms,
        );
        for category in Category::ALL {
            let index = category as usize;
            let pct = totals.category_ms[index] / total_ms * 100.0;
            eprintln!(
                "vulkan_profile phase={} category={} gpu_ms={:.3} pct={:.2}",
                phase.name(),
                category.name(),
                totals.category_ms[index],
                pct
            );
        }
    }
}
