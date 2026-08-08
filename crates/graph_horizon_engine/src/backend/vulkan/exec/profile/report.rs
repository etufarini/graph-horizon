/*
 * Vulkan profiler report: formats accumulated prefill totals on stderr using
 * the stable diagnostic schema consumed during performance investigations.
 * It owns no profiler state, Vulkan resources, or kernel classification.
 */

use super::Totals;
use super::category::Category;

pub(super) fn write(totals: &Totals) {
    let accounted = totals.gpu_ms[1] / totals.gpu_ms[0].max(f64::MIN_POSITIVE) * 100.0;
    eprintln!(
        "vulkan_profile phase=prefill counts_command_dispatch_barrier={:?} gpu_total_kernel_ms={:?} residual_ms={:.3} accounted_pct={:.2} cpu_record_submit_wait_ms={:?} allocation_count_bytes_host={:?}",
        totals.counts,
        totals.gpu_ms,
        (totals.gpu_ms[0] - totals.gpu_ms[1]).max(0.0),
        accounted,
        totals.cpu_ms,
        totals.allocations
    );
    for category in Category::ALL {
        let index = category as usize;
        let pct = totals.category_ms[index] / totals.gpu_ms[0].max(f64::MIN_POSITIVE) * 100.0;
        eprintln!(
            "vulkan_profile phase=prefill category={} gpu_ms={:.3} pct={:.2}",
            category.name(),
            totals.category_ms[index],
            pct
        );
    }
}
