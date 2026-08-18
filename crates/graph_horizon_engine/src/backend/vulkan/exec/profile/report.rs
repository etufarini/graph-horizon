/*
 * Vulkan profiler report: formats phase-separated totals on stderr for
 * performance investigations. It owns no profiler state, Vulkan resources,
 * or kernel classification.
 */

use super::category::{Category, Phase};
use super::summary::PhaseTotals;

pub(super) fn write(totals: &PhaseTotals, allocations: [u64; 3], allocation_ms: f64) {
    eprintln!(
        "vulkan_profile allocation_count_bytes_host={allocations:?} allocation_ms={allocation_ms:.3}"
    );
    for phase in Phase::ALL {
        let totals = &totals[phase as usize];
        if totals.counts[0] == 0 {
            continue;
        }
        let total_ms = totals.gpu_ms[0].max(f64::MIN_POSITIVE);
        let accounted = totals.gpu_ms[1] / total_ms * 100.0;
        eprintln!(
            "vulkan_profile phase={} counts_command_dispatch_barrier={:?} gpu_total_kernel_ms={:?} residual_ms={:.3} accounted_pct={:.2} cpu_record_submit_wait_descriptor_ms={:?}",
            phase.name(),
            totals.counts,
            totals.gpu_ms,
            (totals.gpu_ms[0] - totals.gpu_ms[1]).max(0.0),
            accounted,
            totals.cpu_ms,
        );
        for category in Category::ALL {
            let index = category as usize;
            if totals.category_count[index] == 0 {
                continue;
            }
            let pct = totals.category_ms[index] / total_ms * 100.0;
            let count = totals.category_count[index];
            let average = totals.category_ms[index] / count.max(1) as f64;
            let kernel = totals.category_kernel[index].map_or("none", |kernel| kernel.name());
            eprintln!(
                "vulkan_profile phase={} category={} kernel={} gpu_ms={:.3} pct={:.2} invocations={} avg_ms={:.6} max_groups_xy={:?}",
                phase.name(),
                category.name(),
                kernel,
                totals.category_ms[index],
                pct,
                count,
                average,
                totals.category_groups[index],
            );
        }
        let mut layers = totals
            .layer_ms
            .iter()
            .zip(totals.layer_attention_count)
            .filter_map(|(&ms, count)| (count > 0).then_some(ms))
            .collect::<Vec<_>>();
        layers.sort_by(f64::total_cmp);
        if let (Some(min), Some(max)) = (layers.first(), layers.last()) {
            eprintln!(
                "vulkan_profile phase={} layer_count={} layer_gpu_min_median_max_ms=[{:.3},{:.3},{:.3}]",
                phase.name(),
                layers.len(),
                min,
                layers[layers.len() / 2],
                max,
            );
        }
    }
}
