/*
 * Metal profiler report
 * Formats phase and category aggregates on stderr. It owns no sampling,
 * classification, runtime state, or performance decisions.
 */

use super::{category, summary::Totals};

pub(super) fn write(totals: &[Totals; category::Phase::COUNT]) {
    for phase in category::Phase::ALL {
        let total = &totals[phase as usize];
        if total.commands == 0 {
            continue;
        }
        let residual = total.gpu_ms - total.kernel_ms;
        let accounted = if total.gpu_ms > 0.0 {
            total.kernel_ms * 100.0 / total.gpu_ms
        } else {
            0.0
        };
        eprintln!(
            "metal_profile phase={} commands={} dispatches={} gpu_total_ms={:.3} kernel_ms={:.3} residual_ms={:.3} accounted_pct={:.2} cpu_record_submit_wait_ms={:?}",
            phase.name(),
            total.commands,
            total.dispatches,
            total.gpu_ms,
            total.kernel_ms,
            residual,
            accounted,
            total.cpu_ms,
        );
        for category in category::Category::ALL {
            let index = category as usize;
            if total.category_count[index] == 0 {
                continue;
            }
            eprintln!(
                "metal_profile phase={} category={} gpu_ms={:.3} kernel_pct={:.2} invocations={} avg_ms={:.6}",
                phase.name(),
                category.name(),
                total.category_ms[index],
                total.category_ms[index] * 100.0 / total.kernel_ms,
                total.category_count[index],
                total.category_ms[index] / total.category_count[index] as f64,
            );
        }
    }
}
