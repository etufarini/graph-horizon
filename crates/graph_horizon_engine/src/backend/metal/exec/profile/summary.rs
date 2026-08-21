/*
 * Metal profiler aggregation records
 * Holds sampled category runs and accumulates validated command totals. It owns
 * no Metal resources, operation classification, or output formatting.
 */

use color_eyre::eyre::{Result, eyre};
use objc2_metal::MTLCounterResultTimestamp;

use super::category::Category;

const ERROR_VALUE: u64 = u64::MAX;

pub(super) struct Mark {
    pub(super) category: Category,
    pub(super) start: usize,
    pub(super) end: usize,
    pub(super) dispatches: u64,
}

#[derive(Clone, Copy)]
pub(super) struct Totals {
    pub(super) commands: u64,
    pub(super) dispatches: u64,
    pub(super) gpu_ms: f64,
    pub(super) kernel_ms: f64,
    pub(super) category_ms: [f64; Category::COUNT],
    pub(super) category_count: [u64; Category::COUNT],
    pub(super) cpu_ms: [f64; 3],
}

impl Default for Totals {
    fn default() -> Self {
        Self {
            commands: 0,
            dispatches: 0,
            gpu_ms: 0.0,
            kernel_ms: 0.0,
            category_ms: [0.0; Category::COUNT],
            category_count: [0; Category::COUNT],
            cpu_ms: [0.0; 3],
        }
    }
}

impl Totals {
    pub(super) fn add_command(
        &mut self,
        gpu_ms: f64,
        cpu_ms: [f64; 3],
        marks: &[Mark],
        values: &[MTLCounterResultTimestamp],
    ) -> Result<()> {
        for mark in marks {
            let start = values[mark.start].timestamp;
            let end = values[mark.end].timestamp;
            if start == ERROR_VALUE || end == ERROR_VALUE || end < start {
                return Err(eyre!("metal profile: invalid timestamp value"));
            }
            let ms = (end - start) as f64 / 1_000_000.0;
            let index = mark.category as usize;
            self.dispatches += mark.dispatches;
            self.kernel_ms += ms;
            self.category_ms[index] += ms;
            self.category_count[index] += mark.dispatches;
        }
        self.commands += 1;
        self.gpu_ms += gpu_ms;
        for (total, value) in self.cpu_ms.iter_mut().zip(cpu_ms) {
            *total += value;
        }
        Ok(())
    }
}
