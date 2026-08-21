/*
 * Vulkan profiler aggregation records: hold per-command marks and accumulated
 * phase/category/layer totals. Vulkan query ownership and report formatting stay
 * in sibling modules.
 */

use super::category::{Category, Phase};
use crate::backend::vulkan::pipeline::Kernel;

pub(super) const LAYER_LIMIT: usize = 128;

#[derive(Clone, Copy)]
pub(super) struct Mark {
    pub(super) category: Category,
    pub(super) kernel: Kernel,
    pub(super) start: u32,
    pub(super) end: u32,
    pub(super) groups: [u32; 2],
    pub(super) layer: Option<u8>,
}

pub(super) struct Totals {
    pub(super) counts: [u64; 3],
    pub(super) gpu_ms: [f64; 2],
    pub(super) category_ms: [f64; Category::COUNT],
    pub(super) category_count: [u64; Category::COUNT],
    pub(super) category_kernel: [Option<Kernel>; Category::COUNT],
    pub(super) category_groups: [[u32; 2]; Category::COUNT],
    pub(super) layer_ms: [f64; LAYER_LIMIT],
    pub(super) layer_attention_count: [u64; LAYER_LIMIT],
    pub(super) cpu_ms: [f64; 4],
}

impl Default for Totals {
    fn default() -> Self {
        Self {
            counts: [0; 3],
            gpu_ms: [0.0; 2],
            category_ms: [0.0; Category::COUNT],
            category_count: [0; Category::COUNT],
            category_kernel: [None; Category::COUNT],
            category_groups: [[0; 2]; Category::COUNT],
            layer_ms: [0.0; LAYER_LIMIT],
            layer_attention_count: [0; LAYER_LIMIT],
            cpu_ms: [0.0; 4],
        }
    }
}

impl Totals {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn add_command(
        &mut self,
        gpu_ms: f64,
        dispatches: u64,
        barriers: u64,
        cpu_ms: [f64; 4],
        marks: &[Mark],
        values: &[u64],
        tick_ms: f64,
    ) {
        let mut kernel_ms = 0.0;
        for mark in marks {
            let elapsed = values[mark.end as usize].wrapping_sub(values[mark.start as usize])
                as f64
                * tick_ms;
            let index = mark.category as usize;
            kernel_ms += elapsed;
            self.category_ms[index] += elapsed;
            self.category_count[index] += 1;
            self.category_kernel[index] = Some(mark.kernel);
            for axis in 0..2 {
                self.category_groups[index][axis] =
                    self.category_groups[index][axis].max(mark.groups[axis]);
            }
            if let Some(layer) = mark
                .layer
                .map(usize::from)
                .filter(|&layer| layer < LAYER_LIMIT)
            {
                self.layer_ms[layer] += elapsed;
                self.layer_attention_count[layer] +=
                    u64::from(mark.category == Category::Attention);
            }
        }
        self.counts[0] += 1;
        self.counts[1] += dispatches;
        self.counts[2] += barriers;
        self.gpu_ms[0] += gpu_ms;
        self.gpu_ms[1] += kernel_ms;
        for (total, value) in self.cpu_ms.iter_mut().zip(cpu_ms) {
            *total += value;
        }
    }
}

pub(super) type PhaseTotals = [Totals; Phase::COUNT];
