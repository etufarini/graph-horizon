/*
 * Vulkan profiler aggregation records: hold per-command marks and accumulated
 * phase/category/layer totals. Vulkan query ownership and report formatting stay
 * in sibling modules.
 */

use super::category::{Category, Phase};
use crate::backend::vulkan::pipeline::Kernel;

pub(super) const LAYER_LIMIT: usize = 128;

#[derive(Clone, Copy)]
pub(super) enum MatmulPath {
    Q4Coopmat,
    Q4Metadata,
    Q6Coopmat,
    Q4Matrix2,
    Q6Matrix2,
    Q4Fallback,
    Q6Fallback,
}

impl MatmulPath {
    pub(super) const COUNT: usize = 7;
    pub(super) const ALL: [Self; Self::COUNT] = [
        Self::Q4Coopmat,
        Self::Q4Metadata,
        Self::Q6Coopmat,
        Self::Q4Matrix2,
        Self::Q6Matrix2,
        Self::Q4Fallback,
        Self::Q6Fallback,
    ];

    pub(super) const fn name(self) -> &'static str {
        match self {
            Self::Q4Coopmat => "matmul_q4k_coopmat_f16",
            Self::Q4Metadata => "matmul_q4k_coopmat_metadata_f16",
            Self::Q6Coopmat => "matmul_q6k_coopmat_f16",
            Self::Q4Matrix2 => "matmul_q4k_matrix2_wg256_m32_n32_k128",
            Self::Q6Matrix2 => "matmul_q6k_matrix2_wg256_m32_n32_k128",
            Self::Q4Fallback => "matmul_q4k_batch_f16",
            Self::Q6Fallback => "matmul_q6k_batch_f16",
        }
    }

    const fn from_kernel(kernel: Kernel) -> Option<Self> {
        match kernel {
            Kernel::MatmulQ4KCoopmatF16Out => Some(Self::Q4Coopmat),
            Kernel::MatmulQ4KCoopmatMetadataF16Out => Some(Self::Q4Metadata),
            Kernel::MatmulQ6KCoopmatF16Out => Some(Self::Q6Coopmat),
            Kernel::MatmulQ4KMatrix2F16Out => Some(Self::Q4Matrix2),
            Kernel::MatmulQ6KMatrix2F16Out => Some(Self::Q6Matrix2),
            Kernel::MatmulQ4KBatchF16Out => Some(Self::Q4Fallback),
            Kernel::MatmulQ6KBatchF16Out => Some(Self::Q6Fallback),
            _ => None,
        }
    }
}

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
    pub(super) matmul_path_ms: [[f64; MatmulPath::COUNT]; Category::COUNT],
    pub(super) matmul_path_count: [[u64; MatmulPath::COUNT]; Category::COUNT],
    pub(super) layer_ms: [f64; LAYER_LIMIT],
    pub(super) layer_attention_count: [u64; LAYER_LIMIT],
    pub(super) gap_ms: f64,
    pub(super) gap_count: u64,
    pub(super) gap_max_ms: f64,
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
            matmul_path_ms: [[0.0; MatmulPath::COUNT]; Category::COUNT],
            matmul_path_count: [[0; MatmulPath::COUNT]; Category::COUNT],
            layer_ms: [0.0; LAYER_LIMIT],
            layer_attention_count: [0; LAYER_LIMIT],
            gap_ms: 0.0,
            gap_count: 0,
            gap_max_ms: 0.0,
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
        let mut previous_end = None;
        for mark in marks {
            if let Some(end) = previous_end {
                let gap = values[mark.start as usize].wrapping_sub(values[end]) as f64 * tick_ms;
                if gap >= 0.05 {
                    self.gap_ms += gap;
                    self.gap_count += 1;
                    self.gap_max_ms = self.gap_max_ms.max(gap);
                }
            }
            let elapsed = values[mark.end as usize].wrapping_sub(values[mark.start as usize])
                as f64
                * tick_ms;
            let index = mark.category as usize;
            kernel_ms += elapsed;
            self.category_ms[index] += elapsed;
            self.category_count[index] += 1;
            self.category_kernel[index] = Some(mark.kernel);
            if let Some(path) = MatmulPath::from_kernel(mark.kernel) {
                self.matmul_path_ms[index][path as usize] += elapsed;
                self.matmul_path_count[index][path as usize] += 1;
            }
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
            previous_end = Some(mark.end as usize);
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefill_matmul_paths_are_disjoint() {
        assert!(matches!(
            MatmulPath::from_kernel(Kernel::MatmulQ4KCoopmatF16Out),
            Some(MatmulPath::Q4Coopmat)
        ));
        assert!(matches!(
            MatmulPath::from_kernel(Kernel::MatmulQ6KCoopmatF16Out),
            Some(MatmulPath::Q6Coopmat)
        ));
        assert!(matches!(
            MatmulPath::from_kernel(Kernel::MatmulQ4KMatrix2F16Out),
            Some(MatmulPath::Q4Matrix2)
        ));
        assert!(matches!(
            MatmulPath::from_kernel(Kernel::MatmulQ6KMatrix2F16Out),
            Some(MatmulPath::Q6Matrix2)
        ));
        assert!(matches!(
            MatmulPath::from_kernel(Kernel::MatmulQ4KBatchF16Out),
            Some(MatmulPath::Q4Fallback)
        ));
        assert!(matches!(
            MatmulPath::from_kernel(Kernel::MatmulQ6KBatchF16Out),
            Some(MatmulPath::Q6Fallback)
        ));
        assert!(MatmulPath::from_kernel(Kernel::MatmulQ4KTiled).is_none());
    }

    #[test]
    fn matmul_path_time_sums_to_parent_category() {
        let marks = [
            (Kernel::MatmulQ4KCoopmatF16Out, 0, 1),
            (Kernel::MatmulQ4KBatchF16Out, 1, 2),
            (Kernel::MatmulQ6KBatchF16Out, 2, 3),
        ]
        .map(|(kernel, start, end)| Mark {
            category: Category::MlpDown,
            kernel,
            start,
            end,
            groups: [1, 1],
            layer: None,
        });
        let mut totals = Totals::default();
        totals.add_command(61.0, 3, 0, [0.0; 4], &marks, &[0, 10, 30, 60], 1.0);
        let category_ms = totals.category_ms[Category::MlpDown as usize];
        let path_ms: f64 = totals.matmul_path_ms[Category::MlpDown as usize]
            .iter()
            .sum();
        assert_eq!(category_ms, 60.0);
        assert_eq!(path_ms, category_ms);
    }
}
