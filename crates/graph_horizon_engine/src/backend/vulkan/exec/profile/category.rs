/*
 * Vulkan profiler kernel categories: classifies one recorded dispatch and
 * preserves the fixed seven-matmul order used to separate prefill projections
 * from MLP work. It owns no timestamps, resources, or output formatting.
 */

use crate::backend::vulkan::pipeline::Kernel;

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Category {
    Attention,
    ProjectionQkv,
    ProjectionOutput,
    Mlp,
    Matmul,
    Normalization,
    Rope,
    KvCache,
    Elementwise,
    Embedding,
    Logits,
    Reduction,
}

impl Category {
    pub(super) const COUNT: usize = 12;
    pub(super) const ALL: [Self; Self::COUNT] = [
        Self::Attention,
        Self::ProjectionQkv,
        Self::ProjectionOutput,
        Self::Mlp,
        Self::Matmul,
        Self::Normalization,
        Self::Rope,
        Self::KvCache,
        Self::Elementwise,
        Self::Embedding,
        Self::Logits,
        Self::Reduction,
    ];

    pub(super) const fn name(self) -> &'static str {
        match self {
            Self::Attention => "attention",
            Self::ProjectionQkv => "projection_qkv",
            Self::ProjectionOutput => "projection_output",
            Self::Mlp => "mlp",
            Self::Matmul => "matmul_other",
            Self::Normalization => "normalization",
            Self::Rope => "rope",
            Self::KvCache => "kv_cache",
            Self::Elementwise => "elementwise",
            Self::Embedding => "embedding",
            Self::Logits => "logits",
            Self::Reduction => "reduction",
        }
    }
}

const fn direct(kernel: Kernel) -> Category {
    match kernel {
        Kernel::AttentionDecode
        | Kernel::AttentionDecodeWide
        | Kernel::AttentionDecode1024
        | Kernel::AttentionPrefill
        | Kernel::AttentionPrefillWide
        | Kernel::AttentionDecodeInt8
        | Kernel::AttentionPrefillInt8 => Category::Attention,
        Kernel::MatmulF16
        | Kernel::MatmulQ4KTiled
        | Kernel::MatmulQ5K
        | Kernel::MatmulQ6K
        | Kernel::MatmulQ4KBatchF16Out
        | Kernel::MatmulQ6KBatchF16Out
        | Kernel::MatmulQ4KCoopmatF16Out
        | Kernel::QuantAQ8F16
        | Kernel::MatmulQ4KMmvqF16Out => Category::Matmul,
        Kernel::Logits | Kernel::LogitsQ4K | Kernel::LogitsQ5K | Kernel::LogitsQ6K => {
            Category::Logits
        }
        Kernel::EmbedF16 | Kernel::EmbedQ4K | Kernel::EmbedQ5K | Kernel::EmbedQ6K => {
            Category::Embedding
        }
        Kernel::RmsNormX => Category::Normalization,
        Kernel::Rope => Category::Rope,
        Kernel::Residual | Kernel::SiluMul => Category::Elementwise,
        Kernel::KvWrite | Kernel::KvWriteInt8 => Category::KvCache,
        Kernel::Argmax | Kernel::TopkPartial => Category::Reduction,
    }
}

const fn is_batched_matmul(kernel: Kernel) -> bool {
    matches!(
        kernel,
        Kernel::MatmulQ4KBatchF16Out
            | Kernel::MatmulQ6KBatchF16Out
            | Kernel::MatmulQ4KCoopmatF16Out
    )
}

pub(super) const fn is_prefill_attention(kernel: Kernel) -> bool {
    matches!(
        kernel,
        Kernel::AttentionPrefill | Kernel::AttentionPrefillWide | Kernel::AttentionPrefillInt8
    )
}

pub(super) fn profiled(kernel: Kernel, batched: &mut usize) -> Category {
    if !is_batched_matmul(kernel) {
        return direct(kernel);
    }
    // Every prefill layer records Q, K, V, output, gate, up, then down.
    let category = match *batched % 7 {
        0..=2 => Category::ProjectionQkv,
        3 => Category::ProjectionOutput,
        _ => Category::Mlp,
    };
    *batched += 1;
    category
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batch_sequence_splits_attention_projections_and_mlp() {
        let mut slot = 0;
        let names = (0..7)
            .map(|_| profiled(Kernel::MatmulQ4KBatchF16Out, &mut slot).name())
            .collect::<Vec<_>>();
        assert_eq!(
            names,
            [
                "projection_qkv",
                "projection_qkv",
                "projection_qkv",
                "projection_output",
                "mlp",
                "mlp",
                "mlp"
            ]
        );
    }
}
